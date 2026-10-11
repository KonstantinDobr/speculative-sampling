"""Этап 1 дообучения драфта: ответы таргета сохраняются на диск, чтобы таргет и драфт не делили GPU.

Каждый пример — последовательность токенов «промпт + продолжение». Для каждой позиции продолжения
сохраняется распределение таргета на следующий токен. Откуда берётся продолжение (--source):

    generated — его сэмплирует сам таргет (с --temperature и --top-p, как в SpS): драфт учится на тех
                контекстах, которые встретит при спекулятивном декодировании
    reference — настоящий текст из датасета (teacher forcing): генерации нет, в разы дешевле

HumanEval (prompt + canonical_solution) всегда идёт как reference: по нему finetune.py считает метрики.

От распределения хранится top-k токенов: ln p при temperature = 1 (для KL) и вероятности сэмплера
(с temperature и top-p, для α). Всё, что в top-k не вошло, считается одним «остальным» токеном.

Датасет — jsonl либо с полем text, как further_training.jsonl (первые --prompt-tokens токенов — промпт,
дальше продолжение), либо с полями prompt и reference, как eval_set.jsonl. Результат:
<out-dir>/train_<source>.pt и <out-dir>/humaneval.pt.

Каждые --save-every батчей сделанное пишется в <файл кэша>.part: если задачу сняли (лимит времени), тот же
запуск продолжит с этого места. С --limit кэш уходит в отдельную папку cache/<pair>_limit<N>.
"""
import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, DynamicCache

from eval_domains import load_rows
from eval_humaneval import load_humaneval
from specdec import (DEFAULT_PAIR, DEVICE, MODEL_PAIRS, describe, load_model, make_generator, resolve_model,
                     sample, to_probs)

SOURCES = ["generated", "reference"]
# HumanEval — первым: он считается за минуты и не должен пропасть, если train не уложится в лимит времени
PARTS = ["humaneval", "train"]
# Параметры, от которых зависит содержимое кэша: недописанный кэш продолжается только при тех же
KEY_ARGS = ["max_new_tokens", "top_k", "temperature", "top_p", "batch_size", "dtype", "seed"]


def parse_args():
    parser = argparse.ArgumentParser(description="Кэш распределений таргета для дообучения драфта")
    parser.add_argument("--pair", choices=MODEL_PAIRS, default=DEFAULT_PAIR, help="пара (таргет, драфт)")
    parser.add_argument("--target", default=None, help="переопределяет таргет из --pair")
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=PARTS, help="какие кэши строить")
    parser.add_argument("--train", default="further_training.jsonl", help="jsonl с text или prompt + reference")
    parser.add_argument("--humaneval", default="openai/openai_humaneval",
                        help="имя на HF Hub, локальная папка или файл .jsonl/.jsonl.gz")
    parser.add_argument("--source", choices=SOURCES, default="generated", help="продолжение для train")
    parser.add_argument("--domains", nargs="+", default=None, help="только эти домены train")
    parser.add_argument("--limit", type=int, default=None, help="взять только первые N примеров")
    parser.add_argument("--prompt-tokens", type=int, default=128, help="длина промпта для примеров с полем text")
    parser.add_argument("--max-prompt-tokens", type=int, default=512, help="от поля prompt остаётся хвост")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="токенов продолжения на пример")
    parser.add_argument("--top-k", type=int, default=64, help="сколько токенов распределения хранить")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=512,
                        help="сколько позиций за раз проходит через lm_head (векторы по размеру словаря — память)")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25, help="батчей между сохранениями недописанного кэша")
    parser.add_argument("--out-dir", default=None,
                        help="по умолчанию cache/<pair>, а с --limit — cache/<pair>_limit<N>")
    return parser.parse_args()


def encode(tokenizer, row: dict, source: str, args) -> tuple[list[int], int]:
    """Токены примера и число токенов промпта. Для generated продолжения ещё нет — только промпт."""
    if "text" in row:
        # Сплошной текст: промпт — его начало, но не больше половины (в коротком тексте тоже нужно продолжение)
        ids = tokenizer(row["text"])["input_ids"]
        n_prompt = min(args.prompt_tokens, len(ids) // 2)
    else:
        # Как в eval_domains: токенизируем prompt + reference целиком (на стыке токены могут склеиться),
        # продолжение — токены, которые целиком лежат в reference
        encoding = tokenizer(row["prompt"] + row.get("reference", ""), return_offsets_mapping=True)
        starts = [start for start, _ in encoding["offset_mapping"]]
        n_prompt = next((i for i, start in enumerate(starts) if i > 0 and start >= len(row["prompt"])), len(starts))
        first = max(0, n_prompt - args.max_prompt_tokens)
        ids, n_prompt = encoding["input_ids"][first:], n_prompt - first
    return ids[:n_prompt if source == "generated" else n_prompt + args.max_new_tokens], n_prompt


def collate(sequences: list[torch.Tensor], n_prompt, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Батч с паддингом справа: токены, маска внимания и маска позиций, предсказывающих продолжение."""
    lengths = torch.tensor([len(s) for s in sequences])[:, None]
    input_ids = pad_sequence(sequences, batch_first=True)
    positions = torch.arange(input_ids.shape[1])
    # Позиция i предсказывает токен i + 1
    predict = (positions >= torch.as_tensor(n_prompt)[:, None] - 1) & (positions < lengths - 1)
    return input_ids.to(device), (positions < lengths).long().to(device), predict.to(device)


def hidden_at(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Скрытые состояния только в позициях маски: lm_head по всему батчу при словаре Qwen (~150 тыс.) —
    это гигабайты логитов для промпта и паддинга, которые никому не нужны."""
    return model.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[positions]


@torch.no_grad()
def generate(model, prompts: list[list[int]], max_new_tokens: int, temperature: float, top_p: float,
             eos_token_id: int, generator: torch.Generator | None = None) -> list[list[int]]:
    """Продолжения для батча промптов (вместе с EOS, если он выпал). Батчевый autoregressive_generate."""
    # Паддинг слева: последний токен промпта у всех в одной позиции, новые токены дописываются справа
    width = max(map(len, prompts))
    input_ids = torch.tensor([[0] * (width - len(p)) + list(p) for p in prompts], device=model.device)
    attention_mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=model.device)

    cache = DynamicCache()
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=model.device)
    new_tokens = []
    for _ in range(max_new_tokens):
        logits = model(input_ids, attention_mask=attention_mask, past_key_values=cache, use_cache=True,
                       logits_to_keep=1).logits[:, -1]
        # Дальше в модель идёт только новый токен: всё остальное уже в кэше
        input_ids = sample(to_probs(logits, temperature, top_p), generator)
        new_tokens.append(input_ids)

        finished |= input_ids[:, 0] == eos_token_id
        if finished.all():
            break
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(len(prompts), 1)], dim=1)

    rows = torch.cat(new_tokens, dim=1).tolist()
    return [row[:row.index(eos_token_id) + 1] if eos_token_id in row else row for row in rows]


@torch.no_grad()
def teacher_topk(target, sequences: list[torch.Tensor], n_prompt, args) -> tuple[torch.Tensor, ...]:
    """Top-k таргета в каждой позиции продолжения: индексы, ln p и вероятности сэмплера."""
    lm_head = target.get_output_embeddings()
    # k строго меньше словаря: «остальной» токен не должен оказаться пустым
    k = min(args.top_k, target.config.vocab_size - 1)

    parts = []
    for hidden in hidden_at(target, *collate(sequences, n_prompt, target.device)).split(args.chunk):
        logits = lm_head(hidden).float()
        top = torch.log_softmax(logits, dim=-1).topk(k, dim=-1)
        sampler_probs = to_probs(logits, args.temperature, args.top_p).gather(-1, top.indices)
        parts.append((top.indices.int().cpu(), top.values.half().cpu(), sampler_probs.half().cpu()))
    return tuple(torch.cat(tensors) for tensors in zip(*parts))


def save(data: dict, path: Path):
    # Сначала во временный файл: если задачу снимут посреди записи, прежний файл останется целым
    tmp = path.with_name(path.name + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)


def build(target, examples: list[tuple[list[int], int]], source: str, args, eos_token_id: int,
          generator: torch.Generator | None = None, partial: Path | None = None, key: dict | None = None) -> dict:
    """Кэш для примеров вида (токены, число токенов промпта). partial — файл для недописанного кэша,
    key — параметры запуска: продолжаем только кэш, начатый с теми же."""
    # Самые длинные — первыми: в батче примеры близкой длины, а нехватка памяти выясняется сразу
    examples = sorted(examples, key=lambda example: len(example[0]), reverse=True)
    sequences, n_prompts, topk = [], [], []
    if partial is not None and partial.exists():
        state = torch.load(partial, weights_only=True)
        if state["key"] == key:
            sequences, n_prompts, topk = state["sequences"], state["n_prompts"], state["topk"]
            print(f"Продолжаю {partial}: готово {len(sequences)} примеров из {len(examples)}", flush=True)
            # Другое зерно, чтобы не повторять сэмплы, которые уже выпали в начале прошлого запуска
            if generator is not None:
                generator.manual_seed(args.seed + len(sequences))
        else:
            print(f"{partial} начат с другими параметрами — строю кэш заново", flush=True)
    start_time = time.perf_counter()

    for batch, start in enumerate(range(len(sequences), len(examples), args.batch_size), 1):
        ids, n_prompt = zip(*examples[start:start + args.batch_size])
        if source == "generated":
            new = generate(target, ids, args.max_new_tokens, args.temperature, args.top_p, eos_token_id, generator)
            ids = [prompt + continuation for prompt, continuation in zip(ids, new)]
        ids = [torch.tensor(sequence) for sequence in ids]

        topk.append(teacher_topk(target, ids, n_prompt, args))
        sequences += ids
        n_prompts += n_prompt
        print(f"[{len(sequences)}/{len(examples)}] {time.perf_counter() - start_time:.0f} с", flush=True)
        if partial is not None and batch % args.save_every == 0:
            save({"key": key, "sequences": sequences, "n_prompts": n_prompts, "topk": topk}, partial)

    indices, logp, probs = (torch.cat(tensors) for tensors in zip(*topk))
    return {"tokens": torch.cat(sequences).int(), "lengths": torch.tensor([len(s) for s in sequences]),
            "n_prompt": torch.tensor(n_prompts), "indices": indices, "logp": logp, "probs": probs}


@dataclass
class Batch:
    input_ids: torch.Tensor       # [примеры, токены], паддинг справа
    attention_mask: torch.Tensor
    predict: torch.Tensor         # маска позиций, предсказывающих продолжение; всего их N на батч
    labels: torch.Tensor          # [N] настоящий следующий токен
    indices: torch.Tensor         # [N, k] top-k токенов таргета по убыванию вероятности
    logp: torch.Tensor            # [N, k] их ln p при temperature = 1
    probs: torch.Tensor           # [N, k] их вероятности у сэмплера (temperature и top-p из meta)


class TeacherCache:
    """Кэш в плоских тензорах: пример i — срез токенов и срез строк top-k."""

    def __init__(self, data: dict):
        self.meta = data["meta"]
        self.tokens, self.lengths, self.n_prompt = data["tokens"], data["lengths"], data["n_prompt"]
        self.indices, self.logp, self.probs = data["indices"], data["logp"], data["probs"]

        zero = torch.zeros(1, dtype=torch.long)
        self.token_start = torch.cat([zero, self.lengths.cumsum(0)]).tolist()
        self.row_start = torch.cat([zero, (self.lengths - self.n_prompt).cumsum(0)]).tolist()

    @classmethod
    def load(cls, path: str) -> "TeacherCache":
        return cls(torch.load(path, weights_only=True))

    def __len__(self) -> int:
        return len(self.lengths)

    def batch(self, examples: list[int], device: torch.device) -> Batch:
        sequences = [self.tokens[self.token_start[i]:self.token_start[i + 1]].long() for i in examples]
        # Строки top-k идут в том же порядке, что и позиции в маске predict: по примерам, внутри — по возрастанию
        rows = torch.cat([torch.arange(self.row_start[i], self.row_start[i + 1]) for i in examples])
        input_ids, attention_mask, predict = collate(sequences, self.n_prompt[examples], device)
        return Batch(input_ids, attention_mask, predict, labels=input_ids[:, 1:][predict[:, :-1]],
                     indices=self.indices[rows].to(device, torch.long), logp=self.logp[rows].to(device, torch.float32),
                     probs=self.probs[rows].to(device, torch.float32))

    def batches(self, batch_size: int, device: torch.device, generator: torch.Generator | None = None):
        """Батчи из примеров близкой длины. С generator — в случайном порядке (для обучения)."""
        if generator is None:
            order = self.lengths.argsort(descending=True)
        else:
            # Перемешиваем и сортируем по длине только внутри крупных кусков: состав батчей случайный,
            # а паддинга почти нет
            order = torch.randperm(len(self), generator=generator)
            order = torch.cat([chunk[self.lengths[chunk].argsort()] for chunk in order.split(batch_size * 50)])

        batches = order.split(batch_size)
        if generator is not None:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=generator)]
        for examples in batches:
            yield self.batch(examples.tolist(), device)


def read_rows(part: str, args) -> list[dict]:
    if part == "humaneval":
        rows = [{"prompt": problem["prompt"], "reference": problem["canonical_solution"]}
                for problem in load_humaneval(args.humaneval)]
    else:
        rows = load_rows(args.train, args.domains)
        if rows and not {"text", "prompt"} & rows[0].keys():
            raise SystemExit(f"{args.train}: нужно поле text или поля prompt и reference, а в файле {sorted(rows[0])}")
    return rows[:args.limit]


def main():
    args = parse_args()
    args.target = args.target or MODEL_PAIRS[args.pair][0]
    # Пробный запуск с --limit не должен попасть в папку полного кэша: finetune.py молча взял бы обрезанный
    limit = "" if args.limit is None else f"_limit{args.limit}"
    out_dir = Path(args.out_dir or f"cache/{args.pair}{limit}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Датасеты читаем до загрузки таргета: опечатка в пути не должна стоить минут ожидания
    parts = {part: read_rows(part, args) for part in args.parts}

    target_path = resolve_model(args.target)
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    target = load_model(target_path, dtype, len(tokenizer), DEVICE)
    print(f"Таргет {args.target} на {DEVICE} в {dtype}: {describe(target)}")

    for part, rows in parts.items():
        source = args.source if part == "train" else "reference"
        examples = [encode(tokenizer, row, source, args) for row in rows]
        # Нужен хотя бы один токен контекста и, если продолжение уже есть, хотя бы один токен в нём
        examples = [(ids, n_prompt) for ids, n_prompt in examples
                    if n_prompt > 0 and (source == "generated" or len(ids) > n_prompt)]
        if not examples:
            raise SystemExit(f"{part}: нет ни одного подходящего примера")
        print(f"\n=== {part}: {len(examples)} примеров из {len(rows)}, продолжение — {source} ===")

        path = out_dir / (f"train_{source}.pt" if part == "train" else "humaneval.pt")
        partial = path.with_suffix(".part")
        key = {"target": args.target, "source": source, "examples": len(examples),
               "tokens": sum(len(ids) for ids, _ in examples), **{name: getattr(args, name) for name in KEY_ARGS}}
        cache = build(target, examples, source, args, tokenizer.eos_token_id, make_generator(args.seed), partial, key)
        cache["meta"] = {"target": args.target, "part": part, "source": source, "n_vocab": len(tokenizer),
                         "temperature": args.temperature, "top_p": args.top_p}
        save(cache, path)
        partial.unlink(missing_ok=True)

        rest_p = 1 - cache["logp"].float().exp().sum(dim=-1).mean().item()
        rest_sampler = 1 - cache["probs"].float().sum(dim=-1).mean().item()
        print(f"{path}: {len(cache['lengths'])} примеров, {len(cache['indices'])} позиций, "
              f"{path.stat().st_size / 2 ** 20:.1f} МБ\n"
              f"Вне top-{cache['indices'].shape[1]}: {rest_p:.2%} массы p и {rest_sampler:.2%} массы сэмплера "
              f"(α в finetune.py занижена не больше чем на вторую величину)")


if __name__ == "__main__":
    main()
