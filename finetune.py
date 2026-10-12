"""Этап 2 дообучения драфта: на GPU только драфт, ответы таргета берутся из кэша teacher_cache.py.

Два варианта (--loss), запускаются отдельно на одном и том же кэше. Обозначения: p — таргет, q — драфт.

    kl — KL(p ‖ q) в каждой позиции продолжения: драфт учится на всём распределении таргета (мягкие метки).
         От p известен top-k, остальные токены объединены в один — это нижняя оценка настоящей KL
    ce — кросс-энтропия −ln q(x) на токене x из текста (жёсткие метки). Если продолжение сэмплировал таргет
         (--source generated), это оценка CE(p, q) = KL(p ‖ q) + H(p) по одному сэмплу. Сэмпл взят
         с temperature и top-p, поэтому драфт тянется к уже заострённому распределению таргета

Оба лосса считаются на сырых распределениях (temperature = 1), как CE и KL в metrics.py.

Каждые --eval-every шагов драфт сравнивается с таргетом на HumanEval, метрики как в metrics.py: KL(p ‖ q),
α = Σ min(p, q) и top-1 на распределениях сэмплера, perplexity драфта на эталонных решениях. Лучший по α
драфт сохраняется в <out-dir>/best, последний — в <out-dir>/last. Папку можно подставить в любой скрипт:

    python eval_humaneval.py --draft checkpoints/<pair>_kl/best
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from specdec import DEFAULT_PAIR, DEVICE, MODEL_PAIRS, describe, load_model, load_tokenizer, resolve_model, to_probs
from teacher_cache import Batch, TeacherCache, hidden_at

LOSSES = ["kl", "ce"]


def parse_args():
    parser = argparse.ArgumentParser(description="Дообучение драфта по кэшу таргета: KL или CE")
    parser.add_argument("--loss", choices=LOSSES, required=True)
    parser.add_argument("--pair", choices=MODEL_PAIRS, default=DEFAULT_PAIR, help="пара (таргет, драфт)")
    parser.add_argument("--draft", default=None, help="переопределяет драфт из --pair")
    parser.add_argument("--train-cache", default=None, help="по умолчанию cache/<pair>/train_generated.pt")
    parser.add_argument("--eval-cache", default=None, help="по умолчанию cache/<pair>/humaneval.pt")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="учиться на N случайных примерах кэша; шагов столько же, сколько на полном кэше")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="переопределяет --epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.05, help="доля шагов на линейный разогрев lr")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100, help="шагов между оценками на HumanEval")
    parser.add_argument("--chunk", type=int, default=512,
                        help="сколько позиций за раз оценивать (векторы по размеру словаря — память)")
    # Веса всегда float32, bfloat16 — это autocast на прямом проходе
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None, help="по умолчанию checkpoints/<pair>_<loss>")
    return parser.parse_args()


def autocast(model, args) -> torch.autocast:
    return torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=args.dtype == "bfloat16")


def kl_to_teacher(logits: torch.Tensor, indices: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
    """KL(p ‖ q) в каждой позиции. logits: [позиции, словарь]; indices, logp: [позиции, k] — top-k таргета."""
    log_q = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    kl = (p * (logp - log_q.gather(-1, indices))).sum(dim=-1)

    # «Остальной» токен: всё, что не вошло в top-k. У драфта его масса считается через logsumexp,
    # а не как 1 − Σ q: при q ≈ 1 на top-k разность теряет точность
    p_rest = (1 - p.sum(dim=-1)).clamp_min(0)
    log_q_rest = log_q.scatter(-1, indices, float("-inf")).logsumexp(dim=-1)
    return kl + torch.xlogy(p_rest, p_rest) - p_rest * log_q_rest


def position_loss(name: str, logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    if name == "kl":
        return kl_to_teacher(logits, batch.indices, batch.logp)
    return F.cross_entropy(logits, batch.labels, reduction="none")


@torch.no_grad()
def evaluate(draft, cache: TeacherCache, args) -> dict[str, float]:
    """Сходство драфта с таргетом на кэше: средние по всем позициям продолжения."""
    draft.eval()
    lm_head = draft.get_output_embeddings()
    temperature, top_p = cache.meta["temperature"], cache.meta["top_p"]
    kl, alpha, top1, ce = (torch.zeros((), device=draft.device) for _ in range(4))

    for batch in cache.batches(args.eval_batch_size, draft.device):
        with autocast(draft, args):
            hidden = hidden_at(draft, batch.input_ids, batch.attention_mask, batch.predict)
        # top-p сортирует весь словарь в каждой позиции, поэтому по кускам
        for start in range(0, len(hidden), args.chunk):
            rows = slice(start, start + args.chunk)
            with autocast(draft, args):
                logits = lm_head(hidden[rows])
            logits = logits.float()
            indices = batch.indices[rows]

            kl += kl_to_teacher(logits, indices, batch.logp[rows]).sum()
            ce += F.cross_entropy(logits, batch.labels[rows], reduction="sum")
            # top-k таргета отсортирован по убыванию: первый токен — его argmax
            top1 += (logits.argmax(dim=-1) == indices[:, 0]).sum()
            q = to_probs(logits, temperature, top_p)
            alpha += torch.minimum(batch.probs[rows], q.gather(-1, indices)).sum()

    n = len(cache.indices)
    return {"kl": kl.item() / n, "alpha": alpha.item() / n, "top1": top1.item() / n,
            "perplexity": math.exp(ce.item() / n)}


def lr_schedule(step: int, total: int, warmup: int) -> float:
    # Линейный разогрев, потом косинус до нуля
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def train(draft, train_cache: TeacherCache, eval_cache: TeacherCache, args, save, log=print) -> list[dict]:
    """Дообучает драфт на месте. save(имя) сохраняет текущие веса. Возвращает историю оценок."""
    total = args.max_steps or args.epochs * math.ceil(len(train_cache) / args.batch_size)
    warmup = int(args.warmup * total)
    optimizer = torch.optim.AdamW(draft.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_schedule(step, total, warmup))
    lm_head = draft.get_output_embeddings()
    order = torch.Generator().manual_seed(args.seed)

    history = []
    losses = []
    start_time = time.perf_counter()

    def checkpoint(step: int):
        record = {"step": step, "train_loss": torch.stack(losses).mean().item() if losses else None,
                  **evaluate(draft, eval_cache, args)}
        losses.clear()
        # Исходный драфт (шаг 0) не сохраняем: он и так лежит на диске
        is_best = step > 0 and record["alpha"] > max(r["alpha"] for r in history)
        if is_best:
            save("best")
        history.append(record)

        loss = "—" if record["train_loss"] is None else f"{record['train_loss']:.4f}"
        log(f"[{step}/{total}] {args.loss} {loss} | HumanEval: KL {record['kl']:.4f}, α {record['alpha']:.4f}, "
            f"top-1 {record['top1']:.4f}, PPL {record['perplexity']:.3f} | {time.perf_counter() - start_time:.0f} с"
            + (" | лучший" if is_best else ""))

    checkpoint(0)
    step = 0
    while step < total:
        for batch in train_cache.batches(args.batch_size, draft.device, order):
            draft.train()
            with autocast(draft, args):
                logits = lm_head(hidden_at(draft, batch.input_ids, batch.attention_mask, batch.predict))
            loss = position_loss(args.loss, logits.float(), batch).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Лосс остаётся тензором на устройстве: .item() на каждом шаге — лишняя синхронизация с GPU
            losses.append(loss.detach())
            step += 1
            if step % args.eval_every == 0 or step == total:
                checkpoint(step)
            if step == total:
                break

    save("last")
    return history


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    args.draft = args.draft or MODEL_PAIRS[args.pair][1]
    train_cache = TeacherCache.load(args.train_cache or f"cache/{args.pair}/train_generated.pt")
    eval_cache = TeacherCache.load(args.eval_cache or f"cache/{args.pair}/humaneval.pt")
    args.target = train_cache.meta["target"]
    assert eval_cache.meta["target"] == args.target, "Кэши train и HumanEval построены разными таргетами"
    if args.train_limit is not None:
        # Шагов — как на полном кэше: от размера выборки зависит только разнообразие данных, а не объём обучения.
        # Кэш отсортирован по длине, поэтому берём не начало, а случайные примеры; зерно одно, так что
        # меньшая выборка вложена в большую
        args.max_steps = args.max_steps or args.epochs * math.ceil(len(train_cache) / args.batch_size)
        order = torch.randperm(len(train_cache), generator=torch.Generator().manual_seed(args.seed))
        train_cache = train_cache.subset(order[:args.train_limit].tolist())

    draft_path = resolve_model(args.draft)
    tokenizer = load_tokenizer(resolve_model(args.target), draft_path)
    assert len(tokenizer) == train_cache.meta["n_vocab"], "Кэш построен с другим токенизатором"
    draft = load_model(draft_path, torch.float32, len(tokenizer), DEVICE)
    print(f"Драфт {args.draft} на {DEVICE}: {describe(draft)}. Таргет из кэша: {args.target}\n"
          f"Обучение: {len(train_cache)} примеров, {len(train_cache.indices)} позиций "
          f"(продолжение — {train_cache.meta['source']}). HumanEval: {len(eval_cache)} задач, "
          f"{len(eval_cache.indices)} позиций, α при T={eval_cache.meta['temperature']}, "
          f"top_p={eval_cache.meta['top_p']}")

    limit = "" if args.train_limit is None else f"_n{args.train_limit}"
    out_dir = Path(args.out_dir or f"checkpoints/{args.pair}_{args.loss}{limit}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str):
        draft.save_pretrained(out_dir / name)
        tokenizer.save_pretrained(out_dir / name)

    history = train(draft, train_cache, eval_cache, args, save)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        # temperature и top-p — сэмплера из кэша: с ними таргет генерировал продолжения и с ними посчитана α
        json.dump({"args": vars(args), "train_examples": len(train_cache),
                   "temperature": eval_cache.meta["temperature"], "top_p": eval_cache.meta["top_p"],
                   "history": history}, f, indent=2, ensure_ascii=False)

    first, best = history[0], max(history, key=lambda record: record["alpha"])
    print(f"\n=== {args.loss}: α на HumanEval {first['alpha']:.4f} → {best['alpha']:.4f} (шаг {best['step']}), "
          f"KL {first['kl']:.4f} → {best['kl']:.4f} ===")
    if best["step"] == 0:
        print(f"Дообучение α не улучшило: {out_dir / 'best'} не создан")
    print(f"Чекпойнты: {out_dir}")


if __name__ == "__main__":
    main()
