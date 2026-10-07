import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from specdec import (DEFAULT_PAIR, DEVICE, MODEL_PAIRS, autoregressive_generate, load_models,
                     make_generator, speculative_generate, sync)

# Стандартные стоп-последовательности HumanEval (как в Codex): функция закончилась
STOP_SEQUENCES = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
MODES = ["ars", "sps"]


def parse_args():
    parser = argparse.ArgumentParser(description="ArS vs SpS на HumanEval")
    parser.add_argument("--pair", choices=MODEL_PAIRS, default=DEFAULT_PAIR, help="пара (таргет, драфт)")
    parser.add_argument("--target", default=None, help="переопределяет таргет из --pair")
    parser.add_argument("--draft", default=None, help="переопределяет драфт из --pair")
    parser.add_argument("--dataset", default="openai/openai_humaneval",
                        help="имя на HF Hub, локальная папка или файл .jsonl/.jsonl.gz")
    parser.add_argument("--limit", type=int, default=None, help="взять только первые N задач")
    parser.add_argument("--n-samples", type=int, default=1, help="сэмплов на задачу")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    # float32 — для проверок корректности (greedy-совпадение ArS/SpS), большие таргеты в нём не влезут
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=10.0, help="секунд на прогон тестов")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--check-canonical", action="store_true",
                        help="только проверить тесты на эталонных решениях, без моделей")
    return parser.parse_args()


def load_humaneval(source: str) -> list[dict]:
    if source.endswith((".jsonl", ".jsonl.gz")):
        opener = gzip.open if source.endswith(".gz") else open
        with opener(source, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    from datasets import load_dataset
    return list(load_dataset(source, split="test"))


def truncate(text: str) -> str:
    cuts = [i for i in (text.find(s) for s in STOP_SEQUENCES) if i != -1]
    return text[:min(cuts)] if cuts else text


def make_stop_fn(tokenizer):
    def stop_fn(generated_ids: torch.Tensor) -> bool:
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return any(s in text for s in STOP_SEQUENCES)

    return stop_fn


def get_prompt(problem: dict, tokenizer) -> str:
    # Промпт кончается на "\n". Если токенизатор выделяет его в отдельный токен (SmolLM2, OPT), модель такой
    # одиночный токен почти не видела (обычно "\n" склеен с отступом) и сразу выдаёт EOS — тогда хвостовые
    # пробелы срезаем, как в bigcode-evaluation-harness. Если "\n" склеен с кавычками (Qwen: ' """\n'), срезать
    # нельзя: без него модель считает функцию законченной и начинает новую def — ответ получается пустым
    prompt = problem["prompt"]
    last_token = tokenizer.decode(tokenizer(prompt).input_ids[-1:])
    return prompt.rstrip() if last_token.strip() == "" else prompt


def run_tests(problem: dict, completion: str, timeout: float) -> bool:
    program = (problem["prompt"] + completion + "\n\n" + problem["test"] + "\n\n"
               + f"check({problem['entry_point']})\n")

    # Код от модели исполняется в отдельном процессе во временной папке
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "program.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            result = subprocess.run([sys.executable, path], cwd=tmp, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False

    return result.returncode == 0


def generate(mode: str, target, draft, tokenizer, input_ids: torch.Tensor, args, stop_fn, seed: int) -> dict:
    # Своё зерно на каждую пару (задача, сэмпл), одинаковое для ArS и SpS: оба стартуют из одного состояния
    generator = make_generator(seed)
    sync()
    start = time.perf_counter()

    if mode == "ars":
        output = autoregressive_generate(target, input_ids, args.max_new_tokens, args.temperature,
                                         args.top_p, tokenizer.eos_token_id, stop_fn, generator=generator)
        stats = None
    else:
        output, stats = speculative_generate(target, draft, input_ids, args.max_new_tokens, args.K,
                                             args.temperature, args.top_p, tokenizer.eos_token_id, stop_fn,
                                             generator=generator)

    sync()
    elapsed = time.perf_counter() - start

    generated = output[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    return {"completion": truncate(text), "n_tokens": len(generated), "time": elapsed, "stats": stats}


def summarize(records: list[dict]) -> dict:
    summary = {}
    for mode in MODES:
        rs = [r for r in records if r["mode"] == mode]
        tokens = sum(r["n_tokens"] for r in rs)
        seconds = sum(r["time"] for r in rs)
        summary[mode] = {
            "pass@1": sum(r["passed"] for r in rs) / len(rs),
            "ms_per_token": seconds / tokens * 1000,
            "tokens": tokens,
            "seconds": seconds,
        }

    sps = [r["stats"] for r in records if r["mode"] == "sps"]
    accepted = sum(s["accepted"] for s in sps)
    rejected = sum(s["rejected"] for s in sps)
    drafted = sum(s["drafted"] for s in sps)
    cycles = sum(s["cycles"] for s in sps)
    summary["sps"]["acceptance_rate"] = accepted / max(drafted, 1)  # при K=0 черновиков нет
    # α для формулы ускорения: доля принятых среди проверенных (после первого отказа черновики не проверяются)
    summary["sps"]["alpha"] = accepted / max(accepted + rejected, 1)
    summary["sps"]["tokens_per_cycle"] = summary["sps"]["tokens"] / cycles
    summary["speedup"] = summary["ars"]["ms_per_token"] / summary["sps"]["ms_per_token"]

    pairs = {}
    for r in records:
        pairs.setdefault((r["task_id"], r["sample"]), {})[r["mode"]] = r["completion"]
    summary["identical_completions"] = sum(p["ars"] == p["sps"] for p in pairs.values()) / len(pairs)

    return summary


def print_summary(summary: dict, args):
    print(f"\n=== HumanEval: {args.target} + {args.draft}, {args.dtype}, K={args.K}, "
          f"temperature={args.temperature}, top_p={args.top_p} ===")
    print(f"{'Метод':<6} {'pass@1':>8} {'ms/token':>10} {'speedup':>8}")
    print(f"{'ArS':<6} {summary['ars']['pass@1']:>8.3f} {summary['ars']['ms_per_token']:>10.1f} {1.0:>7.2f}x")
    print(f"{'SpS':<6} {summary['sps']['pass@1']:>8.3f} {summary['sps']['ms_per_token']:>10.1f} "
          f"{summary['speedup']:>7.2f}x")
    print(f"Acceptance rate: {summary['sps']['acceptance_rate']:.3f}, α: {summary['sps']['alpha']:.3f}, "
          f"токенов за цикл: {summary['sps']['tokens_per_cycle']:.2f}")
    print(f"Одинаковых ответов ArS/SpS: {summary['identical_completions']:.1%}"
          + (" (при temperature=0 в float32 должно быть ~100%)" if args.temperature == 0 else ""))


def main():
    args = parse_args()
    problems = load_humaneval(args.dataset)[:args.limit]
    print(f"Задач: {len(problems)}")

    if args.check_canonical:
        passed = sum(run_tests(p, p["canonical_solution"], args.timeout) for p in problems)
        print(f"Эталонные решения: {passed}/{len(problems)} проходят тесты")
        return

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    pair_target, pair_draft = MODEL_PAIRS[args.pair]
    args.target, args.draft = args.target or pair_target, args.draft or pair_draft
    tokenizer, target, draft = load_models(args.target, args.draft, dtype)
    stop_fn = make_stop_fn(tokenizer)

    # Прогрев: первые вызовы на GPU медленные (инициализация CUDA, аллокатор), в замеры не идут
    warmup_ids = tokenizer(get_prompt(problems[0], tokenizer), return_tensors="pt").input_ids.to(DEVICE)
    autoregressive_generate(target, warmup_ids, 8, args.temperature, args.top_p)
    speculative_generate(target, draft, warmup_ids, 8, args.K, args.temperature, args.top_p)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"humaneval_{args.pair}_K{args.K}_T{args.temperature}_{time.strftime('%Y%m%d-%H%M%S')}"
    records_path = out_dir / f"{run_name}.jsonl"

    records = []
    with open(records_path, "w", encoding="utf-8") as f:
        for i, problem in enumerate(problems):
            input_ids = tokenizer(get_prompt(problem, tokenizer), return_tensors="pt").input_ids.to(DEVICE)

            for sample_idx in range(args.n_samples):
                for mode in MODES:
                    seed = args.seed * 1_000_000 + i * 1000 + sample_idx
                    result = generate(mode, target, draft, tokenizer, input_ids, args, stop_fn, seed)
                    result["passed"] = run_tests(problem, result["completion"], args.timeout)
                    record = {"task_id": problem["task_id"], "sample": sample_idx, "mode": mode, **result}
                    records.append(record)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

            done = [r for r in records if r["task_id"] == problem["task_id"]]
            print(f"[{i + 1}/{len(problems)}] {problem['task_id']}: "
                  + ", ".join(f"{r['mode']} {'OK' if r['passed'] else 'fail'} "
                              f"{r['n_tokens']} tok {r['time']:.1f}s" for r in done), flush=True)

    summary = summarize(records)
    summary["args"] = vars(args)
    with open(out_dir / f"{run_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_summary(summary, args)
    print(f"\nРезультаты: {records_path}")


if __name__ == "__main__":
    main()
