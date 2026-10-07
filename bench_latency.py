import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from eval_humaneval import get_prompt, load_humaneval, make_stop_fn
from specdec import (DEVICE, MODEL_PAIRS, describe, load_model, load_models, resolve_model, rollback,
                     speculative_generate, sync)

ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9]


def parse_args():
    parser = argparse.ArgumentParser(description="Латентность draft/target и теоретическое ускорение SpS")
    parser.add_argument("--pairs", nargs="+", choices=MODEL_PAIRS, default=list(MODEL_PAIRS))
    parser.add_argument("--prefix-lens", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--max-K", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    # Короткий прогон SpS на HumanEval, чтобы подставить в формулу реальную α, а не только сетку
    parser.add_argument("--alpha-tasks", type=int, default=0, help="задач для оценки α (0 — не оценивать)")
    parser.add_argument("--alpha-K", type=int, default=4)
    parser.add_argument("--alpha-max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dataset", default="openai/openai_humaneval")
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def time_call(fn, repeats: int, warmup: int, reset=None) -> dict:
    # sync + perf_counter, как в eval_humaneval: в замер входят и накладные расходы Python/HF,
    # они при декодировании тоже реальны
    times = []
    for i in range(warmup + repeats):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1000)
        if reset is not None:
            reset()

    return {"median_ms": statistics.median(times), "p90_ms": statistics.quantiles(times, n=10)[-1]}


@torch.no_grad()
def measure_model(name: str, dtype: torch.dtype, args) -> dict:
    path = resolve_model(name)
    n_vocab = len(AutoTokenizer.from_pretrained(path))
    model = load_model(path, dtype, n_vocab, DEVICE)
    print(f"\n{name}: {describe(model)}", flush=True)

    result = {"description": describe(model), "num_layers": model.config.num_hidden_layers,
              "hidden_size": model.config.hidden_size, "decode": {}, "no_cache": {}}
    generator = torch.Generator().manual_seed(args.seed)

    for L in args.prefix_lens:
        # Содержимое токенов на скорость не влияет, поэтому префикс случайный
        ids = torch.randint(n_vocab, (1, L + args.max_K + 1), generator=generator).to(DEVICE)
        cache = model(ids[:, :L], use_cache=True).past_key_values

        # Проход по n новым токенам поверх кэша префикса длины L: n=1 — шаг ArS/драфта,
        # n=K+1 — проверка черновика таргетом. После каждого замера кэш откатываем обратно до L
        decode = {}
        for n in range(1, args.max_K + 2):
            decode[n] = time_call(lambda n=n: model(ids[:, L:L + n], past_key_values=cache, use_cache=True),
                                  args.repeats, args.warmup, reset=lambda: rollback(cache, L))

        # Для справки: один токен без KV-кэша, как сейчас работает specdec.py
        no_cache = time_call(lambda: model(ids[:, :L + 1]), args.repeats, args.warmup)

        result["decode"][L] = decode
        result["no_cache"][L] = no_cache
        print(f"  L={L}: 1 токен {decode[1]['median_ms']:.2f} ms, {args.max_K + 1} токенов "
              f"{decode[args.max_K + 1]['median_ms']:.2f} ms, без кэша {no_cache['median_ms']:.2f} ms", flush=True)

        del cache
        free_memory()

    del model
    free_memory()
    return result


def measure_alpha(target_name: str, draft_name: str, dtype: torch.dtype, problems: list[dict], args) -> float:
    tokenizer, target, draft = load_models(target_name, draft_name, dtype)
    stop_fn = make_stop_fn(tokenizer)

    accepted = rejected = 0
    for problem in problems:
        input_ids = tokenizer(get_prompt(problem, tokenizer), return_tensors="pt").input_ids.to(DEVICE)
        _, stats = speculative_generate(target, draft, input_ids, args.alpha_max_new_tokens, args.alpha_K,
                                        args.temperature, args.top_p, tokenizer.eos_token_id, stop_fn)
        accepted += stats["accepted"]
        rejected += stats["rejected"]

    del target, draft
    free_memory()
    # Делим на проверенные черновики: после первого отказа в цикле остальные не проверяются
    return accepted / max(accepted + rejected, 1)


def expected_tokens(alpha: float, K: int) -> float:
    # Среднее число токенов за цикл (Leviathan et al., 2023): (1 - α^(K+1)) / (1 - α)
    if alpha >= 1:
        return K + 1
    return (1 - alpha ** (K + 1)) / (1 - alpha)


def pair_report(target: dict, draft: dict, alphas: list[float], args) -> dict:
    report = {}
    for L in args.prefix_lens:
        t_target = target["decode"][L][1]["median_ms"]
        t_draft = draft["decode"][L][1]["median_ms"]
        c = t_draft / t_target

        speedup = {}
        for alpha in alphas:
            # ideal: проверка K+1 токенов стоит как один проход таргета (допущение статей)
            # measured: подставляем измеренную стоимость проверки K+1 токенов
            ideal = {K: expected_tokens(alpha, K) / (K * c + 1) for K in range(args.max_K + 1)}
            measured = {K: expected_tokens(alpha, K) * t_target
                        / (K * t_draft + target["decode"][L][K + 1]["median_ms"])
                        for K in range(args.max_K + 1)}
            best_K = max(measured, key=measured.get)
            speedup[round(alpha, 3)] = {"ideal": ideal, "measured": measured, "best_K": best_K}

        report[L] = {
            "c": c,
            "verify_ratio": {K: target["decode"][L][K + 1]["median_ms"] / t_target for K in range(1, args.max_K + 1)},
            "speedup": speedup,
        }
    return report


def print_pair(pair: str, report: dict, alpha_measured: float | None, args):
    print(f"\n=== {pair} ===")
    if alpha_measured is not None:
        print(f"Измеренная α (K={args.alpha_K}, T={args.temperature}, top_p={args.top_p}): {alpha_measured:.3f}")
    for L, r in report.items():
        print(f"L={L}: c = {r['c']:.3f}, проверка {args.max_K + 1} токенов = "
              f"{r['verify_ratio'][args.max_K]:.2f} × один проход таргета")

    # Подробную таблицу печатаем для средней длины префикса, остальное — в json
    L = args.prefix_lens[len(args.prefix_lens) // 2]
    print(f"Ожидаемое ускорение (с измеренной стоимостью проверки), L={L}:")
    print(f"{'α':>6} " + " ".join(f"{'K=' + str(K):>6}" for K in range(1, args.max_K + 1)) + "  лучшее K")
    for alpha, s in report[L]["speedup"].items():
        print(f"{alpha:>6.3f} " + " ".join(f"{s['measured'][K]:>6.2f}" for K in range(1, args.max_K + 1))
              + f"  {s['best_K']}")


def is_available(name: str) -> bool:
    try:
        resolve_model(name)
        return True
    except FileNotFoundError as e:
        print(f"Пропуск: {e}")
        return False


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    pairs = [p for p in args.pairs if all(is_available(name) for name in MODEL_PAIRS[p])]
    # Каждую модель меряем один раз и по отдельности, даже если она входит в несколько пар
    models = list(dict.fromkeys(name for p in pairs for name in MODEL_PAIRS[p]))
    latency = {name: measure_model(name, dtype, args) for name in models}

    problems = load_humaneval(args.dataset)[:args.alpha_tasks] if args.alpha_tasks > 0 else []

    results = {}
    for pair in pairs:
        target_name, draft_name = MODEL_PAIRS[pair]
        alpha_measured = measure_alpha(target_name, draft_name, dtype, problems, args) if problems else None
        alphas = ALPHAS + ([alpha_measured] if alpha_measured is not None else [])

        report = pair_report(latency[target_name], latency[draft_name], alphas, args)
        results[pair] = {"target": target_name, "draft": draft_name, "alpha_measured": alpha_measured,
                         "report": report}
        print_pair(pair, report, alpha_measured, args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bench_{time.strftime('%Y%m%d-%H%M%S')}.json"
    env = {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
           "torch": torch.__version__, "transformers": transformers.__version__}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "env": env, "latency": latency, "pairs": results},
                  f, indent=2, ensure_ascii=False)
    print(f"\nРезультаты: {out_path}")


if __name__ == "__main__":
    main()
