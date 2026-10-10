"""Таргет, драфт и пара (SpS) на бенчмарке с доменами (eval_set.jsonl): итоги по каждому домену.

Две части:
1. Teacher forcing на эталонном продолжении (поле reference): perplexity и top-1 точность таргета и драфта
   на настоящем тексте домена и метрики их сходства из metrics.py (α = Σ min(p, q), top-1, KL). Один проход
   каждой модели на пример, поэтому считается по всем примерам.
2. Генерация продолжения prompt тремя способами: таргет (ArS), драфт (ArS) и пара (SpS). Скорость (мс/токен,
   P90 по примерам, ускорение SpS относительно таргета, α, токенов за цикл) и качество (ROUGE-L с эталоном).
   Дорого, поэтому по --gen-per-domain примеров на домен.
"""
import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from metrics import position_metrics
from specdec import (DEFAULT_PAIR, DEVICE, MODEL_PAIRS, autoregressive_generate, load_models, make_generator,
                     speculative_generate, sync)

TF_METRICS = ["ce_domain_p", "ce_domain_q", "acc_p", "acc_q", "alpha", "top1", "kl_pq"]
MODES = ["target", "draft", "sps"]
ALL_CODE = "code (все)"
ALL = "всё"


def parse_args():
    parser = argparse.ArgumentParser(description="Таргет, драфт и SpS по доменам eval_set.jsonl")
    parser.add_argument("--pair", choices=MODEL_PAIRS, default=DEFAULT_PAIR, help="пара (таргет, драфт)")
    parser.add_argument("--target", default=None, help="переопределяет таргет из --pair")
    parser.add_argument("--draft", default=None, help="переопределяет драфт из --pair")
    parser.add_argument("--dataset", default="eval_set.jsonl")
    parser.add_argument("--domains", nargs="+", default=None, help="только эти домены")
    parser.add_argument("--tf-per-domain", type=int, default=None,
                        help="примеров на домен для teacher forcing (по умолчанию все, 0 — пропустить)")
    parser.add_argument("--gen-per-domain", type=int, default=30, help="примеров на домен для генерации (0 — пропустить)")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--max-ref-tokens", type=int, default=256, help="сколько токенов reference оценивать")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def load_rows(path: str, domains: list[str] | None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if domains is None or r["domain"] in domains]


def per_domain(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return rows
    taken = defaultdict(int)
    result = []
    for row in rows:
        if taken[row["domain"]] < limit:
            taken[row["domain"]] += 1
            result.append(row)
    return result


@torch.no_grad()
def teacher_forced(target, draft, tokenizer, row: dict, args) -> dict | None:
    # Токенизируем prompt + reference целиком (на стыке токены могут склеиться) и оцениваем только токены,
    # которые целиком лежат в reference
    encoding = tokenizer(row["prompt"] + row["reference"], return_offsets_mapping=True)
    prompt_chars = len(row["prompt"])
    positions = [i for i, (start, _) in enumerate(encoding["offset_mapping"]) if i > 0 and start >= prompt_chars]
    positions = positions[:args.max_ref_tokens]
    if not positions:
        return None

    first, last = positions[0], positions[-1]
    input_ids = torch.tensor([encoding["input_ids"][:last + 1]], device=DEVICE)
    # Позиция i предсказывает токен i + 1
    target_logits = target(input_ids).logits[0, first - 1:last]
    draft_logits = draft(input_ids).logits[0, first - 1:last]
    next_tokens = input_ids[0, first:last + 1]

    metrics = position_metrics(target_logits, draft_logits, next_tokens, args.temperature, args.top_p)
    metrics["acc_p"] = (target_logits.argmax(dim=-1) == next_tokens).float()
    metrics["acc_q"] = (draft_logits.argmax(dim=-1) == next_tokens).float()
    return {"tokens": len(next_tokens), **{name: metrics[name].sum().item() for name in TF_METRICS}}


def rouge_l(candidate: str, reference: str) -> float:
    # F1 по наибольшей общей подпоследовательности слов
    cand, ref = candidate.split(), reference.split()
    if not cand or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for word in cand:
        cur = [0]
        for j, ref_word in enumerate(ref):
            cur.append(prev[j] + 1 if word == ref_word else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(cand), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def generate(mode: str, target, draft, tokenizer, input_ids: torch.Tensor, args, seed: int) -> dict:
    # Одно зерно на пример для всех режимов: все стартуют из одного состояния генератора
    generator = make_generator(seed)
    stats = None
    sync()
    start = time.perf_counter()

    if mode == "sps":
        output, stats = speculative_generate(target, draft, input_ids, args.max_new_tokens, args.K,
                                             args.temperature, args.top_p, tokenizer.eos_token_id,
                                             generator=generator)
    else:
        model = target if mode == "target" else draft
        output = autoregressive_generate(model, input_ids, args.max_new_tokens, args.temperature, args.top_p,
                                         tokenizer.eos_token_id, generator=generator)

    sync()
    elapsed = time.perf_counter() - start
    generated = output[0, input_ids.shape[1]:]
    return {"text": tokenizer.decode(generated, skip_special_tokens=True), "n_tokens": len(generated),
            "time": elapsed, "stats": stats}


def domain_groups(domain: str) -> list[str]:
    groups = [domain, ALL]
    if domain.startswith("code_"):
        groups.append(ALL_CODE)
    return groups


def p90(values: list[float]) -> float:
    return statistics.quantiles(values, n=10)[-1] if len(values) >= 2 else values[0]


def summarize(records: list[dict], modes: list[str]) -> dict:
    tf = defaultdict(lambda: defaultdict(float))
    gen = defaultdict(lambda: defaultdict(list))
    for r in records:
        for group in domain_groups(r["domain"]):
            if r["kind"] == "tf":
                tf[group]["examples"] += 1
                for name in ["tokens", *TF_METRICS]:
                    tf[group][name] += r[name]
            else:
                gen[group][r["mode"]].append(r)

    summary = {}
    for group in set(tf) | set(gen):
        s = {}
        if group in tf:
            t = tf[group]
            n = t["tokens"]
            s["tf"] = {"examples": int(t["examples"]), "tokens": int(n),
                       "ppl_target": math.exp(t["ce_domain_p"] / n), "ppl_draft": math.exp(t["ce_domain_q"] / n),
                       "acc_target": t["acc_p"] / n, "acc_draft": t["acc_q"] / n,
                       "alpha": t["alpha"] / n, "top1": t["top1"] / n, "kl": t["kl_pq"] / n}
        if group in gen:
            g = {}
            for mode in modes:
                rs = gen[group].get(mode, [])
                if not rs:
                    continue
                tokens = sum(r["n_tokens"] for r in rs)
                g[mode] = {"examples": len(rs), "ms_per_token": sum(r["time"] for r in rs) / max(tokens, 1) * 1000,
                           "p90_ms_per_token": p90([r["time"] / max(r["n_tokens"], 1) * 1000 for r in rs]),
                           "rouge_l": statistics.mean(r["rouge_l"] for r in rs)}
                if mode == "sps":
                    accepted = sum(r["stats"]["accepted"] for r in rs)
                    rejected = sum(r["stats"]["rejected"] for r in rs)
                    g[mode]["alpha"] = accepted / max(accepted + rejected, 1)
                    g[mode]["tokens_per_cycle"] = tokens / max(sum(r["stats"]["cycles"] for r in rs), 1)
            if "target" in g and "sps" in g:
                g["speedup"] = g["target"]["ms_per_token"] / g["sps"]["ms_per_token"]
            s["gen"] = g
        summary[group] = s
    return summary


def ordered_groups(summary: dict) -> list[str]:
    domains = sorted((d for d in summary if d not in (ALL_CODE, ALL)),
                     key=lambda d: (d.startswith("code_"), d))
    return domains + [g for g in (ALL_CODE, ALL) if g in summary]


def print_summary(summary: dict, args):
    groups = ordered_groups(summary)
    tf_groups = [g for g in groups if "tf" in summary[g]]
    if tf_groups:
        print(f"\n=== Teacher forcing на reference (до {args.max_ref_tokens} токенов; α и top-1 при "
              f"T={args.temperature}, top_p={args.top_p}) ===")
        print(f"{'домен':<16} {'прим.':>5} {'PPL тарг':>9} {'PPL драфт':>9} {'acc тарг':>8} {'acc драфт':>9} "
              f"{'α':>6} {'top-1':>6} {'KL':>6}")
        for g in tf_groups:
            t = summary[g]["tf"]
            print(f"{g:<16} {t['examples']:>5} {t['ppl_target']:>9.2f} {t['ppl_draft']:>9.2f} {t['acc_target']:>8.3f} "
                  f"{t['acc_draft']:>9.3f} {t['alpha']:>6.3f} {t['top1']:>6.3f} {t['kl']:>6.3f}")

    gen_groups = [g for g in groups if "gen" in summary[g]]
    if gen_groups:
        print(f"\n=== Генерация {args.max_new_tokens} токенов, K={args.K}, T={args.temperature}, "
              f"top_p={args.top_p} (мс/токен, в скобках P90 по примерам) ===")
        print(f"{'домен':<16} {'прим.':>5} {'таргет':>14} {'драфт':>14} {'SpS':>14} {'ускор.':>7} {'α':>6} "
              f"{'ток/цикл':>8} {'ROUGE-L т/д/SpS':>17}")
        for g in gen_groups:
            gs = summary[g]["gen"]
            cell = lambda m: f"{gs[m]['ms_per_token']:.1f} ({gs[m]['p90_ms_per_token']:.1f})" if m in gs else "—"
            rouge = "/".join(f"{gs[m]['rouge_l']:.3f}" if m in gs else "—" for m in MODES)
            n = max(v["examples"] for k, v in gs.items() if k in MODES)
            sps = gs.get("sps", {})
            print(f"{g:<16} {n:>5} {cell('target'):>14} {cell('draft'):>14} {cell('sps'):>14} "
                  f"{gs['speedup'] if 'speedup' in gs else float('nan'):>6.2f}x {sps.get('alpha', float('nan')):>6.3f} "
                  f"{sps.get('tokens_per_cycle', float('nan')):>8.2f} {rouge:>17}")


def main():
    args = parse_args()
    rows = load_rows(args.dataset, args.domains)
    pair_target, pair_draft = MODEL_PAIRS[args.pair]
    args.target, args.draft = args.target or pair_target, args.draft or pair_draft
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    tokenizer, target, draft = load_models(args.target, args.draft, dtype)

    tf_rows = per_domain(rows, args.tf_per_domain) if args.tf_per_domain != 0 else []
    gen_rows = per_domain(rows, args.gen_per_domain) if args.gen_per_domain != 0 else []
    print(f"Примеров: {len(rows)}, teacher forcing: {len(tf_rows)}, генерация: {len(gen_rows)} × {args.modes}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"domains_{args.pair}_K{args.K}_T{args.temperature}_{time.strftime('%Y%m%d-%H%M%S')}"
    records = []

    with open(out_dir / f"{run_name}.jsonl", "w", encoding="utf-8") as f:
        def save(record):
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

        for i, row in enumerate(tf_rows):
            result = teacher_forced(target, draft, tokenizer, row, args)
            if result is not None:
                save({"id": row["id"], "domain": row["domain"], "kind": "tf", **result})
            if (i + 1) % 100 == 0:
                print(f"teacher forcing: {i + 1}/{len(tf_rows)}", flush=True)

        if gen_rows:
            # Прогрев: первые вызовы на GPU медленные, в замеры не идут
            warmup_ids = tokenizer(gen_rows[0]["prompt"], return_tensors="pt").input_ids.to(DEVICE)
            for mode in args.modes:
                generate(mode, target, draft, tokenizer, warmup_ids, argparse.Namespace(**{**vars(args),
                         "max_new_tokens": 8}), seed=0)

        for i, row in enumerate(gen_rows):
            input_ids = tokenizer(row["prompt"], return_tensors="pt").input_ids.to(DEVICE)
            reference = tokenizer.decode(tokenizer(row["reference"], add_special_tokens=False).input_ids
                                         [:args.max_new_tokens])
            seed = args.seed * 1_000_000 + i
            line = []
            for mode in args.modes:
                result = generate(mode, target, draft, tokenizer, input_ids, args, seed)
                result["rouge_l"] = rouge_l(result["text"], reference)
                save({"id": row["id"], "domain": row["domain"], "kind": "gen", "mode": mode, **result})
                line.append(f"{mode} {result['n_tokens'] / result['time']:.1f} ток/с")
            print(f"[{i + 1}/{len(gen_rows)}] {row['id']}: " + ", ".join(line), flush=True)

    summary = summarize(records, args.modes)
    with open(out_dir / f"{run_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "domains": summary}, f, indent=2, ensure_ascii=False)

    print(f"\nПара: {args.target} + {args.draft}, {args.dtype}")
    print_summary(summary, args)
    print(f"\nРезультаты: {out_dir / run_name}.jsonl")


if __name__ == "__main__":
    main()
