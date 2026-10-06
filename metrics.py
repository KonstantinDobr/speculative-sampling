"""Метрики похожести драфта на таргет и качества таргета на домене.

Обозначения: p — распределение таргета, q — распределение драфта на следующий токен.
Каждая метрика считается в каждом контексте (префиксе текста), потом усредняется по всем
контекстам всех текстов датасета:

    TV        = ½ · Σₓ |p(x) − q(x)|               — полная вариация
    α         = Σₓ min(p(x), q(x)) = 1 − TV         — вероятность принятия догадки драфта
    top1      = [argmax p = argmax q]               — совпадение самых вероятных токенов
    CE(p, q)  = −Σₓ p(x) · ln q(x)                  — удивление драфта текстом таргета
    H(p)      = −Σₓ p(x) · ln p(x)                  — неуверенность таргета
    KL(p‖q)   = CE(p, q) − H(p)                     — чистое несовпадение драфта с таргетом
    CE_domain = −ln p(настоящий следующий токен)    — насколько таргет хорош на реальном тексте

TV, α и top1 считаются на распределениях сэмплера (с --temperature и --top-p), как в спекдеке.
CE, H, KL и CE_domain — на «сырых» распределениях моделей (temperature = 1, без top-p):
иначе нули от top-p или one-hot при temperature = 0 дают ln 0 = −∞.
"""
import argparse
import gzip
import json
import math
from pathlib import Path

import torch

from specdec import DEVICE, DRAFT_NAME, TARGET_NAME, load_models, to_probs

METRICS = ["tv", "alpha", "top1", "ce_pq", "h_p", "kl_pq", "ce_domain_p", "ce_domain_q"]


def parse_args():
    parser = argparse.ArgumentParser(description="TV, α, top-1, CE, H на датасете")
    parser.add_argument("--target", default=TARGET_NAME)
    parser.add_argument("--draft", default=DRAFT_NAME)
    parser.add_argument("--dataset", required=True,
                        help="имя на HF Hub, файл .jsonl/.jsonl.gz или .txt (весь файл — один текст)")
    parser.add_argument("--split", default="test", help="сплит для датасета с HF Hub")
    parser.add_argument("--fields", nargs="+", default=["text"],
                        help="поля записи, которые склеиваются в текст (HumanEval: prompt canonical_solution)")
    parser.add_argument("--limit", type=int, default=None, help="взять только первые N текстов")
    parser.add_argument("--max-length", type=int, default=1024, help="обрезать текст до N токенов")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--chunk", type=int, default=256,
                        help="сколько позиций обрабатывать за раз (векторы по 49152 — память)")
    parser.add_argument("--out", default=None, help="куда сохранить итог в json")
    return parser.parse_args()


def load_texts(source: str, split: str, fields: list[str]) -> list[str]:
    if source.endswith(".txt"):
        return [Path(source).read_text(encoding="utf-8")]

    if source.endswith((".jsonl", ".jsonl.gz")):
        opener = gzip.open if source.endswith(".gz") else open
        with opener(source, "rt", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        from datasets import load_dataset
        records = list(load_dataset(source, split=split))

    return ["".join(record[field] for field in fields) for record in records]


def position_metrics(target_logits: torch.Tensor, draft_logits: torch.Tensor, next_tokens: torch.Tensor,
                     temperature: float, top_p: float) -> dict[str, torch.Tensor]:
    """Метрики для каждой позиции. Логиты: [позиции, словарь], next_tokens: [позиции]."""
    target_logits = target_logits.float()
    draft_logits = draft_logits.float()

    # Распределения сэмплера — для метрик принятия
    p = to_probs(target_logits, temperature, top_p)
    q = to_probs(draft_logits, temperature, top_p)

    tv = 0.5 * (p - q).abs().sum(dim=-1)
    alpha = torch.minimum(p, q).sum(dim=-1)
    top1 = (p.argmax(dim=-1) == q.argmax(dim=-1)).float()

    # Сырые распределения моделей — для информационных метрик.
    # log_softmax вместо log(softmax): не получим ln 0 = −∞ из-за округления маленьких вероятностей
    log_p = torch.log_softmax(target_logits, dim=-1)
    log_q = torch.log_softmax(draft_logits, dim=-1)
    p_raw = log_p.exp()

    ce_pq = -(p_raw * log_q).sum(dim=-1)
    h_p = -(p_raw * log_p).sum(dim=-1)
    kl_pq = ce_pq - h_p

    # Настоящий следующий токен из текста: берём его логарифм вероятности у каждой модели
    ce_domain_p = -log_p.gather(-1, next_tokens[:, None]).squeeze(-1)
    ce_domain_q = -log_q.gather(-1, next_tokens[:, None]).squeeze(-1)

    return {"tv": tv, "alpha": alpha, "top1": top1, "ce_pq": ce_pq, "h_p": h_p, "kl_pq": kl_pq,
            "ce_domain_p": ce_domain_p, "ce_domain_q": ce_domain_q}


@torch.no_grad()
def text_metrics(target, draft, input_ids: torch.Tensor, args) -> dict[str, float]:
    """Суммы метрик по всем контекстам одного текста. Один прогон каждой модели на весь текст."""
    # Позиция i предсказывает токен i + 1, поэтому последнюю позицию отбрасываем:
    # для неё нет настоящего следующего токена
    target_logits = target(input_ids).logits[0, :-1]
    draft_logits = draft(input_ids).logits[0, :-1]
    next_tokens = input_ids[0, 1:]

    sums = {name: 0.0 for name in METRICS}
    for start in range(0, len(next_tokens), args.chunk):
        end = start + args.chunk
        chunk = position_metrics(target_logits[start:end], draft_logits[start:end], next_tokens[start:end],
                                 args.temperature, args.top_p)
        for name in METRICS:
            sums[name] += chunk[name].sum().item()

    return sums


def main():
    args = parse_args()
    texts = load_texts(args.dataset, args.split, args.fields)[:args.limit]
    print(f"Текстов: {len(texts)}")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    tokenizer, target, draft = load_models(args.target, args.draft, dtype)

    totals = {name: 0.0 for name in METRICS}
    n_contexts = 0
    for i, text in enumerate(texts):
        input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=args.max_length).input_ids.to(DEVICE)
        if input_ids.shape[1] < 2:
            continue  # из одного токена не получить ни одного контекста с ответом

        sums = text_metrics(target, draft, input_ids, args)
        for name in METRICS:
            totals[name] += sums[name]
        n_contexts += input_ids.shape[1] - 1

        print(f"[{i + 1}/{len(texts)}] {input_ids.shape[1]} токенов, "
              f"α = {sums['alpha'] / (input_ids.shape[1] - 1):.3f}", flush=True)

    # Среднее по ВСЕМ контекстам всех текстов: (1/N) · Σₜ, как в определениях
    summary = {name: totals[name] / n_contexts for name in METRICS}
    summary["perplexity_p"] = math.exp(summary["ce_domain_p"])
    summary["perplexity_q"] = math.exp(summary["ce_domain_q"])
    summary["n_contexts"] = n_contexts

    print(f"\n=== {args.dataset}: {n_contexts} контекстов, "
          f"temperature={args.temperature}, top_p={args.top_p} ===")
    print(f"TV                       {summary['tv']:.4f}")
    print(f"α = Σ min(p, q)          {summary['alpha']:.4f}   (1 − TV = {1 - summary['tv']:.4f})")
    print(f"top-1 acceptance         {summary['top1']:.4f}")
    print(f"CE(p, q)                 {summary['ce_pq']:.4f} нат")
    print(f"H(p)                     {summary['h_p']:.4f} нат")
    print(f"KL(p ‖ q) = CE − H       {summary['kl_pq']:.4f} нат")
    print(f"CE таргета на домене     {summary['ce_domain_p']:.4f} нат  (perplexity {summary['perplexity_p']:.2f})")
    print(f"CE драфта на домене      {summary['ce_domain_q']:.4f} нат  (perplexity {summary['perplexity_q']:.2f})")

    if args.out:
        summary["args"] = vars(args)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
