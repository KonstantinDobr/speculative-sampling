# Проверки дообучения драфта: python -m pytest test_finetune.py
# Модели — крошечные Llama со случайными весами, как в test_specdec.py
import argparse
import copy

import pytest
import torch

from finetune import evaluate, kl_to_teacher, train
from metrics import position_metrics
from specdec import autoregressive_generate, make_generator
from teacher_cache import TeacherCache, build, collate, encode, generate, hidden_at
from test_specdec import perturbed, tiny_llama

torch.set_num_threads(1)

VOCAB = 64
TEMPERATURE, TOP_P = 0.7, 0.8


def make_args(**overrides) -> argparse.Namespace:
    defaults = dict(top_k=VOCAB, temperature=TEMPERATURE, top_p=TOP_P, max_new_tokens=12, batch_size=4,
                    eval_batch_size=4, chunk=7, dtype="float32", loss="kl", epochs=1, max_steps=None, lr=3e-3,
                    weight_decay=0.0, warmup=0.1, max_grad_norm=1.0, eval_every=10, seed=0)
    return argparse.Namespace(**{**defaults, **overrides})


def make_cache(target, examples, source: str, args) -> TeacherCache:
    cache = build(target, examples, source, args, eos_token_id=-1, generator=make_generator(0, "cpu"))
    cache["meta"] = {"temperature": args.temperature, "top_p": args.top_p}
    return TeacherCache(cache)


def random_examples(n: int, seed: int) -> list[tuple[list[int], int]]:
    # Последовательности разной длины (чтобы был паддинг) с промптом разной длины
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(6, 30, (n,), generator=generator).tolist()
    return [(torch.randint(0, VOCAB, (length,), generator=generator).tolist(), 1 + i % 4)
            for i, length in enumerate(lengths)]


@pytest.fixture(scope="module")
def models():
    target = tiny_llama(VOCAB, 3, seed=0, sharpness=20.0)
    return target, perturbed(target, 0.02, seed=1)


def exact_metrics(target, draft, examples) -> dict[str, float]:
    # То же, что считает evaluate, но напрямую из двух моделей и по всему словарю
    sums, n = {}, 0
    for ids, n_prompt in examples:
        input_ids = torch.tensor([ids])
        with torch.no_grad():
            target_logits = target(input_ids).logits[0, n_prompt - 1:-1]
            draft_logits = draft(input_ids).logits[0, n_prompt - 1:-1]
        metrics = position_metrics(target_logits, draft_logits, input_ids[0, n_prompt:], TEMPERATURE, TOP_P)
        for name, values in metrics.items():
            sums[name] = sums.get(name, 0.0) + values.sum().item()
        n += len(ids) - n_prompt
    return {name: total / n for name, total in sums.items()}


class CharTokenizer:
    # Токен — два символа текста, перед ними BOS (как у OPT)
    def __call__(self, text, return_offsets_mapping=False):
        spans = [(i, min(i + 2, len(text))) for i in range(0, len(text), 2)]
        return {"input_ids": ["<s>"] + [text[a:b] for a, b in spans], "offset_mapping": [(0, 0)] + spans}


@pytest.mark.parametrize("row, source, expected", [
    # Сплошной текст: промпт — первые prompt_tokens токенов, но не больше половины
    ({"text": "abcdefghijklmnopqr"}, "reference", (["<s>", "ab", "cd", "ef", "gh", "ij", "kl"], 3)),
    ({"text": "abcdefghijklmnopqr"}, "generated", (["<s>", "ab", "cd"], 3)),
    ({"text": "abcdef"}, "reference", (["<s>", "ab", "cd", "ef"], 2)),
    # prompt + reference: токен "ef" лежит на стыке и остаётся в промпте, от промпта берётся хвост
    ({"prompt": "abcde", "reference": "fghijklmnopq"}, "reference", (["ab", "cd", "ef", "gh", "ij", "kl", "mn"], 3)),
    ({"prompt": "abcde", "reference": "fghijklmnopq"}, "generated", (["ab", "cd", "ef"], 3)),
    ({"prompt": "abcd"}, "generated", (["<s>", "ab", "cd"], 3)),
])
def test_encode(row, source, expected):
    args = make_args(prompt_tokens=3, max_prompt_tokens=3, max_new_tokens=4)
    assert encode(CharTokenizer(), row, source, args) == expected


def test_generate_matches_autoregressive(models):
    target, _ = models
    prompts = [[1, 5, 9, 2, 7], [3], [10, 20, 30, 40, 50, 60, 11, 12]]
    expected = [autoregressive_generate(target, torch.tensor([p]), 12, 0.0)[0, len(p):].tolist() for p in prompts]
    assert generate(target, prompts, 12, 0.0, 1.0, eos_token_id=-1) == expected

    # EOS обрывает только свой пример, остальные генерируются дальше
    eos = expected[1][4]
    cut = [row[:row.index(eos) + 1] if eos in row else row for row in expected]
    assert len(cut[1]) < 12
    assert generate(target, prompts, 12, 0.0, 1.0, eos_token_id=eos) == cut


def test_hidden_at_matches_full_logits(models):
    target, _ = models
    examples = random_examples(5, seed=2)
    input_ids, attention_mask, predict = collate([torch.tensor(ids) for ids, _ in examples],
                                                 [n for _, n in examples], "cpu")
    assert predict.sum() == sum(len(ids) - n for ids, n in examples)
    with torch.no_grad():
        logits = target.get_output_embeddings()(hidden_at(target, input_ids, attention_mask, predict))
        expected = target(input_ids, attention_mask=attention_mask).logits[predict]
    assert torch.allclose(logits, expected, atol=1e-5)


def test_cache_batches_cover_every_example(models, tmp_path):
    target, _ = models
    examples = random_examples(11, seed=3)
    cache = make_cache(target, examples, "reference", make_args())
    torch.save({"meta": cache.meta, "tokens": cache.tokens, "lengths": cache.lengths, "n_prompt": cache.n_prompt,
                "indices": cache.indices, "logp": cache.logp, "probs": cache.probs}, tmp_path / "cache.pt")
    cache = TeacherCache.load(tmp_path / "cache.pt")

    for generator in (None, torch.Generator().manual_seed(0)):
        seen = []
        for batch in cache.batches(4, "cpu", generator):
            assert len(batch.labels) == len(batch.indices) == batch.predict.sum()
            for row, length in zip(batch.input_ids, batch.attention_mask.sum(dim=1)):
                seen.append(row[:length].tolist())
        assert sorted(seen) == sorted(ids for ids, _ in examples)


def test_cache_subset_keeps_examples_with_their_targets(models):
    target, _ = models
    examples = random_examples(11, seed=3)
    cache = make_cache(target, examples, "reference", make_args())
    picked = [7, 2, 9]
    subset = cache.subset(picked)

    assert len(subset) == 3
    # Пример j подвыборки — это пример picked[j] полного кэша вместе с его top-k таргета
    for j, i in enumerate(picked):
        part, full = subset.batch([j], "cpu"), cache.batch([i], "cpu")
        assert torch.equal(part.input_ids, full.input_ids) and torch.equal(part.predict, full.predict)
        assert torch.equal(part.labels, full.labels) and torch.equal(part.indices, full.indices)
        assert torch.equal(part.logp, full.logp) and torch.equal(part.probs, full.probs)


def test_evaluate_matches_metrics(models):
    target, draft = models
    examples = random_examples(9, seed=4)
    # top-k = словарь без одного токена: «остальной» токен — ровно один настоящий, оценки точные
    # (расхождение — только float16 в кэше)
    cache = make_cache(target, examples, "reference", make_args())
    assert cache.indices.shape[1] == VOCAB - 1

    result = evaluate(draft, cache, make_args())
    exact = exact_metrics(target, draft, examples)
    assert exact["kl_pq"] > 0.05 and exact["alpha"] < 0.95, "драфт неотличим от таргета — тест ничего не проверяет"
    assert result["kl"] == pytest.approx(exact["kl_pq"], abs=2e-3)
    assert result["alpha"] == pytest.approx(exact["alpha"], abs=2e-3)
    assert result["top1"] == pytest.approx(exact["top1"])
    assert result["perplexity"] == pytest.approx(torch.tensor(exact["ce_domain_q"]).exp().item(), rel=1e-4)

    same = evaluate(target, cache, make_args())
    assert same["kl"] == pytest.approx(0, abs=2e-3)
    assert same["alpha"] == pytest.approx(1, abs=2e-3)
    assert same["top1"] == 1


def test_small_top_k_gives_lower_bounds(models):
    target, draft = models
    examples = random_examples(9, seed=4)
    exact = exact_metrics(target, draft, examples)
    result = evaluate(draft, make_cache(target, examples, "reference", make_args(top_k=4)), make_args())
    assert 0 < result["kl"] <= exact["kl_pq"] + 2e-3
    assert 0 < result["alpha"] <= exact["alpha"] + 2e-3


def test_kl_gradient_is_finite_when_draft_ignores_the_rest():
    # Драфт кладёт всю массу в top-k: ln q «остального» токена около −100, но градиент обязан остаться конечным
    logits = torch.full((1, 8), -100.0)
    logits[0, :2] = 0.0
    logits.requires_grad_()
    kl = kl_to_teacher(logits, torch.tensor([[0, 1]]), torch.tensor([[0.5, 0.3]]).log())
    kl.sum().backward()
    assert torch.isfinite(kl).all() and torch.isfinite(logits.grad).all()
    # Остаток p = 0.2 против почти нулевого у драфта: градиент поднимает токены вне top-k
    assert (logits.grad[0, 2:] < 0).all()


@pytest.mark.parametrize("loss", ["kl", "ce"])
def test_training_brings_draft_closer(models, loss):
    target, _ = models
    draft = copy.deepcopy(tiny_llama(VOCAB, 1, seed=5))
    prompts = [(ids[:n_prompt + 2], n_prompt + 2) for ids, n_prompt in random_examples(48, seed=6)]
    args = make_args(loss=loss, max_new_tokens=24, batch_size=8, epochs=5)
    train_cache = make_cache(target, prompts, "generated", args)
    eval_cache = make_cache(target, random_examples(16, seed=7), "reference", args)

    saved = []
    history = train(draft, train_cache, eval_cache, args, save=saved.append, log=lambda line: None)

    assert [record["step"] for record in history] == [0, 10, 20, 30]
    assert history[-1]["kl"] < 0.9 * history[0]["kl"]
    assert history[-1]["alpha"] > history[0]["alpha"] + 0.03
    assert saved[-1] == "last" and "best" in saved
