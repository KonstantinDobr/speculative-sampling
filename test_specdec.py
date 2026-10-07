# Проверки корректности SpS: python -m pytest test_specdec.py
# Модели — крошечные Llama со случайными весами, ничего не скачивается, хватает CPU
import copy
import math
from collections import Counter

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from specdec import (autoregressive_generate, get_probs, make_generator, residual_distribution, sample,
                     speculative_generate, verify)

torch.set_num_threads(1)


def chi2_ok(counts: Counter, probs: dict, n: int) -> tuple[bool, float]:
    # χ²-критерий согласия: клетки с ожидаемым числом < 5 объединяем в одну, порог — среднее + 5σ
    # распределения χ² (ложное срабатывание практически исключено, а сломанный алгоритм даёт сотни и тысячи)
    assert all(probs.get(k, 0) > 0 for k in counts), "выпал исход, у которого вероятность 0"
    big = {k: p for k, p in probs.items() if n * p >= 5}
    observed = [counts[k] for k in big] + [n - sum(counts[k] for k in big)]
    expected = [n * p for p in big.values()] + [n * (1 - sum(big.values()))]
    cells = [(o, e) for o, e in zip(observed, expected) if e > 0]
    stat = sum((o - e) ** 2 / e for o, e in cells)
    df = len(cells) - 1
    return stat < df + 5 * math.sqrt(2 * df), stat


# ---------- rejection sampling на игрушечных распределениях, без моделей ----------

# Обозначения как в Chen et al.: p — таргет, q — драфт
P = torch.tensor([0.4, 0.3, 0.2, 0.1])  # таргет
Q = torch.tensor([0.1, 0.2, 0.3, 0.4])  # драфт, β = Σ min(p, q) = 0.6


def iid_speculative(p_target: torch.Tensor, q_draft: torch.Tensor, K: int, length: int) -> tuple:
    # Распределения не зависят от контекста: тогда SpS обязан выдавать i.i.d. токены из p
    tokens = []
    while len(tokens) < length:
        guesses = torch.cat([sample(q_draft) for _ in range(K)])
        new_tokens, _ = verify(guesses, q_draft.expand(K, -1), p_target.expand(K + 1, -1))
        tokens += new_tokens.tolist()
    return tuple(tokens[:length])


def test_residual_distribution():
    r = residual_distribution(P, Q)
    assert torch.all(r >= 0) and torch.isclose(r.sum(), torch.tensor(1.0))
    # max(0, p - q) = (0.3, 0.1, 0, 0), нормируем на 0.4
    assert torch.allclose(r, torch.tensor([0.75, 0.25, 0.0, 0.0]))


def test_residual_distribution_equal_falls_back_to_target():
    r = residual_distribution(P, P.clone())
    assert not torch.isnan(r).any()
    assert torch.equal(r, P)


@pytest.mark.parametrize("K", [1, 3])
def test_verify_matches_target_distribution(K):
    torch.manual_seed(0)
    n, length = 20000, 3
    counts = Counter(iid_speculative(P, Q, K, length) for _ in range(n))

    joint = {}
    for i in range(4):
        for j in range(4):
            for k in range(4):
                joint[(i, j, k)] = (P[i] * P[j] * P[k]).item()
    ok, stat = chi2_ok(counts, joint, n)
    assert ok, f"χ² = {stat:.1f}: совместное распределение 3 токенов не совпадает с p"

    # Контроль мощности теста: сэмплы драфта (как если бы всё принималось) тест обязан отвергнуть
    draft_counts = Counter(tuple(sample(Q).item() for _ in range(length)) for _ in range(n))
    assert not chi2_ok(draft_counts, joint, n)[0]


def test_acceptance_rate_equals_beta():
    torch.manual_seed(1)
    n = 20000
    accepted = sum(verify(sample(Q), Q[None], P.expand(2, -1))[1] for _ in range(n))
    beta = torch.minimum(P, Q).sum().item()
    sigma = math.sqrt(beta * (1 - beta) / n)
    assert abs(accepted / n - beta) < 5 * sigma, f"доля принятия {accepted / n:.3f}, β = {beta:.3f}"


# ---------- генерация на крошечных моделях ----------

def tiny_llama(vocab: int, layers: int, seed: int, sharpness: float = 1.0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    model = LlamaForCausalLM(config).eval()
    # Случайная инициализация даёт почти равномерные распределения; усиливаем выход, чтобы top-p что-то резал
    with torch.no_grad():
        model.lm_head.weight.mul_(sharpness)
    return model


def perturbed(model: LlamaForCausalLM, scale: float, seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    copy_model = copy.deepcopy(model)
    with torch.no_grad():
        for param in copy_model.parameters():
            param.add_(scale * torch.randn_like(param))
    return copy_model


@pytest.fixture(scope="module")
def models():
    target = tiny_llama(64, 3, seed=0)
    # Разная доля принятия: ~0 (отказ на первом черновике), ~1 (всё принимается) и посередине
    drafts = {"random": tiny_llama(64, 1, seed=1), "same": target, "close": perturbed(target, 0.004, seed=2)}
    return target, drafts


PROMPTS = [[1, 5, 9, 2, 7], [3], [10, 20, 30, 40, 50, 60, 11, 12]]


@pytest.mark.parametrize("prompt", PROMPTS)
def test_ars_cache_matches_no_cache(models, prompt):
    target, _ = models
    input_ids = torch.tensor([prompt])
    with_cache = autoregressive_generate(target, input_ids, 40, 0.0)
    without_cache = autoregressive_generate(target, input_ids, 40, 0.0, use_cache=False)
    assert torch.equal(with_cache, without_cache)


@pytest.mark.parametrize("draft_name", ["random", "same", "close"])
@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_matches_ars(models, draft_name, use_cache, prompt):
    target, drafts = models
    input_ids = torch.tensor([prompt])
    max_new_tokens = 40
    expected = autoregressive_generate(target, input_ids, max_new_tokens, 0.0)

    for K in range(8):
        output, stats = speculative_generate(target, drafts[draft_name], input_ids, max_new_tokens, K, 0.0,
                                             use_cache=use_cache)
        assert torch.equal(output, expected), f"K={K}"  # в том числе ровно max_new_tokens токенов
        assert stats["drafted"] <= K * stats["cycles"]
        if K == 0:
            assert stats["accepted"] == stats["rejected"] == 0


def test_greedy_eos(models):
    target, drafts = models
    input_ids = torch.tensor([PROMPTS[0]])
    eos = int(autoregressive_generate(target, input_ids, 40, 0.0)[0, input_ids.shape[1] + 10])
    expected = autoregressive_generate(target, input_ids, 40, 0.0, eos_token_id=eos)
    assert expected[0, -1] == eos

    for K in (1, 3, 6):
        output, _ = speculative_generate(target, drafts["close"], input_ids, 40, K, 0.0, eos_token_id=eos)
        assert torch.equal(output, expected), f"K={K}"


def exact_two_token_distribution(model, input_ids: torch.Tensor, temperature: float, top_p: float) -> dict:
    first = get_probs(model, input_ids, temperature, top_p)[-1]
    distribution = {}
    for a in first.nonzero().flatten().tolist():
        extended = torch.cat([input_ids, torch.tensor([[a]])], dim=1)
        second = get_probs(model, extended, temperature, top_p)[-1]
        for b in second.nonzero().flatten().tolist():
            distribution[(a, b)] = (first[a] * second[b]).item()
    return distribution


@pytest.mark.parametrize("temperature, top_p", [(1.0, 1.0), (0.7, 0.8)])
@pytest.mark.parametrize("K", [1, 3])
def test_sampling_matches_target(temperature, top_p, K):
    # Распределение первых двух токенов SpS (с кэшем) против точного распределения таргета
    target = tiny_llama(6, 2, seed=3, sharpness=30.0)
    draft = tiny_llama(6, 1, seed=4, sharpness=30.0)
    input_ids = torch.tensor([[1, 2, 3]])
    exact = exact_two_token_distribution(target, input_ids, temperature, top_p)
    if top_p < 1.0:
        assert len(exact) < 36, "top-p ничего не отрезал — тест не проверяет фильтрацию"

    torch.manual_seed(5)
    n = 2000
    # max_new_tokens = K + 1, чтобы в первом цикле было ровно K черновиков (последний цикл их урезает)
    counts = Counter(
        tuple(speculative_generate(target, draft, input_ids, K + 1, K, temperature, top_p)[0][0, 3:5].tolist())
        for _ in range(n)
    )
    ok, stat = chi2_ok(counts, exact, n)
    assert ok, f"χ² = {stat:.1f}"


def test_generator_reproducible(models):
    target, drafts = models
    input_ids = torch.tensor([PROMPTS[0]])
    for run in (
        lambda g: autoregressive_generate(target, input_ids, 30, 1.0, generator=g),
        lambda g: speculative_generate(target, drafts["close"], input_ids, 30, 4, 1.0, generator=g)[0],
    ):
        first, second = run(make_generator(7, "cpu")), run(make_generator(7, "cpu"))
        assert torch.equal(first, second)
        assert not torch.equal(first, run(make_generator(8, "cpu")))


@pytest.mark.parametrize("K", [1, 3, 7])
def test_sps_never_exceeds_max_new_tokens(models, K):
    target, drafts = models
    input_ids = torch.tensor([PROMPTS[0]])
    for max_new_tokens in (1, 2, 5, 13):
        output, _ = speculative_generate(target, drafts["same"], input_ids, max_new_tokens, K, 1.0)
        assert output.shape[1] == input_ids.shape[1] + max_new_tokens


class IgnoresLogitsToKeep(torch.nn.Module):
    # Как OPT в transformers 4.57: принимает logits_to_keep, но возвращает логиты по всем позициям
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, logits_to_keep=None, **kwargs):
        return self.model(*args, **kwargs)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_model_ignoring_logits_to_keep(models, temperature):
    target, drafts = models
    input_ids = torch.tensor([PROMPTS[0]])
    wrapped_target, wrapped_draft = IgnoresLogitsToKeep(target), IgnoresLogitsToKeep(drafts["close"])
    for K in (0, 1, 4):
        expected, _ = speculative_generate(target, drafts["close"], input_ids, 30, K, temperature,
                                           generator=make_generator(3, "cpu"))
        output, _ = speculative_generate(wrapped_target, wrapped_draft, input_ids, 30, K, temperature,
                                         generator=make_generator(3, "cpu"))
        assert torch.equal(output, expected), f"K={K}"
