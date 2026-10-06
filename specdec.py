import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# Пары (таргет, драфт), только base-модели. Какую брать — решаем по замеру скоростей draft/target
MODEL_PAIRS = {
    "smollm2-135m-1.7b": ("HuggingFaceTB/SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-135M"),
    "smollm2-360m-1.7b": ("HuggingFaceTB/SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-360M"),
    "qwen2.5-0.5b-1.5b": ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B"),
    "qwen2.5-0.5b-3b": ("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-0.5B"),
    "qwen2.5-0.5b-7b": ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-0.5B"),
    "qwen2.5-coder-0.5b-7b": ("Qwen/Qwen2.5-Coder-7B", "Qwen/Qwen2.5-Coder-0.5B"),
}
DEFAULT_PAIR = "smollm2-135m-1.7b"
TARGET_NAME, DRAFT_NAME = MODEL_PAIRS[DEFAULT_PAIR]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_model(name: str) -> str:
    # На кластере интернета нет: если задан MODELS_DIR (несколько папок через ":", как PATH),
    # модель берётся из первой папки, где есть подпапка с последней частью имени
    models_dirs = os.environ.get("MODELS_DIR")
    if not models_dirs or os.path.exists(name):
        return name

    for models_dir in models_dirs.split(os.pathsep):
        path = os.path.join(models_dir, name.split("/")[-1])
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Нет модели {name} ни в одной из папок MODELS_DIR={models_dirs}: "
                            f"скачайте её через download_models.sh")


def load_model(name: str, dtype: torch.dtype, n_vocab: int, device: str):
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)

    # У Qwen выходной слой дополнен до «круглого» размера, и у моделей разного размера по-разному.
    # Обрезаем до словаря токенизатора: распределения target и draft должны быть над одними токенами
    assert model.config.vocab_size >= n_vocab, (
        f"{name}: выход {model.config.vocab_size} меньше словаря токенизатора {n_vocab}"
    )
    if model.config.vocab_size > n_vocab:
        model.resize_token_embeddings(n_vocab)

    return model.to(device).eval()


def describe(model) -> str:
    config = model.config
    return (f"{model.num_parameters() / 1e6:.0f}M параметров, {config.num_hidden_layers} слоёв, "
            f"hidden {config.hidden_size}")


def load_models(target_name: str = TARGET_NAME, draft_name: str = DRAFT_NAME,
                dtype: torch.dtype = torch.float32, device: str = DEVICE):
    target_name, draft_name = resolve_model(target_name), resolve_model(draft_name)
    tokenizer = AutoTokenizer.from_pretrained(target_name)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_name)

    assert tokenizer.get_vocab() == draft_tokenizer.get_vocab(), "У моделей разные токенизаторы!"
    # У instruct-версий другой EOS (например, <|im_end|> у Qwen): так ловим смешанную пару base + instruct
    assert (tokenizer.eos_token_id, tokenizer.bos_token_id) == \
           (draft_tokenizer.eos_token_id, draft_tokenizer.bos_token_id), (
        "Разные EOS/BOS: возможно, одна из моделей instruct-версия"
    )

    n_vocab = len(tokenizer)
    target = load_model(target_name, dtype, n_vocab, device)
    draft = load_model(draft_name, dtype, n_vocab, device)

    print(f"Загружено на {device} в {dtype}. Словарь: {n_vocab} токенов\n"
          f"  таргет {target_name}: {describe(target)}\n"
          f"  драфт  {draft_name}: {describe(draft)}")

    return tokenizer, target, draft


def sync():
    # На GPU операции асинхронные: без синхронизации замер времени врёт
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Оставляем минимальный набор токенов с суммарной вероятностью >= top_p (top-1 остаётся всегда)
    sorted_probs = sorted_probs.masked_fill(cumulative - sorted_probs > top_p, 0.0)
    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return filtered / filtered.sum(dim=-1, keepdim=True)


def to_probs(logits: torch.Tensor, temperature: float, top_p: float = 1.0) -> torch.Tensor:
    assert temperature >= 0, "Отрицательная температура"
    logits = logits.float()

    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(dim=-1), logits.shape[-1]).float()

    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        probs = top_p_filter(probs, top_p)
    return probs


@torch.no_grad()
def get_probs(model, input_ids: torch.Tensor, temperature: float, top_p: float = 1.0,
              cache: DynamicCache | None = None) -> torch.Tensor:
    # С кэшем подаём только токены, которых в нём ещё нет, и распределения получаем только для них
    if cache is not None:
        input_ids = input_ids[:, cache.get_seq_length():]
    token_tensor = model(input_ids, past_key_values=cache, use_cache=cache is not None).logits[0]
    return to_probs(token_tensor, temperature, top_p)


def rollback(cache: DynamicCache, length: int):
    # crop(-n) убирает n последних токенов и работает во всех версиях transformers
    # (crop с положительной длиной в новых устарел). crop(0) обнулил бы кэш, поэтому только при n > 0
    extra = cache.get_seq_length() - length
    if extra > 0:
        cache.crop(-extra)


def sample(probs: torch.Tensor) -> int:
    return torch.multinomial(probs, num_samples=1).item()


def autoregressive_generate(model, input_ids: torch.Tensor, max_new_tokens: int,
                            temperature: float, top_p: float = 1.0,
                            eos_token_id: int | None = None, stop_fn=None,
                            use_cache: bool = True) -> torch.Tensor:
    start_seq_len = input_ids.shape[1]
    cache = DynamicCache() if use_cache else None

    for _ in range(max_new_tokens):
        probs = get_probs(model, input_ids, temperature, top_p, cache)[-1]
        new_token = sample(probs)

        input_ids = torch.cat([input_ids, torch.tensor([[new_token]], device=input_ids.device)], dim=1)

        if new_token == eos_token_id or (stop_fn is not None and stop_fn(input_ids[0, start_seq_len:])):
            break

    return input_ids


def residual_distribution(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    diff = torch.clamp(q - p, min=0)
    total = diff.sum()
    # При q ≈ p остаток численно нулевой (отказ тогда почти невозможен): сэмплируем из q, а не из NaN
    if total <= 0:
        return q
    return diff / total


def verify(guesses: list[int], draft_probs: list[torch.Tensor],
           target_probs: torch.Tensor) -> tuple[list[int], int]:
    # Rejection sampling из Chen et al.: guesses[i] сэмплирован из draft_probs[i], target_probs — K+1
    # распределений таргета (для каждого черновика и ещё одно для бонусного токена).
    # Возвращает новые токены (принятые черновики + исправленный или бонусный токен) и число принятых
    for idx, guess_num in enumerate(guesses):
        q_t = target_probs[idx]
        p_t = draft_probs[idx]
        a = min(1.0, (q_t[guess_num] / p_t[guess_num]).item())
        r = torch.rand(1).item()

        if r >= a:
            return guesses[:idx] + [sample(residual_distribution(q_t, p_t))], idx

    return guesses + [sample(target_probs[len(guesses)])], len(guesses)


def speculative_generate(target, draft, input_ids: torch.Tensor, max_new_tokens: int,
                         K: int, temperature: float, top_p: float = 1.0,
                         eos_token_id: int | None = None, stop_fn=None, use_cache: bool = True):
    device = input_ids.device
    start_seq_len = input_ids.shape[1]
    stats = {'cycles': 0,
             'accepted': 0,
             'rejected': 0,
             'drafted': 0}

    # Инвариант между циклами: в кэше каждой модели лежат принятые токены, кроме последнего (или меньше).
    # Всё, чего в кэше нет, get_probs досчитает сам
    target_cache = DynamicCache() if use_cache else None
    draft_cache = DynamicCache() if use_cache else None

    while input_ids.shape[1] - start_seq_len < max_new_tokens:

        stats['cycles'] += 1

        input_draft = input_ids
        guesses = []
        draft_probs = []
        for _ in range(K):
            stats['drafted'] += 1
            p_t = get_probs(draft, input_draft, temperature, top_p, draft_cache)[-1]

            draft_guess = sample(p_t)
            guesses.append(draft_guess)
            draft_probs.append(p_t)

            input_draft = torch.cat([input_draft, torch.tensor([[draft_guess]], device=device)], dim=1)

        target_probs = get_probs(target, input_draft, temperature, top_p, target_cache)[-(K + 1):]

        new_tokens, n_accepted = verify(guesses, draft_probs, target_probs)
        stats['accepted'] += n_accepted
        stats['rejected'] += int(n_accepted < K)

        # Всё, что после EOS, выбрасываем
        if eos_token_id in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(eos_token_id) + 1]

        input_ids = torch.cat([input_ids, torch.tensor([new_tokens], device=device)], dim=1)

        # Откат кэшей: отбрасываем отклонённые черновики и восстанавливаем инвариант. Если приняты все K,
        # драфт ещё не видел последний черновик — в следующем цикле он получит его вместе с бонусным токеном
        if use_cache:
            rollback(target_cache, input_ids.shape[1] - 1)
            rollback(draft_cache, input_ids.shape[1] - 1)

        if eos_token_id in new_tokens or (stop_fn is not None and stop_fn(input_ids[0, start_seq_len:])):
            break

    return input_ids, stats


def main():
    torch.manual_seed(0)
    tokenizer, target, draft = load_models()

    prompt = "def fibonacci(n):"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    max_new_tokens = 40
    temperature = 0.0
    K = 4

    sync()
    start = time.perf_counter()
    ar_tokens = autoregressive_generate(target, input_ids, max_new_tokens, temperature)
    sync()
    ar_time = time.perf_counter() - start

    start = time.perf_counter()
    sps_tokens, stats = speculative_generate(target, draft, input_ids, max_new_tokens, K, temperature)
    sync()
    sps_time = time.perf_counter() - start

    print("=== ArS ===")
    print(tokenizer.decode(ar_tokens[0]))
    print("=== SpS ===")
    print(tokenizer.decode(sps_tokens[0]))

    print(f"ArS: {ar_time / max_new_tokens * 1000:.1f} ms/token")
    print(f"SpS: {sps_time / max_new_tokens * 1000:.1f} ms/token")
    print(f"Speedup: {ar_time / sps_time:.2f}x")
    # При K=0 черновиков нет: делим на max(..., 1)
    print(f"Acceptance rate: {stats['accepted'] / max(stats['drafted'], 1):.2f}")

    if temperature == 0.0:
        n = input_ids.shape[1] + max_new_tokens
        same = torch.equal(ar_tokens[0, :n], sps_tokens[0, :n])
        print("Greedy check:", "OK, совпадает" if same else "ОШИБКА, тексты разные")


if __name__ == "__main__":
    main()
