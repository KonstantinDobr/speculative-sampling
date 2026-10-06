import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_NAME = "HuggingFaceTB/SmolLM2-1.7B"
DRAFT_NAME = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(target_name: str = TARGET_NAME, draft_name: str = DRAFT_NAME,
                dtype: torch.dtype = torch.float32, device: str = DEVICE):
    tokenizer = AutoTokenizer.from_pretrained(target_name)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_name)

    assert tokenizer.get_vocab() == draft_tokenizer.get_vocab(), "У моделей разные токенизаторы!"

    target = AutoModelForCausalLM.from_pretrained(target_name, dtype=dtype).to(device)
    draft = AutoModelForCausalLM.from_pretrained(draft_name, dtype=dtype).to(device)

    target.eval()
    draft.eval()

    assert target.config.vocab_size == draft.config.vocab_size, (
        f"Разный размер выхода: {target.config.vocab_size} vs {draft.config.vocab_size}"
    )

    print(f"Загружено на {device}. Словарь: {target.config.vocab_size} токенов, "
          f"таргет: {target.num_parameters() / 1e6:.0f}M параметров, "
          f"драфт: {draft.num_parameters() / 1e6:.0f}M параметров")

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
def get_probs(model, input_ids: torch.Tensor, temperature: float, top_p: float = 1.0) -> torch.Tensor:
    token_tensor = model(input_ids).logits[0]
    return to_probs(token_tensor, temperature, top_p)


def sample(probs: torch.Tensor) -> int:
    return torch.multinomial(probs, num_samples=1).item()


def autoregressive_generate(model, input_ids: torch.Tensor, max_new_tokens: int,
                            temperature: float, top_p: float = 1.0,
                            eos_token_id: int | None = None, stop_fn=None) -> torch.Tensor:
    start_seq_len = input_ids.shape[1]

    for _ in range(max_new_tokens):
        probs = get_probs(model, input_ids, temperature, top_p)[-1]
        new_token = sample(probs)

        input_ids = torch.cat([input_ids, torch.tensor([[new_token]], device=input_ids.device)], dim=1)

        if new_token == eos_token_id or (stop_fn is not None and stop_fn(input_ids[0, start_seq_len:])):
            break

    return input_ids


def residual_distribution(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    diff = torch.clamp(q - p, min=0)
    diff = diff / diff.sum()
    return diff


def speculative_generate(target, draft, input_ids: torch.Tensor, max_new_tokens: int,
                         K: int, temperature: float, top_p: float = 1.0,
                         eos_token_id: int | None = None, stop_fn=None):
    device = input_ids.device
    start_seq_len = input_ids.shape[1]
    stats = {'cycles': 0,
             'accepted': 0,
             'drafted': 0}

    while input_ids.shape[1] - start_seq_len < max_new_tokens:

        stats['cycles'] += 1

        input_draft = input_ids.clone()
        guesses = []
        draft_probs = []
        for _ in range(K):
            stats['drafted'] += 1
            p_t = get_probs(draft, input_draft, temperature, top_p)[-1]

            draft_guess = sample(p_t)
            guesses.append(draft_guess)
            draft_probs.append(p_t)

            input_draft = torch.cat([input_draft, torch.tensor([[draft_guess]], device=device)], dim=1)

        target_probs = get_probs(target, input_draft, temperature, top_p)

        new_tokens = []
        for idx in range(K):
            guess_num = guesses[idx]
            q_t = target_probs[-(K + 1 - idx)]
            p_t = draft_probs[idx]
            a = min(1.0, (q_t[guess_num] / p_t[guess_num]).item())
            r = torch.rand(1).item()

            if r < a:
                new_tokens.append(guess_num)
                stats['accepted'] += 1
            else:
                new_tokens.append(sample(residual_distribution(q_t, p_t)))
                break
        else:
            new_tokens.append(sample(target_probs[-1]))

        # Всё, что после EOS, выбрасываем
        if eos_token_id in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(eos_token_id) + 1]

        input_ids = torch.cat([input_ids, torch.tensor([new_tokens], device=device)], dim=1)

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
    print(f"Acceptance rate: {stats['accepted'] / stats['drafted']:.2f}")

    if temperature == 0.0:
        n = input_ids.shape[1] + max_new_tokens
        same = torch.equal(ar_tokens[0, :n], sps_tokens[0, :n])
        print("Greedy check:", "OK, совпадает" if same else "ОШИБКА, тексты разные")


if __name__ == "__main__":
    main()
