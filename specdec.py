import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_NAME = "HuggingFaceTB/SmolLM2-1.7B"
DRAFT_NAME = "HuggingFaceTB/SmolLM2-135M"


def load_models():
    tokenizer = AutoTokenizer.from_pretrained(TARGET_NAME)
    draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT_NAME)

    assert tokenizer.get_vocab() == draft_tokenizer.get_vocab(), "У моделей разные токенизаторы!"

    target = AutoModelForCausalLM.from_pretrained(TARGET_NAME, dtype=torch.float32)
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_NAME, dtype=torch.float32)

    target.eval()
    draft.eval()

    assert target.config.vocab_size == draft.config.vocab_size, (
        f"Разный размер выхода: {target.config.vocab_size} vs {draft.config.vocab_size}"
    )

    print(f"Загружено. Словарь: {target.config.vocab_size} токенов, "
          f"таргет: {target.num_parameters() / 1e6:.0f}M параметров, "
          f"драфт: {draft.num_parameters() / 1e6:.0f}M параметров")

    return tokenizer, target, draft


def to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    assert temperature >= 0, "Отрицательная температура"

    if temperature > 0:
        return torch.softmax(logits / temperature, dim=-1)

    return torch.nn.functional.one_hot(logits.argmax(dim=-1), logits.shape[-1]).float()


@torch.no_grad()
def get_probs(model, input_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    token_tensor = model(input_ids).logits[0]
    return to_probs(token_tensor, temperature)


def sample(probs: torch.Tensor) -> int:
    return torch.multinomial(probs, num_samples=1).item()


def autoregressive_generate(model, input_ids: torch.Tensor, max_new_tokens: int,
                            temperature: float) -> torch.Tensor:
    for _ in range(max_new_tokens):
        probs = get_probs(model, input_ids, temperature)[-1]
        new_token = torch.tensor(sample(probs))
        new_token = new_token.reshape((1, 1))

        input_ids = torch.cat([input_ids, new_token], dim=1)

    return input_ids


def residual_distribution(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    diff = torch.clamp(q - p, min=0)
    diff = diff / diff.sum()
    return diff


def speculative_generate(target, draft, input_ids: torch.Tensor, max_new_tokens: int,
                         K: int, temperature: float):
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
            p_t = get_probs(draft, input_draft, temperature)[-1]

            draft_guess = sample(p_t)
            guesses.append(draft_guess)
            draft_probs.append(p_t)

            input_draft = torch.cat([input_draft, torch.tensor([[draft_guess]])], dim=1)

        target_probs = get_probs(target, input_draft, temperature)

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

        input_ids = torch.cat([input_ids, torch.tensor([new_tokens])], dim=1)

    return input_ids, stats


def main():
    torch.manual_seed(0)
    tokenizer, target, draft = load_models()

    prompt = "def fibonacci(n):"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    max_new_tokens = 40
    temperature = 0.0
    K = 4

    start = time.perf_counter()
    ar_tokens = autoregressive_generate(target, input_ids, max_new_tokens, temperature)
    ar_time = time.perf_counter() - start

    start = time.perf_counter()
    sps_tokens, stats = speculative_generate(target, draft, input_ids, max_new_tokens, K, temperature)
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
