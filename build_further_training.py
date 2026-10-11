import argparse
import bisect
import builtins
import hashlib
import json
import keyword
import random
import re
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer

SOURCE = "RealTimeData/code_alltime"
HUMANEVAL = "openai/openai_humaneval"
HUMANEVAL_FILE = "openai_humaneval/test-00000-of-00001.parquet"
DOMAIN = "code_python"

# Упоминание бенчмарка в файле — повод выбросить файл целиком
HUMANEVAL_NAME = re.compile(r"human[\s_\-]?eval", re.IGNORECASE)
# Лексемы для сравнения с HumanEval: идентификаторы, числа и одиночные знаки, пробелы и переносы не важны
LEXEME = re.compile(r"[A-Za-z_]\w*|\d+|\S")
# Ключевые слова и встроенные имена есть в любом коде: совпадение из них одних ничего не значит
COMMON_WORDS = set(keyword.kwlist) | set(dir(builtins))
# С этих строк начинаются окна из середины файла: определение верхнего уровня
TOP_LEVEL = re.compile(r"(def |async def |class |@)")


def parse_args():
    parser = argparse.ArgumentParser(description="Обучающая выборка для дистилляции драфта: Python из code_alltime")
    parser.add_argument("--eval-set", default="eval_set.jsonl", help="тестовый набор: его файлы в обучение не идут")
    parser.add_argument("--out", default="further_training.jsonl")
    parser.add_argument("--first-month", default="2017-01")
    parser.add_argument("--last-month", default="2024-12", help="тест собран из 2025 года, он остаётся нетронутым")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B", help="имя на HF Hub или путь к tokenizer.json")
    parser.add_argument("--max-tokens", type=int, default=512, help="длина окна в токенах")
    parser.add_argument("--min-head-tokens", type=int, default=64)
    parser.add_argument("--min-body-tokens", type=int, default=128)
    parser.add_argument("--body-windows", type=int, default=2, help="окон из середины файла, кроме начала")
    parser.add_argument("--files-per-repo", type=int, default=1000,
                        help="не больше стольких файлов из одного репозитория (берутся самые свежие)")
    parser.add_argument("--eval-overlap", type=float, default=0.3,
                        help="окно выбрасывается, если в нём есть такая доля строк какого-то тестового примера")
    parser.add_argument("--eval-run", type=int, default=80,
                        help="окно выбрасывается, если делит с тестовым примером столько символов подряд "
                             "(без пробельных)")
    parser.add_argument("--humaneval-ngram", type=int, default=8,
                        help="файл выбрасывается, если делит с HumanEval столько лексем подряд")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_tokenizer(name: str) -> Tokenizer:
    if name.endswith(".json"):
        return Tokenizer.from_file(name)
    return Tokenizer.from_file(hf_hub_download(name, "tokenizer.json"))


def load_python_files(first_month: str, last_month: str) -> list[dict]:
    files = []
    for path in sorted(list_repo_files(SOURCE, repo_type="dataset")):
        month = path.split("/")[0]
        if not path.endswith(".parquet") or not first_month <= month <= last_month:
            continue
        for row in pq.read_table(hf_hub_download(SOURCE, path, repo_type="dataset")).to_pylist():
            if row["file_path"].endswith(".py") and row["code"]:
                files.append({**row, "month": month})
    return files


def latest_versions(files: list[dict]) -> list[dict]:
    # Один и тот же файл попадает в датасет в разных коммитах — оставляем последнюю версию
    latest = {}
    for row in files:
        key = (row["repo_name"], row["file_path"])
        if key not in latest or row["commit_date"] > latest[key]["commit_date"]:
            latest[key] = row
    return list(latest.values())


def looks_like_code(text: str) -> bool:
    # Отсекаем таблицы данных, минифицированный и сгенерированный текст (фильтры как в Codex)
    lines = text.splitlines()
    if not lines:
        return False
    lengths = [len(line) for line in lines]
    alnum = sum(ch.isalnum() for ch in text) / len(text)
    return max(lengths) <= 1000 and sum(lengths) / len(lengths) <= 100 and alnum >= 0.25


def lexemes(text: str) -> list[str]:
    return LEXEME.findall(text)


def ngrams(tokens: list[str], n: int) -> set[tuple]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def informative(gram: tuple) -> bool:
    # Списки чисел и пунктуация (`1 , 2 , 3 , 4`, `) - > List [ int ] :`) совпадают с чем угодно:
    # учитываем только отрезки, где есть хотя бы три разных слова помимо общеупотребимых
    words = {t for t in gram if len(t) >= 2 and (t[0].isalpha() or t[0] == "_") and t not in COMMON_WORDS}
    return len(words) >= 3


def humaneval_ngrams(n: int) -> set[tuple]:
    path = hf_hub_download(HUMANEVAL, HUMANEVAL_FILE, repo_type="dataset")
    grams = set()
    for task in pq.read_table(path).to_pylist():
        # Условие с решением — как цельная функция; тесты отдельно
        grams |= ngrams(lexemes(task["prompt"] + task["canonical_solution"]), n)
        grams |= ngrams(lexemes(task["test"]), n)
    return {gram for gram in grams if informative(gram)}


def code_lines(text: str) -> set[str]:
    # Содержательные строки: короткие (`import os`, `else:`) есть в любом файле и ничего не значат
    return {line.strip() for line in text.splitlines() if len(line.strip()) >= 20}


def char_runs(text: str, n: int) -> set[str]:
    # Отрезки по n символов без пробельных: переформатирование и отступы совпадению не мешают
    text = "".join(text.split())
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def load_eval(path: str, run: int) -> tuple[set, list[set[str]], set[str]]:
    keys, examples, runs = set(), [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["domain"] != DOMAIN:
                continue
            keys.add((row["meta"]["repo_name"], row["meta"]["file_path"]))
            # Последняя строка reference обрезана посередине — её не сравниваем
            text = row["prompt"] + row["reference"]
            examples.append(code_lines(text[:text.rfind("\n") + 1]))
            runs |= char_runs(text, run)
    return keys, examples, runs


def eval_overlap(text: str, line_index: dict[str, list[int]], sizes: list[int]) -> float:
    # Наибольшая по тестовым примерам доля их строк, найденных в тексте
    hits = Counter()
    for line in code_lines(text):
        for example in line_index.get(line, ()):
            hits[example] += 1
    return max((count / sizes[example] for example, count in hits.items()), default=0.0)


def cut_windows(code: str, tokenizer: Tokenizer, args, rng: random.Random) -> list[dict]:
    lines = code.splitlines(keepends=True)
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))
    token_starts = [start for start, _ in tokenizer.encode(code, add_special_tokens=False).offsets]
    # Сколько токенов начинается до каждой строки
    tokens_before = [bisect.bisect_left(token_starts, start) for start in line_starts]

    def window(first: int, part: str, min_tokens: int) -> dict | None:
        last = bisect.bisect_right(tokens_before, tokens_before[first] + args.max_tokens) - 1
        while last > first:
            text = "".join(lines[first:last])
            n_tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
            if n_tokens <= args.max_tokens:
                break
            # Токен на стыке строк мог разрезаться иначе — убираем по строке, пока не влезет
            last -= 1
        if last <= first or n_tokens < min_tokens or not looks_like_code(text):
            return None
        return {"part": part, "first_line": first, "last_line": last, "n_tokens": n_tokens, "text": text}

    # Начало файла — как в eval_set.jsonl: без шапки из комментариев (лицензия, shebang) и пустых строк
    head_start = 0
    while head_start < len(lines) and (not lines[head_start].strip() or lines[head_start].lstrip().startswith("#")):
        head_start += 1
    if head_start == len(lines):
        return []

    windows = []
    head = window(head_start, "head", args.min_head_tokens)
    if head:
        windows.append(head)
    body_from = head["last_line"] if head else head_start

    starts = [i for i in range(body_from, len(lines)) if TOP_LEVEL.match(lines[i])]
    # Декоратор и его функция — одно определение: начинаем с первого декоратора
    starts = [i for i in starts if i == body_from or not lines[i - 1].startswith("@")]
    rng.shuffle(starts)
    taken = []
    for start in starts:
        if len(taken) == args.body_windows:
            break
        body = window(start, "body", args.min_body_tokens)
        if body and all(body["last_line"] <= a or body["first_line"] >= b for a, b in taken):
            taken.append((body["first_line"], body["last_line"]))
            windows.append(body)
    return windows


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    eval_keys, eval_examples, eval_runs = load_eval(args.eval_set, args.eval_run)
    line_index = defaultdict(list)
    for example, example_lines in enumerate(eval_examples):
        for line in example_lines:
            line_index[line].append(example)
    eval_sizes = [len(example_lines) for example_lines in eval_examples]
    he_grams = humaneval_ngrams(args.humaneval_ngram)

    dropped = Counter()
    files = load_python_files(args.first_month, args.last_month)
    print(f"Python-файлов за {args.first_month}..{args.last_month}: {len(files)}")

    # Тестовые файлы убираем во всех версиях, а не только последнюю
    kept = [row for row in files if (row["repo_name"], row["file_path"]) not in eval_keys]
    dropped["версия файла из eval_set"] = len(files) - len(kept)
    files = latest_versions(kept)
    dropped["старая версия того же файла"] = len(kept) - len(files)

    by_repo = defaultdict(list)
    for row in files:
        if HUMANEVAL_NAME.search(row["code"]) or HUMANEVAL_NAME.search(row["file_path"]):
            dropped["упоминание HumanEval"] += 1
        elif ngrams(lexemes(row["code"]), args.humaneval_ngram) & he_grams:
            dropped["общий фрагмент с HumanEval"] += 1
        elif len(row["code"]) < 200 or not looks_like_code(row["code"]):
            dropped["не похоже на код"] += 1
        else:
            by_repo[row["repo_name"]].append(row)

    rows, seen = [], set()
    for repo, repo_files in sorted(by_repo.items()):
        # Свежие файлы ближе к тестовому периоду; при равной дате порядок фиксируем по пути
        repo_files.sort(key=lambda row: (row["commit_date"], row["file_path"]), reverse=True)
        dropped["сверх лимита на репозиторий"] += max(len(repo_files) - args.files_per_repo, 0)
        for row in repo_files[:args.files_per_repo]:
            rng = random.Random(f"{args.seed}:{repo}:{row['file_path']}")
            for win in cut_windows(row["code"], tokenizer, args, rng):
                digest = hashlib.sha1(" ".join(win["text"].split()).encode()).digest()
                if digest in seen:
                    dropped["окно: дубликат"] += 1
                elif eval_overlap(win["text"], line_index, eval_sizes) >= args.eval_overlap:
                    dropped["окно: похоже на пример из eval_set"] += 1
                elif not eval_runs.isdisjoint(char_runs(win["text"], args.eval_run)):
                    # Шаблонные докстринги и импорты кочуют между файлами репозитория дословно
                    dropped["окно: общий кусок с примером из eval_set"] += 1
                else:
                    seen.add(digest)
                    rows.append({
                        "domain": DOMAIN, "source": SOURCE, "month": row["month"],
                        "meta": {"repo_name": repo, "file_path": row["file_path"],
                                 "commit_date": row["commit_date"], "sha": row["sha"], "lang": "python",
                                 "part": win["part"], "first_line": win["first_line"],
                                 "n_tokens": win["n_tokens"]},
                        "text": win["text"],
                    })

    # Перемешиваем, чтобы любой префикс файла был представительной подвыборкой
    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            f.write(json.dumps({"id": f"ft_{DOMAIN}-{i:05d}", **row}, ensure_ascii=False) + "\n")

    print("Отброшено:")
    for reason, count in dropped.items():
        print(f"  {reason}: {count}")
    n_tokens = sum(row["meta"]["n_tokens"] for row in rows)
    print(f"Записано {len(rows)} окон, {n_tokens / 1e6:.2f}M токенов -> {args.out}")
    print("  части:", dict(Counter(row["meta"]["part"] for row in rows)))
    for repo, count in Counter(row["meta"]["repo_name"] for row in rows).most_common():
        print(f"  {repo:<32} {count:>6} {count / len(rows):>6.1%}")


if __name__ == "__main__":
    main()
