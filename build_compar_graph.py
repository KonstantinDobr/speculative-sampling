"""Тестовый набор для графиков сравнения дообученных драфтов: только Python, формат как у eval_set.jsonl.

Три слоя по удалённости от обучающих данных (поле domain):
    code_py_samerepo — те же репозитории, что в further_training.jsonl, но файлы, которых не было ни там,
                       ни в eval_set.jsonl (RealTimeData/code_alltime за 2025 год)
    code_py_newrepo  — чужие репозитории (CrossCodeEval: созданы в марте — июне 2023)
    code_py_tasks    — задачи в стиле HumanEval, но не HumanEval (BigCodeBench: условие → решение)

Первые два слоя нарезаны так же, как eval_set.jsonl: начало файла без шапки, 128 токенов SmolLM2 — prompt,
следующие 256 — reference (этим правилом eval_set.jsonl воспроизводится посимвольно). У задач prompt — условие,
reference — эталонное решение целиком.

В набор не попадает ничего из further_training.jsonl, eval_set.jsonl и HumanEval (по нему finetune.py выбирает
лучший чекпойнт): ни те же файлы, ни тексты с общими с ними кусками.
"""
import argparse
import ast
import hashlib
import json
import random
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer

from build_further_training import (HUMANEVAL_NAME, char_runs, code_lines, humaneval_ngrams, latest_versions,
                                    lexemes, load_tokenizer, looks_like_code, ngrams)

CODE_ALLTIME = "RealTimeData/code_alltime"
CCEVAL = "amazon-science/cceval"
CCEVAL_URL = "https://raw.githubusercontent.com/amazon-science/cceval/main/data/crosscodeeval_data.tar.xz"
CCEVAL_MEMBER = "python/line_completion.jsonl"
BIGCODEBENCH = "bigcode/bigcodebench"
SAME_REPO, NEW_REPO, TASKS = "code_py_samerepo", "code_py_newrepo", "code_py_tasks"


def parse_args():
    parser = argparse.ArgumentParser(description="Тестовый набор для графиков: Python, которого не было в обучении")
    parser.add_argument("--eval-set", default="eval_set.jsonl", help="исходный тест: в набор не идёт")
    parser.add_argument("--train", default="further_training.jsonl", help="обучающая выборка: в набор не идёт")
    parser.add_argument("--out", default="compar_graph.jsonl")
    parser.add_argument("--per-domain", type=int, default=300, help="примеров в каждом слое")
    parser.add_argument("--first-month", default="2025-01", help="обучение собрано по 2024-12")
    parser.add_argument("--last-month", default="2025-06")
    parser.add_argument("--cceval", default="cache/crosscodeeval_data.tar.xz",
                        help="архив CrossCodeEval; если файла нет, скачивается с GitHub")
    parser.add_argument("--bigcodebench-version", default="v0.1.4")
    parser.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M",
                        help="им нарезан eval_set.jsonl; имя на HF Hub или путь к tokenizer.json")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--reference-tokens", type=int, default=256)
    parser.add_argument("--min-task-tokens", type=int, default=32, help="короче — решение задачи не берём")
    parser.add_argument("--max-task-prompt-tokens", type=int, default=1024, help="длиннее — условие задачи не берём")
    parser.add_argument("--overlap", type=float, default=0.3,
                        help="пример выбрасывается, если такая доля его строк есть в одном обучающем окне "
                             "или тестовом примере")
    parser.add_argument("--run", type=int, default=80,
                        help="пример выбрасывается, если делит с обучением или тестом столько символов подряд "
                             "(без пробельных)")
    parser.add_argument("--humaneval-ngram", type=int, default=8,
                        help="пример выбрасывается, если делит с HumanEval столько лексем подряд")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_seen(eval_path: str, train_path: str) -> tuple[set, list[str]]:
    # Всё, что драфт уже видел при обучении или на чём его уже оценивали
    keys, texts = set(), []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["domain"].startswith("code_"):
                keys.add((row["meta"]["repo_name"], row["meta"]["file_path"]))
                texts.append(row["prompt"] + row["reference"])
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            keys.add((row["meta"]["repo_name"], row["meta"]["file_path"]))
            texts.append(row["text"])
    return keys, texts


def strip_header(code: str) -> str:
    # Как в eval_set.jsonl: без шапки из комментариев (лицензия, shebang) и пустых строк
    lines = code.splitlines(keepends=True)
    start = 0
    while start < len(lines) and (not lines[start].strip() or lines[start].lstrip().startswith("#")):
        start += 1
    return "".join(lines[start:])


def cut(code: str, tokenizer: Tokenizer, args) -> tuple[str, str] | None:
    text = strip_header(code)
    offsets = tokenizer.encode(text, add_special_tokens=False).offsets
    total = args.prompt_tokens + args.reference_tokens
    if len(offsets) <= total:
        return None
    middle, end = offsets[args.prompt_tokens][0], offsets[total][0]
    # Отступ на стыке уходит в reference: prompt не кончается пробелами
    while text[middle - 1] in " \t":
        middle -= 1
    return text[:middle], text[middle:end]


def same_repo_candidates(tokenizer: Tokenizer, seen_keys: set, args) -> tuple[list[dict], Counter]:
    dropped = Counter()
    files = []
    for path in sorted(list_repo_files(CODE_ALLTIME, repo_type="dataset")):
        month = path.split("/")[0]
        if not path.endswith(".parquet") or not args.first_month <= month <= args.last_month:
            continue
        for row in pq.read_table(hf_hub_download(CODE_ALLTIME, path, repo_type="dataset")).to_pylist():
            if row["file_path"].endswith(".py") and row["code"]:
                files.append({**row, "month": month})

    kept = [row for row in files if (row["repo_name"], row["file_path"]) not in seen_keys]
    dropped["файл есть в обучении или в eval_set"] = len(files) - len(kept)
    files = latest_versions(kept)
    dropped["старая версия того же файла"] = len(kept) - len(files)

    candidates = []
    for row in sorted(files, key=lambda row: (row["repo_name"], row["file_path"])):
        parts = cut(row["code"], tokenizer, args)
        if parts is None:
            dropped["файл короче prompt + reference"] += 1
            continue
        candidates.append({
            "domain": SAME_REPO, "source": CODE_ALLTIME, "month": row["month"], "group": row["repo_name"],
            "mentions": row["file_path"] + row["code"],
            "meta": {"repo_name": row["repo_name"], "file_path": row["file_path"],
                     "commit_date": row["commit_date"], "lang": "python"},
            "prompt": parts[0], "reference": parts[1],
        })
    return candidates, dropped


def new_repo_candidates(tokenizer: Tokenizer, args) -> tuple[list[dict], Counter]:
    dropped = Counter()
    archive = Path(args.cceval)
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CCEVAL_URL, archive)

    # Задача CrossCodeEval — дописать строку посреди файла; prompt + groundtruth + right_context — файл целиком
    files = {}
    with tarfile.open(archive) as tar:
        for line in tar.extractfile(CCEVAL_MEMBER):
            row = json.loads(line)
            key = (row["metadata"]["repository"], row["metadata"]["file"])
            files[key] = row["prompt"] + row["groundtruth"] + row["right_context"]

    candidates = []
    for (repo, file_path), code in sorted(files.items()):
        parts = cut(code, tokenizer, args)
        if parts is None:
            dropped["файл короче prompt + reference"] += 1
            continue
        candidates.append({
            # Дат у файлов нет: репозитории созданы с марта по июнь 2023. Имя репозитория — owner-repo-коммит
            "domain": NEW_REPO, "source": CCEVAL, "month": None, "group": repo, "mentions": file_path + code,
            "meta": {"repo_name": repo, "file_path": file_path, "lang": "python"},
            "prompt": parts[0], "reference": parts[1],
        })
    return candidates, dropped


def task_candidates(tokenizer: Tokenizer, args) -> tuple[list[dict], Counter]:
    dropped = Counter()
    path = hf_hub_download(BIGCODEBENCH, f"data/{args.bigcodebench_version}-00000-of-00001.parquet",
                           repo_type="dataset")
    candidates = []
    for task in pq.read_table(path).to_pylist():
        prompt, reference = task["complete_prompt"], task["canonical_solution"]
        n_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
        if n_tokens(reference) < args.min_task_tokens:
            dropped["решение слишком короткое"] += 1
        elif n_tokens(prompt) > args.max_task_prompt_tokens:
            dropped["условие слишком длинное"] += 1
        else:
            candidates.append({
                # Каждая задача — сама себе группа: ограничивать тут нечего
                "domain": TASKS, "source": BIGCODEBENCH, "month": None, "group": task["task_id"],
                "mentions": prompt + reference,
                "meta": {"task_id": task["task_id"], "entry_point": task["entry_point"],
                         "libs": sorted(ast.literal_eval(task["libs"])), "lang": "python"},
                "prompt": prompt, "reference": reference,
            })
    return candidates, dropped


def run_hashes(text: str, n: int) -> set[int]:
    # Хэши вместо самих отрезков: отрезков в обучающей выборке миллионы
    return {hash(run) for run in char_runs(text, n)}


def shared_runs(candidates: list[dict], seen_texts: list[str], n: int) -> set[int]:
    wanted = set()
    for row in candidates:
        wanted |= run_hashes(row["prompt"] + row["reference"], n)
    shared = set()
    for text in seen_texts:
        shared |= wanted & run_hashes(text, n)
    return shared


def max_overlap(text: str, line_index: dict[str, list[int]]) -> float:
    # Наибольшая по виденным текстам доля строк примера, найденных в одном из них
    lines = code_lines(text)
    hits = Counter()
    for line in lines:
        for seen in line_index.get(line, ()):
            hits[seen] += 1
    return max(hits.values(), default=0) / max(len(lines), 1)


def pick(candidates: list[dict], limit: int, rng: random.Random) -> list[dict]:
    # По кругу по репозиториям: в любом префиксе слоя репозитории представлены поровну, насколько хватает файлов
    # (eval_domains.py берёт для генерации первые примеры домена)
    groups = defaultdict(list)
    for row in candidates:
        groups[row["group"]].append(row)
    queues = [groups[name] for name in sorted(groups)]
    for queue in queues:
        rng.shuffle(queue)
    rng.shuffle(queues)

    picked = []
    while queues and len(picked) < limit:
        for queue in queues:
            if len(picked) < limit:
                picked.append(queue.pop())
        queues = [queue for queue in queues if queue]
    return picked


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    seen_keys, seen_texts = load_seen(args.eval_set, args.train)
    he_grams = humaneval_ngrams(args.humaneval_ngram)

    layers = [same_repo_candidates(tokenizer, seen_keys, args), new_repo_candidates(tokenizer, args),
              task_candidates(tokenizer, args)]
    candidates = [row for layer, _ in layers for row in layer]

    line_index = defaultdict(list)
    for i, text in enumerate(seen_texts):
        for line in code_lines(text):
            line_index[line].append(i)
    shared = shared_runs(candidates, seen_texts, args.run)

    rows, seen_digests = [], set()
    for (layer, dropped), domain in zip(layers, [SAME_REPO, NEW_REPO, TASKS]):
        kept = []
        for row in layer:
            text = row["prompt"] + row["reference"]
            digest = hashlib.sha1(" ".join(text.split()).encode()).digest()
            if HUMANEVAL_NAME.search(row["mentions"]):
                dropped["упоминание HumanEval"] += 1
            elif ngrams(lexemes(text), args.humaneval_ngram) & he_grams:
                dropped["общий фрагмент с HumanEval"] += 1
            elif not looks_like_code(text):
                dropped["не похоже на код"] += 1
            elif digest in seen_digests:
                dropped["дубликат"] += 1
            elif max_overlap(text, line_index) >= args.overlap:
                dropped["похоже на обучающее окно или пример из eval_set"] += 1
            elif not shared.isdisjoint(run_hashes(text, args.run)):
                dropped["общий кусок с обучением или eval_set"] += 1
            else:
                seen_digests.add(digest)
                kept.append(row)

        picked = pick(kept, args.per_domain, random.Random(f"{args.seed}:{domain}"))
        dropped["сверх лимита на слой"] = len(kept) - len(picked)
        for i, row in enumerate(picked):
            rows.append({"id": f"{domain}-{i:04d}", "domain": domain, "source": row["source"],
                         "month": row["month"], "meta": row["meta"], "prompt": row["prompt"],
                         "reference": row["reference"]})

        print(f"{domain}: кандидатов {len(layer)}, записано {len(picked)}")
        for reason, count in dropped.items():
            print(f"  отброшено — {reason}: {count}")
        if domain != TASKS:
            repos = Counter(row["group"] for row in picked)
            top = ", ".join(f"{repo} {count}" for repo, count in repos.most_common(8))
            print(f"  репозиториев: {len(repos)} ({top})")

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Записано {len(rows)} примеров -> {args.out}")


if __name__ == "__main__":
    main()
