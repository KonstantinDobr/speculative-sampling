"""Три набора для дообучения драфта на одном домене — языке C (файлы .c из RealTimeData/code_alltime):

    further_training_c.jsonl — обучение: окна до 512 токенов Qwen в поле text, формат как у further_training.jsonl
    validation_c.jsonl       — валидация по ходу дообучения (вместо HumanEval): prompt + reference
    compar_graph_c.jsonl     — тест для графиков «до и после дообучения»: prompt + reference, два слоя (поле domain)
        code_c_samerepo — репозитории те же, что в обучении, файлы новые: последняя версия файла — 2025 год
        code_c_newrepo  — репозитории, которых в обучении нет совсем (--new-repos)

Каждый файл целиком (во всех версиях) попадает только в один набор:
    репозиторий из --new-repos            -> тест, слой code_c_newrepo
    последняя версия с --test-first-month -> тест, слой code_c_samerepo
    случайные файлы из оставшихся         -> валидация (--val-examples примеров)
    всё остальное                         -> обучение

Валидация и тест нарезаны как eval_set.jsonl: начало файла без шапки из комментариев, 128 токенов SmolLM2 — prompt,
следующие 256 — reference. Файлов из eval_set.jsonl нет ни в одном наборе. Общих кусков у наборов тоже нет:
из обучения выброшены окна, похожие на пример валидации, теста или eval_set (в репозиториях лежат копии одних
и тех же библиотек: zlib, libjpeg, freetype), из слоя code_c_newrepo — примеры, чей текст есть в файлах
репозиториев обучения.

Кэш таргета и дообучение:
    python teacher_cache.py --parts train --train further_training_c.jsonl --out-dir cache/<pair>_c
    python teacher_cache.py --parts train --train validation_c.jsonl --source reference --out-dir cache/<pair>_c
    python finetune.py --loss kl --train-cache cache/<pair>_c/train_generated.pt \
        --eval-cache cache/<pair>_c/train_reference.pt --out-dir checkpoints/<pair>_c_kl
"""
import argparse
import bisect
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from build_compar_graph import max_overlap, pick
from build_further_training import char_runs, code_lines, eval_overlap, latest_versions, load_tokenizer, looks_like_code

SOURCE = "RealTimeData/code_alltime"
DOMAIN = "code_c"
SAME_REPO, NEW_REPO = "code_c_samerepo", "code_c_newrepo"
# Небольшие репозитории: вместе около десятой части файлов, обучение почти не теряет
NEW_REPOS = ["sqlite_sqlite", "pytorch_pytorch", "apache_hadoop", "aseprite_aseprite", "NationalSecurityAgency_ghidra",
             "pandas-dev_pandas", "lua_lua", "overleaf_overleaf", "flynn_flynn", "microsoft_PowerToys"]

# Шапка файла: пустые строки, строки // и блоки /* */ (лицензия, описание файла)
HEADER = re.compile(r"(?:[ \t]*\n|[ \t]*//[^\n]*\n?|[ \t]*/\*.*?\*/[ \t]*\n?)*", re.DOTALL)
# Незакрытый /* в конце окна — тоже комментарий
COMMENT = re.compile(r"//[^\n]*|/\*.*?(?:\*/|\Z)", re.DOTALL)
NUMBER = re.compile(r"\b(?:0[xX][0-9a-fA-F]+|\d+)[uUlL]*\b")
# С этих строк окно из середины файла не начинается: хвост предыдущего определения или ветка препроцессора
NOT_A_START = re.compile(r"[\s{})]|#\s*(?:else|elif|endif)")


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение, валидация и тест для графиков: только язык C")
    parser.add_argument("--eval-set", default="eval_set.jsonl", help="исходный тест: его файлы никуда не идут")
    parser.add_argument("--train-out", default="further_training_c.jsonl")
    parser.add_argument("--val-out", default="validation_c.jsonl")
    parser.add_argument("--test-out", default="compar_graph_c.jsonl")
    parser.add_argument("--first-month", default="2017-01")
    parser.add_argument("--last-month", default="2025-06")
    parser.add_argument("--test-first-month", default="2025-01",
                        help="файлы с последней версией не раньше этого месяца — слой code_c_samerepo")
    parser.add_argument("--new-repos", nargs="+", default=NEW_REPOS, help="репозитории слоя code_c_newrepo")
    parser.add_argument("--val-examples", type=int, default=200)
    parser.add_argument("--per-domain", type=int, default=300, help="не больше стольких примеров в слое теста")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B",
                        help="им меряются обучающие окна; имя на HF Hub или путь к tokenizer.json")
    parser.add_argument("--cut-tokenizer", default="HuggingFaceTB/SmolLM2-135M",
                        help="им нарезаны eval_set.jsonl, валидация и тест")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--reference-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=512, help="длина обучающего окна в токенах")
    parser.add_argument("--min-head-tokens", type=int, default=64)
    parser.add_argument("--min-body-tokens", type=int, default=128)
    parser.add_argument("--body-windows", type=int, default=8, help="окон из середины файла, кроме начала")
    parser.add_argument("--files-per-repo", type=int, default=1000,
                        help="не больше стольких файлов из одного репозитория в обучении (берутся самые свежие)")
    parser.add_argument("--max-comments", type=float, default=0.8,
                        help="текст выбрасывается, если комментариев в нём больше такой доли символов")
    parser.add_argument("--max-numbers", type=float, default=0.3,
                        help="текст выбрасывается, если чисел в нём больше такой доли символов (таблицы констант)")
    parser.add_argument("--overlap", type=float, default=0.3,
                        help="тексты считаются похожими, если такая доля строк примера есть в другом тексте")
    parser.add_argument("--run", type=int, default=80,
                        help="тексты считаются похожими, если делят столько символов подряд (без пробельных)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_c_files(first_month: str, last_month: str) -> list[dict]:
    root = Path(snapshot_download(SOURCE, repo_type="dataset", allow_patterns="*.parquet"))
    files = []
    for path in sorted(root.glob("*/*.parquet")):
        month = path.parent.name
        if not first_month <= month <= last_month:
            continue
        table = pq.read_table(path, columns=["file_path", "code", "repo_name", "commit_date", "sha"])
        # Только .c: заголовков .h в архиве нет, а .cpp, .cc и .m — другие языки
        table = table.filter([name.endswith(".c") for name in table.column("file_path").to_pylist()])
        files += [{**row, "month": month} for row in table.to_pylist() if row["code"]]
    return files


def load_eval(path: str) -> tuple[set, list[str]]:
    keys, texts = set(), []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["domain"].startswith("code_"):
                keys.add((row["meta"]["repo_name"], row["meta"]["file_path"]))
                texts.append(row["prompt"] + row["reference"])
    return keys, texts


def strip_header(code: str) -> str:
    return code[HEADER.match(code).end():]


def share(pattern: re.Pattern, text: str) -> float:
    # Доля непробельных символов текста, попавших в совпадения
    matched = sum(len("".join(match.group().split())) for match in pattern.finditer(text))
    return matched / max(len("".join(text.split())), 1)


def looks_like_c(text: str, args) -> bool:
    # Сплошные комментарии (в тестах clang это проверки FileCheck с LLVM IR) и таблицы чисел — не код на C
    return (looks_like_code(text) and share(COMMENT, text) <= args.max_comments
            and share(NUMBER, text) <= args.max_numbers)


def complete_lines(text: str) -> str:
    # Последняя строка reference обрезана посередине — её не сравниваем
    return text[:text.rfind("\n") + 1]


def digest(text: str) -> bytes:
    return hashlib.sha1(" ".join(text.split()).encode()).digest()


def run_hashes(text: str, n: int, wanted: set[int] | None = None) -> set[int]:
    # Хэши отрезков по n символов без пробельных. С wanted — только те, что есть в нём: файлы бывают
    # по несколько мегабайт, все их отрезки в память не складываем
    text = "".join(text.split())
    hashes = (hash(text[i:i + n]) for i in range(len(text) - n + 1))
    return set(hashes) if wanted is None else {h for h in hashes if h in wanted}


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


def example(row: dict, domain: str, tokenizer: Tokenizer, args) -> dict | None:
    parts = cut(row["code"], tokenizer, args)
    if parts is None:
        return None
    return {"domain": domain, "source": SOURCE, "month": row["month"], "group": row["repo_name"],
            "meta": {"repo_name": row["repo_name"], "file_path": row["file_path"],
                     "commit_date": row["commit_date"], "lang": "c"},
            "prompt": parts[0], "reference": parts[1]}


def top_level_starts(lines: list[str]) -> list[int]:
    # Определение верхнего уровня (вместе с комментарием перед ним): строка без отступа после пустой, не внутри /* */
    starts, in_comment = [], False
    for i, line in enumerate(lines):
        if not in_comment and i > 0 and not lines[i - 1].strip() and not NOT_A_START.match(line):
            starts.append(i)
        rest = line
        while True:
            if in_comment:
                end = rest.find("*/")
                if end == -1:
                    break
                rest, in_comment = rest[end + 2:], False
            else:
                start = rest.find("/*")
                if start == -1 or "//" in rest[:start]:
                    break
                rest, in_comment = rest[start + 2:], True
    return starts


def cut_windows(code: str, tokenizer: Tokenizer, args, rng: random.Random) -> list[dict]:
    # Как в build_further_training.py, но шапка и начала определений — по правилам C
    text = strip_header(code)
    skipped_lines = code[:len(code) - len(text)].count("\n")
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))
    token_starts = [start for start, _ in tokenizer.encode(text, add_special_tokens=False).offsets]
    # Сколько токенов начинается до каждой строки
    tokens_before = [bisect.bisect_left(token_starts, start) for start in line_starts]

    def window(first: int, part: str, min_tokens: int) -> dict | None:
        last = bisect.bisect_right(tokens_before, tokens_before[first] + args.max_tokens) - 1
        while last > first:
            piece = "".join(lines[first:last])
            n_tokens = len(tokenizer.encode(piece, add_special_tokens=False).ids)
            if n_tokens <= args.max_tokens:
                break
            # Токен на стыке строк мог разрезаться иначе — убираем по строке, пока не влезет
            last -= 1
        if last <= first or n_tokens < min_tokens or not looks_like_c(piece, args):
            return None
        return {"part": part, "first_line": skipped_lines + first, "last_line": skipped_lines + last,
                "n_tokens": n_tokens, "text": piece}

    windows = []
    head = window(0, "head", args.min_head_tokens)
    if head:
        windows.append(head)
    body_from = head["last_line"] - skipped_lines if head else 0

    starts = [i for i in top_level_starts(lines) if i >= body_from]
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


def write(path: str, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def report(dropped: Counter):
    for reason, count in dropped.items():
        print(f"  отброшено — {reason}: {count}")


def repos(rows: list[dict], key=lambda row: row["meta"]["repo_name"]) -> str:
    return ", ".join(f"{repo} {count}" for repo, count in Counter(map(key, rows)).most_common())


def main():
    args = parse_args()
    tokenizer, cut_tokenizer = load_tokenizer(args.tokenizer), load_tokenizer(args.cut_tokenizer)
    eval_keys, eval_texts = load_eval(args.eval_set)

    dropped = Counter()
    files = load_c_files(args.first_month, args.last_month)
    print(f"Файлов .c за {args.first_month}..{args.last_month}: {len(files)}")
    # Файлы eval_set убираем во всех версиях, а не только последнюю
    kept = [row for row in files if (row["repo_name"], row["file_path"]) not in eval_keys]
    dropped["версия файла из eval_set"] = len(files) - len(kept)
    files = latest_versions(kept)
    dropped["старая версия того же файла"] = len(kept) - len(files)
    kept = [row for row in files if len(row["code"]) >= 200 and looks_like_code(row["code"])]
    dropped["не похоже на код"] = len(files) - len(kept)
    files = sorted(kept, key=lambda row: (row["repo_name"], row["file_path"]))
    report(dropped)

    new_repo = [row for row in files if row["repo_name"] in args.new_repos]
    known = [row for row in files if row["repo_name"] not in args.new_repos]
    same_repo = [row for row in known if row["month"] >= args.test_first_month]
    known = [row for row in known if row["month"] < args.test_first_month]

    # --- Тест. Чужой репозиторий не должен повторять свои: его примеры сверяются с их файлами целиком ---
    layers = [(SAME_REPO, same_repo, []), (NEW_REPO, new_repo, [row["code"] for row in known])]
    held_out, seen_digests = [], set()
    for domain, layer, seen_texts in layers:
        dropped = Counter()
        candidates = []
        for row in layer:
            candidate = example(row, domain, cut_tokenizer, args)
            if candidate is None:
                dropped["файл короче prompt + reference"] += 1
            else:
                candidates.append(candidate)

        line_index = defaultdict(list)
        for i, text in enumerate(seen_texts):
            for line in code_lines(text):
                line_index[line].append(i)
        wanted = set()
        for row in candidates:
            wanted |= run_hashes(row["prompt"] + row["reference"], args.run)
        shared = set()
        for text in seen_texts:
            shared |= run_hashes(text, args.run, wanted)

        kept = []
        for row in candidates:
            text = row["prompt"] + row["reference"]
            if not looks_like_c(text, args):
                dropped["не похоже на код на C"] += 1
            elif digest(text) in seen_digests:
                dropped["дубликат"] += 1
            elif max_overlap(complete_lines(text), line_index) >= args.overlap:
                dropped["похоже на файл из репозиториев обучения"] += 1
            elif not shared.isdisjoint(run_hashes(text, args.run)):
                dropped["общий кусок с файлом из репозиториев обучения"] += 1
            else:
                seen_digests.add(digest(text))
                kept.append(row)
        picked = pick(kept, args.per_domain, random.Random(f"{args.seed}:{domain}"))
        dropped["сверх лимита на слой"] = len(kept) - len(picked)
        held_out.append(picked)
        print(f"{domain}: файлов {len(layer)}, записано {len(picked)}")
        report(dropped)
        print(f"  репозитории: {repos(picked)}")
    test = [{"id": f"{row['domain']}-{i:04d}", **{k: v for k, v in row.items() if k != "group"}}
            for picked in held_out for i, row in enumerate(picked)]
    write(args.test_out, test)

    # --- Валидация: случайные файлы своих репозиториев, не похожие на тест ---
    seen_texts = eval_texts + [row["prompt"] + row["reference"] for row in test]
    line_index = defaultdict(list)
    for i, text in enumerate(seen_texts):
        for line in code_lines(text):
            line_index[line].append(i)
    seen_runs = set()
    for text in seen_texts:
        seen_runs |= run_hashes(text, args.run)

    dropped = Counter()
    order = list(known)
    random.Random(f"{args.seed}:val").shuffle(order)
    val, val_keys = [], set()
    for row in order:
        if len(val) == args.val_examples:
            break
        candidate = example(row, DOMAIN, cut_tokenizer, args)
        if candidate is None:
            dropped["файл короче prompt + reference (ушёл в обучение)"] += 1
            continue
        text = candidate["prompt"] + candidate["reference"]
        if not looks_like_c(text, args):
            dropped["не похоже на код на C (ушёл в обучение)"] += 1
        elif digest(text) in seen_digests:
            dropped["дубликат (ушёл в обучение)"] += 1
        elif max_overlap(complete_lines(text), line_index) >= args.overlap:
            dropped["похоже на тест или eval_set (ушёл в обучение)"] += 1
        elif not seen_runs.isdisjoint(run_hashes(text, args.run)):
            dropped["общий кусок с тестом или eval_set (ушёл в обучение)"] += 1
        else:
            seen_digests.add(digest(text))
            val_keys.add((row["repo_name"], row["file_path"]))
            val.append({"id": f"val_{DOMAIN}-{len(val):04d}",
                        **{k: v for k, v in candidate.items() if k != "group"}})
    write(args.val_out, val)
    print(f"{DOMAIN} (валидация): записано {len(val)}")
    report(dropped)
    print(f"  репозитории: {repos(val)}")

    # --- Обучение: всё остальное, без окон, похожих на валидацию, тест и eval_set ---
    held_texts = seen_texts + [row["prompt"] + row["reference"] for row in val]
    held_lines = [code_lines(complete_lines(text)) for text in held_texts]
    line_index = defaultdict(list)
    for i, lines in enumerate(held_lines):
        for line in lines:
            line_index[line].append(i)
    held_sizes = [len(lines) for lines in held_lines]
    held_runs = set()
    for text in held_texts:
        held_runs |= char_runs(text, args.run)

    dropped = Counter()
    by_repo = defaultdict(list)
    for row in known:
        if (row["repo_name"], row["file_path"]) not in val_keys:
            by_repo[row["repo_name"]].append(row)

    rows = []
    for repo, repo_files in sorted(by_repo.items()):
        # Свежие файлы ближе к тестовому периоду; при равной дате порядок фиксируем по пути
        repo_files.sort(key=lambda row: (row["commit_date"], row["file_path"]), reverse=True)
        dropped["сверх лимита на репозиторий"] += max(len(repo_files) - args.files_per_repo, 0)
        for row in repo_files[:args.files_per_repo]:
            rng = random.Random(f"{args.seed}:{repo}:{row['file_path']}")
            for win in cut_windows(row["code"], tokenizer, args, rng):
                if digest(win["text"]) in seen_digests:
                    dropped["окно: дубликат"] += 1
                elif eval_overlap(win["text"], line_index, held_sizes) >= args.overlap:
                    dropped["окно: похоже на пример валидации, теста или eval_set"] += 1
                elif not held_runs.isdisjoint(char_runs(win["text"], args.run)):
                    dropped["окно: общий кусок с примером валидации, теста или eval_set"] += 1
                else:
                    seen_digests.add(digest(win["text"]))
                    rows.append({
                        "domain": DOMAIN, "source": SOURCE, "month": row["month"],
                        "meta": {"repo_name": repo, "file_path": row["file_path"],
                                 "commit_date": row["commit_date"], "sha": row["sha"], "lang": "c",
                                 "part": win["part"], "first_line": win["first_line"],
                                 "n_tokens": win["n_tokens"]},
                        "text": win["text"],
                    })

    # Перемешиваем, чтобы любой префикс файла был представительной подвыборкой
    random.Random(args.seed).shuffle(rows)
    write(args.train_out, [{"id": f"ft_{DOMAIN}-{i:05d}", **row} for i, row in enumerate(rows)])
    n_tokens = sum(row["meta"]["n_tokens"] for row in rows)
    n_files = len({(row["meta"]["repo_name"], row["meta"]["file_path"]) for row in rows})
    print(f"{DOMAIN} (обучение): записано {len(rows)} окон из {n_files} файлов, {n_tokens / 1e6:.2f}M токенов")
    report(dropped)
    print("  части:", dict(Counter(row["meta"]["part"] for row in rows)))
    print(f"  репозитории: {repos(rows)}")
    print(f"Обучение -> {args.train_out}, валидация -> {args.val_out}, тест -> {args.test_out}")


if __name__ == "__main__":
    main()
