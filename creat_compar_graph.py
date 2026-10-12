"""Графики сравнения трёх драфтов — без дообучения, с KL-дистилляцией и с CE — по числам из
run_compar_graph.sbatch (папка results/compar_graph: по подпапке на драфт, в каждой итог eval_domains.py
на compar_graph.jsonl, а у дообученных ещё history.json от finetune.py).

Все метрики посчитаны с той же температурой и top-p, с которыми таргет генерировал продолжения для
дообучения: программа сверяет это по cache.json и history.json и при расхождении графики не строит.

Что получается:

    accept_vs_test_prefix.png   по оси X — первые N примеров compar_graph.jsonl, по оси Y — доля принятых
                                токенов драфта в SpS на этих N примерах; три драфта
    alpha_vs_test_prefix.png    то же для α (teacher forcing, все примеры набора)
    training_curves.png         по оси X — шаг дообучения, по оси Y — α, top-1 и KL на HumanEval,
                                как их записал finetune.py по ходу обучения
    final_comparison.png        три драфта по слоям теста: доля принятых токенов, α и KL с интервалами
    <метрика>_vs_train_size.png по оси X — на скольких примерах further_training.jsonl дообучен драфт
                                (шагов обучения везде поровну); строится, если размеров больше одного
    <метрика>_gain_vs_train_size.png  прирост главной метрики к драфту без дообучения
    summary.csv                 все точки графиков числами

Метрики: доля принятых токенов — принятые таргетом из предложенных драфтом при генерации SpS;
α = Σ min(p, q) — вероятность принять токен драфта на тексте из набора; top-1 — доля позиций, где самый
вероятный токен у драфта и таргета один; KL(p ‖ q). Интервалы — 95%-й бутстрап по примерам теста.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator

BASE = "base"
LOSSES = {
    "kl": {"label": "KL-дистилляция (мягкие метки)", "short": "KL", "color": "#2a78d6", "marker": "o"},
    "ce": {"label": "Кросс-энтропия (жёсткие метки)", "short": "CE", "color": "#eb6834", "marker": "s"},
}
BASE_STYLE = {"label": "Без дообучения", "color": "#6f6e69", "marker": "D"}
# Метрика — отношение сумм по примерам: kind — из каких записей eval_domains.py (tf — teacher forcing,
# sps — генерация SpS), num и den — числитель и знаменатель одного примера
METRICS = {
    "accept": {"kind": "sps", "num": lambda r: r["stats"]["accepted"], "den": lambda r: r["stats"]["drafted"],
               "name": "Доля принятых токенов драфта в SpS", "gain": "Прирост доли принятых токенов",
               "better": "больше — лучше"},
    "alpha": {"kind": "tf", "num": lambda r: r["alpha"], "den": lambda r: r["tokens"],
              "name": "α — вероятность принять токен драфта", "gain": "Прирост α", "better": "больше — лучше"},
    "top1": {"kind": "tf", "num": lambda r: r["top1"], "den": lambda r: r["tokens"],
             "name": "Совпадение top-1 драфта и таргета", "gain": "Прирост совпадения top-1",
             "better": "больше — лучше"},
    "kl": {"kind": "tf", "num": lambda r: r["kl_pq"], "den": lambda r: r["tokens"],
           "name": "KL(p ‖ q), нат", "gain": "Изменение KL", "better": "меньше — лучше"},
}
# Метрика -> поле, в которое finetune.py пишет её в history.json на каждом шаге оценки
HISTORY_FIELDS = {"alpha": "alpha", "top1": "top1", "kl": "kl"}
DOMAIN_TITLES = {
    "code_py_samerepo": "Те же репозитории, новые файлы",
    "code_py_newrepo": "Чужие репозитории",
    "code_py_tasks": "Задачи (BigCodeBench)",
    "code_c_samerepo": "Те же репозитории, новые файлы",
    "code_c_newrepo": "Чужие репозитории",
}
DOMAIN_TICKS = {"code_py_samerepo": "Те же\nрепозитории", "code_py_newrepo": "Чужие\nрепозитории",
                "code_py_tasks": "Задачи", "code_c_samerepo": "Те же\nрепозитории",
                "code_c_newrepo": "Чужие\nрепозитории"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5df"
DASHED = (0, (5, 3))
# Доля по первым нескольким примерам скачет и растягивает шкалу — кривая начинается отсюда
FIRST_PREFIX = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Графики сравнения дообученных драфтов на compar_graph.jsonl")
    parser.add_argument("--results", default="results/compar_graph", help="папка из run_compar_graph.sbatch")
    parser.add_argument("--out", default="graphs")
    parser.add_argument("--bootstrap", type=int, default=2000, help="выборок бутстрапа для интервалов")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--watermark", default=None, help="надпись поверх каждого графика (черновик, проба)")
    return parser.parse_args()


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_variant(folder: Path, cache: dict) -> dict | None:
    summaries = sorted(folder.glob("*_summary.json"))
    if not summaries:
        return None
    # Имя кончается временем запуска: последний файл — самый свежий
    summary = summaries[-1]
    with open(summary.with_name(summary.name.replace("_summary.json", ".jsonl")), encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    variant = {"folder": folder.name, "args": read_json(summary)["args"], "loss": BASE, "size": 0, "exact": True,
               "training": None,
               "tf": {r["id"]: r for r in records if r["kind"] == "tf"},
               "sps": {r["id"]: r for r in records if r["kind"] == "gen" and r["mode"] == "sps"}}

    # history.json кладёт только дообучение: без него это исходный драфт
    if (folder / "history.json").exists():
        training = variant["training"] = read_json(folder / "history.json")
        run = training["args"]
        variant["loss"] = run["loss"]
        if "train_examples" in training:
            variant["size"] = training["train_examples"]
        elif "train_examples" in cache:
            # history.json от прежней версии finetune.py: тогда учились только на всём кэше
            variant["size"] = cache["train_examples"]
        else:
            # Шагов в эпохе — примеров на батч с округлением вверх: оценка завышена меньше чем на батч
            variant["size"] = training["history"][-1]["step"] * run["batch_size"] // run["epochs"]
            variant["exact"] = False
    return variant


def load_results(path: str) -> tuple[dict[tuple[str, int], dict], dict]:
    root = Path(path)
    if not root.is_dir():
        raise SystemExit(f"Нет папки {root}. Числа считаются на сервере: sbatch run_compar_graph.sbatch, "
                         f"потом скопируйте results/compar_graph сюда")
    cache = read_json(root / "cache.json") if (root / "cache.json").exists() else {}
    variants = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        variant = load_variant(folder, cache)
        if variant is not None:
            variants[variant["loss"], variant["size"]] = variant
    if (BASE, 0) not in variants:
        raise SystemExit(f"В {root} нет оценки драфта без дообучения (подпапка без history.json)")
    if len(variants) == 1:
        raise SystemExit(f"В {root} только драфт без дообучения: сравнивать не с чем")
    return variants, cache


def check_sampler(variants: dict, cache: dict) -> tuple[float, float]:
    """Температура и top-p оценки. Они обязаны быть одни у всех драфтов и те же, что при генерации."""
    used = {(v["args"]["temperature"], v["args"]["top_p"]) for v in variants.values()}
    if len(used) > 1:
        raise SystemExit(f"Драфты оценены с разными (temperature, top-p): {sorted(used)} — сравнивать нельзя")
    sampler = used.pop()

    # С чем таргет генерировал продолжения: из кэша таргета и из history.json новых запусков finetune.py
    generated = {}
    if "temperature" in cache:
        generated[cache.get("cache", "cache.json")] = (cache["temperature"], cache["top_p"])
    for v in variants.values():
        if v["training"] and "temperature" in v["training"]:
            generated[f"{v['folder']}/history.json"] = (v["training"]["temperature"], v["training"]["top_p"])
    wrong = {source: value for source, value in generated.items() if value != sampler}
    if wrong:
        raise SystemExit(f"Оценка сделана с (temperature, top-p) = {sampler}, а таргет генерировал продолжения "
                         f"с другими: {wrong}. Пересчитайте оценку с теми же значениями")
    if not generated:
        print(f"Внимание: с чем генерировал таргет, в результатах не записано — совпадение с оценкой "
              f"(temperature {sampler[0]}, top-p {sampler[1]}) проверьте сами")
    return sampler


def main_keys(variants: dict) -> list[tuple[str, int]]:
    # Три основных драфта: без дообучения и по одному на лосс — обученный на самой большой выборке
    return [(BASE, 0)] + [max(key for key in variants if key[0] == loss) for loss in LOSSES
                          if any(key[0] == loss for key in variants)]


def per_example(variants: dict, keys: list, metric: str) -> dict[str, dict]:
    """По доменам и драфтам: числители и знаменатели метрики для примеров, посчитанных у всех драфтов
    из keys. Порядок примеров — как в compar_graph.jsonl."""
    spec = METRICS[metric]
    records = {key: variants[key][spec["kind"]] for key in keys}
    shared = set.intersection(*(set(r) for r in records.values()))
    domains = {}
    for example, record in records[BASE, 0].items():
        if example in shared:
            domains.setdefault(record["domain"], []).append(example)
    # У SpS знаменатель у каждого драфта свой: сколько токенов он предложил
    return {domain: {key: (np.array([spec["num"](records[key][i]) for i in examples], dtype=float),
                           np.array([spec["den"](records[key][i]) for i in examples], dtype=float))
                     for key in keys}
            for domain, examples in domains.items()}


def bootstrap(variants: dict, keys: list, metrics: list[str], n_samples: int, seed: int) -> tuple[dict, dict]:
    """Значение метрики и интервалы для каждого (домен, драфт, метрика), плюс число примеров."""
    stats, counts = {}, {}
    for metric in metrics:
        for domain, data in per_example(variants, keys, metric).items():
            n = counts[domain, metric] = len(data[BASE, 0][0])
            # Одни и те же выборки примеров для всех драфтов: разности считаются попарно, без шума от состава
            picks = np.random.default_rng(seed).integers(0, n, size=(n_samples, n))
            values = {key: num.sum() / den.sum() for key, (num, den) in data.items()}
            samples = {key: num[picks].sum(axis=1) / den[picks].sum(axis=1) for key, (num, den) in data.items()}
            for key in keys:
                gain = samples[key] - samples[BASE, 0]
                stats[domain, key, metric] = {
                    "value": values[key],
                    "low": np.percentile(samples[key], 2.5), "high": np.percentile(samples[key], 97.5),
                    "gain": values[key] - values[BASE, 0],
                    "gain_low": np.percentile(gain, 2.5), "gain_high": np.percentile(gain, 97.5),
                }
    return stats, counts


def format_y(ax, sign: bool = False):
    # Круглый шаг делений, десятичная запятая, знаков столько, чтобы соседние подписи различались,
    # а в приростах — явный знак
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ticks = ax.get_yticks()
    digits = max(2, int(np.ceil(-np.log10(ticks[1] - ticks[0]) - 1e-9)))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: f"{value:{'+' if sign and abs(value) > 1e-12 else ''}.{digits}f}".replace(".", ",")))


def thousands(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def panels(n: int, sharey: bool = False, wspace: float = 0.24):
    # Не уже двух панелей: иначе подзаголовок не помещается над графиком
    fig, axes = plt.subplots(1, n, figsize=(4.6 * max(n, 2), 4.1), sharey=sharey, squeeze=False)
    fig.subplots_adjust(top=0.76, bottom=0.22, wspace=wspace)
    for ax in axes[0]:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0, labelsize=9)
    return fig, axes[0]


def panel_title(ax, title: str, note: str):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=20)
    ax.text(0, 1.025, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=MUTED)


def finish(fig, title: str, subtitle: str, xlabel: str | None, handles: list, path: Path, args):
    fig.suptitle(title, x=0.01, y=1.0, ha="left", va="bottom", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.985, subtitle, ha="left", va="top", fontsize=9, color=MUTED, linespacing=1.4)
    if xlabel:
        fig.supxlabel(xlabel, y=0.08, fontsize=9.5, color=MUTED)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(handles), frameon=False,
               fontsize=10, labelcolor=INK)
    if args.watermark:
        fig.text(0.5, 0.5, args.watermark, ha="center", va="center", fontsize=34, fontweight="bold",
                 color="#9a9993", alpha=0.35, rotation=18, zorder=100)
    fig.savefig(path, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def setup_note(variants: dict, sampler: tuple[float, float]) -> str:
    run = variants[BASE, 0]["args"]
    temperature, top_p = (str(value).replace(".", ",") for value in sampler)
    return (f"Таргет {run['target'].split('/')[-1]}, драфт {run['draft'].split('/')[-1]}. "
            f"T = {temperature}, top-p = {top_p} — как при генерации продолжений для дообучения.\n")


def look(key: tuple[str, int]) -> dict:
    return BASE_STYLE if key[0] == BASE else LOSSES[key[0]]


def plot_test_prefix(metric: str, variants: dict, sampler, path: Path, args):
    """Метрика на первых N примерах каждого слоя теста: три основных драфта."""
    spec = METRICS[metric]
    keys = main_keys(variants)
    data = per_example(variants, keys, metric)
    fig, axes = panels(len(data))
    handles = []
    for ax, (domain, arrays) in zip(axes, data.items()):
        n = len(arrays[BASE, 0][0])
        first = min(FIRST_PREFIX, n)
        handles = []
        for key in keys:
            num, den = arrays[key]
            # Накопленное значение: по всем примерам от первого до N
            running = num.cumsum() / den.cumsum()
            line, = ax.plot(np.arange(first, n + 1), running[first - 1:], color=look(key)["color"], linewidth=2,
                            linestyle=DASHED if key[0] == BASE else "-", label=look(key)["label"])
            # Конец линии — значение на всём слое
            ax.plot(n, running[-1], look(key)["marker"], color=look(key)["color"], markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            handles.append(line)
        ax.set_xlim(first, n + max(1, n * 0.03))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        format_y(ax)
        panel_title(ax, DOMAIN_TITLES.get(domain, domain), f"всего {n} примеров")

    axes[0].set_ylabel(spec["name"], fontsize=9.5, color=MUTED)
    how = "Генерация SpS" if spec["kind"] == "sps" else "Teacher forcing"
    subtitle = (setup_note(variants, sampler)
                + f"{how}; значение накопленное: по всем примерам от первого до N; {spec['better']}.")
    finish(fig, f"{spec['name']} на префиксе тестового набора", subtitle,
           "Первые N примеров слоя compar_graph.jsonl, по которым посчитана метрика", handles, path, args)


def plot_training(variants: dict, sampler, path: Path, args):
    """Ход дообучения: метрики на HumanEval, которые finetune.py записывал каждые несколько шагов."""
    runs = {key: variants[key]["training"] for key in main_keys(variants)[1:]}
    fig, axes = panels(len(HISTORY_FIELDS))
    handles = []
    for ax, (metric, field) in zip(axes, HISTORY_FIELDS.items()):
        # Шаг 0 — драфт до дообучения: у обоих лоссов он один
        start = next(iter(runs.values()))["history"][0][field]
        handles = [ax.axhline(start, color=BASE_STYLE["color"], linewidth=1.5, linestyle=DASHED,
                              label=BASE_STYLE["label"])]
        for key, training in runs.items():
            line, = ax.plot([record["step"] for record in training["history"]],
                            [record[field] for record in training["history"]], color=look(key)["color"],
                            linewidth=2, marker=look(key)["marker"], markersize=5, markeredgecolor=SURFACE,
                            markeredgewidth=1, label=look(key)["label"])
            handles.append(line)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: thousands(value)))
        format_y(ax)
        panel_title(ax, METRICS[metric]["name"], METRICS[metric]["better"])

    batches = {training["args"]["batch_size"] for training in runs.values()}
    batch = f" (один шаг — батч из {batches.pop()} примеров)" if len(batches) == 1 else ""
    subtitle = setup_note(variants, sampler) + ("Метрики на HumanEval, записаны finetune.py по ходу обучения "
                                               "(по этому же набору выбирался лучший чекпойнт).")
    finish(fig, "Ход дообучения: сходство драфта с таргетом по шагам", subtitle, f"Шаг дообучения{batch}",
           handles, path, args)


def plot_final(metrics: list[str], variants: dict, stats: dict, counts: dict, sampler, path: Path, args):
    keys = main_keys(variants)
    offsets = np.linspace(-0.24, 0.24, len(keys))
    fig, axes = panels(len(metrics), wspace=0.26)
    handles = []
    for ax, metric in zip(axes, metrics):
        spec = METRICS[metric]
        domains = [domain for domain, m in counts if m == metric]
        handles = []
        for key, offset in zip(keys, offsets):
            points = [stats[domain, key, metric] for domain in domains]
            values = np.array([p["value"] for p in points])
            errors = [values - [p["low"] for p in points], [p["high"] for p in points] - values]
            handles.append(ax.errorbar(np.arange(len(domains)) + offset, values, yerr=errors,
                                       fmt=look(key)["marker"], color=look(key)["color"], markersize=8,
                                       markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=2,
                                       label=look(key)["label"]))
        ax.set_xticks(range(len(domains)), [DOMAIN_TICKS.get(d, d) for d in domains], fontsize=9, color=INK)
        ax.set_xlim(-0.6, len(domains) - 0.4)
        format_y(ax)
        per_layer = sorted({counts[domain, metric] for domain in domains})
        examples = f"{per_layer[0]}" if len(per_layer) == 1 else f"{per_layer[0]}–{per_layer[-1]}"
        panel_title(ax, spec["name"], f"{spec['better']}; примеров на слой: {examples}")

    sizes = ", ".join(f"{look(key)['short']} — {'' if variants[key]['exact'] else '≈ '}{thousands(key[1])}"
                      for key in keys[1:])
    subtitle = (setup_note(variants, sampler) + f"Примеров в дообучении: {sizes}. "
                f"Отрезки — 95%-й интервал (бутстрап по примерам теста).")
    finish(fig, "Три драфта на коде, которого не было в обучении", subtitle, None, handles, path, args)


def plot_train_size(metric: str, gain: bool, variants: dict, stats: dict, counts: dict, sampler, path: Path, args):
    spec = METRICS[metric]
    domains = [domain for domain, m in counts if m == metric]
    sizes = sorted({size for loss, size in variants if loss != BASE})
    value, low, high = ("gain", "gain_low", "gain_high") if gain else ("value", "low", "high")

    # Приросты — на общей шкале: видно, в каком слое теста дообучение дало больше
    fig, axes = panels(len(domains), sharey=gain, wspace=0.12 if gain else 0.24)
    handles = []
    for ax, domain in zip(axes, domains):
        base = stats[domain, (BASE, 0), metric]
        if not gain:
            ax.axhspan(base["low"], base["high"], color=BASE_STYLE["color"], alpha=0.10, linewidth=0)
        handles = [ax.axhline(0 if gain else base["value"], color=BASE_STYLE["color"], linewidth=1.5,
                              linestyle=DASHED, label=BASE_STYLE["label"])]
        for loss in LOSSES:
            xs = [size for size in sizes if (loss, size) in variants]
            if not xs:
                continue
            points = [stats[domain, (loss, size), metric] for size in xs]
            ax.fill_between(xs, [p[low] for p in points], [p[high] for p in points], color=LOSSES[loss]["color"],
                            alpha=0.15, linewidth=0)
            line, = ax.plot(xs, [p[value] for p in points], color=LOSSES[loss]["color"], linewidth=2,
                            marker=LOSSES[loss]["marker"], markersize=7, markeredgecolor=SURFACE,
                            markeredgewidth=1.5, label=LOSSES[loss]["label"])
            handles.append(line)

        ax.set_xscale("log")
        ax.set_xlim(sizes[0] / 1.25, sizes[-1] * 1.25)
        ax.set_xticks(sizes, [thousands(size) for size in sizes])
        ax.xaxis.set_minor_locator(NullLocator())
        panel_title(ax, DOMAIN_TITLES.get(domain, domain), f"{counts[domain, metric]} примеров теста")
    # После отрисовки всех панелей: при общей шкале пределы оси знает только последняя
    for ax in axes:
        format_y(ax, sign=gain)

    axes[0].set_ylabel(spec["gain"] if gain else spec["name"], fontsize=9.5, color=MUTED)
    kind = f"{spec['gain']} после дообучения" if gain else spec["name"]
    subtitle = (setup_note(variants, sampler)
                + f"Полосы — 95%-й интервал (бутстрап по примерам теста); {spec['better']}.")
    finish(fig, f"{kind}: сколько обучающих данных нужно каждому варианту", subtitle,
           "Примеров further_training.jsonl, на которых дообучен драфт (шкала логарифмическая)", handles, path,
           args)


def write_table(tables: dict[str, tuple[dict, dict]], path: Path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["graph", "domain", "examples", "loss", "train_examples", "metric", "value", "low", "high",
                         "gain", "gain_low", "gain_high"])
        for graph, (stats, counts) in tables.items():
            for (domain, (loss, size), metric), point in stats.items():
                writer.writerow([graph, domain, counts[domain, metric], loss, size, metric]
                                + [f"{point[name]:.5f}" for name in
                                   ("value", "low", "high", "gain", "gain_low", "gain_high")])
    print(f"  {path}")


def print_table(metric: str, keys: list, stats: dict, counts: dict):
    domains = [domain for domain, m in counts if m == metric]
    print(f"\n{METRICS[metric]['name']} по слоям теста "
          f"(в скобках изменение к драфту без дообучения и его 95%-й интервал)")
    print(f"{'драфт':<8} {'примеров':>8}  " + "  ".join(f"{domain:<34}" for domain in domains))
    for loss, size in keys:
        cells = []
        for domain in domains:
            p = stats[domain, (loss, size), metric]
            cells.append(f"{p['value']:.4f} ({p['gain']:+.4f} [{p['gain_low']:+.4f}; {p['gain_high']:+.4f}])"
                         if loss != BASE else f"{p['value']:.4f}")
        print(f"{loss:<8} {size or '—':>8}  " + "  ".join(f"{cell:<34}" for cell in cells))


def main():
    args = parse_args()
    variants, cache = load_results(args.results)
    sampler = check_sampler(variants, cache)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def measured(keys: list) -> list[str]:
        # Долю принятых токенов знает только генерация SpS: без неё остаются метрики teacher forcing
        return [metric for metric in ("accept", "alpha", "kl")
                if all(variants[key][METRICS[metric]["kind"]] for key in keys)]

    main_set = main_keys(variants)
    metrics = measured(main_set)
    print(f"Драфтов: {len(variants)}, temperature {sampler[0]}, top-p {sampler[1]}. Графики:")
    for metric in (metric for metric in metrics if metric != "kl"):
        plot_test_prefix(metric, variants, sampler, out / f"{metric}_vs_test_prefix.png", args)
    plot_training(variants, sampler, out / "training_curves.png", args)
    final = bootstrap(variants, main_set, metrics, args.bootstrap, args.seed)
    plot_final(metrics, variants, *final, sampler, out / "final_comparison.png", args)
    tables = {"final": final}

    # Кривая по размеру выборки — когда драфт дообучали на выборках разного размера
    if len({size for loss, size in variants if loss != BASE}) > 1:
        every = sorted(variants, key=lambda key: (key[0] != BASE, key))
        sweep_metrics = measured(every)
        sweep = tables["train_size"] = bootstrap(variants, every, sweep_metrics, args.bootstrap, args.seed)
        for metric in sweep_metrics:
            plot_train_size(metric, False, variants, *sweep, sampler, out / f"{metric}_vs_train_size.png", args)
        plot_train_size(sweep_metrics[0], True, variants, *sweep, sampler,
                        out / f"{sweep_metrics[0]}_gain_vs_train_size.png", args)
    write_table(tables, out / "summary.csv")
    print_table(metrics[0], main_set, *final)


if __name__ == "__main__":
    main()
