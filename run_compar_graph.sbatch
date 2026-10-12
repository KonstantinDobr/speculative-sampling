#!/bin/bash
#SBATCH --job-name=specdec-compar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH -t 6:00:00
#SBATCH -o logs/compar_%j.log

# Числа для creat_compar_graph.py: три драфта (без дообучения, KL, CE) сравниваются с таргетом
# на compar_graph.jsonl, потом то же для драфтов, дообученных на части выборки. Температура и top-p везде
# берутся из кэша таргета — те самые, с которыми таргет генерировал продолжения для дообучения.
# Запуск: mkdir -p logs && sbatch run_compar_graph.sbatch [доп. аргументы finetune.py]
#   (доп. аргументы должны совпадать с теми, с которыми обучен полный драфт в checkpoints/<пара>_<лосс>)
# Только три основных драфта, без выборок меньшего размера: SIZES="" sbatch run_compar_graph.sbatch
# Другие размеры: SIZES="500 2000" sbatch run_compar_graph.sbatch
# Один лосс: LOSSES=kl sbatch run_compar_graph.sbatch
# Примеров на слой теста для генерации SpS: GEN=100 у основных драфтов, GEN_SWEEP=30 у остальных
# Не уложилась в лимит времени: запустите ту же команду ещё раз — готовые варианты пропускаются
# Веса драфтов, дообученных на части выборки, после оценки удаляются (около 2 ГБ на драфт);
#   оставить: KEEP_CHECKPOINTS=1 sbatch run_compar_graph.sbatch
# Потом скопируйте results/compar_graph к себе и запустите python creat_compar_graph.py
# Снять задачу: scancel <job_id>

cd "$SLURM_SUBMIT_DIR"

ENV_DIR=/data/shared/specdec_team/.venv
export HF_HOME=/data/shared/hf
export HF_HUB_OFFLINE=1
# Модели: сначала папка команды (download_models.sh), потом общая read-only
TEAM_DIR=${TEAM_DIR:-/data/teams/specdec_team}
export MODELS_DIR="$TEAM_DIR/models:/data/shared/hf"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PAIR=${PAIR:-qwen2.5-0.5b-14b}
LOSSES=${LOSSES:-"kl ce"}
SIZES=${SIZES-"350 700 1400 2800 5600"}
GEN=${GEN:-100}
GEN_SWEEP=${GEN_SWEEP:-30}
CACHE=${CACHE:-cache/$PAIR/train_generated.pt}
OUT=${OUT:-results/compar_graph}

nvidia-smi
# Сэмплер, которым таргет генерировал продолжения: с ним же считаются все метрики и идёт SpS
read -r TEMPERATURE TOP_P TRAIN_EXAMPLES < <("$ENV_DIR/bin/python" -c "
import sys, torch
cache = torch.load(sys.argv[1], weights_only=True)
print(cache['meta']['temperature'], cache['meta']['top_p'], len(cache['lengths']))" "$CACHE")
if [ -z "$TRAIN_EXAMPLES" ]; then
    echo "Не прочитался кэш таргета $CACHE: сначала run_teacher.sbatch"
    exit 1
fi
echo "=== Сэмплер из $CACHE: temperature $TEMPERATURE, top-p $TOP_P; примеров в кэше: $TRAIN_EXAMPLES ==="
mkdir -p "$OUT"
echo "{\"cache\": \"$CACHE\", \"temperature\": $TEMPERATURE, \"top_p\": $TOP_P, \"train_examples\": $TRAIN_EXAMPLES}" \
    > "$OUT/cache.json"

# evaluate <вариант> <примеров на слой для SpS> [драфт]: teacher forcing на всём compar_graph.jsonl
# и генерация SpS на первых примерах каждого слоя
evaluate() {
    if ls "$OUT/$1"/*_summary.json > /dev/null 2>&1; then
        echo "=== $1: уже оценён ==="
        return
    fi
    "$ENV_DIR/bin/python" eval_domains.py --pair "$PAIR" --dataset compar_graph.jsonl \
        --temperature "$TEMPERATURE" --top-p "$TOP_P" --gen-per-domain "$2" --modes sps \
        --out-dir "$OUT/$1" ${3:+--draft "$3"} || exit 1
}

# 1. Три основных драфта: без дообучения и дообученные на всей выборке
evaluate base "$GEN"
for loss in $LOSSES; do
    variant=${loss}_all
    [ -f "$OUT/$variant/history.json" ] && continue
    # Готовый запуск run_finetune.sbatch берём как есть. Если его нет или он не дописан (history.json
    # появляется в самом конце), учим отдельно, чужую папку не трогаем
    checkpoint=checkpoints/${PAIR}_$loss
    if [ ! -f "$checkpoint/history.json" ] || [ ! -d "$checkpoint/last" ]; then
        checkpoint=checkpoints/${PAIR}_${loss}_nall
        if [ ! -f "$checkpoint/history.json" ]; then
            "$ENV_DIR/bin/python" finetune.py --loss "$loss" --pair "$PAIR" --train-cache "$CACHE" \
                --out-dir "$checkpoint" "$@" || exit 1
        fi
    fi
    # last, а не best: best выбран по HumanEval, а сравнивать надо чекпойнты, взятые одинаково
    evaluate "$variant" "$GEN" "$checkpoint/last"
    # Из history.json программа графиков берёт лосс, размер выборки и ход обучения по шагам
    cp "$checkpoint/history.json" "$OUT/$variant/"
done

# 2. Драфты, дообученные на части выборки (шагов обучения столько же, сколько на всей)
for loss in $LOSSES; do
    for size in $SIZES; do
        variant=${loss}_n$size
        [ -f "$OUT/$variant/history.json" ] && continue
        checkpoint=checkpoints/${PAIR}_$variant
        if [ ! -d "$checkpoint/last" ]; then
            "$ENV_DIR/bin/python" finetune.py --loss "$loss" --pair "$PAIR" --train-cache "$CACHE" \
                --train-limit "$size" --out-dir "$checkpoint" "$@" || exit 1
        fi
        evaluate "$variant" "$GEN_SWEEP" "$checkpoint/last"
        cp "$checkpoint/history.json" "$OUT/$variant/"
        [ -n "$KEEP_CHECKPOINTS" ] || rm -rf "$checkpoint/best" "$checkpoint/last"
    done
done
