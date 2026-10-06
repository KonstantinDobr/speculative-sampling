#!/bin/bash
#SBATCH --job-name=specdec-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o logs/bench_%j.log

# Запуск: mkdir -p logs && sbatch run_bench.sbatch [доп. аргументы bench_latency.py]
# Только часть пар: sbatch run_bench.sbatch --pairs smollm2-135m-1.7b qwen2.5-0.5b-7b
# Пары сравниваем только на одном типе GPU: при необходимости sbatch --gres=gpu:<тип>:1 run_bench.sbatch
# Пары, для которых модели не скачаны (download_models.sh), пропускаются с предупреждением

cd "$SLURM_SUBMIT_DIR"

ENV_DIR=/data/shared/specdec_team/.venv
export HF_HOME=/data/shared/hf
export HF_HUB_OFFLINE=1
# Модели: сначала папка команды (download_models.sh), потом общая read-only
TEAM_DIR=${TEAM_DIR:-/data/teams/specdec_team}
export MODELS_DIR="$TEAM_DIR/models:/data/shared/hf"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nvidia-smi
"$ENV_DIR/bin/python" bench_latency.py \
    --dataset data/HumanEval.jsonl.gz \
    --alpha-tasks 20 \
    "$@"
