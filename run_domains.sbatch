#!/bin/bash
#SBATCH --job-name=specdec-domains
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH -t 3:00:00
#SBATCH -o logs/domains_%j.log

# Запуск: mkdir -p logs && sbatch run_domains.sbatch [доп. аргументы eval_domains.py]
# Другая пара: PAIR=opt-125m-13b sbatch run_domains.sbatch
# Быстрая проверка: sbatch -t 0:20:00 run_domains.sbatch --tf-per-domain 5 --gen-per-domain 2
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

nvidia-smi
"$ENV_DIR/bin/python" eval_domains.py \
    --pair "${PAIR:-qwen2.5-0.5b-14b}" \
    --dataset eval_set.jsonl \
    "$@"
