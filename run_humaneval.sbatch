#!/bin/bash
#SBATCH --job-name=specdec-humaneval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH -t 2:00:00
#SBATCH -o logs/humaneval_%j.log

# Запуск: mkdir -p logs && sbatch run_humaneval.sbatch [доп. аргументы eval_humaneval.py]
# Другая пара моделей (список — MODEL_PAIRS в specdec.py): PAIR=opt-125m-13b sbatch run_humaneval.sbatch
# Пример отладки: sbatch -t 0:20:00 run_humaneval.sbatch --limit 5 --temperature 0 --dtype float32
# Снять задачу: scancel <job_id>

cd "$SLURM_SUBMIT_DIR"

# Общее окружение и модели команды (см. /data/shared/specdec_team/README.md), в интернет не ходим.
# Модели лежат в <папка из MODELS_DIR>/<имя без организации>, например /data/teams/<команда>/models/Qwen2.5-7B
ENV_DIR=/data/shared/specdec_team/.venv
export HF_HOME=/data/shared/hf
export HF_HUB_OFFLINE=1
# Модели: сначала папка команды (download_models.sh), потом общая read-only
TEAM_DIR=${TEAM_DIR:-/data/teams/specdec_team}
export MODELS_DIR="$TEAM_DIR/models:/data/shared/hf"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nvidia-smi
"$ENV_DIR/bin/python" eval_humaneval.py \
    --pair "${PAIR:-qwen2.5-0.5b-14b}" \
    --dataset data/HumanEval.jsonl.gz \
    --K 4 --temperature 0.8 --top-p 0.95 --max-new-tokens 512 \
    "$@"
