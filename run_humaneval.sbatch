#!/bin/bash
#SBATCH --job-name=specdec-humaneval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o logs/humaneval_%j.log

# Запуск: mkdir -p logs && sbatch run_humaneval.sbatch [доп. аргументы eval_humaneval.py]
# Другая пара моделей (список — MODEL_PAIRS в specdec.py): PAIR=qwen2.5-0.5b-7b sbatch run_humaneval.sbatch
# Пример отладки: sbatch -t 0:20:00 run_humaneval.sbatch --limit 5 --temperature 0 --dtype float32
# Снять задачу: scancel <job_id>

cd "$SLURM_SUBMIT_DIR"

# Общее окружение и модели команды (см. /data/shared/specdec_team/README.md), в интернет не ходим.
# Модели лежат в $MODELS_DIR/<имя без организации>, например /data/shared/hf/Qwen2.5-7B
ENV_DIR=/data/shared/specdec_team/.venv
export HF_HOME=/data/shared/hf
export HF_HUB_OFFLINE=1
export MODELS_DIR=/data/shared/hf
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nvidia-smi
"$ENV_DIR/bin/python" eval_humaneval.py \
    --pair "${PAIR:-smollm2-135m-1.7b}" \
    --dataset data/HumanEval.jsonl.gz \
    --K 4 --temperature 0.8 --top-p 0.95 --max-new-tokens 512 \
    "$@"
