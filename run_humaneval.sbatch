#!/bin/bash
#SBATCH --job-name=specdec-humaneval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o logs/humaneval_%j.log

# Запуск: mkdir -p logs && sbatch run_humaneval.sbatch [доп. аргументы eval_humaneval.py]
# Пример отладки: sbatch -t 0:20:00 run_humaneval.sbatch --limit 5 --temperature 0
# Снять задачу: scancel <job_id>

cd "$SLURM_SUBMIT_DIR"

# Общее окружение и модели команды (см. /data/shared/specdec_team/README.md), в интернет не ходим
ENV_DIR=/data/shared/specdec_team/.venv
export HF_HOME=/data/shared/hf
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nvidia-smi
"$ENV_DIR/bin/python" eval_humaneval.py \
    --target /data/shared/hf/SmolLM2-1.7B \
    --draft /data/shared/hf/SmolLM2-135M \
    --dataset data/HumanEval.jsonl.gz \
    --K 4 --temperature 0.8 --top-p 0.95 --max-new-tokens 512 \
    "$@"
