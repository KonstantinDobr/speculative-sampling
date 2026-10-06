#!/bin/bash
#SBATCH --job-name=specdec-metrics
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH -o logs/metrics_%j.log

# Запуск: mkdir -p logs && sbatch run_metrics.sbatch [доп. аргументы metrics.py]
# Другой датасет: sbatch run_metrics.sbatch --dataset data/my.jsonl --fields text
# Как в спекдеке на HumanEval: sbatch run_metrics.sbatch --temperature 0.8 --top-p 0.95
# Снять задачу: scancel <job_id>

cd "$SLURM_SUBMIT_DIR"

# Общее окружение и модели команды (см. /data/shared/specdec_team/README.md), в интернет не ходим
ENV_DIR=/data/shared/specdec_team/.venv
export HF_HOME=/data/shared/hf
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nvidia-smi
"$ENV_DIR/bin/python" metrics.py \
    --target /data/shared/hf/SmolLM2-1.7B \
    --draft /data/shared/hf/SmolLM2-135M \
    --dataset data/HumanEval.jsonl.gz \
    --fields prompt canonical_solution \
    --out "results/metrics_${SLURM_JOB_ID}.json" \
    "$@"
