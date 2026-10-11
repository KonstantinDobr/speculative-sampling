#!/bin/bash
#SBATCH --job-name=specdec-finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/finetune_%j.log

# Этап 2 дообучения драфта: драфт учится по кэшу из run_teacher.sbatch (таргета на GPU нет).
# Два варианта запускаются отдельно:
#   mkdir -p logs && LOSS=kl sbatch run_finetune.sbatch [доп. аргументы finetune.py]
#   mkdir -p logs && LOSS=ce sbatch run_finetune.sbatch
# Другая пара: PAIR=opt-125m-13b LOSS=kl sbatch run_finetune.sbatch
# Кэш на reference: LOSS=ce sbatch run_finetune.sbatch --train-cache cache/qwen2.5-0.5b-14b/train_reference.pt
# Быстрая проверка: LOSS=kl sbatch -t 0:20:00 run_finetune.sbatch --max-steps 20 --eval-every 10
# Дообученный драфт в SpS: sbatch run_humaneval.sbatch --draft checkpoints/qwen2.5-0.5b-14b_kl/best
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
"$ENV_DIR/bin/python" finetune.py \
    --loss "${LOSS:-kl}" \
    --pair "${PAIR:-qwen2.5-0.5b-14b}" \
    "$@"
