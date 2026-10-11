#!/bin/bash
#SBATCH --job-name=specdec-teacher
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH -t 6:00:00
#SBATCH -o logs/teacher_%j.log

# Этап 1 дообучения драфта: таргет пишет свои распределения в cache/<пара>/ (драфта на GPU нет).
# Запуск: mkdir -p logs && sbatch run_teacher.sbatch [доп. аргументы teacher_cache.py]
# Другая пара: PAIR=opt-125m-13b sbatch run_teacher.sbatch
# Без генерации, на тексте из датасета (в разы быстрее): sbatch run_teacher.sbatch --source reference
# Дешевле генерация: --limit 4000 (меньше примеров) или --max-new-tokens 128 (короче продолжения)
# Быстрая проверка: sbatch -t 0:20:00 run_teacher.sbatch --limit 32
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
"$ENV_DIR/bin/python" teacher_cache.py \
    --pair "${PAIR:-qwen2.5-0.5b-14b}" \
    --train further_training.jsonl \
    --humaneval data/HumanEval.jsonl.gz \
    "$@"
