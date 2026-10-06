#!/bin/bash
# Скачивание моделей из MODEL_PAIRS (specdec.py) в папку команды. Общая /data/shared/hf только для чтения,
# то, что уже лежит там, не качаем. Запускать на login-узле, где есть интернет:
#   bash download_models.sh                                    # все недостающие модели
#   bash download_models.sh Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-7B  # только указанные
# Своя папка команды: TEAM_DIR=/data/teams/<команда> bash download_models.sh
set -euo pipefail
cd "$(dirname "$0")"

TEAM_DIR=${TEAM_DIR:-/data/teams/specdec_team}
ENV_DIR=/data/shared/specdec_team/.venv
export MODELS_DIR="$TEAM_DIR/models:/data/shared/hf"
# Кэш и служебные файлы HF — в папку команды: в общую писать нельзя
export HF_HOME="$TEAM_DIR/hf_home"
unset HF_HUB_OFFLINE
mkdir -p "$TEAM_DIR/models" "$HF_HOME"

"$ENV_DIR/bin/python" - "$TEAM_DIR/models" "$@" <<'EOF'
import os
import sys

from huggingface_hub import snapshot_download

from specdec import MODEL_PAIRS, resolve_model

out_dir, requested = sys.argv[1], sys.argv[2:]
names = requested or list(dict.fromkeys(name for pair in MODEL_PAIRS.values() for name in pair))

for name in names:
    try:
        print(f"Уже есть: {name} -> {resolve_model(name)}")
        continue
    except FileNotFoundError:
        pass

    local_dir = os.path.join(out_dir, name.split("/")[-1])
    print(f"Качаю {name} -> {local_dir}", flush=True)
    # Только веса, конфиги и токенизатор: без .bin-дубликатов, onnx и прочего
    snapshot_download(name, local_dir=local_dir,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
EOF

du -sh "$TEAM_DIR"/models/* 2>/dev/null || true
