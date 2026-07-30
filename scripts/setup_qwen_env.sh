#!/usr/bin/env bash
set -euo pipefail

BASE_PYTHON="${BASE_PYTHON:-/opt/conda/envs/torch-base/bin/python}"
UV="${UV:-/opt/conda/envs/torch-base/bin/uv}"
ENV_DIR="${ENV_DIR:-/root/hiro-env}"
INDEX="${UV_DEFAULT_INDEX:-https://mirrors.cloud.tencent.com/pypi/simple}"

"$UV" venv --system-site-packages "$ENV_DIR"
UV_DEFAULT_INDEX="$INDEX" "$UV" pip install --python "$ENV_DIR/bin/python" \
    --upgrade transformers accelerate pandas matplotlib
# Pin the CUDA 12.8 builds compatible with the installed 535-series driver.
UV_DEFAULT_INDEX="$INDEX" "$UV" pip install --python "$ENV_DIR/bin/python" \
    --reinstall torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

"$ENV_DIR/bin/python" - <<'PY'
import torch
import transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
