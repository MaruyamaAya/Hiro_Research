#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/root/hiro-env/bin/python}"
cd "$ROOT"

exec "$ROOT/scripts/with_gpu_hold.sh" \
    "$PYTHON" -m llm_rl.evaluate_checkpoints "$@"
