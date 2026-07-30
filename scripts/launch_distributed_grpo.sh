#!/usr/bin/env bash
# Launch one node-wide GRPO job and preserve code, logs, and checkpoints.
set -euo pipefail

MODE="${1:?usage: launch_distributed_grpo.sh REWARD_MODE RUN_NAME [extra train args...]}"
RUN_NAME="${2:?usage: launch_distributed_grpo.sh REWARD_MODE RUN_NAME [extra train args...]}"
shift 2

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/root/hiro-env/bin/python}"
NPROC="${NPROC:-8}"
PERSIST_ROOT="${HIRO_PERSIST_ROOT:-${TAIJI_BASIC_OUTPUT_PATH:-/root}/hiro_rl}"
RUN_DIR="$PERSIST_ROOT/$RUN_NAME"
LOCAL_LOG_DIR="$ROOT/results/remote_logs"
mkdir -p "$RUN_DIR/code_snapshot" "$LOCAL_LOG_DIR"

# Save exact launch context before allocating GPUs.
date -Is > "$RUN_DIR/launch_time.txt"
hostname > "$RUN_DIR/hostname.txt"
printf '%q ' "$0" "$MODE" "$RUN_NAME" "$@" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"
env | sort > "$RUN_DIR/environment.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi_at_launch.txt"
cp -a "$ROOT/llm_rl" "$ROOT/configs" "$ROOT/docs" "$ROOT/scripts" \
    "$RUN_DIR/code_snapshot/" 2>/dev/null || true

cd "$ROOT"
DATA="${HIRO_TRAIN_DATA:-$RUN_DIR/math_curriculum.jsonl}"
if [[ -z "${HIRO_TRAIN_DATA:-}" ]]; then
    python3 -m llm_rl.generate_math_curriculum \
        --output "$DATA" --per-level 500
fi

exec "$ROOT/scripts/with_gpu_hold.sh" \
    "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node="$NPROC" \
    -m llm_rl.train_grpo \
    --mode "$MODE" \
    --data "$DATA" \
    --output "$RUN_DIR/checkpoints" \
    "$@"
