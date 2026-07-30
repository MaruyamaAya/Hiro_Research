#!/usr/bin/env bash
# Stop gpu_hold, run one GPU experiment, and restore gpu_hold on every exit path.
set -euo pipefail

HOLD_DIR="${HOLD_DIR:-/root/gpu_hold}"
MEM_FRAC="${HOLD_MEM_FRAC:-0.80}"
MATMUL_DIM="${HOLD_MATMUL_DIM:-16384}"
SLEEP="${HOLD_SLEEP:-0.03}"
PYTHON="${HOLD_PYTHON:-/opt/conda/envs/torch-base/bin/python}"

restore_hold() {
    if [[ -d "$HOLD_DIR" ]] && ! pgrep -f '[g]pu_hold.py' >/dev/null 2>&1; then
        (
            cd "$HOLD_DIR"
            env -u CUDA_VISIBLE_DEVICES \
                MEM_FRAC="$MEM_FRAC" MATMUL_DIM="$MATMUL_DIM" SLEEP="$SLEEP" \
                PYTHON="$PYTHON" ./start_hold.sh
        ) || true
    fi
}

trap restore_hold EXIT HUP INT TERM

if [[ -x "$HOLD_DIR/stop_hold.sh" ]]; then
    "$HOLD_DIR/stop_hold.sh"
fi

if [[ "$#" -eq 0 ]]; then
    echo "usage: $0 <gpu experiment command...>" >&2
    exit 2
fi

"$@"
