#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:?usage: run_grpo.sh MODE OUTPUT [extra args...]}"
OUTPUT="${2:?usage: run_grpo.sh MODE OUTPUT [extra args...]}"
shift 2
python3 -m llm_rl.generate_math_curriculum
./scripts/with_gpu_hold.sh /root/hiro-env/bin/python -m llm_rl.train_grpo \
    --mode "$MODE" --output "$OUTPUT" "$@"
