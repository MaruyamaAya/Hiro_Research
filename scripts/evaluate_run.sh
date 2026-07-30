#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: evaluate_run.sh RUN_DIR DATA SPLIT [extra evaluator args...]}"
DATA="${2:?usage: evaluate_run.sh RUN_DIR DATA SPLIT [extra evaluator args...]}"
SPLIT="${3:?usage: evaluate_run.sh RUN_DIR DATA SPLIT [extra evaluator args...]}"
shift 3
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT="$RUN_DIR/evaluation/$SPLIT"

exec "$ROOT/scripts/run_math_eval.sh" \
  --data "$DATA" \
  --split "$SPLIT" \
  --checkpoint-root "$RUN_DIR/checkpoints" \
  --output "$OUTPUT" \
  --max-new-tokens 1024 \
  --temperature 0 \
  "$@"
