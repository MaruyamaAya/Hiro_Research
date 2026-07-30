#!/usr/bin/env bash
set -euo pipefail

DATA="${1:?usage: evaluate_base.sh DATA SPLIT OUTPUT [extra evaluator args...]}"
SPLIT="${2:?usage: evaluate_base.sh DATA SPLIT OUTPUT [extra evaluator args...]}"
OUTPUT="${3:?usage: evaluate_base.sh DATA SPLIT OUTPUT [extra evaluator args...]}"
shift 3
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

exec "$ROOT/scripts/run_math_eval.sh" \
  --data "$DATA" \
  --split "$SPLIT" \
  --output "$OUTPUT" \
  --max-new-tokens 1024 \
  --temperature 0 \
  "$@"
