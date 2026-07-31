#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/root/hiro-env/bin/python}"
OUTPUT="${1:?usage: run_sharded_math_eval.sh OUTPUT [evaluator args...]}"
shift
mkdir -p "$OUTPUT/shards"

cleanup() {
    jobs -pr | xargs -r kill 2>/dev/null || true
}
trap cleanup INT TERM

for device in {0..7}; do
    CUDA_VISIBLE_DEVICES="$device" "$PYTHON" -m llm_rl.evaluate_checkpoints \
        --output "$OUTPUT/shards/shard-$device" \
        --device cuda:0 \
        --num-shards 8 \
        --shard-index "$device" \
        "$@" \
        >"$OUTPUT/shard-$device.log" 2>&1 &
done
wait

"$PYTHON" -m llm_rl.merge_eval_shards \
    --input "$OUTPUT/shards/shard-*/*/predictions.jsonl" \
    --output "$OUTPUT/merged"
