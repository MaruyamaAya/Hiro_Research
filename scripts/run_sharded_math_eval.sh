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

mapfile -t labels < <(
    find "$OUTPUT/shards/shard-0" -mindepth 1 -maxdepth 1 -type d \
        -exec basename {} \; | sort
)
if [[ "${#labels[@]}" -eq 0 ]]; then
    echo "No completed checkpoint labels found in shard-0" >&2
    exit 1
fi

summaries=()
for label in "${labels[@]}"; do
    "$PYTHON" -m llm_rl.merge_eval_shards \
        --input "$OUTPUT/shards/shard-*/$label/predictions.jsonl" \
        --output "$OUTPUT/$label"
    summaries+=("$OUTPUT/$label/summary.json")
done

"$PYTHON" - "${summaries[@]}" >"$OUTPUT/all_summaries.json" <<'PY'
import json
import sys

print(json.dumps([json.load(open(path)) for path in sys.argv[1:]], indent=2))
PY
