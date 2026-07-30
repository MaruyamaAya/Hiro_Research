#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m qwen_study.generate_pairs
./scripts/with_gpu_hold.sh /root/hiro-env/bin/python -m qwen_study.evaluate_zero_shot "$@"
