#!/usr/bin/env bash
set -euo pipefail
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
echo
ps -eo pid,ni,pcpu,pmem,args --sort=-pcpu | grep '[g]pu_hold.py' || true
