#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p data/raw data/manifests

DAPO_REV="65877096c24ffa7abc4e4fa5edb95cf3413a5674"
MATH500_REV="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
GSM8K_REV="740312add88f781978c0658806c59bc2815b9866"

curl -L --fail --retry 5 -sS \
  "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/$MATH500_REV/test.jsonl" \
  -o data/raw/math-500-test.jsonl
curl -L --fail --retry 5 -sS \
  "https://huggingface.co/datasets/openai/gsm8k/resolve/$GSM8K_REV/main/test-00000-of-00001.parquet" \
  -o data/raw/gsm8k-test.parquet

python3 - <<'PY'
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

output = Path("data/raw/dapo-math-17k-api.jsonl")
base = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
    {
        "dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
        "config": "default",
        "split": "train",
    }
)
with output.open("w") as handle:
    for offset in range(0, 17_000, 100):
        url = base + f"&offset={offset}&length=100"
        for attempt in range(8):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = json.load(response)
                break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(2**attempt)
        for item in payload["rows"]:
            handle.write(json.dumps(item["row"], ensure_ascii=False) + "\n")
        if offset % 1000 == 0:
            print(f"DAPO rows: {offset + len(payload['rows'])}", flush=True)
PY

python3 -m llm_rl.prepare_real_math_data
