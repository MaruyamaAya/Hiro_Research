from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from llm_rl.eval_metrics import summarize_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    for value in args.input:
        for path_value in sorted(glob.glob(value)):
            path = Path(path_value)
            records.extend(json.loads(line) for line in path.open() if line.strip())
    if not records:
        parser.error("No prediction records found")
    records.sort(key=lambda x: (str(x["id"]), int(x["sample_index"])))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize_predictions(records)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
