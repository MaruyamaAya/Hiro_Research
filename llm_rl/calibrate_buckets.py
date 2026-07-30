from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def calibrate(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("difficulty", "unknown"))].append(record)
    rows = []
    for bucket in sorted(grouped):
        values = grouped[bucket]
        rows.append(
            {
                "bucket": bucket,
                "samples": len(values),
                "pass_rate": sum(bool(x["correct"]) for x in values) / len(values),
                "mean_tokens": sum(float(x["completion_tokens"]) for x in values) / len(values),
                "truncation_rate": sum(bool(x["truncated"]) for x in values) / len(values),
            }
        )
    pass_rates = [row["pass_rate"] for row in rows]
    monotonic_nonincreasing = all(
        left >= right for left, right in zip(pass_rates, pass_rates[1:])
    )
    return {
        "buckets": rows,
        "monotonic_nonincreasing": monotonic_nonincreasing,
        "violations": [
            {"easier": rows[i], "harder": rows[i + 1]}
            for i in range(len(rows) - 1)
            if rows[i]["pass_rate"] < rows[i + 1]["pass_rate"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in open(args.predictions) if line.strip()]
    # Calibration uses one fixed sample per prompt.
    records = [x for x in records if int(x.get("sample_index", 0)) == 0]
    report = calibrate(records)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["monotonic_nonincreasing"]:
        raise SystemExit("Proxy buckets are not monotonic under base-model pass rate")


if __name__ == "__main__":
    main()
