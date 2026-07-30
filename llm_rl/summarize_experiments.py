from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_run(run_dir: Path) -> dict[str, Any]:
    summaries_path = run_dir / "evaluation/eval/all_summaries.json"
    summaries = json.loads(summaries_path.read_text())
    final = next((x for x in summaries if str(x.get("checkpoint", "")).endswith("/final")), None)
    if final is None and summaries:
        final = summaries[-1]
    if final is None:
        raise ValueError(f"No evaluation summary in {summaries_path}")
    name = run_dir.name
    seed_text = name.rsplit("_seed", 1)[-1]
    condition = name.rsplit("_seed", 1)[0].removeprefix("final_")
    return {
        "run": name,
        "condition": condition,
        "seed": int(seed_text),
        "pass_at_1": float(final["pass_at_1"]),
        "format_valid_rate": float(final["format_valid_rate"]),
        "truncation_rate": float(final["truncation_rate"]),
        "mean_completion_tokens": float(final["mean_completion_tokens"]),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conditions = sorted({row["condition"] for row in rows})
    output = []
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        item: dict[str, Any] = {"condition": condition, "seeds": len(selected)}
        for metric in ["pass_at_1", "format_valid_rate", "truncation_rate", "mean_completion_tokens"]:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = [
        load_run(path)
        for path in sorted(root.glob("final_*_seed*"))
        if (path / "evaluation/eval/all_summaries.json").exists()
    ]
    if not rows:
        raise SystemExit("No completed final runs found")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    aggregates = aggregate(rows)
    (output / "condition_summary.json").write_text(json.dumps(aggregates, indent=2) + "\n")
    with (output / "condition_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
