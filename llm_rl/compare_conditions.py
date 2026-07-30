from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from llm_rl.bootstrap_stats import paired_bootstrap


def load_correct(path: Path) -> dict[str, bool]:
    output = {}
    for line in path.open():
        row = json.loads(line)
        if int(row.get("sample_index", 0)) == 0:
            output[str(row["id"])] = bool(row["correct"])
    return output


def compare_items(baseline: dict[str, bool], treatment: dict[str, bool], samples: int, seed: int) -> dict[str, Any]:
    shared = sorted(set(baseline) & set(treatment))
    if not shared:
        raise ValueError("No shared evaluation item ids")
    return paired_bootstrap(
        [float(baseline[x]) for x in shared],
        [float(treatment[x]) for x in shared],
        samples=samples,
        seed=seed,
    ) | {"shared_items": len(shared)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, action="append")
    parser.add_argument("--treatment", required=True, action="append")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.baseline) != len(args.treatment):
        parser.error("Provide paired --baseline/--treatment files in seed order")

    per_seed = []
    seed_differences = []
    for baseline_path, treatment_path in zip(args.baseline, args.treatment):
        report = compare_items(
            load_correct(Path(baseline_path)),
            load_correct(Path(treatment_path)),
            args.samples,
            args.seed,
        )
        report.update({"baseline": baseline_path, "treatment": treatment_path})
        per_seed.append(report)
        seed_differences.append(report["mean_difference"])

    # Seed-level uncertainty is the primary independent-replicate analysis.
    rng = np.random.default_rng(args.seed)
    seed_values = np.asarray(seed_differences)
    if len(seed_values) > 1:
        indexes = rng.integers(0, len(seed_values), size=(args.samples, len(seed_values)))
        means = seed_values[indexes].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
    else:
        low = high = seed_values[0]
    output = {
        "paired_seeds": len(per_seed),
        "seed_mean_difference": float(seed_values.mean()),
        "seed_ci95_low": float(low),
        "seed_ci95_high": float(high),
        "per_seed_item_bootstrap": per_seed,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
