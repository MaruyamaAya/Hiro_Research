from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def paired_bootstrap(
    baseline: Iterable[float],
    treatment: Iterable[float],
    samples: int = 10_000,
    seed: int = 20260730,
) -> dict[str, float]:
    baseline = np.asarray(list(baseline), dtype=float)
    treatment = np.asarray(list(treatment), dtype=float)
    if baseline.shape != treatment.shape or baseline.ndim != 1 or len(baseline) == 0:
        raise ValueError("paired arrays must be non-empty one-dimensional arrays of equal length")
    differences = treatment - baseline
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrap_means = differences[indexes].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "n": int(len(differences)),
        "baseline_mean": float(baseline.mean()),
        "treatment_mean": float(treatment.mean()),
        "mean_difference": float(differences.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="JSON list of paired values")
    parser.add_argument("--treatment", required=True, help="JSON list of paired values")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = paired_bootstrap(
        json.loads(Path(args.baseline).read_text()),
        json.loads(Path(args.treatment).read_text()),
        args.samples,
        args.seed,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
