from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = [
    "skill",
    "external_return",
    "learning_progress",
    "easy_ratio",
    "challenge_ratio",
    "noisy_tv_ratio",
    "treadmill_ratio",
    "impossible_ratio",
    "meaningless_suffering_ratio",
    "boredom_rate",
    "challenge_appropriateness_error",
    "mean_selected_challenge_difficulty",
    "total_damage",
]


def bootstrap_ci(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/main")
    parser.add_argument("--output", default="results/main_summary")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)
    config = json.load((src / "resolved_config.json").open())
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    curves: list[dict[str, str]] = []
    for path in sorted((src / "runs").glob("*.csv")):
        rows = list(csv.DictReader(path.open()))
        grouped[rows[-1]["agent"]].append(rows[-1])
        curves.extend(rows)

    rng = np.random.default_rng(20260730)
    summary: list[dict[str, float | str | int]] = []
    for agent, rows in sorted(grouped.items()):
        record: dict[str, float | str | int] = {"agent": agent, "n_seeds": len(rows)}
        for metric in METRICS:
            values = np.array([float(r[metric]) for r in rows])
            lo, hi = bootstrap_ci(values, int(config["bootstrap_samples"]), rng)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_ci_low"] = lo
            record[f"{metric}_ci_high"] = hi
        summary.append(record)

    with (dst / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    # Long-format aggregate learning curves.
    by_curve: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in curves:
        by_curve[(row["agent"], int(row["step"]))].append(row)
    curve_summary = []
    for (agent, step), rows in sorted(by_curve.items()):
        rec: dict[str, float | str | int] = {"agent": agent, "step": step}
        for metric in METRICS:
            vals = np.array([float(r[metric]) for r in rows])
            rec[f"{metric}_mean"] = float(vals.mean())
            rec[f"{metric}_sem"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        curve_summary.append(rec)
    with (dst / "curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_summary[0]))
        writer.writeheader()
        writer.writerows(curve_summary)

    try:
        import matplotlib.pyplot as plt

        agents = [r["agent"] for r in summary]
        metrics = [
            "challenge_ratio",
            "noisy_tv_ratio",
            "treadmill_ratio",
            "impossible_ratio",
            "skill",
            "external_return",
        ]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, metric in zip(axes.flat, metrics):
            means = [float(r[f"{metric}_mean"]) for r in summary]
            lows = [float(r[f"{metric}_ci_low"]) for r in summary]
            highs = [float(r[f"{metric}_ci_high"]) for r in summary]
            err = np.array([np.array(means) - lows, np.array(highs) - means])
            ax.bar(range(len(agents)), means, yerr=err, capsize=2)
            ax.set_title(metric)
            ax.set_xticks(range(len(agents)), agents, rotation=65, ha="right", fontsize=8)
        fig.tight_layout()
        fig.savefig(dst / "main_metrics.png", dpi=180)
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; skipped plots")

    print(f"Wrote {dst / 'summary.csv'}")


if __name__ == "__main__":
    main()
