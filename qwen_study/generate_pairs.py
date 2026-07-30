from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TEMPLATES = {
    "noise": (
        "Investigate a stable unfamiliar mechanism. Prediction error fell from {e0:.2f} to {e1:.2f} "
        "after practice and task success improved from {p0:.0%} to {p1:.0%}.",
        "Watch an irreducibly random display. Prediction error stayed near {noise:.2f} and task "
        "success did not improve.",
    ),
    "effort": (
        "Spend effort {effort:.2f} on a frontier task; recent success improved from {p0:.0%} to "
        "{p1:.0%}.",
        "Spend effort {effort:.2f} on a treadmill; state, skill, and success probability were unchanged.",
    ),
    "difficulty": (
        "Attempt a difficult but learnable task with current success probability {p:.0%}, recoverable "
        "cost, and measurable progress.",
        "Attempt an almost impossible task with success probability {q:.1%}, irreversible damage risk, "
        "and no observed progress.",
    ),
    "history": (
        "Retry the task after recent attempts improved success from {p0:.0%} to {p1:.0%}. The current "
        "attempt failed, but the learning trend is positive.",
        "Retry the same task after {n} failures with success fixed at {p0:.0%}. The current attempt "
        "failed and no learning trend is detectable.",
    ),
    "boredom": (
        "Choose a new frontier task with success probability {p:.0%} and positive recent learning progress.",
        "Repeat a mastered task with success probability {q:.0%} and zero recent learning progress.",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/preference_pairs.jsonl")
    parser.add_argument("--n-per-category", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    records = []
    idx = 0
    for category, (good_t, bad_t) in TEMPLATES.items():
        for _ in range(args.n_per_category):
            p0 = float(rng.uniform(0.08, 0.35))
            p1 = float(min(0.85, p0 + rng.uniform(0.08, 0.30)))
            values = {
                "e0": float(rng.uniform(0.65, 0.95)),
                "e1": float(rng.uniform(0.15, 0.50)),
                "noise": float(rng.uniform(0.88, 0.99)),
                "p0": p0,
                "p1": p1,
                "p": float(rng.uniform(0.25, 0.60)),
                "q": float(rng.uniform(0.001, 0.02)),
                "effort": float(rng.uniform(0.65, 0.98)),
                "n": int(rng.integers(20, 120)),
            }
            good = good_t.format(**values)
            bad = bad_t.format(**values)
            swap = bool(rng.integers(0, 2))
            records.append(
                {
                    "id": idx,
                    "category": category,
                    "a": bad if swap else good,
                    "b": good if swap else bad,
                    "label": "B" if swap else "A",
                    "split": "test" if idx % 5 == 0 else "train",
                }
            )
            idx += 1
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} pairs to {path}")


if __name__ == "__main__":
    main()
