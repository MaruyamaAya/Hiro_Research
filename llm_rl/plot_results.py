from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    rows = json.loads(Path(args.summary).read_text())
    labels = [row["condition"] for row in rows]
    means = [row["pass_at_1_mean"] for row in rows]
    errors = [row["pass_at_1_std"] for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, means, yerr=errors, capsize=3, color="#3b82f6")
    axis.set_ylabel("Held-out pass@1")
    axis.set_xlabel("Condition")
    axis.set_ylim(0, max(1.0, max(means, default=0) * 1.15))
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    if output.suffix.lower() != ".pdf":
        figure.savefig(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
