from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/publication_matrix.json")
    parser.add_argument("--stage", choices=["integration", "calibration", "final"], required=True)
    parser.add_argument("--run-root", default="${HIRO_PERSIST_ROOT:-${TAIJI_BASIC_OUTPUT_PATH}/hiro_rl}")
    parser.add_argument("--train-data", default="${HIRO_TRAIN_DATA:?set persistent prepared DAPO JSONL}")
    parser.add_argument("--test-data", default="${HIRO_TEST_DATA:?set persistent MATH-500/GSM8K JSONL}")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    seeds = config["seeds"][:1] if args.stage != "final" else config["seeds"]
    for condition in config["conditions"]:
        for seed in seeds:
            name = f"{args.stage}_{condition['id']}_seed{seed}"
            run = f'{args.run_root}/{name}'
            data = args.test_data if args.stage == "final" else args.train_data
            split = "eval" if args.stage == "final" else "validation"
            print(
                f'./scripts/evaluate_run.sh "{run}" "{data}" {split}'
            )


if __name__ == "__main__":
    main()
