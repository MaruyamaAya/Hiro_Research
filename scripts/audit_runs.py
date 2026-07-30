from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/publication_matrix.json")
    parser.add_argument("--stage", choices=["integration", "calibration", "final"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    seeds = config["seeds"][:1] if args.stage != "final" else config["seeds"]
    root = Path(args.root)
    runs = []
    complete = 0
    for condition in config["conditions"]:
        for seed in seeds:
            name = f"{args.stage}_{condition['id']}_seed{seed}"
            directory = root / name
            required = {
                "launch_time": directory / "launch_time.txt",
                "environment": directory / "environment.txt",
                "command": directory / "command.txt",
                "final_adapter": directory / "checkpoints/final/adapter_config.json",
                "curriculum": directory / "checkpoints/final/curriculum_state.json",
                "evaluation": directory / "evaluation/all_summaries.json",
            }
            missing = [key for key, path in required.items() if not path.exists()]
            is_complete = not missing
            complete += int(is_complete)
            runs.append({"run": name, "complete": is_complete, "missing": missing})
    report = {
        "stage": args.stage,
        "root": str(root),
        "expected_runs": len(runs),
        "complete_runs": complete,
        "all_complete": complete == len(runs),
        "runs": runs,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")
    raise SystemExit(0 if report["all_complete"] else 1)


if __name__ == "__main__":
    main()
