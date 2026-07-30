from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def option(name: str, value: object) -> list[str]:
    flag = "--" + name.replace("_", "-")
    if isinstance(value, bool):
        return [flag if value else "--no-" + name.replace("_", "-")]
    return [flag, str(value)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/publication_matrix.json")
    parser.add_argument("--stage", choices=["integration", "calibration", "final"], required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    overrides = {
        "integration": {"max_steps": 5, "save_steps": 5},
        "calibration": {"max_steps": 200, "save_steps": 20},
        "final": {},
    }[args.stage]
    seeds = config["seeds"][:1] if args.stage != "final" else config["seeds"]
    commands = []
    for condition in config["conditions"]:
        for seed in seeds:
            run_name = f"{args.stage}_{condition['id']}_seed{seed}"
            values = {**config["shared"], **condition, **overrides, "seed": seed}
            reward_mode = values.pop("mode")
            values.pop("id")
            pieces = [
                "./scripts/launch_distributed_grpo.sh",
                reward_mode,
                run_name,
            ]
            for key, value in values.items():
                pieces.extend(option(key, value))
            command = " ".join(shlex.quote(x) for x in pieces)
            commands.append(
                "NPROC=8 "
                'HIRO_TRAIN_DATA="${HIRO_TRAIN_DATA:?set persistent train JSONL}" '
                + command
            )
    text = "\n".join(commands) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
