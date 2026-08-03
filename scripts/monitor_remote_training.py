from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace") if path.exists() else ""
    steps = [int(x) for x in re.findall(r"(\d+)/200", text)]
    records = []
    for line in text.splitlines():
        if "'rewards/outcome_reward/mean'" not in line:
            continue
        start = line.find("{'loss'")
        if start < 0:
            continue
        payload = line[start:].replace("'", '"')
        try:
            records.append(json.loads(payload))
        except json.JSONDecodeError:
            pass
    return {
        "current_step": max(steps, default=0),
        "logged_steps": len(records),
        "latest": records[-1] if records else None,
        "zero_std_fraction": (
            sum(float(x.get("frac_reward_zero_std", 0)) for x in records) / len(records)
            if records
            else None
        ),
        "mean_clipped_ratio": (
            sum(float(x.get("completions/clipped_ratio", 0)) for x in records)
            / len(records)
            if records
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run = Path(args.run)
    report = parse_log(run / "console.log")
    checkpoints = sorted(
        int(path.name.split("-")[-1])
        for path in (run / "checkpoints").glob("checkpoint-*")
        if path.name.split("-")[-1].isdigit()
    )
    report["checkpoints"] = checkpoints
    report["latest_checkpoint"] = checkpoints[-1] if checkpoints else None
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
