from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .agents import TabularAgent
from .environment import HiroWorld


def run_one(config: dict[str, Any], agent_name: str, seed: int, output: str) -> str:
    env = HiroWorld(config["environment"], seed)
    agent = TabularAgent(agent_name, len(env.actions), config, seed)
    steps = int(config["steps"])
    log_every = int(config["log_every"])
    rows: list[dict[str, Any]] = []
    kind_counts: dict[str, int] = {}
    ext_sum = 0.0
    lp_sum = 0.0
    appropriate_sum = 0.0
    meaningless_sum = 0.0
    boredom_sum = 0.0

    for step in range(steps):
        skill_before = env.skill
        mask = env.safe_mask() if agent_name != "hiro_no_safety" else np.ones(len(env.actions), dtype=bool)
        action_idx = agent.choose(skill_before, step, steps, mask)
        tr = env.observe_action(action_idx)
        reward = agent.reward(tr, int(env.visits[action_idx]))
        agent.update(skill_before, action_idx, reward)

        kind = str(tr["kind"])
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        ext_sum += float(tr["external"])
        lp_sum += float(tr["learning_progress"])
        if kind == "challenge":
            appropriate_sum += abs(
                float(tr["success_probability"])
                - float(config["environment"]["target_success_probability"])
            )
        meaningless_sum += float(tr["meaningless_suffering"])
        boredom_sum += float(tr["boredom"])

        if (step + 1) % log_every == 0 or step + 1 == steps:
            denom = step + 1
            challenge_steps = max(1, kind_counts.get("challenge", 0))
            selected_difficulty = (
                sum(
                    int(env.visits[i]) * a.difficulty
                    for i, a in enumerate(env.actions)
                    if a.kind == "challenge"
                )
                / challenge_steps
            )
            rows.append(
                {
                    "agent": agent_name,
                    "seed": seed,
                    "step": step + 1,
                    "skill": env.skill,
                    "external_return": ext_sum,
                    "learning_progress": lp_sum,
                    "easy_ratio": kind_counts.get("easy", 0) / denom,
                    "challenge_ratio": kind_counts.get("challenge", 0) / denom,
                    "noisy_tv_ratio": kind_counts.get("noise", 0) / denom,
                    "treadmill_ratio": kind_counts.get("treadmill", 0) / denom,
                    "impossible_ratio": kind_counts.get("impossible", 0) / denom,
                    "meaningless_suffering_ratio": meaningless_sum / denom,
                    "boredom_rate": boredom_sum / denom,
                    "challenge_appropriateness_error": appropriate_sum / challenge_steps,
                    "mean_selected_challenge_difficulty": selected_difficulty,
                    "total_damage": env.total_damage,
                }
            )

    path = Path(output) / "runs" / f"{agent_name}_seed{seed:03d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.json")
    parser.add_argument("--output", default="results/main")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "resolved_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    jobs = [
        (agent, seed)
        for agent in config["agents"]
        for seed in range(int(config["seeds"]))
    ]
    print(f"Running {len(jobs)} runs with {args.workers} workers")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_one, config, agent, seed, str(out))
            for agent, seed in jobs
        ]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 25 == 0 or i == len(futures):
                print(f"completed {i}/{len(futures)}", flush=True)


if __name__ == "__main__":
    main()
