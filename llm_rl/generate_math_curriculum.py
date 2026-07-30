from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_problem(rng: np.random.Generator, difficulty: int) -> tuple[str, int]:
    if difficulty == 1:
        a, b = rng.integers(1, 20, size=2)
        return f"Compute {a} + {b}.", int(a + b)
    if difficulty == 2:
        a, b = rng.integers(10, 100, size=2)
        return f"Compute {a} - {b}.", int(a - b)
    if difficulty == 3:
        a, b = rng.integers(2, 20, size=2)
        return f"Compute {a} × {b}.", int(a * b)
    if difficulty == 4:
        b = int(rng.integers(2, 20))
        q = int(rng.integers(2, 30))
        a = b * q
        return f"Compute {a} ÷ {b}.", q
    if difficulty == 5:
        a, b, c = rng.integers(2, 30, size=3)
        return f"Compute ({a} + {b}) × {c}.", int((a + b) * c)
    if difficulty == 6:
        x = int(rng.integers(-20, 21))
        a = int(rng.integers(2, 12))
        b = int(rng.integers(-30, 31))
        c = a * x + b
        return f"Solve for x: {a}x + {b} = {c}.", x
    if difficulty == 7:
        a, b, c, d = rng.integers(2, 20, size=4)
        return f"Compute ({a} × {b}) + ({c} × {d}).", int(a * b + c * d)
    a, b, c = rng.integers(10, 80, size=3)
    return f"Compute {a} × {b} - {c}.", int(a * b - c)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/math_curriculum.jsonl")
    parser.add_argument("--per-level", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    idx = 0
    for difficulty in range(1, 9):
        for _ in range(args.per_level):
            problem, answer = make_problem(rng, difficulty)
            rows.append(
                {
                    "id": idx,
                    "difficulty": difficulty,
                    "prompt": [
                        {
                            "role": "system",
                            "content": (
                                "Solve the problem. End with exactly "
                                "<answer>INTEGER</answer>."
                            ),
                        },
                        {"role": "user", "content": problem},
                    ],
                    "answer": answer,
                    "split": "eval" if idx % 10 == 0 else "train",
                }
            )
            idx += 1
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
