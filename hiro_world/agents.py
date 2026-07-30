from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RewardWeights:
    external: float = 1.0
    novelty: float = 0.0
    surprise: float = 0.0
    information_gain: float = 0.0
    learning_progress: float = 0.0
    challenge_match: float = 0.0
    productive_effort: float = 0.0
    effort: float = 0.0
    difficulty: float = 0.0
    boredom: float = 0.0
    repetition: float = 0.0


AGENT_WEIGHTS: dict[str, RewardWeights] = {
    "extrinsic": RewardWeights(external=1.0),
    "novelty": RewardWeights(external=0.45, novelty=0.85),
    "surprise": RewardWeights(external=0.25, surprise=1.15),
    "suffering": RewardWeights(external=0.20, effort=1.25),
    "difficulty": RewardWeights(external=0.05, difficulty=2.50),
    "hiro": RewardWeights(
        external=0.35,
        novelty=0.08,
        information_gain=1.25,
        learning_progress=28.0,
        challenge_match=0.30,
        productive_effort=42.0,
        boredom=0.18,
        repetition=0.08,
    ),
    "hiro_no_lp": RewardWeights(
        external=0.35,
        novelty=0.08,
        surprise=0.65,
        challenge_match=0.30,
        boredom=0.12,
    ),
    "hiro_no_challenge": RewardWeights(
        external=0.35,
        novelty=0.08,
        information_gain=1.25,
        learning_progress=28.0,
        effort=0.08,
        boredom=0.12,
    ),
    "hiro_no_safety": RewardWeights(
        external=0.35,
        novelty=0.08,
        information_gain=1.25,
        learning_progress=28.0,
        challenge_match=0.30,
        productive_effort=42.0,
        difficulty=0.35,
        boredom=0.18,
        repetition=0.08,
    ),
    "hiro_no_external": RewardWeights(
        novelty=0.08,
        information_gain=1.25,
        learning_progress=28.0,
        challenge_match=0.30,
        productive_effort=42.0,
        boredom=0.18,
        repetition=0.08,
    ),
    "hiro_additive_effort": RewardWeights(
        external=0.35,
        novelty=0.08,
        information_gain=1.25,
        learning_progress=28.0,
        challenge_match=0.30,
        effort=0.85,
        boredom=0.12,
    ),
}


class TabularAgent:
    def __init__(
        self,
        name: str,
        n_actions: int,
        config: dict[str, Any],
        seed: int,
    ):
        if name not in AGENT_WEIGHTS:
            raise ValueError(f"Unknown agent: {name}")
        self.name = name
        self.weights = AGENT_WEIGHTS[name]
        self.rng = np.random.default_rng(seed + 10_000)
        self.n_actions = n_actions
        self.n_bins = int(config["skill_bins"])
        self.alpha = float(config["learning_rate"])
        self.q = np.zeros((self.n_bins, n_actions), dtype=np.float64)
        self.counts = np.zeros((self.n_bins, n_actions), dtype=np.int64)
        self.config = config

    def state_bin(self, skill: float) -> int:
        return int(np.clip(round(skill * (self.n_bins - 1)), 0, self.n_bins - 1))

    def epsilon(self, step: int, total_steps: int) -> float:
        start = float(self.config["epsilon_start"])
        end = float(self.config["epsilon_end"])
        horizon = max(1, int(total_steps * float(self.config["epsilon_decay_fraction"])))
        frac = min(1.0, step / horizon)
        return start + frac * (end - start)

    def choose(
        self,
        skill: float,
        step: int,
        total_steps: int,
        safe_mask: np.ndarray,
    ) -> int:
        b = self.state_bin(skill)
        allowed = np.flatnonzero(safe_mask)
        if self.rng.random() < self.epsilon(step, total_steps):
            return int(self.rng.choice(allowed))
        values = self.q[b].copy()
        values[~safe_mask] = -np.inf
        best = np.flatnonzero(np.isclose(values, np.max(values)))
        return int(self.rng.choice(best))

    def reward(self, transition: dict[str, Any], action_visits: int) -> float:
        w = self.weights
        difficulty = 1.0 - float(transition["success_probability"])
        repetition = np.log1p(action_visits) / 10.0
        return float(
            w.external * transition["external"]
            + w.novelty * transition["novelty"]
            + w.surprise * transition["surprise"]
            + w.information_gain * transition["information_gain"]
            + w.learning_progress * transition["learning_progress"]
            + w.challenge_match * transition["challenge_match"]
            + w.productive_effort * transition["productive_effort"]
            + w.effort * transition["effort"]
            + w.difficulty * difficulty
            - w.boredom * transition["boredom"]
            - w.repetition * repetition
        )

    def update(self, skill_before: float, action: int, reward: float) -> None:
        b = self.state_bin(skill_before)
        self.counts[b, action] += 1
        # Constant-step-size bandit update tracks the non-stationary reward.
        self.q[b, action] += self.alpha * (reward - self.q[b, action])
