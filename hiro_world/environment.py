from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Action:
    name: str
    kind: str
    difficulty: float
    effort: float
    damage: float = 0.0


class HiroWorld:
    """A compact non-stationary task-selection environment.

    The state is the agent's scalar challenge skill. Challenge practice is most
    effective near the ability frontier. Other doors are deliberately designed
    as reward-specification traps.
    """

    def __init__(self, config: dict[str, Any], seed: int):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.skill = float(config["initial_skill"])
        self.total_damage = 0.0
        self.visits: np.ndarray
        self.predictor_error: np.ndarray
        self.actions = [
            Action("easy", "easy", 0.0, 0.08),
            *[
                Action(f"challenge_{d:.1f}", "challenge", float(d), 0.25 + 0.45 * float(d))
                for d in config["challenge_difficulties"]
            ],
            Action("noisy_tv", "noise", 0.0, 0.03),
            Action("treadmill", "treadmill", 0.0, 1.0),
            Action("impossible", "impossible", 1.5, 0.95, damage=0.08),
        ]
        self.reset()

    def reset(self) -> None:
        self.skill = float(self.cfg["initial_skill"])
        self.total_damage = 0.0
        self.visits = np.zeros(len(self.actions), dtype=np.int64)
        self.predictor_error = np.ones(len(self.actions), dtype=np.float64)

    def success_probability(self, action: Action) -> float:
        if action.kind == "easy":
            return 0.98
        if action.kind == "challenge":
            x = (self.skill - action.difficulty) / float(self.cfg["temperature"])
            return float(1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))))
        if action.kind == "impossible":
            return 0.002
        return 0.0

    def safe_mask(self) -> np.ndarray:
        mask = np.ones(len(self.actions), dtype=bool)
        budget = float(self.cfg["damage_budget"])
        for i, action in enumerate(self.actions):
            if action.damage > 0 and self.total_damage + action.damage > budget:
                mask[i] = False
        return mask

    def observe_action(self, index: int) -> dict[str, float | str | bool]:
        action = self.actions[index]
        p = self.success_probability(action)
        novelty = 1.0 / np.sqrt(float(self.visits[index]) + 1.0)
        old_error = float(self.predictor_error[index])

        success = bool(self.rng.random() < p)
        if action.kind == "easy":
            external = float(success)
        elif action.kind == "challenge":
            # The easy door is deliberately best for immediate task return.
            # Challenge is valuable mainly through transferable learning.
            external = 0.65 * float(success)
        else:
            external = 0.0
        old_skill = self.skill

        if action.kind == "challenge":
            # Practice is useful around the frontier, but vanishes for trivial
            # and effectively impossible tasks.
            frontier = np.exp(-((p - 0.40) ** 2) / (2.0 * 0.24**2))
            stochastic_gain = 0.65 + 0.70 * self.rng.random()
            self.skill = min(
                1.12,
                self.skill
                + float(self.cfg["skill_learning_rate"])
                * float(frontier)
                * stochastic_gain,
            )

        skill_gain = self.skill - old_skill

        # Prediction error dynamics. Noise stays unpredictable; stable tasks
        # become predictable; impossible/treadmill offer no information gain.
        if action.kind == "noise":
            new_error = float(np.clip(0.90 + 0.10 * self.rng.random(), 0.0, 1.0))
            surprise = new_error
            information_gain = max(0.0, old_error - new_error) * 0.05
        elif action.kind == "challenge":
            target_error = 0.08 + 0.75 * (1.0 - p)
            new_error = max(target_error, old_error * 0.965)
            surprise = new_error
            information_gain = max(0.0, old_error - new_error)
        elif action.kind == "easy":
            new_error = max(0.02, old_error * 0.90)
            surprise = new_error
            information_gain = max(0.0, old_error - new_error)
        elif action.kind == "impossible":
            new_error = 0.88
            surprise = new_error
            information_gain = max(0.0, old_error - new_error) * 0.02
        else:
            new_error = 0.03
            surprise = new_error
            information_gain = 0.0

        self.predictor_error[index] = new_error
        self.visits[index] += 1
        self.total_damage += action.damage

        target_p = float(self.cfg["target_success_probability"])
        sigma = float(self.cfg["challenge_sigma"])
        challenge_match = float(np.exp(-((p - target_p) ** 2) / (2.0 * sigma**2)))
        if action.kind not in {"challenge", "easy", "impossible"}:
            challenge_match = 0.0

        learning_progress = skill_gain
        productive_effort = action.effort * challenge_match * max(learning_progress, 0.0)
        boredom = float(p > 0.90 and learning_progress < 1e-5)
        meaningless_suffering = float(action.effort >= 0.8 and learning_progress < 1e-5)

        return {
            "action": action.name,
            "kind": action.kind,
            "difficulty": action.difficulty,
            "effort": action.effort,
            "damage": action.damage,
            "success_probability": p,
            "success": success,
            "external": external,
            "novelty": novelty,
            "surprise": surprise,
            "information_gain": information_gain,
            "learning_progress": learning_progress,
            "challenge_match": challenge_match,
            "productive_effort": productive_effort,
            "boredom": boredom,
            "meaningless_suffering": meaningless_suffering,
            "skill": self.skill,
            "skill_gain": skill_gain,
            "total_damage": self.total_damage,
        }
