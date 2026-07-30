from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Iterator, Sized

import torch
from torch.utils.data import Sampler


class CurriculumState:
    """Synchronized bucket-level competence statistics for Hiro sampling."""

    def __init__(
        self,
        buckets: Iterable[str],
        window_size: int = 128,
        target_success: float = 0.45,
        challenge_bandwidth: float = 0.20,
    ):
        self.buckets = tuple(sorted({str(x) for x in buckets}))
        self.window_size = window_size
        self.target_success = target_success
        self.challenge_bandwidth = challenge_bandwidth
        self.attempts = defaultdict(int)
        self.correct = defaultdict(int)
        self.recent = {
            bucket: deque(maxlen=window_size) for bucket in self.buckets
        }
        self.previous_window = {bucket: 0.5 for bucket in self.buckets}
        self.zero_gradient_groups = defaultdict(int)
        self.groups = defaultdict(int)

    def update(
        self,
        bucket: str,
        outcomes: Iterable[float],
        zero_gradient: bool = False,
    ) -> None:
        bucket = str(bucket)
        values = [float(x) for x in outcomes]
        if bucket not in self.recent:
            self.buckets = tuple(sorted(set(self.buckets) | {bucket}))
            self.recent[bucket] = deque(maxlen=self.window_size)
            self.previous_window[bucket] = 0.5
        old_rate = self.pass_rate(bucket)
        self.previous_window[bucket] = old_rate
        self.recent[bucket].extend(values)
        self.attempts[bucket] += len(values)
        self.correct[bucket] += int(sum(values))
        self.groups[bucket] += 1
        self.zero_gradient_groups[bucket] += int(zero_gradient)

    def pass_rate(self, bucket: str) -> float:
        values = self.recent.get(str(bucket))
        return sum(values) / len(values) if values else 0.5

    def progress(self, bucket: str) -> float:
        bucket = str(bucket)
        return self.pass_rate(bucket) - self.previous_window.get(bucket, 0.5)

    def sampling_weights(
        self,
        mode: str,
        min_probability: float = 0.02,
        progress_weight: float = 1.0,
        coverage_weight: float = 0.15,
    ) -> dict[str, float]:
        if not self.buckets:
            return {}
        scores = {}
        max_attempts = max([self.attempts[x] for x in self.buckets] + [1])
        for bucket in self.buckets:
            ability = self.pass_rate(bucket)
            challenge = math.exp(
                -((ability - self.target_success) ** 2)
                / (2 * self.challenge_bandwidth**2)
            )
            positive_progress = max(0.0, self.progress(bucket))
            coverage = 1.0 - self.attempts[bucket] / max_attempts
            if mode == "uniform":
                score = 1.0
            elif mode == "challenge":
                score = challenge
            elif mode == "progress":
                score = positive_progress
            elif mode == "hiro":
                score = challenge + progress_weight * positive_progress
            else:
                raise ValueError(f"Unknown curriculum mode: {mode}")
            scores[bucket] = max(0.0, score + coverage_weight * coverage)
        total = sum(scores.values())
        if total <= 0:
            scores = {bucket: 1.0 for bucket in self.buckets}
            total = float(len(scores))
        probabilities = {bucket: value / total for bucket, value in scores.items()}
        floor = min(min_probability, 1.0 / len(probabilities))
        remaining = 1.0 - floor * len(probabilities)
        return {
            bucket: floor + remaining * probability
            for bucket, probability in probabilities.items()
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "buckets": list(self.buckets),
            "window_size": self.window_size,
            "target_success": self.target_success,
            "challenge_bandwidth": self.challenge_bandwidth,
            "attempts": dict(self.attempts),
            "correct": dict(self.correct),
            "recent": {key: list(value) for key, value in self.recent.items()},
            "previous_window": self.previous_window,
            "zero_gradient_groups": dict(self.zero_gradient_groups),
            "groups": dict(self.groups),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "CurriculumState":
        obj = cls(
            state["buckets"],
            window_size=int(state["window_size"]),
            target_success=float(state["target_success"]),
            challenge_bandwidth=float(state["challenge_bandwidth"]),
        )
        for key in obj.buckets:
            obj.attempts[key] = int(state["attempts"].get(key, 0))
            obj.correct[key] = int(state["correct"].get(key, 0))
            obj.recent[key].extend(state["recent"].get(key, []))
            obj.previous_window[key] = float(
                state["previous_window"].get(key, 0.5)
            )
            obj.zero_gradient_groups[key] = int(
                state["zero_gradient_groups"].get(key, 0)
            )
            obj.groups[key] = int(state["groups"].get(key, 0))
        return obj

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state_dict(), indent=2) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "CurriculumState":
        return cls.from_state_dict(json.loads(Path(path).read_text()))


class CurriculumRepeatSampler(Sampler[int]):
    """TRL-compatible repeated sampler with bucket-weighted prompt selection."""

    def __init__(
        self,
        data_source: Sized,
        bucket_by_index: list[str],
        state: CurriculumState,
        mode: str,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: int = 0,
    ):
        if len(data_source) != len(bucket_by_index):
            raise ValueError("bucket_by_index must align with data_source")
        self.data_source = data_source
        self.bucket_by_index = [str(x) for x in bucket_by_index]
        self.state = state
        self.mode = mode
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.generator = torch.Generator().manual_seed(seed)
        self.indices_by_bucket: dict[str, list[int]] = defaultdict(list)
        for index, bucket in enumerate(self.bucket_by_index):
            self.indices_by_bucket[bucket].append(index)

    def __iter__(self) -> Iterator[int]:
        num_chunks = self.num_samples // self.batch_size
        for _ in range(num_chunks):
            # Recompute at every generation batch so newly synchronized
            # competence statistics affect subsequent prompt selection.
            weights_by_bucket = self.state.sampling_weights(self.mode)
            per_index_weights = torch.tensor(
                [
                    weights_by_bucket[bucket] / len(self.indices_by_bucket[bucket])
                    for bucket in self.bucket_by_index
                ],
                dtype=torch.double,
            )
            chunk = torch.multinomial(
                per_index_weights,
                self.batch_size,
                replacement=True,
                generator=self.generator,
            ).tolist()
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        usable = (self.num_samples // self.batch_size) * self.batch_size
        return usable * self.mini_repeat_count * self.repeat_count
