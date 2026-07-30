from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_rl.curriculum import CurriculumRepeatSampler, CurriculumState


class CurriculumTest(unittest.TestCase):
    def test_hiro_prefers_frontier_bucket(self) -> None:
        state = CurriculumState(["easy", "frontier", "impossible"])
        state.update("easy", [1] * 100)
        state.update("frontier", [1] * 45 + [0] * 55)
        state.update("impossible", [0] * 100)
        weights = state.sampling_weights("hiro", coverage_weight=0.0)
        self.assertGreater(weights["frontier"], weights["easy"])
        self.assertGreater(weights["frontier"], weights["impossible"])

    def test_round_trip(self) -> None:
        state = CurriculumState(["a"], window_size=4)
        state.update("a", [1, 0, 1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state.save(path)
            restored = CurriculumState.load(path)
        self.assertEqual(state.state_dict(), restored.state_dict())

    def test_sampler_repeats_each_prompt_as_a_group(self) -> None:
        state = CurriculumState(["a", "b"])
        sampler = CurriculumRepeatSampler(
            list(range(4)),
            ["a", "a", "b", "b"],
            state,
            mode="uniform",
            mini_repeat_count=3,
            batch_size=2,
            repeat_count=1,
            seed=7,
        )
        values = list(sampler)
        self.assertEqual(len(values), 12)
        for start in range(0, len(values), 3):
            self.assertEqual(len(set(values[start : start + 3])), 1)


if __name__ == "__main__":
    unittest.main()
