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

    def test_progress_compares_disjoint_windows(self) -> None:
        state = CurriculumState(["a"], window_size=4)
        state.update("a", [0, 0, 0, 0])
        self.assertEqual(state.previous_pass_rate("a"), 0.5)
        state.update("a", [1, 1, 1, 1])
        self.assertEqual(state.previous_pass_rate("a"), 0.0)
        self.assertEqual(state.pass_rate("a"), 1.0)
        self.assertEqual(state.progress("a"), 1.0)

    def test_progress_mode_revisits_regressing_bucket(self) -> None:
        state = CurriculumState(["stable", "regressing"], window_size=4)
        state.update("stable", [1, 1, 1, 1, 1, 1, 1, 1])
        state.update("regressing", [1, 1, 1, 1, 0, 0, 0, 0])
        weights = state.sampling_weights("progress", coverage_weight=0.0)
        self.assertGreater(weights["regressing"], weights["stable"])

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
            ["0", "1", "2", "3"],
            state,
            mode="uniform",
            dynamic_sampling=False,
            mini_repeat_count=3,
            batch_size=2,
            repeat_count=1,
            seed=7,
        )
        values = list(sampler)
        self.assertEqual(len(values), 12)
        for start in range(0, len(values), 3):
            self.assertEqual(len(set(values[start : start + 3])), 1)

    def test_sampler_resume_continues_exact_random_stream(self) -> None:
        state = CurriculumState(["a", "b"])
        first_sampler = CurriculumRepeatSampler(
            list(range(6)),
            ["a", "a", "a", "b", "b", "b"],
            ["0", "1", "2", "3", "4", "5"],
            state,
            mode="uniform",
            dynamic_sampling=False,
            mini_repeat_count=1,
            batch_size=2,
            seed=19,
        )
        iterator = iter(first_sampler)
        prefix = [next(iterator) for _ in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state.save(path)
            restored = CurriculumState.load(path)
        original_suffix = [next(iterator) for _ in range(4)]
        resumed_sampler = CurriculumRepeatSampler(
            list(range(6)),
            ["a", "a", "a", "b", "b", "b"],
            ["0", "1", "2", "3", "4", "5"],
            restored,
            mode="uniform",
            dynamic_sampling=False,
            mini_repeat_count=1,
            batch_size=2,
            seed=19,
        )
        resumed_iterator = iter(resumed_sampler)
        resumed_suffix = [next(resumed_iterator) for _ in range(4)]
        self.assertEqual(len(prefix), 2)
        self.assertEqual(original_suffix, resumed_suffix)

    def test_dynamic_sampling_downweights_zero_gradient_prompt(self) -> None:
        state = CurriculumState(["a"])
        for _ in range(10):
            state.update("a", [1, 1, 1, 1], zero_gradient=True, prompt_id="stale")
            state.update("a", [1, 0, 1, 0], zero_gradient=False, prompt_id="useful")
        self.assertLess(
            state.prompt_dynamic_weight("stale"),
            state.prompt_dynamic_weight("useful"),
        )


if __name__ == "__main__":
    unittest.main()
