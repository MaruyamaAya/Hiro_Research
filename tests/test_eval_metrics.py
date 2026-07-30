from __future__ import annotations

import unittest

from llm_rl.eval_metrics import summarize_predictions


class EvalMetricsTest(unittest.TestCase):
    def test_pass_at_k_and_strata(self) -> None:
        records = [
            {
                "id": "a",
                "sample_index": 0,
                "correct": False,
                "valid": True,
                "truncated": False,
                "completion_tokens": 10,
                "status": "incorrect",
                "difficulty": 1,
                "source": "toy",
            },
            {
                "id": "a",
                "sample_index": 1,
                "correct": True,
                "valid": True,
                "truncated": False,
                "completion_tokens": 12,
                "status": "correct",
                "difficulty": 1,
                "source": "toy",
            },
            {
                "id": "b",
                "sample_index": 0,
                "correct": True,
                "valid": True,
                "truncated": True,
                "completion_tokens": 20,
                "status": "correct",
                "difficulty": 2,
                "source": "toy",
            },
            {
                "id": "b",
                "sample_index": 1,
                "correct": False,
                "valid": False,
                "truncated": False,
                "completion_tokens": 8,
                "status": "no_answer",
                "difficulty": 2,
                "source": "toy",
            },
        ]
        summary = summarize_predictions(records)
        self.assertEqual(summary["problems"], 2)
        self.assertEqual(summary["generated_completions"], 4)
        self.assertEqual(summary["pass_at_k"]["1"], 0.5)
        self.assertEqual(summary["pass_at_k"]["2"], 1.0)
        self.assertEqual(summary["by_difficulty"]["1"]["pass_at_1"], 0.0)
        self.assertEqual(summary["by_difficulty"]["2"]["pass_at_1"], 1.0)
        self.assertEqual(summary["truncation_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
