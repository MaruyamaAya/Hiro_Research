from __future__ import annotations

import unittest

from llm_rl.summarize_experiments import aggregate


class SummarizeExperimentsTest(unittest.TestCase):
    def test_aggregate_seed_metrics(self) -> None:
        rows = [
            {"condition": "a", "pass_at_1": 0.4, "format_valid_rate": 1.0, "truncation_rate": 0.2, "mean_completion_tokens": 10},
            {"condition": "a", "pass_at_1": 0.6, "format_valid_rate": 0.8, "truncation_rate": 0.0, "mean_completion_tokens": 14},
        ]
        summary = aggregate(rows)[0]
        self.assertEqual(summary["seeds"], 2)
        self.assertAlmostEqual(summary["pass_at_1_mean"], 0.5)
        self.assertAlmostEqual(summary["mean_completion_tokens_mean"], 12.0)


if __name__ == "__main__":
    unittest.main()
