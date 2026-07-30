from __future__ import annotations

import unittest

from llm_rl.bootstrap_stats import paired_bootstrap


class BootstrapStatsTest(unittest.TestCase):
    def test_constant_paired_gain(self) -> None:
        report = paired_bootstrap([0, 1, 2], [1, 2, 3], samples=1000, seed=1)
        self.assertEqual(report["mean_difference"], 1.0)
        self.assertEqual(report["ci95_low"], 1.0)
        self.assertEqual(report["ci95_high"], 1.0)

    def test_rejects_unpaired_input(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap([1], [1, 2])


if __name__ == "__main__":
    unittest.main()
