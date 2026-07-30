from __future__ import annotations

import unittest

from llm_rl.compare_conditions import compare_items


class CompareConditionsTest(unittest.TestCase):
    def test_paired_item_comparison(self) -> None:
        report = compare_items({"a": False, "b": True}, {"a": True, "b": True}, 1000, 1)
        self.assertEqual(report["shared_items"], 2)
        self.assertEqual(report["mean_difference"], 0.5)


if __name__ == "__main__":
    unittest.main()
