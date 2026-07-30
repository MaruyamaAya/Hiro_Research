from __future__ import annotations

import unittest

from llm_rl.rewards import soft_overlong_penalty


class RewardTest(unittest.TestCase):
    def test_soft_overlong_penalty(self) -> None:
        self.assertEqual(soft_overlong_penalty(700, 1024, 256, 1.0), 0.0)
        self.assertEqual(soft_overlong_penalty(768, 1024, 256, 1.0), 0.0)
        self.assertAlmostEqual(
            soft_overlong_penalty(896, 1024, 256, 1.0), 0.5
        )
        self.assertEqual(soft_overlong_penalty(1024, 1024, 256, 1.0), 1.0)


if __name__ == "__main__":
    unittest.main()
