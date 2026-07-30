from __future__ import annotations

import unittest

from llm_rl.prepare_real_math_data import (
    canonical_text,
    heuristic_difficulty_bucket,
    text_hash,
)


class PrepareRealMathDataTest(unittest.TestCase):
    def test_canonicalization_ignores_cosmetic_math_formatting(self) -> None:
        self.assertEqual(canonical_text(r"Find $x+1$."), canonical_text("Find x + 1"))

    def test_hash_is_stable(self) -> None:
        self.assertEqual(text_hash("A  B"), text_hash("a-b"))

    def test_proxy_difficulty_is_monotonic_for_clear_examples(self) -> None:
        easy = heuristic_difficulty_bucket("Compute 1+1.", "2")
        hard = heuristic_difficulty_bucket(
            r"Let a polynomial function satisfy \sum_{k=1}^{100} "
            r"\frac{x^k}{k}. Determine the complex roots using a matrix.",
            r"\frac{\sqrt{2}}{3}",
        )
        self.assertLess(int(easy[-1]), int(hard[-1]))


if __name__ == "__main__":
    unittest.main()
