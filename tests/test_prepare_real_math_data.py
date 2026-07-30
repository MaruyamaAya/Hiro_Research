from __future__ import annotations

import unittest

from llm_rl.prepare_real_math_data import (
    canonical_text,
    heuristic_difficulty_bucket,
    near_duplicate_eval_id,
    token_ngrams,
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

    def test_near_duplicate_detection(self) -> None:
        eval_ngrams = [("eval:1", token_ngrams("Find the value of x if x plus two equals five"))]
        index = {}
        for gram in eval_ngrams[0][1]:
            index.setdefault(gram, set()).add(0)
        match, score = near_duplicate_eval_id(
            "Find the value of x if x plus two equals five.",
            eval_ngrams,
            index,
            0.8,
        )
        self.assertEqual(match, "eval:1")
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
