from __future__ import annotations

import unittest

from llm_rl.prepare_real_math_data import canonical_text, text_hash


class PrepareRealMathDataTest(unittest.TestCase):
    def test_canonicalization_ignores_cosmetic_math_formatting(self) -> None:
        self.assertEqual(canonical_text(r"Find $x+1$."), canonical_text("Find x + 1"))

    def test_hash_is_stable(self) -> None:
        self.assertEqual(text_hash("A  B"), text_hash("a-b"))


if __name__ == "__main__":
    unittest.main()
