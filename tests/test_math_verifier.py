from __future__ import annotations

import unittest

from llm_rl.math_verifier import (
    VerificationStatus,
    answers_equivalent,
    extract_answer_candidates,
    verify_answer,
)


class MathVerifierTest(unittest.TestCase):
    def test_answer_tag_integer(self) -> None:
        result = verify_answer("Work... <answer>-12</answer>", -12)
        self.assertEqual(result.status, VerificationStatus.CORRECT)
        self.assertTrue(result.valid)

    def test_boxed_fraction_matches_decimal(self) -> None:
        result = verify_answer(r"Thus \boxed{\frac{1}{2}}.", "0.5")
        self.assertEqual(result.status, VerificationStatus.CORRECT)

    def test_simple_symbolic_equivalence(self) -> None:
        self.assertTrue(answers_equivalent("(x+1)^2", "x^2 + 2*x + 1"))

    def test_assignment_form(self) -> None:
        result = verify_answer(r"\boxed{x = 7}", 7)
        self.assertEqual(result.status, VerificationStatus.CORRECT)

    def test_incorrect_but_valid(self) -> None:
        result = verify_answer("Final answer: 41", 42)
        self.assertEqual(result.status, VerificationStatus.INCORRECT)
        self.assertTrue(result.valid)

    def test_no_implicit_number_extraction(self) -> None:
        result = verify_answer("The prompt says 42, but I am unsure.", 42)
        self.assertEqual(result.status, VerificationStatus.NO_ANSWER)

    def test_conflicting_tags_are_ambiguous(self) -> None:
        result = verify_answer("<answer>2</answer> then <answer>3</answer>", 3)
        self.assertEqual(result.status, VerificationStatus.AMBIGUOUS)

    def test_repeated_equivalent_tags_are_allowed(self) -> None:
        result = verify_answer(
            r"<answer>1/2</answer> and finally <answer>\frac{2}{4}</answer>",
            "0.5",
        )
        self.assertEqual(result.status, VerificationStatus.CORRECT)

    def test_rejects_unsupported_identifier(self) -> None:
        result = verify_answer("<answer>open_file(1)</answer>", 1)
        self.assertEqual(result.status, VerificationStatus.INVALID)

    def test_allows_single_letter_symbolic_variables(self) -> None:
        result = verify_answer("<answer>2k + 2</answer>", "2*(k+1)")
        self.assertEqual(result.status, VerificationStatus.CORRECT)

    def test_complex_number(self) -> None:
        result = verify_answer("<answer>6 - 5i</answer>", "6-5*i")
        self.assertEqual(result.status, VerificationStatus.CORRECT)

    def test_answer_tag_has_priority_over_reasoning_box(self) -> None:
        candidates, source = extract_answer_candidates(
            r"Intermediate \boxed{3}; <answer>4</answer>"
        )
        self.assertEqual(candidates, ["4"])
        self.assertEqual(source, "answer_tag")


if __name__ == "__main__":
    unittest.main()
