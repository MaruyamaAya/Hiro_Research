from __future__ import annotations

import unittest

from llm_rl.manual_verifier_audit import deterministic_sample, score_annotations


class ManualVerifierAuditTest(unittest.TestCase):
    def test_stratified_sampling(self) -> None:
        records = [
            {"id": str(i), "source": source, "status": status}
            for i, (source, status) in enumerate(
                [("a", "correct"), ("a", "correct"), ("a", "incorrect"), ("b", "correct")]
            )
        ]
        sample = deterministic_sample(records, per_stratum=1, seed=1)
        self.assertEqual(len(sample), 3)

    def test_annotation_scoring(self) -> None:
        report = score_annotations([
            {"verifier_status_hidden": "correct", "human_label": "correct"},
            {"verifier_status_hidden": "correct", "human_label": "incorrect"},
            {"verifier_status_hidden": "incorrect", "human_label": "correct"},
        ])
        self.assertAlmostEqual(report["binary_agreement"], 1 / 3)
        self.assertEqual(report["false_accepts"], 1)
        self.assertEqual(report["false_rejects"], 1)


if __name__ == "__main__":
    unittest.main()
