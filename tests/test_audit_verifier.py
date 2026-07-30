from __future__ import annotations

import unittest

from llm_rl.audit_verifier import audit


class AuditVerifierTest(unittest.TestCase):
    def test_audit_positive_and_negative_controls(self) -> None:
        report = audit(
            [
                {"id": "a", "source": "toy", "answer": "1/2"},
                {"id": "b", "source": "toy", "answer": r"\text{east}"},
            ]
        )
        self.assertEqual(report["reference_coverage"], 1.0)
        self.assertEqual(report["negative_false_accepts"], 0)
        self.assertEqual(report["failures"], [])


if __name__ == "__main__":
    unittest.main()
