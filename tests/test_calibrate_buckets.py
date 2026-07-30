from __future__ import annotations

import unittest

from llm_rl.calibrate_buckets import calibrate


class CalibrateBucketsTest(unittest.TestCase):
    def test_detects_monotonic_buckets(self) -> None:
        records = []
        for bucket, outcomes in [("proxy_1", [1, 1]), ("proxy_2", [1, 0]), ("proxy_3", [0, 0])]:
            for outcome in outcomes:
                records.append({"difficulty": bucket, "correct": outcome, "completion_tokens": 10, "truncated": False})
        report = calibrate(records)
        self.assertTrue(report["monotonic_nonincreasing"])

    def test_detects_violation(self) -> None:
        report = calibrate([
            {"difficulty": "proxy_1", "correct": False, "completion_tokens": 1, "truncated": False},
            {"difficulty": "proxy_2", "correct": True, "completion_tokens": 1, "truncated": False},
        ])
        self.assertFalse(report["monotonic_nonincreasing"])
        self.assertEqual(len(report["violations"]), 1)


if __name__ == "__main__":
    unittest.main()
