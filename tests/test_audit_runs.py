from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class AuditRunsTest(unittest.TestCase):
    def test_reports_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run(
                ["python3", "scripts/audit_runs.py", "--stage", "integration", "--root", directory],
                text=True,
                capture_output=True,
            )
        self.assertEqual(process.returncode, 1)
        report = json.loads(process.stdout)
        self.assertEqual(report["expected_runs"], 8)
        self.assertEqual(report["complete_runs"], 0)


if __name__ == "__main__":
    unittest.main()
