from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.monitor_remote_training import parse_log


class MonitorTrainingTest(unittest.TestCase):
    def test_parse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "console.log"
            path.write_text("1/200 {'loss': '0.1', 'rewards/outcome_reward/mean': '0.5', 'frac_reward_zero_std': '0', 'completions/clipped_ratio': '0.25'}\n")
            report = parse_log(path)
        self.assertEqual(report["current_step"], 1)
        self.assertEqual(report["latest"]["rewards/outcome_reward/mean"], "0.5")


if __name__ == "__main__":
    unittest.main()
