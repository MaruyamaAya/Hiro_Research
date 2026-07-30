from __future__ import annotations

import subprocess
import unittest


class GenerateEvaluationCommandsTest(unittest.TestCase):
    def test_final_uses_test_data_and_eval_split(self) -> None:
        output = subprocess.check_output(
            ["python3", "scripts/generate_evaluation_commands.py", "--stage", "final"],
            text=True,
        )
        lines = output.splitlines()
        self.assertEqual(len(lines), 24)
        self.assertTrue(all("HIRO_TEST_DATA" in line for line in lines))
        self.assertTrue(all(line.endswith(" eval") for line in lines))

    def test_calibration_uses_validation(self) -> None:
        output = subprocess.check_output(
            ["python3", "scripts/generate_evaluation_commands.py", "--stage", "calibration"],
            text=True,
        )
        self.assertTrue(all(line.endswith(" validation") for line in output.splitlines()))


if __name__ == "__main__":
    unittest.main()
