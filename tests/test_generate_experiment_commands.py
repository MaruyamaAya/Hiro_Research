from __future__ import annotations

import subprocess
import unittest


class GenerateCommandsTest(unittest.TestCase):
    def test_final_matrix_has_24_runs(self) -> None:
        output = subprocess.check_output(
            ["python3", "scripts/generate_experiment_commands.py", "--stage", "final"],
            text=True,
        )
        lines = [x for x in output.splitlines() if x]
        self.assertEqual(len(lines), 24)
        self.assertTrue(any("dapo_hiro_seed42" in line for line in lines))
        self.assertTrue(any("dapo_histdyn_seed42" in line for line in lines))
        self.assertTrue(any("--dynamic-sampling" in line for line in lines))
        self.assertTrue(any("--no-dynamic-sampling" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
