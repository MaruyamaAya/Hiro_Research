from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class MergeEvalShardsTest(unittest.TestCase):
    def test_merge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shard, record in enumerate(
                [
                    {"id": "a", "sample_index": 0, "correct": True, "valid": True, "truncated": False, "completion_tokens": 1, "status": "correct"},
                    {"id": "b", "sample_index": 0, "correct": False, "valid": True, "truncated": False, "completion_tokens": 2, "status": "incorrect"},
                ]
            ):
                path = root / f"shard-{shard}.jsonl"
                path.write_text(json.dumps(record) + "\n")
            output = root / "merged"
            subprocess.check_call(
                [
                    "python3",
                    "-m",
                    "llm_rl.merge_eval_shards",
                    "--input",
                    str(root / "shard-*.jsonl"),
                    "--output",
                    str(output),
                ]
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["problems"], 2)
            self.assertEqual(summary["pass_at_1"], 0.5)


if __name__ == "__main__":
    unittest.main()
