from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from llm_rl.curriculum import CurriculumState


class CurriculumCallbackTest(unittest.TestCase):
    def test_only_world_zero_writes_checkpoint_state(self) -> None:
        try:
            from llm_rl.curriculum_callback import CurriculumCheckpointCallback
        except ModuleNotFoundError:
            self.skipTest("transformers is only installed in the remote training env")
        curriculum = CurriculumState(["a"])
        callback = CurriculumCheckpointCallback(curriculum)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=directory)
            control = object()
            callback.on_save(
                args,
                SimpleNamespace(global_step=1, is_world_process_zero=False),
                control,
            )
            self.assertFalse(
                (Path(directory) / "checkpoint-1/curriculum_state.json").exists()
            )
            callback.on_save(
                args,
                SimpleNamespace(global_step=1, is_world_process_zero=True),
                control,
            )
            self.assertTrue(
                (Path(directory) / "checkpoint-1/curriculum_state.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
