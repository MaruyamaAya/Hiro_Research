from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from llm_rl.curriculum import CurriculumState


class CurriculumCheckpointCallback(TrainerCallback):
    def __init__(self, curriculum_state: CurriculumState):
        self.curriculum_state = curriculum_state

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any):
        path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self.curriculum_state.save(path / "curriculum_state.json")
        return control
