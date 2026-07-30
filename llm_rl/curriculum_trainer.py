from __future__ import annotations

from typing import Any

from trl import GRPOTrainer

from llm_rl.curriculum import CurriculumRepeatSampler, CurriculumState


class CurriculumGRPOTrainer(GRPOTrainer):
    """GRPO trainer whose prompt sampler reads synchronized competence state."""

    def __init__(
        self,
        *args: Any,
        curriculum_state: CurriculumState,
        curriculum_mode: str,
        **kwargs: Any,
    ):
        self.curriculum_state = curriculum_state
        self.curriculum_mode = curriculum_mode
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, dataset=None):
        if dataset is None:
            dataset = self.train_dataset
        buckets = [str(x) for x in dataset["bucket"]]
        return CurriculumRepeatSampler(
            data_source=dataset,
            bucket_by_index=buckets,
            state=self.curriculum_state,
            mode=self.curriculum_mode,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            seed=self.args.seed,
        )
