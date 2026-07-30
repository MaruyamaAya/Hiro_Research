from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText
from trl import GRPOConfig

from llm_rl.curriculum import CurriculumState
from llm_rl.curriculum_callback import CurriculumCheckpointCallback
from llm_rl.curriculum_trainer import CurriculumGRPOTrainer
from llm_rl.math_verifier import verify_answer


class RewardState:
    def __init__(
        self,
        mode: str,
        curriculum_state: CurriculumState,
        state_output: Path,
    ):
        self.mode = mode
        self.curriculum_state = curriculum_state
        self.state_output = state_output

    def __call__(
        self,
        completions: list[Any],
        answer: list[Any],
        bucket: list[str],
        id: list[str],
        **_: Any,
    ) -> list[float]:
        rewards = []
        local_records = []
        for completion, target, sample_bucket, sample_id in zip(
            completions, answer, bucket, id
        ):
            text = (
                completion[-1]["content"]
                if isinstance(completion, list)
                else str(completion)
            )
            verification = verify_answer(text, target)
            correct = float(verification.correct)
            valid = float(verification.valid)
            if self.mode == "outcome":
                reward = correct + 0.05 * valid
            elif self.mode == "effort":
                reward = correct + 0.002 * min(len(text), 800)
            else:
                raise ValueError(self.mode)
            rewards.append(float(reward))
            local_records.append((str(sample_id), str(sample_bucket), correct))
        self._synchronized_update(local_records)
        return rewards

    def _synchronized_update(self, local_records: list[tuple[str, str, float]]) -> None:
        gathered = [local_records]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local_records)
        groups: dict[tuple[str, str], list[float]] = {}
        for rank_records in gathered:
            for sample_id, bucket, correct in rank_records:
                groups.setdefault((sample_id, bucket), []).append(correct)
        for (_, bucket), outcomes in sorted(groups.items()):
            zero_gradient = all(x == outcomes[0] for x in outcomes)
            self.curriculum_state.update(bucket, outcomes, zero_gradient)
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            self.curriculum_state.save(self.state_output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["outcome", "effort"], default="outcome")
    parser.add_argument(
        "--curriculum",
        choices=["uniform", "challenge", "progress", "hiro"],
        default="uniform",
    )
    parser.add_argument(
        "--loss-type",
        choices=["grpo", "dr_grpo", "dapo"],
        default="dapo",
    )
    parser.add_argument("--data", default="data/math_curriculum.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=96)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    model_path = args.model or os.environ["TAIJI_BASIC_MODEL_PATH"]
    rows = [json.loads(x) for x in open(args.data) if x.strip()]
    train_rows = [r for r in rows if r["split"] == "train"]
    for row in train_rows:
        row["id"] = str(row["id"])
        row["bucket"] = str(row.get("bucket", row.get("difficulty", "unknown")))
    dataset = Dataset.from_list(train_rows)
    state_path = Path(args.output) / "curriculum_state.json"
    resume_state_path = (
        Path(args.resume_from_checkpoint) / "curriculum_state.json"
        if args.resume_from_checkpoint
        else None
    )
    if resume_state_path and resume_state_path.exists():
        curriculum_state = CurriculumState.load(resume_state_path)
    else:
        curriculum_state = CurriculumState(row["bucket"] for row in train_rows)

    loss_type = {
        "grpo": "grpo",
        "dr_grpo": "dr_grpo",
        "dapo": "dapo",
    }[args.loss_type]
    scale_rewards = args.loss_type != "dr_grpo"

    config = GRPOConfig(
        output_dir=args.output,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=5e-6,
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=4,
        save_only_model=False,
        report_to="none",
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=0.8,
        top_p=0.95,
        beta=0.02,
        loss_type=loss_type,
        scale_rewards=scale_rewards,
        epsilon_high=0.28 if args.loss_type == "dapo" else None,
        seed=args.seed,
        model_init_kwargs={
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
        },
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    reward_fn = RewardState(args.mode, curriculum_state, state_path)
    reward_fn.__name__ = f"{args.mode}_reward"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    # Transformers 5.x moved this bookkeeping field away from some multimodal
    # architectures, while current TRL still expects it.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    trainer = CurriculumGRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset,
        peft_config=lora,
        curriculum_state=curriculum_state,
        curriculum_mode=args.curriculum,
        callbacks=[CurriculumCheckpointCallback(curriculum_state)],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(Path(args.output) / "final")
    curriculum_state.save(Path(args.output) / "final" / "curriculum_state.json")


if __name__ == "__main__":
    main()
