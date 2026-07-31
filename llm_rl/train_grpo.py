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
from llm_rl.rewards import soft_overlong_penalty


class RewardState:
    def __init__(
        self,
        mode: str,
        curriculum_state: CurriculumState,
        state_output: Path,
        max_completion_length: int,
        overlong_buffer: int,
        overlong_penalty: float,
    ):
        self.mode = mode
        self.curriculum_state = curriculum_state
        self.state_output = state_output
        self.max_completion_length = max_completion_length
        self.overlong_buffer = overlong_buffer
        self.overlong_penalty = overlong_penalty

    def __call__(
        self,
        completions: list[Any],
        answer: list[Any],
        bucket: list[str],
        id: list[str],
        completion_ids: list[Any] | None = None,
        **_: Any,
    ) -> list[float]:
        rewards = []
        local_records = []
        completion_ids = completion_ids or [None] * len(completions)
        for completion, target, sample_bucket, sample_id, token_ids in zip(
            completions, answer, bucket, id, completion_ids
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
            if token_ids is not None and self.overlong_buffer > 0:
                reward -= soft_overlong_penalty(
                    len(token_ids),
                    self.max_completion_length,
                    self.overlong_buffer,
                    self.overlong_penalty,
                )
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
        for (sample_id, bucket), outcomes in sorted(groups.items()):
            zero_gradient = all(x == outcomes[0] for x in outcomes)
            self.curriculum_state.update(
                bucket, outcomes, zero_gradient, prompt_id=sample_id
            )
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
        "--dynamic-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Downweight prompts whose synchronized rollout groups have zero variance.",
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
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-high", type=float, default=None)
    parser.add_argument("--overlong-buffer", type=int, default=256)
    parser.add_argument("--overlong-penalty", type=float, default=1.0)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the model's explicit thinking trace. Disabled by default for bounded RL rollouts.",
    )
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
    if args.overlong_buffer < 0 or args.overlong_buffer >= args.max_completion_length:
        parser.error("--overlong-buffer must be in [0, max-completion-length)")
    epsilon_high = (
        args.epsilon_high
        if args.epsilon_high is not None
        else (0.28 if args.loss_type == "dapo" else args.epsilon)
    )

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
        chat_template_kwargs={"enable_thinking": args.enable_thinking},
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=0.8,
        top_p=0.95,
        beta=0.02,
        epsilon=args.epsilon,
        loss_type=loss_type,
        scale_rewards=scale_rewards,
        epsilon_high=epsilon_high,
        ignore_data_skip=bool(args.resume_from_checkpoint),
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
    reward_fn = RewardState(
        args.mode,
        curriculum_state,
        state_path,
        args.max_completion_length,
        args.overlong_buffer,
        args.overlong_penalty,
    )
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
        dynamic_sampling=args.dynamic_sampling,
        callbacks=[CurriculumCheckpointCallback(curriculum_state)],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(Path(args.output) / "final")
    if trainer.is_world_process_zero():
        curriculum_state.save(Path(args.output) / "final" / "curriculum_state.json")


if __name__ == "__main__":
    main()
