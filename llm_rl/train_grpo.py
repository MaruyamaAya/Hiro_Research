from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText
from trl import GRPOConfig, GRPOTrainer


ANSWER_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>", re.I)


class RewardState:
    def __init__(self, mode: str):
        self.mode = mode
        self.history: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=128))
        self.previous: dict[int, float] = defaultdict(lambda: 0.5)

    def __call__(
        self,
        completions: list[Any],
        answer: list[int],
        difficulty: list[int],
        **_: Any,
    ) -> list[float]:
        rewards = []
        grouped_correct: dict[int, list[float]] = defaultdict(list)
        for completion, target, d in zip(completions, answer, difficulty):
            text = (
                completion[-1]["content"]
                if isinstance(completion, list)
                else str(completion)
            )
            match = ANSWER_RE.search(text)
            correct = float(bool(match) and int(match.group(1)) == int(target))
            valid = float(bool(match))
            recent = self.history[int(d)]
            ability = float(np.mean(recent)) if recent else 0.5
            challenge = math.exp(-((ability - 0.45) ** 2) / (2 * 0.20**2))
            progress = max(0.0, ability - self.previous[int(d)])
            if self.mode == "outcome":
                reward = correct + 0.05 * valid
            elif self.mode == "difficulty":
                reward = correct + 0.15 * valid + 0.12 * int(d)
            elif self.mode == "effort":
                reward = correct + 0.002 * min(len(text), 800)
            elif self.mode == "hiro":
                reward = (
                    0.70 * correct
                    + 0.08 * valid
                    + 0.25 * challenge
                    + 3.0 * progress
                    - 0.15 * float(len(text) > 700)
                )
            elif self.mode == "hiro_no_progress":
                reward = 0.70 * correct + 0.08 * valid + 0.25 * challenge
            else:
                raise ValueError(self.mode)
            rewards.append(float(reward))
            grouped_correct[int(d)].append(correct)
        # Update after scoring the group to avoid within-group leakage.
        for d, values in grouped_correct.items():
            old = float(np.mean(self.history[d])) if self.history[d] else 0.5
            self.previous[d] = old
            self.history[d].extend(values)
        return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["outcome", "difficulty", "effort", "hiro", "hiro_no_progress"], required=True)
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
    dataset = Dataset.from_list(train_rows)

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
        loss_type="dapo",
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
    reward_fn = RewardState(args.mode)
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
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset,
        peft_config=lora,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(Path(args.output) / "final")


if __name__ == "__main__":
    main()
