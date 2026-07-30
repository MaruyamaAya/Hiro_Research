from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch

from llm_rl.eval_metrics import summarize_predictions
from llm_rl.math_verifier import verify_answer


def checkpoint_label(path: str | None) -> str:
    if path is None:
        return "base"
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path).name)
    digest = hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:8]
    return f"{clean}-{digest}"


def discover_checkpoints(explicit: list[str], roots: list[str]) -> list[str | None]:
    checkpoints: list[str | None] = list(explicit)
    for root_value in roots:
        root = Path(root_value)
        candidates = list(root.glob("checkpoint-*"))
        candidates.sort(
            key=lambda p: (
                0,
                int(p.name.split("-")[-1]),
            )
            if p.name.split("-")[-1].isdigit()
            else (1, p.name)
        )
        checkpoints.extend(str(x) for x in candidates if x.is_dir())
        final = root / "final"
        if final.is_dir():
            checkpoints.append(str(final))
    if not checkpoints:
        return [None]
    # Stable de-duplication matters when both explicit and root discovery are used.
    return list(dict.fromkeys(checkpoints))


def load_model(
    model_path: str,
    checkpoint: str | None,
    device: str,
) -> tuple[Any, Any]:
    # Keep heavyweight remote-environment dependencies lazy so metric and
    # checkpoint-discovery utilities remain usable on CPU development hosts.
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32,
        "attn_implementation": "sdpa",
    }
    if device.startswith("cuda"):
        kwargs["device_map"] = device
    model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    if checkpoint is not None:
        model = PeftModel.from_pretrained(
            model,
            checkpoint,
            local_files_only=True,
            is_trainable=False,
        )
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return processor, model


def generate_one(
    processor: Any,
    model: Any,
    prompt: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[str, int, bool]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inputs = processor.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs.update(temperature=temperature, top_p=top_p)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    input_length = inputs["input_ids"].shape[1]
    completion_ids = output_ids[0, input_length:]
    eos_id = processor.tokenizer.eos_token_id
    truncated = (
        len(completion_ids) >= max_new_tokens
        and (eos_id is None or int(completion_ids[-1]) != int(eos_id))
    )
    text = processor.tokenizer.decode(completion_ids, skip_special_tokens=True)
    return text, int(len(completion_ids)), bool(truncated)


def evaluate_checkpoint(
    model_path: str,
    checkpoint: str | None,
    rows: list[dict[str, Any]],
    output_root: Path,
    samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    label = checkpoint_label(checkpoint)
    output_dir = output_root / label
    output_dir.mkdir(parents=True, exist_ok=True)
    processor, model = load_model(model_path, checkpoint, device)
    records: list[dict[str, Any]] = []
    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("w") as handle:
        for row_index, row in enumerate(rows):
            for sample_index in range(samples):
                sample_seed = seed + row_index * 100_003 + sample_index
                text, length, truncated = generate_one(
                    processor,
                    model,
                    row["prompt"],
                    max_new_tokens,
                    temperature,
                    top_p,
                    sample_seed,
                )
                verification = verify_answer(text, row["answer"])
                record = {
                    "id": row["id"],
                    "source": row.get("source", "unknown"),
                    "difficulty": row.get("difficulty", "unknown"),
                    "sample_index": sample_index,
                    "seed": sample_seed,
                    "reference_answer": row["answer"],
                    "completion": text,
                    "completion_tokens": length,
                    "truncated": truncated,
                    "status": verification.status.value,
                    "correct": verification.correct,
                    "valid": verification.valid,
                    "extracted_answer": verification.candidate,
                    "answer_source": verification.source,
                }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{label}] {row_index + 1}/{len(rows)}",
                flush=True,
            )

    summary = summarize_predictions(records)
    summary.update(
        {
            "model": model_path,
            "checkpoint": checkpoint,
            "checkpoint_label": label,
            "data_problems": len(rows),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "base_seed": seed,
        }
    )
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the base model and/or LoRA checkpoints on held-out math."
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-root", action="append", default=[])
    parser.add_argument("--data", default="data/math_curriculum.jsonl")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--output", default="results/math_eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.samples > 1 and args.temperature <= 0:
        parser.error("--samples > 1 requires --temperature > 0")

    model_path = args.model or os.environ["TAIJI_BASIC_MODEL_PATH"]
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    rows = [row for row in rows if row.get("split") == args.split]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        parser.error(f"No rows found for split {args.split!r} in {args.data}")

    checkpoints = discover_checkpoints(args.checkpoint, args.checkpoint_root)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for checkpoint in checkpoints:
        all_summaries.append(
            evaluate_checkpoint(
                model_path=model_path,
                checkpoint=checkpoint,
                rows=rows,
                output_root=output_root,
                samples=args.samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
                device=args.device,
            )
        )
    with (output_root / "all_summaries.json").open("w") as handle:
        json.dump(all_summaries, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
