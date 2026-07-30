from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


SYSTEM = """You judge task choices using this principle:
Prefer effortful, appropriately difficult experiences when effort can become
learning or useful growth. Do not reward random noise, stagnant repetition,
meaningless effort, or irreversible danger."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default="data/preference_pairs.jsonl")
    parser.add_argument("--output", default="results/qwen_zero_shot/predictions.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    model_path = args.model or __import__("os").environ["TAIJI_BASIC_MODEL_PATH"]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    rows = [json.loads(x) for x in open(args.data) if x.strip()]
    rows = [r for r in rows if r["split"] == "test"][: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    with output.open("w") as f:
        for i, row in enumerate(rows, 1):
            prompt = (
                f"Option A: {row['a']}\n\nOption B: {row['b']}\n\n"
                "Which is preferable? Give a short reason, then end with exactly "
                "'Final choice: A' or 'Final choice: B'."
            )
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda:0")
            # Score complete answer suffixes, rather than bare letters, because
            # Qwen3.5 is a reasoning model and bare next-token probabilities
            # have a strong format prior unrelated to the pair's semantics.
            candidate_scores = {}
            with torch.inference_mode():
                for choice in ("A", "B"):
                    candidate = (
                        f"The option better matches productive difficulty and avoids "
                        f"unproductive failure modes. Final choice: {choice}"
                    )
                    candidate_ids = processor.tokenizer(
                        candidate, add_special_tokens=False, return_tensors="pt"
                    )["input_ids"].to("cuda:0")
                    full_ids = torch.cat([inputs["input_ids"], candidate_ids], dim=1)
                    full_mask = torch.ones_like(full_ids)
                    logits = model(input_ids=full_ids, attention_mask=full_mask).logits
                    start = inputs["input_ids"].shape[1] - 1
                    token_logits = logits[:, start : start + candidate_ids.shape[1], :]
                    log_probs = token_logits.log_softmax(dim=-1)
                    score = log_probs.gather(
                        -1, candidate_ids.unsqueeze(-1)
                    ).mean().item()
                    candidate_scores[choice] = score
            pred = max(candidate_scores, key=candidate_scores.get)
            correct += int(pred == row["label"])
            record = {**row, "prediction": pred, "scores": candidate_scores}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{i}/{len(rows)} accuracy={correct / i:.3f}", flush=True)


if __name__ == "__main__":
    main()
