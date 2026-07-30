from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


SYSTEM_PROMPT = (
    "Solve the problem carefully. End with exactly one explicit final answer "
    "using <answer>YOUR_ANSWER</answer>."
)


def canonical_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def text_hash(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode()).hexdigest()


def heuristic_difficulty_bucket(problem: str, answer: str) -> str:
    """Deterministic proxy buckets until source-native difficulty is available."""

    canonical = canonical_text(problem)
    words = len(canonical.split())
    latex_ops = len(
        re.findall(
            r"\\(?:frac|sqrt|sum|prod|int|log|sin|cos|tan|binom|begin)|"
            r"\^|[{}]",
            problem,
        )
    )
    structural = len(
        re.findall(
            r"\b(?:triangle|polynomial|probability|integer|function|sequence|"
            r"geometry|circle|prime|divisible|matrix|complex)\b",
            canonical,
        )
    )
    answer_complexity = min(10, len(canonical_text(answer).split()))
    score = words / 30 + latex_ops / 6 + structural / 2 + answer_complexity / 5
    if score < 2.0:
        return "proxy_1"
    if score < 3.5:
        return "proxy_2"
    if score < 5.5:
        return "proxy_3"
    if score < 8.0:
        return "proxy_4"
    return "proxy_5"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            line = (json.dumps(row, ensure_ascii=False) + "\n").encode()
            handle.write(line)
            digest.update(line)
            count += 1
    return count, digest.hexdigest()


def load_math500(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.open():
        raw = json.loads(line)
        rows.append(
            {
                "id": f"math500:{raw['unique_id']}",
                "source": "math500",
                "difficulty": int(raw["level"]),
                "subject": raw["subject"],
                "problem": raw["problem"],
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": raw["problem"]},
                ],
                "answer": raw["answer"],
                "split": "eval",
            }
        )
    return rows


def load_gsm8k(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    rows = []
    for index, raw in enumerate(table.to_pylist()):
        match = re.search(r"####\s*(.+?)\s*$", raw["answer"])
        if not match:
            raise ValueError(f"GSM8K row {index} has no #### answer")
        answer = match.group(1).replace(",", "").strip()
        rows.append(
            {
                "id": f"gsm8k:test:{index}",
                "source": "gsm8k",
                "difficulty": "gsm8k",
                "problem": raw["question"],
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": raw["question"]},
                ],
                "answer": answer,
                "split": "eval",
            }
        )
    return rows


def load_dapo(path: Path, held_out_hashes: set[str]) -> tuple[list[dict[str, Any]], Counter]:
    rows = []
    stats: Counter = Counter()
    seen: set[str] = set()
    for line in path.open():
        raw = json.loads(line)
        user_messages = [x["content"] for x in raw["prompt"] if x["role"] == "user"]
        if len(user_messages) != 1:
            stats["invalid_prompt"] += 1
            continue
        problem = user_messages[0]
        problem_hash = text_hash(problem)
        if problem_hash in held_out_hashes:
            stats["exact_eval_overlap"] += 1
            continue
        if problem_hash in seen:
            stats["duplicate_train_problem"] += 1
            continue
        seen.add(problem_hash)
        answer = str(raw["reward_model"]["ground_truth"]).strip()
        if not answer:
            stats["empty_answer"] += 1
            continue
        index = str(raw["extra_info"]["index"])
        rows.append(
            {
                "id": f"dapo:{index}",
                "source": "dapo_math_17k",
                "difficulty": heuristic_difficulty_bucket(problem, answer),
                "ability": raw.get("ability", "MATH"),
                "problem": problem,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": problem},
                ],
                "answer": answer,
                "split": "train",
            }
        )
    stats["kept"] = len(rows)
    return rows, stats


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapo", default="data/raw/dapo-math-17k-api.jsonl")
    parser.add_argument("--math500", default="data/raw/math-500-test.jsonl")
    parser.add_argument("--gsm8k", default="data/raw/gsm8k-test.parquet")
    parser.add_argument("--train-output", default="data/real_math_train.jsonl")
    parser.add_argument("--eval-output", default="data/real_math_eval.jsonl")
    parser.add_argument("--manifest", default="data/manifests/real_math_manifest.json")
    args = parser.parse_args()

    dapo_path = Path(args.dapo)
    math500_path = Path(args.math500)
    gsm8k_path = Path(args.gsm8k)
    math500 = load_math500(math500_path)
    gsm8k = load_gsm8k(gsm8k_path)
    eval_rows = math500 + gsm8k
    held_out_hashes = {text_hash(row["problem"]) for row in eval_rows}
    train_rows, filter_stats = load_dapo(dapo_path, held_out_hashes)

    train_count, train_sha = write_jsonl(Path(args.train_output), train_rows)
    eval_count, eval_sha = write_jsonl(Path(args.eval_output), eval_rows)
    manifest = {
        "schema_version": 1,
        "sources": {
            "dapo_math_17k": {
                "repository": "BytedTsinghua-SIA/DAPO-Math-17k",
                "revision": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
                "license": "apache-2.0",
                "original_split": "train",
                "input_path": str(dapo_path),
                "input_sha256": sha256_file(dapo_path),
                "selection": "first 17,000 rows from datasets-server rows API",
            },
            "math500": {
                "repository": "HuggingFaceH4/MATH-500",
                "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
                "original_split": "test",
                "input_path": str(math500_path),
                "input_sha256": sha256_file(math500_path),
            },
            "gsm8k": {
                "repository": "openai/gsm8k",
                "revision": "740312add88f781978c0658806c59bc2815b9866",
                "license": "mit",
                "original_split": "test",
                "input_path": str(gsm8k_path),
                "input_sha256": sha256_file(gsm8k_path),
            },
        },
        "processing": {
            "exact_decontamination": (
                "SHA-256 over lowercased alphanumeric canonical problem text"
            ),
            "difficulty_bucket": (
                "Five deterministic proxy buckets from prompt length, LaTeX "
                "operator count, structural math terms, and answer complexity"
            ),
            "filter_counts": dict(filter_stats),
            "train_bucket_counts": dict(
                sorted(Counter(x["difficulty"] for x in train_rows).items())
            ),
        },
        "outputs": {
            "train": {
                "path": args.train_output,
                "rows": train_count,
                "sha256": train_sha,
            },
            "eval": {
                "path": args.eval_output,
                "rows": eval_count,
                "sha256": eval_sha,
                "source_counts": dict(Counter(x["source"] for x in eval_rows)),
            },
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest["outputs"], indent=2))
    print(json.dumps(manifest["processing"], indent=2))


if __name__ == "__main__":
    main()
