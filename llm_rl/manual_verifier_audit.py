from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def deterministic_sample(records: list[dict[str, Any]], per_stratum: int, seed: int) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        strata[(str(record.get("source", "unknown")), str(record.get("status", "unknown")))].append(record)
    rng = random.Random(seed)
    sampled = []
    for stratum in sorted(strata):
        values = list(strata[stratum])
        rng.shuffle(values)
        sampled.extend(values[:per_stratum])
    return sampled


def create_annotation_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        audit_id = hashlib.sha256(
            f"{record.get('id')}:{record.get('sample_index', 0)}:{record.get('completion', '')}".encode()
        ).hexdigest()[:16]
        output.append(
            {
                "audit_id": audit_id,
                "problem_id": record.get("id"),
                "source": record.get("source"),
                "reference_answer": record.get("reference_answer"),
                "completion": record.get("completion"),
                "verifier_status_hidden": record.get("status"),
                "human_label": "",
                "annotator": "",
                "notes": "",
            }
        )
    return output


def score_annotations(rows: list[dict[str, str]]) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("human_label") in {"correct", "incorrect", "ambiguous"}]
    if not labeled:
        raise ValueError("No valid human labels")
    agreements = 0
    false_accepts = 0
    false_rejects = 0
    for row in labeled:
        verifier_correct = row["verifier_status_hidden"] == "correct"
        human_correct = row["human_label"] == "correct"
        agreements += int(verifier_correct == human_correct)
        false_accepts += int(verifier_correct and not human_correct)
        false_rejects += int(not verifier_correct and human_correct)
    return {
        "labeled": len(labeled),
        "binary_agreement": agreements / len(labeled),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--predictions", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--per-stratum", type=int, default=25)
    create.add_argument("--seed", type=int, default=20260730)
    score = subparsers.add_parser("score")
    score.add_argument("--annotations", required=True)
    score.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "create":
        records = [json.loads(line) for line in open(args.predictions) if line.strip()]
        rows = create_annotation_rows(deterministic_sample(records, args.per_stratum, args.seed))
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} blinded audit rows to {args.output}")
    else:
        with open(args.annotations, newline="") as handle:
            rows = list(csv.DictReader(handle))
        report = score_annotations(rows)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
