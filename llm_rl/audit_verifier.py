from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from llm_rl.math_verifier import verify_answer


def mutation_cases(answer: Any) -> list[tuple[str, bool]]:
    value = str(answer)
    cases = [
        (f"<answer>{value}</answer>", True),
        (f"Reasoning mentions 123 and {value}, but has no final marker.", False),
        ("<answer>__definitely_not_the_reference__</answer>", False),
        (f"<answer>{value}</answer><answer>__conflict__</answer>", False),
    ]
    return cases


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    failures = []
    by_source = defaultdict(Counter)
    negative_false_accepts = 0
    negative_cases = 0
    for row in rows:
        reference = row["answer"]
        positive = verify_answer(f"<answer>{reference}</answer>", reference)
        status_counts[positive.status.value] += 1
        by_source[str(row.get("source", "unknown"))][positive.status.value] += 1
        if not positive.correct:
            failures.append(
                {
                    "id": row.get("id"),
                    "source": row.get("source"),
                    "answer": reference,
                    "status": positive.status.value,
                    "detail": positive.detail,
                }
            )
        for completion, expected_correct in mutation_cases(reference)[1:]:
            result = verify_answer(completion, reference)
            negative_cases += 1
            if result.correct != expected_correct:
                negative_false_accepts += 1
                failures.append(
                    {
                        "id": row.get("id"),
                        "source": row.get("source"),
                        "mutation": completion,
                        "expected_correct": expected_correct,
                        "status": result.status.value,
                    }
                )
    return {
        "rows": len(rows),
        "reference_status_counts": dict(status_counts),
        "reference_coverage": status_counts["correct"] / len(rows) if rows else 0.0,
        "negative_cases": negative_cases,
        "negative_false_accepts": negative_false_accepts,
        "negative_false_accept_rate": (
            negative_false_accepts / negative_cases if negative_cases else 0.0
        ),
        "by_source": {key: dict(value) for key, value in sorted(by_source.items())},
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--output", default="results/verifier_audit.json")
    args = parser.parse_args()
    rows = []
    for path in args.data:
        rows.extend(json.loads(line) for line in open(path) if line.strip())
    report = audit(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}, indent=2))
    if report["failures"]:
        raise SystemExit(f"Verifier audit failed with {len(report['failures'])} failures")


if __name__ == "__main__":
    main()
