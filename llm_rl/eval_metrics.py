from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one-or-more sampled completions per evaluation problem."""

    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_problem[str(record["id"])].append(record)
    for samples in by_problem.values():
        samples.sort(key=lambda x: int(x["sample_index"]))

    first_samples = [samples[0] for samples in by_problem.values()]
    num_samples = max((len(samples) for samples in by_problem.values()), default=0)
    pass_at_k = {}
    for k in range(1, num_samples + 1):
        pass_at_k[str(k)] = _mean(
            any(bool(sample["correct"]) for sample in samples[:k])
            for samples in by_problem.values()
        )

    def slice_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
        statuses = Counter(str(sample["status"]) for sample in samples)
        return {
            "problems": len(samples),
            "pass_at_1": _mean(bool(sample["correct"]) for sample in samples),
            "format_valid_rate": _mean(bool(sample["valid"]) for sample in samples),
            "truncation_rate": _mean(bool(sample["truncated"]) for sample in samples),
            "mean_completion_tokens": _mean(
                float(sample["completion_tokens"]) for sample in samples
            ),
            "status_counts": dict(sorted(statuses.items())),
        }

    by_difficulty: dict[str, Any] = {}
    difficulties = sorted({str(x.get("difficulty", "unknown")) for x in first_samples})
    for difficulty in difficulties:
        by_difficulty[difficulty] = slice_summary(
            [x for x in first_samples if str(x.get("difficulty", "unknown")) == difficulty]
        )

    by_source: dict[str, Any] = {}
    sources = sorted({str(x.get("source", "unknown")) for x in first_samples})
    for source in sources:
        by_source[source] = slice_summary(
            [x for x in first_samples if str(x.get("source", "unknown")) == source]
        )

    summary = slice_summary(first_samples)
    summary.update(
        {
            "generated_completions": len(records),
            "samples_per_problem": num_samples,
            "pass_at_k": pass_at_k,
            "by_difficulty": by_difficulty,
            "by_source": by_source,
        }
    )
    return summary
