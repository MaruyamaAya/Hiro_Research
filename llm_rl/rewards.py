from __future__ import annotations


def soft_overlong_penalty(
    completion_tokens: int,
    max_completion_length: int,
    buffer_length: int,
    penalty_factor: float,
) -> float:
    if buffer_length <= 0:
        return 0.0
    start = max_completion_length - buffer_length
    overflow_fraction = max(0.0, (completion_tokens - start) / buffer_length)
    return penalty_factor * min(1.0, overflow_fraction)
