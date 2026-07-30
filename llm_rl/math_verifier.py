from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


class VerificationStatus(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    NO_ANSWER = "no_answer"
    INVALID = "invalid"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    candidate: str | None = None
    normalized_candidate: str | None = None
    normalized_reference: str | None = None
    source: str | None = None
    detail: str | None = None

    @property
    def correct(self) -> bool:
        return self.status is VerificationStatus.CORRECT

    @property
    def valid(self) -> bool:
        return self.status in {
            VerificationStatus.CORRECT,
            VerificationStatus.INCORRECT,
        }


_ANSWER_TAG_RE = re.compile(r"<answer\b[^>]*>(.*?)</answer\s*>", re.I | re.S)
_FINAL_RE = re.compile(
    r"(?:final\s+answer|answer\s+is|therefore\s*,?\s*the\s+answer\s+is)"
    r"\s*(?::|=)?\s*([^\n]+)",
    re.I,
)
_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)
_LOCAL_DICT = {
    "pi": sympy.pi,
    "e": sympy.E,
    "i": sympy.I,
    "sqrt": sympy.sqrt,
}
_GLOBAL_DICT = {
    "__builtins__": {},
    "Integer": sympy.Integer,
    "Rational": sympy.Rational,
    "Float": sympy.Float,
    "Symbol": sympy.Symbol,
}


def _balanced_boxed_values(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            values.append(text[start : i - 1])
    return values


def extract_answer_candidates(text: str) -> tuple[list[str], str | None]:
    """Extract only explicit final-answer forms, in decreasing priority.

    Restricting extraction to explicit forms prevents a verifier from accepting
    numbers copied from the prompt or intermediate calculations.
    """

    tagged = [x.strip() for x in _ANSWER_TAG_RE.findall(text) if x.strip()]
    if tagged:
        return tagged, "answer_tag"

    boxed = [x.strip() for x in _balanced_boxed_values(text) if x.strip()]
    if boxed:
        return boxed, "boxed"

    finals = []
    for value in _FINAL_RE.findall(text):
        value = value.strip().rstrip(".。")
        if value:
            finals.append(value)
    if finals:
        return finals, "final_phrase"
    return [], None


def _replace_latex_fractions(value: str) -> str:
    # Repeatedly handle innermost simple \frac{a}{b} forms. This covers the
    # common benchmark answers without evaluating arbitrary LaTeX.
    pattern = re.compile(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    while previous != value:
        previous = value
        value = pattern.sub(r"((\1)/(\2))", value)
    return value


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    text = text.replace("−", "-").replace("–", "-").replace("×", "*").replace("÷", "/")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    text = text.replace("\\circ", "")
    text = text.replace("$", "").replace("，", ",")
    text = _replace_latex_fractions(text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = text.strip().rstrip(".。")

    # Common solution form: x = expression. Do not silently reduce arbitrary
    # equations, because equation equivalence requires a different contract.
    assignment = re.fullmatch(r"\s*[xyz]\s*=\s*(.+)", text, re.I)
    if assignment:
        text = assignment.group(1).strip()
    return text


def _parse_math(value: Any) -> sympy.Expr:
    normalized = normalize_answer(value)
    if not normalized or len(normalized) > 256:
        raise ValueError("empty or overlong answer")
    if re.search(r"[^0-9a-zA-Z_+\-*/^().\s]", normalized):
        raise ValueError("unsupported characters")
    identifiers = set(re.findall(r"[A-Za-z_]+", normalized))
    if "_" in identifiers:
        raise ValueError("unsupported identifiers: ['_']")
    local_dict = dict(_LOCAL_DICT)
    for identifier in identifiers - set(local_dict):
        if len(identifier) != 1:
            raise ValueError(f"unsupported identifiers: {sorted(identifiers)}")
        local_dict[identifier] = sympy.Symbol(identifier)
    expression = parse_expr(
        normalized,
        local_dict=local_dict,
        global_dict=_GLOBAL_DICT,
        transformations=_TRANSFORMATIONS,
        evaluate=True,
    )
    if not isinstance(expression, sympy.Expr):
        raise ValueError("not a mathematical expression")
    return expression


def answers_equivalent(candidate: Any, reference: Any) -> bool:
    left = _parse_math(candidate)
    right = _parse_math(reference)
    difference = sympy.simplify(left - right)
    if difference == 0:
        return True
    if not difference.free_symbols:
        try:
            return abs(float(sympy.N(difference, 16))) <= 1e-9
        except (TypeError, ValueError):
            return False
    return False


def verify_answer(completion: str, reference: Any) -> VerificationResult:
    candidates, source = extract_answer_candidates(completion)
    if not candidates:
        return VerificationResult(
            VerificationStatus.NO_ANSWER,
            detail="No explicit <answer>, boxed, or final-answer phrase found.",
        )

    parsed: list[str] = []
    try:
        for candidate in candidates:
            _parse_math(candidate)
            parsed.append(candidate)
    except (ValueError, TypeError, SyntaxError, sympy.SympifyError) as exc:
        return VerificationResult(
            VerificationStatus.INVALID,
            candidate=candidates[-1],
            source=source,
            detail=str(exc),
        )

    first = parsed[0]
    try:
        if any(not answers_equivalent(first, other) for other in parsed[1:]):
            return VerificationResult(
                VerificationStatus.AMBIGUOUS,
                candidate=" | ".join(parsed),
                source=source,
                detail="Conflicting explicit answers.",
            )
        correct = answers_equivalent(first, reference)
        return VerificationResult(
            VerificationStatus.CORRECT if correct else VerificationStatus.INCORRECT,
            candidate=first,
            normalized_candidate=normalize_answer(first),
            normalized_reference=normalize_answer(reference),
            source=source,
        )
    except (ValueError, TypeError, SyntaxError, sympy.SympifyError) as exc:
        return VerificationResult(
            VerificationStatus.INVALID,
            candidate=first,
            source=source,
            detail=str(exc),
        )
