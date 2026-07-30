from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import sympy
from sympy import FiniteSet, Interval, Matrix
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
    "oo": sympy.oo,
    "sqrt": sympy.sqrt,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "cot": sympy.cot,
    "sec": sympy.sec,
    "csc": sympy.csc,
    "log": sympy.log,
    "ln": sympy.log,
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
    text = text.replace("^\\circ", "").replace("\\circ", "").replace("\\$", "")
    text = text.replace("$", "").replace("，", ",")
    text = re.sub(r"\\(?:text|mbox)\s*\{([^{}]*)\}", r"\1", text)
    text = _replace_latex_fractions(text)
    # Common malformed MATH-style fractions: \frac12, \frac 1 2, \frac1{2}.
    text = re.sub(r"\\(?:d?frac)\s*([\-+]?\d+)\s*\{([^{}]+)\}", r"((\1)/(\2))", text)
    text = re.sub(
        r"\\(?:d?frac)\s*\{([^{}]+)\}\s*([\-+]?\d+)", r"((\1)/(\2))", text
    )
    text = re.sub(
        r"\\(?:d?frac)\s*([\-+]?\d+)\s*([\-+]?\d+)", r"((\1)/(\2))", text
    )
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\sqrt\s*([0-9A-Za-z])", r"sqrt(\1)", text)
    text = text.replace("\\pi", "pi").replace("\\infty", "oo")
    text = re.sub(r"\\(sin|cos|tan|cot|sec|csc|log|ln)\b", r"\1", text)
    text = text.replace("\\pm", "±").replace("\\cup", "∪")
    if re.fullmatch(r"\s*[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*", text):
        text = text.replace(",", "")
    text = re.sub(
        r"\s*(?:degrees?|inches?|inch|cm|mm|meters?|cents?|dollars?)"
        r"(?:\s*\^\s*2)?\s*$",
        "",
        text,
        flags=re.I,
    )
    text = text.strip().rstrip(".。")

    # Common solution form: x = expression. Do not silently reduce arbitrary
    # equations, because equation equivalence requires a different contract.
    assignment = re.fullmatch(r"\s*[xyz]\s*=\s*(.+)", text, re.I)
    if assignment:
        text = assignment.group(1).strip()
    return text


def _strip_outer_braces(value: str) -> str:
    value = value.strip()
    if value.startswith(r"\{") and value.endswith(r"\}"):
        return value[2:-2].strip()
    if value.startswith("{") and value.endswith("}"):
        return value[1:-1].strip()
    return value


def _split_top_level(value: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = set(pairs.values())
    for index, character in enumerate(value):
        if character in pairs:
            depth += 1
        elif character in closers:
            depth -= 1
        elif character == delimiter and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return [x for x in parts if x]


def _parse_math(value: Any) -> sympy.Expr:
    normalized = normalize_answer(value)
    normalized = re.sub(
        r"\b(sin|cos|tan|cot|sec|csc|log|ln)\s+([A-Za-z])\b",
        r"\1(\2)",
        normalized,
    )
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


def _normalize_text(value: Any) -> str:
    text = normalize_answer(value)
    text = re.sub(r"^\(?\s*([A-E])\s*\)?$", r"\1", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _parse_base_number(value: str) -> int | None:
    match = re.fullmatch(r"\s*([0-9]+)\s*(?:_\s*\{?\s*([2-9])\s*\}?)\s*", value)
    if not match:
        return None
    return int(match.group(1), int(match.group(2)))


def _parse_matrix(value: str) -> Matrix | None:
    match = re.fullmatch(
        r"\\begin\{(?:p|b|v)?matrix\}(.*?)\\end\{(?:p|b|v)?matrix\}",
        value.strip(),
        re.S,
    )
    if not match:
        return None
    body = match.group(1)
    rows = [row.strip() for row in re.split(r"\\\\", body) if row.strip()]
    parsed_rows = []
    for row in rows:
        columns = [x.strip() for x in row.split("&")]
        parsed_rows.append([_parse_math(x) for x in columns])
    return Matrix(parsed_rows)


def _parse_interval(value: str) -> Interval | None:
    value = normalize_answer(value)
    if len(value) < 5 or value[0] not in "([" or value[-1] not in ")]":
        return None
    parts = _split_top_level(value[1:-1])
    if len(parts) != 2:
        return None
    left = _parse_math(parts[0])
    right = _parse_math(parts[1])
    if left == sympy.oo:
        left = -sympy.oo
    return Interval(
        left,
        right,
        left_open=value[0] == "(",
        right_open=value[-1] == ")",
    )


def _parse_collection(value: Any) -> Any:
    original = str(value).strip()
    matrix = _parse_matrix(original)
    if matrix is not None:
        return matrix
    normalized = normalize_answer(original)
    normalized = normalized.replace(r"\{", "{").replace(r"\}", "}")
    text_match = re.fullmatch(r"[A-Za-z]+", normalized)
    if text_match and len(normalized) > 1:
        return _normalize_text(normalized)
    choice_match = re.fullmatch(r"\(?\s*([A-E])\s*\)?", normalized, re.I)
    if choice_match:
        return choice_match.group(1).casefold()
    relation = re.fullmatch(r"\s*([A-Za-z])\s*\\in\s*(.+)", normalized)
    if relation:
        return (relation.group(1), _parse_collection(relation.group(2)))
    equation = re.fullmatch(r"\s*(.+?)\s*=\s*(.+)\s*", normalized)
    if equation:
        return sympy.Eq(_parse_math(equation.group(1)), _parse_math(equation.group(2)))
    base_number = _parse_base_number(normalized)
    if base_number is not None:
        return sympy.Integer(base_number)

    if "∪" in normalized:
        parts = [x.strip() for x in normalized.split("∪")]
        parsed = [_parse_collection(x) for x in parts]
        if all(isinstance(x, sympy.Set) for x in parsed):
            return sympy.Union(*parsed)

    if normalized[0:1] in "([" and normalized[-1:] in ")]":
        inner_parts = _split_top_level(normalized[1:-1])
        if len(inner_parts) == 2:
            interval = _parse_interval(normalized)
            if interval is not None:
                return interval
        elif len(inner_parts) > 2 and normalized[0] == "(" and normalized[-1] == ")":
            return sympy.Tuple(*[_parse_collection(x) for x in inner_parts])

    stripped = _strip_outer_braces(normalized)
    if stripped != normalized:
        values = []
        for item in _split_top_level(stripped):
            parsed_item = _parse_collection(item)
            if isinstance(parsed_item, FiniteSet):
                values.extend(list(parsed_item))
            else:
                values.append(parsed_item)
        return FiniteSet(*values)

    if "±" in normalized:
        left, right = normalized.split("±", 1)
        return FiniteSet(
            _parse_math(left + "+" + right), _parse_math(left + "-" + right)
        )

    parts = _split_top_level(normalized)
    if len(parts) > 1:
        return sympy.Tuple(*[_parse_collection(x) for x in parts])
    return _parse_math(normalized)


def answers_equivalent(candidate: Any, reference: Any) -> bool:
    if _normalize_text(candidate) == _normalize_text(reference):
        return True
    try:
        left = _parse_collection(candidate)
        right = _parse_collection(reference)
    except (ValueError, TypeError, SyntaxError, sympy.SympifyError):
        return _normalize_text(candidate) == _normalize_text(reference)
    if isinstance(left, str) or isinstance(right, str):
        return left == right
    if isinstance(left, tuple) or isinstance(right, tuple):
        return left == right
    if isinstance(left, Matrix) or isinstance(right, Matrix):
        return isinstance(left, Matrix) and isinstance(right, Matrix) and left == right
    if isinstance(left, (sympy.Set, sympy.Tuple)) or isinstance(
        right, (sympy.Set, sympy.Tuple)
    ):
        return left == right
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
            _parse_collection(candidate)
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
