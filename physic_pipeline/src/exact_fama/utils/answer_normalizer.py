from __future__ import annotations

import math
import re
from typing import Any

TRUE_ALIASES = {"true", "yes", "entailed", "correct", "1"}
FALSE_ALIASES = {"false", "no", "contradiction", "contradicted", "incorrect", "0"}
UNKNOWN_ALIASES = {
    "unknown",
    "uncertain",
    "undetermined",
    "cannot be determined",
    "can't be determined",
    "not enough information",
    "not enough info",
    "nei",
    "unanswerable",
}

SUPERSCRIPT_TRANSLATION = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
})

# Conservative unit stripping used only when a numeric answer accidentally has a
# trailing unit. Do not add broad words here; conceptual answers must remain text.
_UNIT_SUFFIX_RE = re.compile(
    r"(?i)"
    r"(v/m|n/c|ohms?|ω|volt(?:s)?|v|ampere(?:s)?|a|w|j|mj|μj|uj|"
    r"pf|nf|μf|uf|mf|f|nc|μc|uc|mc|c|hz|khz|mhz|mh|μh|uh|h|"
    r"tesla|t|newton(?:s)?|n|meter(?:s)?|metre(?:s)?|m)$"
)


def _clean_answer(value: Any) -> str:
    """Normalize non-numeric answers for exact/string comparison."""
    text = str(value or "").strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)

    m = re.fullmatch(r"(?:option\s*)?[\(\[]?([a-d])[\)\].:]?", text)
    if m:
        return m.group(1)

    return text


def _normalize_numeric_text(value: Any) -> str:
    """Normalize common scientific-notation spellings without throwing."""
    text = str(value or "").strip().lower()
    text = text.translate(SUPERSCRIPT_TRANSLATION)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("µ", "μ")
    text = text.replace(",", "")

    text = re.sub(r"\b(?:approximately|approx\.?|about)\b", "", text)
    text = text.replace("≈", "")

    text = text.replace("\\times", "*")
    text = text.replace("\\cdot", "*")
    text = text.replace("\\,", "")
    text = text.replace("{", "").replace("}", "")

    text = text.replace("×", "*").replace("·", "*")
    text = re.sub(r"(?<=\d)\s*x\s*(?=10\s*\^?)", "*", text, flags=re.I)
    # Some gold strings use "3.4 . 10^{-7}" for 3.4 × 10^-7.
    text = re.sub(r"(?<=\d)\s*\.\s*(?=10(?:\s*\^|\s*[-+]))", "*", text)

    text = text.replace("\\sqrt", "sqrt")
    text = re.sub(r"\s+", "", text)
    return text


def _safe_pow10(exp: int) -> float | None:
    """Return 10**exp as float, or None if it would overflow/underflow badly."""
    if exp > 308 or exp < -324:
        return None
    try:
        value = 10.0 ** exp
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _strip_simple_unit_suffix(text: str) -> str:
    return _UNIT_SUFFIX_RE.sub("", text)


def _try_float(value: Any) -> float | None:
    """Best-effort numeric parser. It must never raise."""
    try:
        text = _normalize_numeric_text(value)

        if text in {"zero", "approxzero", "approximatelyzero"}:
            return 0.0

        text = _strip_simple_unit_suffix(text)
        if not text:
            return None

        # Simple fractions, e.g. "1/4" or "10/3".
        m = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))/([+-]?(?:\d+(?:\.\d*)?|\.\d+))", text)
        if m:
            denom = float(m.group(2))
            if denom == 0:
                return None
            return float(m.group(1)) / denom

        # sqrt(2), sqrt2, 2sqrt(2), 2√2.
        m = re.fullmatch(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)?)?(?:sqrt\(?([+-]?(?:\d+(?:\.\d*)?|\.\d+))\)?|√([+-]?(?:\d+(?:\.\d*)?|\.\d+)))",
            text,
        )
        if m:
            coef_text = m.group(1)
            rad_text = m.group(2) or m.group(3)
            if coef_text in (None, "", "+"):
                coef = 1.0
            elif coef_text == "-":
                coef = -1.0
            else:
                coef = float(coef_text)
            rad = float(rad_text)
            if rad < 0:
                return None
            return coef * math.sqrt(rad)

        # Plain integer/decimal/e-notation must be parsed before school-style
        # a*10^b forms.  Otherwise values like 10000 or 0.109 can be
        # misread as 1000*10^0 or 0*10^9.
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
            out = float(text)
            if math.isfinite(out):
                return out
            return None

        # Scientific notation: a*10^b, a10^b, a*10b after normalization.
        # The no-star form is kept only for explicit signed exponents so plain
        # integers cannot be split at their last '10'.
        m = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\*10\^?([+-]?\d+)", text)
        if m:
            exp = int(m.group(2))
            pow10 = _safe_pow10(exp)
            if pow10 is None:
                return None
            return float(m.group(1)) * pow10

        # Scientific notation without coefficient: 10^b or -10^b.
        m = re.fullmatch(r"([+-]?)10\^?([+-]?\d+)", text)
        if m:
            exp = int(m.group(2))
            pow10 = _safe_pow10(exp)
            if pow10 is None:
                return None
            sign = -1.0 if m.group(1) == "-" else 1.0
            return sign * pow10

        if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
            return None

        out = float(text)
        if not math.isfinite(out):
            return None
        return out
    except Exception:
        return None


def canonical_answer(value: Any) -> str:
    text = _clean_answer(value)

    if text in TRUE_ALIASES:
        return "true"
    if text in FALSE_ALIASES:
        return "false"
    if text in UNKNOWN_ALIASES:
        return "unknown"

    num = _try_float(value)
    if num is not None:
        if abs(num) < 1e-15:
            return "0"
        return f"{num:.12g}"

    return text


def answers_match(gold: Any, pred: Any, numeric_tolerance: float = 1e-6) -> bool:
    gold_num = _try_float(gold)
    pred_num = _try_float(pred)

    if gold_num is not None and pred_num is not None:
        return math.isclose(
            gold_num,
            pred_num,
            rel_tol=max(numeric_tolerance, 5e-4),
            abs_tol=max(numeric_tolerance, 5e-6),
        )

    return canonical_answer(gold) == canonical_answer(pred)
