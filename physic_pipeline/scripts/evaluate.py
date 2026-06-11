#!/usr/bin/env python
from __future__ import annotations

# Robust evaluator for EXACT-FAMA predictions.
# Goals:
# - deterministic numeric comparison with relative tolerance
# - no dangerous substring matching ("5" != "5000")
# - school/Vietnamese scientific notation support ("5.2323.10^9", "3 × 10^-7")
# - variable labels ignored for numeric lists ("I1 = 0.5; I2 = 1.0" -> [0.5, 1.0])
# - pure textual unit answers handled ("Henry (H)" == "Henry")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
import math
import re
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any

from exact_fama.utils.jsonl import read_jsonl

_SUPERSCRIPT = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-",
})
_SUBSCRIPT = str.maketrans({
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "₊": "+", "₋": "-",
})

_NUM_DEC = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_NUM_TOKEN_RE = re.compile(rf"{_NUM_DEC}(?:[eE][-+]?\d+)?")

# Unit words are removed only for conceptual-text comparison after raw exact
# match fails. Never rely on unit stripping for numeric matching.
_UNIT_WORD_RE = re.compile(
    r"(?i)\b("
    r"v/m|n/c|volt(?:s)?|ampere(?:s)?|ohms?|henr(?:y|ies)|joule(?:s)?|"
    r"farad(?:s)?|coulomb(?:s)?|tesla|newton(?:s)?|watt(?:s)?|meter(?:s)?|metre(?:s)?|"
    r"hz|khz|mhz|pf|nf|μf|uf|mf|f|nc|μc|uc|mc|c|mh|μh|uh|h|"
    r"kv|mv|v|ma|a|ω|ohm|j|w|n|m|cm|mm|kg|g|s|%|percent"
    r")\b"
)


def _ascii_math(value: Any) -> str:
    s = "" if value is None else str(value)
    s = s.translate(_SUPERSCRIPT).translate(_SUBSCRIPT)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("×", "x").replace("·", "*").replace("⋅", "*")
    s = s.replace("µ", "μ").replace("Ω", "Ω")
    s = s.replace("\\times", "x").replace("\\cdot", "*")
    s = s.replace("\\sqrt", "sqrt")
    s = re.sub(r"sqrt\s*\{\s*([^{}]+)\s*\}", r"sqrt(\1)", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _safe_decimal_float(text: str) -> float | None:
    try:
        x = float(Decimal(str(text)))
        return x if math.isfinite(x) else None
    except (InvalidOperation, ValueError, OverflowError):
        return None


def _decimal_sci_to_string(coef: str, exp: str) -> str | None:
    try:
        e = int(str(exp).replace(" ", ""))
        if abs(e) > 308:
            return None
        value = Decimal(str(coef)) * (Decimal(10) ** e)
        x = float(value)
        if math.isfinite(x):
            return f" {value} "
    except Exception:
        return None
    return None


def _replace_scientific_notation(value: Any) -> str:
    s = _ascii_math(value)
    # Normalize exponent braces: 10^{-6} -> 10^-6.
    s = re.sub(r"10\s*\^\s*\{\s*([-+]?\d+)\s*\}", r"10^\1", s, flags=re.I)

    # Vietnamese/school notation: 5.2323.10^9 and 5.10-9.
    # Important: require an explicit exponent marker after `.10` (`^`, `+`, or `-`).
    # Otherwise ordinary decimals like 0.109 or 1.109 would be misread as
    # 0 × 10^9 or 1 × 10^9.
    dot_sci = re.compile(
        rf"(?P<coef>[-+]?\d+(?:\.\d+)?)\s*\.\s*10\s*(?P<exp>\^\s*\{{?\s*[-+]?\d+\s*\}}?|[-+]\d+)",
        flags=re.I,
    )

    def repl_dot(m: re.Match[str]) -> str:
        exp = re.sub(r"^\^\s*\{?\s*", "", m.group("exp"))
        exp = re.sub(r"\s*\}\s*$", "", exp)
        return _decimal_sci_to_string(m.group("coef"), exp) or m.group(0)

    s = dot_sci.sub(repl_dot, s)

    # Explicit multiplication: 3 x 10^-6 / 3*10^6.
    explicit = re.compile(
        rf"(?P<coef>{_NUM_DEC})\s*(?:x|\*)\s*10\s*(?:\^\s*)?(?P<exp>[-+]?\d+)",
        flags=re.I,
    )

    def repl_explicit(m: re.Match[str]) -> str:
        return _decimal_sci_to_string(m.group("coef"), m.group("exp")) or m.group(0)

    s = explicit.sub(repl_explicit, s)

    # Bare powers: 10^-6, -10^6.
    bare = re.compile(r"(?<![\d.])(?P<sign>[-+]?)10\s*\^\s*(?P<exp>[-+]?\d+)", flags=re.I)

    def repl_bare(m: re.Match[str]) -> str:
        coef = "-1" if m.group("sign") == "-" else "1"
        return _decimal_sci_to_string(coef, m.group("exp")) or m.group(0)

    s = bare.sub(repl_bare, s)
    return s


def _strip_variable_labels(text: str) -> str:
    s = _ascii_math(text)
    # Remove labels at item boundaries only, e.g. "I1 = 0.5; I2 = 1.0".
    # The lookahead prevents erasing conceptual formulas like "W = 1/2 L I0^2".
    s = re.sub(
        rf"(?:(?<=^)|(?<=[;\n,]))\s*[A-Za-zΑ-Ωα-ωμΩ]+(?:[_ ]?\d+)?\s*=\s*(?={_NUM_DEC})",
        " ",
        s,
    )
    return s


def _fraction_spans_and_values(s: str) -> tuple[list[tuple[int, int]], list[float]]:
    spans: list[tuple[int, int]] = []
    vals: list[float] = []
    for m in re.finditer(rf"(?<![\w.])({_NUM_DEC})\s*/\s*({_NUM_DEC})(?![\w.])", s):
        num = _safe_decimal_float(m.group(1))
        den = _safe_decimal_float(m.group(2))
        if num is not None and den not in (None, 0.0):
            spans.append(m.span())
            vals.append(num / den)
    return spans, vals


def _extract_numbers(value: Any) -> list[float]:
    s = _strip_variable_labels(_replace_scientific_notation(value))
    spans, nums = _fraction_spans_and_values(s)

    def inside_fraction(pos: int) -> bool:
        return any(a <= pos < b for a, b in spans)

    for m in _NUM_TOKEN_RE.finditer(s):
        if inside_fraction(m.start()):
            continue
        # Avoid picking label indices that survived in words like I1/q2.
        before = s[m.start() - 1] if m.start() > 0 else ""
        after = s[m.end()] if m.end() < len(s) else ""
        if before.isalpha() or after.isalpha():
            continue
        val = _safe_decimal_float(m.group(0))
        if val is not None:
            nums.append(val)
    return nums


def _raw_text_norm(value: Any) -> str:
    s = _ascii_math(value).lower()
    s = s.replace("halved", "halfed")
    s = s.replace("approximately", "approx").replace("approx.", "approx")
    s = re.sub(r"[_:,;=\[\]{}.!?]+", " ", s)
    s = re.sub(r"[()]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _unit_concept_norm(value: Any) -> str:
    s = _raw_text_norm(value)
    # Normalize common parenthetical unit abbreviations by deleting the abbreviation
    # when the full word is present.
    replacements = {
        "volt v": "volt",
        "henry h": "henry",
        "joule j": "joule",
        "tesla t": "tesla",
        "newton n": "newton",
        "farad f": "farad",
        "coulomb c": "coulomb",
        "ohm ω": "ohm",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def _text_for_compare(value: Any) -> str:
    s = _strip_variable_labels(_ascii_math(value)).lower()
    replacements = {
        "halved": "halfed",
        "one half": "half",
        "entirely": "",
        "approximately": "approx",
        "approx.": "approx",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    s = _UNIT_WORD_RE.sub(" ", s)
    s = re.sub(r"[_:,;=()\[\]{}.!?]+", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _has_numeric(value: Any) -> bool:
    return bool(_extract_numbers(value))


def _text_match(gold: Any, pred: Any) -> bool:
    gr = _raw_text_norm(gold)
    pr = _raw_text_norm(pred)
    if gr and pr and gr == pr:
        return True

    # Pure text unit answers: "Henry (H)" should equal "Henry".
    gu = _unit_concept_norm(gold)
    pu = _unit_concept_norm(pred)
    if gu and pu and gu == pu and not _has_numeric(gold) and not _has_numeric(pred):
        return True

    # Do not fuzzy-match numeric answers as text.
    if _has_numeric(gold) or _has_numeric(pred):
        return False

    g = _text_for_compare(gold)
    p = _text_for_compare(pred)
    if g and p and g == p:
        return True
    if not g or not p:
        return False

    # Conservative conceptual-text match only.
    g_tokens = set(g.split())
    p_tokens = set(p.split())
    if len(g_tokens) >= 5 and len(p_tokens) >= 5:
        ratio = SequenceMatcher(None, g, p).ratio()
        jacc = len(g_tokens & p_tokens) / max(len(g_tokens | p_tokens), 1)
        if ratio >= 0.95 or jacc >= 0.90:
            neg = {"not", "no", "never", "không"}
            if bool(g_tokens & neg) == bool(p_tokens & neg):
                return True
    return False


def _numbers_match(gold: Any, pred: Any, numeric_tolerance: float) -> bool:
    g_nums = _extract_numbers(gold)
    p_nums = _extract_numbers(pred)
    if not g_nums or not p_nums or len(g_nums) != len(p_nums):
        return False

    rel_tol = max(float(numeric_tolerance), 0.0)
    # Very small absolute tolerance only; do not use 0.02 as an absolute
    # tolerance, or 0.000379 would incorrectly equal 0.0015.
    abs_tol = 1e-9
    for g, p in zip(g_nums, p_nums):
        if math.isclose(g, p, rel_tol=rel_tol, abs_tol=abs_tol):
            continue
        return False
    return True


def _answers_match_robust(gold: Any, pred: Any, numeric_tolerance: float) -> bool:
    if _text_match(gold, pred):
        return True
    return _numbers_match(gold, pred, numeric_tolerance)


def _canonical_answer_local(value: Any) -> str:
    nums = _extract_numbers(value)
    if nums:
        return "; ".join(f"{x:.12g}" for x in nums)
    # Prefer unit-concept normalization if meaningful.
    unit_norm = _unit_concept_norm(value)
    return unit_norm or _text_for_compare(value) or _raw_text_norm(value)


def _safe_answers_match(gold: Any, pred: Any, numeric_tolerance: float) -> tuple[bool, str | None]:
    try:
        return _answers_match_robust(gold, pred, numeric_tolerance), None
    except Exception as exc:
        return False, f"EVAL_NORMALIZATION_ERROR: {type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--numeric_tolerance", type=float, default=0.02)
    args = parser.parse_args()

    gold_rows = read_jsonl(args.gold)
    pred_rows = read_jsonl(args.pred)
    pred_by_id = {str(row.get("id", str(i))): row for i, row in enumerate(pred_rows)}

    total = 0
    correct = 0
    has_expl = 0
    has_reasoning = 0
    details: list[dict[str, Any]] = []

    for i, gold in enumerate(gold_rows):
        gid = str(gold.get("id", str(i)))
        pred = pred_by_id.get(gid, pred_rows[i] if i < len(pred_rows) else {})

        g_ans = gold.get("answer", "")
        p_ans = pred.get("answer", "")
        if not g_ans:
            continue

        total += 1
        ok, eval_warning = _safe_answers_match(g_ans, p_ans, args.numeric_tolerance)
        correct += int(ok)

        has_expl += int(bool(pred.get("explanation")))
        has_reasoning += int(bool(pred.get("cot") or pred.get("premises") or pred.get("fol")))

        warnings = list(pred.get("warnings", []) or [])
        if eval_warning:
            warnings.append(eval_warning)

        details.append({
            "id": gid,
            "gold": g_ans,
            "pred": p_ans,
            "gold_norm": _canonical_answer_local(g_ans),
            "pred_norm": _canonical_answer_local(p_ans),
            "correct": ok,
            "warnings": warnings,
        })

    report = {
        "total_scored": total,
        "accuracy": correct / total if total else 0.0,
        "explanation_coverage": has_expl / total if total else 0.0,
        "reasoning_field_coverage": has_reasoning / total if total else 0.0,
        "details": details,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
