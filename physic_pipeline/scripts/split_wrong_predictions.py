# scripts/split_wrong_predictions.py
from __future__ import annotations

import argparse
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from exact_fama.utils.jsonl import read_jsonl, write_jsonl
try:
    from evaluate import _answers_match_robust
except Exception:  # pragma: no cover
    from exact_fama.utils.answer_normalizer import answers_match as _project_answers_match

    def _answers_match_robust(gold, pred, numeric_tolerance=0.02):
        return _project_answers_match(gold, pred, numeric_tolerance)


def answers_match(gold, pred):
    return _answers_match_robust(gold, pred, 0.02)


SUPERSCRIPT = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-",
})
SUBSCRIPT = str.maketrans({
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "₊": "+", "₋": "-",
})


def _ascii_math(text: Any) -> str:
    s = "" if text is None else str(text)
    s = s.translate(SUPERSCRIPT).translate(SUBSCRIPT)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("×", "x").replace("·", "*").replace("⋅", "*")
    s = s.replace("µ", "μ")
    s = s.replace("\\times", "x").replace("\\cdot", "*")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _safe_float(x: str) -> float | None:
    try:
        value = float(Decimal(x))
        if math.isfinite(value):
            return value
    except (InvalidOperation, ValueError, OverflowError):
        return None
    return None


def _replace_scientific_notation(text: str) -> str:
    s = _ascii_math(text)

    # Vietnamese / generated data often uses "2.26.10^4" or "45.10^{5}"
    # to mean 2.26 × 10^4 or 45 × 10^5.
    s = re.sub(r"(?<=\d)\.(?=10\s*\^)", "*", s)
    s = re.sub(r"(?<=\d)\.\s+(?=10\s*\^)", "*", s)

    # Require an explicit multiplication sign for coefficient notation, so
    # plain integers such as 91091 are never misread as 9 × 10^91.
    explicit_sci_pat = re.compile(
        r"(?P<coef>[-+]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"\s*(?:x|\*)\s*10\s*\^\s*\{?\s*(?P<exp>[-+]?\d+)\s*\}?",
        flags=re.IGNORECASE,
    )
    bare_power_pat = re.compile(
        r"(?<![\d.])(?P<sign>[-+]?)10\s*\^\s*\{?\s*(?P<exp>[-+]?\d+)\s*\}?",
        flags=re.IGNORECASE,
    )

    def repl_explicit(m: re.Match[str]) -> str:
        try:
            value = Decimal(m.group("coef")) * (Decimal(10) ** int(m.group("exp")))
            return f" {value} "
        except Exception:
            return m.group(0)

    def repl_bare(m: re.Match[str]) -> str:
        try:
            coef = Decimal("-1") if m.group("sign") == "-" else Decimal("1")
            value = coef * (Decimal(10) ** int(m.group("exp")))
            return f" {value} "
        except Exception:
            return m.group(0)

    return bare_power_pat.sub(repl_bare, explicit_sci_pat.sub(repl_explicit, s))


def extract_numbers(text: Any) -> list[float]:
    s = _replace_scientific_notation(text)

    nums: list[float] = []

    # Simple standalone fractions such as 10/3.
    fraction_spans: list[tuple[int, int]] = []
    for m in re.finditer(r"(?<![\w.])([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)(?![\w.])", s):
        den = _safe_float(m.group(2))
        num = _safe_float(m.group(1))
        if den not in (None, 0.0) and num is not None:
            nums.append(num / den)
            fraction_spans.append(m.span())

    def inside_fraction(pos: int) -> bool:
        return any(a <= pos < b for a, b in fraction_spans)

    # Standard numbers, including e notation.
    for m in re.finditer(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", s):
        if inside_fraction(m.start()):
            continue
        val = _safe_float(m.group(0))
        if val is not None:
            nums.append(val)

    return nums


def relative_error(gold: float, pred: float) -> float:
    denom = max(abs(gold), 1e-12)
    return abs(gold - pred) / denom


def decimals_in_answer(text: Any) -> int | None:
    s = _ascii_math(text)
    m = re.search(r"[-+]?\d+\.(\d+)", s)
    if not m:
        return None
    return len(m.group(1))


def numeric_lists_close(
    gold_nums: list[float],
    pred_nums: list[float],
    *,
    rel_tol: float,
    abs_tol: float,
) -> tuple[bool, list[float]]:
    if not gold_nums or not pred_nums or len(gold_nums) != len(pred_nums):
        return False, []

    errs = []
    for g, p in zip(gold_nums, pred_nums):
        err_abs = abs(g - p)
        err_rel = relative_error(g, p)
        errs.append(err_rel)
        if not (err_abs <= abs_tol or err_rel <= rel_tol):
            return False, errs

    return True, errs


def numeric_lists_scale_equivalent(
    gold_nums: list[float],
    pred_nums: list[float],
    *,
    rel_tol: float,
    abs_tol: float,
) -> tuple[bool, float | None, list[float]]:
    if not gold_nums or not pred_nums or len(gold_nums) != len(pred_nums):
        return False, None, []

    common_scales = [
        1e-12, 1e-9, 1e-6, 1e-3,
        1e3, 1e6, 1e9, 1e12,
        100.0, 0.01,
    ]

    for scale in common_scales:
        scaled_pred = [p * scale for p in pred_nums]
        ok, errs = numeric_lists_close(gold_nums, scaled_pred, rel_tol=rel_tol, abs_tol=abs_tol)
        if ok:
            return True, scale, errs

    return False, None, []


def canonical_text(text: Any) -> str:
    s = _ascii_math(text).lower()

    # Remove short unit annotations like "(h)", "(v)", "(j)", "(n/c)".
    s = re.sub(r"\(([a-zμ/Ωohm]{1,8})\)", " ", s, flags=re.IGNORECASE)

    # Remove common variable labels before numbers: "P = 48.0" -> "48.0"
    s = re.sub(r"\b[a-z][a-z0-9_]*\s*=\s*", " ", s)

    # Normalize punctuation/spacing.
    s = re.sub(r"[_:,;=]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def text_equivalent_soft(gold_answer: Any, pred_answer: Any) -> bool:
    g = canonical_text(gold_answer)
    p = canonical_text(pred_answer)

    if not g or not p:
        return False

    if g == p:
        return True

    # "Henry (H)" vs "Henry", "Volt (V)" vs "Volt"
    if g in p or p in g:
        # Avoid marking long missing multi-answer cases as soft by requiring short text.
        return max(len(g), len(p)) <= 40

    return False


def classify_wrong(
    gold_answer: Any,
    pred_answer: Any,
    *,
    rel_tol: float,
    abs_tol: float,
    unit_rel_tol: float,
    allow_unit_scale: bool,
) -> dict[str, Any]:
    reasons: list[str] = []

    gold_nums = extract_numbers(gold_answer)
    pred_nums = extract_numbers(pred_answer)

    if text_equivalent_soft(gold_answer, pred_answer):
        reasons.append("format_or_label_only")
        return {
            "split": "soft_wrong",
            "reasons": reasons,
            "gold_numbers": gold_nums,
            "pred_numbers": pred_nums,
            "relative_errors": [],
        }

    close, errs = numeric_lists_close(
        gold_nums, pred_nums, rel_tol=rel_tol, abs_tol=abs_tol
    )
    if close:
        max_err = max(errs) if errs else 0.0
        if max_err <= 1e-12:
            reasons.append("numeric_equal_but_format_failed")
        elif max_err <= 0.001:
            reasons.append("rounding_or_constant_error_le_0.1_percent")
        elif max_err <= 0.005:
            reasons.append("rounding_or_constant_error_le_0.5_percent")
        elif max_err <= 0.01:
            reasons.append("rounding_or_constant_error_le_1_percent")
        else:
            reasons.append(f"numeric_close_le_{rel_tol:g}_relative")
        return {
            "split": "soft_wrong",
            "reasons": reasons,
            "gold_numbers": gold_nums,
            "pred_numbers": pred_nums,
            "relative_errors": errs,
        }

    # Try comparison after rounding pred to the number of decimals used by gold.
    # This catches cases where the raw relative error is tiny but decimal display differs.
    if gold_nums and pred_nums and len(gold_nums) == len(pred_nums):
        dec = decimals_in_answer(gold_answer)
        if dec is not None:
            rounded_pred = [round(p, dec) for p in pred_nums]
            rounded_gold = [round(g, dec) for g in gold_nums]
            if rounded_pred == rounded_gold:
                reasons.append(f"rounding_to_gold_decimals_{dec}")
                return {
                    "split": "soft_wrong",
                    "reasons": reasons,
                    "gold_numbers": gold_nums,
                    "pred_numbers": pred_nums,
                    "relative_errors": [relative_error(g, p) for g, p in zip(gold_nums, pred_nums)],
                }

    if allow_unit_scale:
        scale_ok, scale, scale_errs = numeric_lists_scale_equivalent(
            gold_nums, pred_nums, rel_tol=unit_rel_tol, abs_tol=abs_tol
        )
        if scale_ok:
            reasons.append(f"unit_scale_or_output_unit_issue_scale_pred_by_{scale:g}")
            return {
                "split": "soft_wrong",
                "reasons": reasons,
                "gold_numbers": gold_nums,
                "pred_numbers": pred_nums,
                "relative_errors": scale_errs,
            }

    reasons.append("hard_formula_or_target_error")
    return {
        "split": "hard_wrong",
        "reasons": reasons,
        "gold_numbers": gold_nums,
        "pred_numbers": pred_nums,
        "relative_errors": errs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Split wrong predictions into soft_wrong "
            "(rounding/format/unit-scale/evaluator false negative) and hard_wrong "
            "(real formula/target errors)."
        )
    )
    ap.add_argument("--gold", required=True, help="Path to gold JSONL.")
    ap.add_argument("--pred", required=True, help="Path to prediction JSONL.")

    ap.add_argument("--out_soft_gold", required=True, help="Gold rows for soft wrong cases.")
    ap.add_argument("--out_hard_gold", required=True, help="Gold rows for hard wrong cases.")

    ap.add_argument("--out_soft_pred", default=None, help="Optional prediction rows for soft wrong cases.")
    ap.add_argument("--out_hard_pred", default=None, help="Optional prediction rows for hard wrong cases.")

    ap.add_argument("--out_soft_mismatches", default=None, help="Optional debug JSONL for soft wrong cases.")
    ap.add_argument("--out_hard_mismatches", default=None, help="Optional debug JSONL for hard wrong cases.")

    ap.add_argument(
        "--rel_tol",
        type=float,
        default=0.02,
        help="Relative tolerance for soft numeric mismatch. Default: 0.02 = 2%%.",
    )
    ap.add_argument(
        "--abs_tol",
        type=float,
        default=1e-12,
        help="Absolute tolerance for very small numeric answers. Default: 1e-12.",
    )
    ap.add_argument(
        "--unit_rel_tol",
        type=float,
        default=0.005,
        help="Relative tolerance after common unit scaling. Default: 0.005 = 0.5%%.",
    )
    ap.add_argument(
        "--no_unit_scale",
        action="store_true",
        help="Disable classifying common 10^3/10^6 unit-scale mismatches as soft wrong.",
    )
    ap.add_argument(
        "--include_correct",
        action="store_true",
        help="Also write already-correct rows to soft split. Default: skip correct rows.",
    )
    ap.add_argument(
        "--skip_missing_pred",
        action="store_true",
        help="Skip gold rows without matching prediction. Default: count as hard wrong.",
    )

    args = ap.parse_args()

    gold_rows = read_jsonl(args.gold)
    pred_rows = read_jsonl(args.pred)
    pred_by_id = {str(r.get("id", i)): r for i, r in enumerate(pred_rows)}

    soft_gold: list[dict[str, Any]] = []
    hard_gold: list[dict[str, Any]] = []
    soft_pred: list[dict[str, Any]] = []
    hard_pred: list[dict[str, Any]] = []
    soft_mismatches: list[dict[str, Any]] = []
    hard_mismatches: list[dict[str, Any]] = []

    counters = {
        "input_gold": len(gold_rows),
        "input_pred": len(pred_rows),
        "already_correct_skipped": 0,
        "missing_pred": 0,
        "soft_wrong": 0,
        "hard_wrong": 0,
    }
    reason_counts: dict[str, int] = {}

    for i, gold in enumerate(gold_rows):
        gid = str(gold.get("id", i))
        pred = pred_by_id.get(gid)

        if pred is None:
            counters["missing_pred"] += 1
            if args.skip_missing_pred:
                continue

            hard_gold.append(gold)
            hard_mismatches.append({
                "id": gid,
                "split": "hard_wrong",
                "reasons": ["missing_prediction"],
                "question": gold.get("question", ""),
                "gold": gold.get("answer", ""),
                "pred": None,
                "warnings": ["MISSING_PREDICTION"],
            })
            counters["hard_wrong"] += 1
            reason_counts["missing_prediction"] = reason_counts.get("missing_prediction", 0) + 1
            continue

        gold_answer = gold.get("answer", "")
        pred_answer = pred.get("answer", "")

        is_correct = answers_match(gold_answer, pred_answer)
        if is_correct and not args.include_correct:
            counters["already_correct_skipped"] += 1
            continue

        if is_correct and args.include_correct:
            result = {
                "split": "soft_wrong",
                "reasons": ["already_correct_included"],
                "gold_numbers": extract_numbers(gold_answer),
                "pred_numbers": extract_numbers(pred_answer),
                "relative_errors": [],
            }
        else:
            result = classify_wrong(
                gold_answer,
                pred_answer,
                rel_tol=args.rel_tol,
                abs_tol=args.abs_tol,
                unit_rel_tol=args.unit_rel_tol,
                allow_unit_scale=not args.no_unit_scale,
            )

        item = {
            "id": gid,
            "split": result["split"],
            "reasons": result["reasons"],
            "question": gold.get("question", pred.get("question", "")),
            "gold": gold_answer,
            "pred": pred_answer,
            "gold_numbers": result.get("gold_numbers", []),
            "pred_numbers": result.get("pred_numbers", []),
            "relative_errors": result.get("relative_errors", []),
            "warnings": pred.get("warnings", []),
        }

        for reason in result["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        if result["split"] == "soft_wrong":
            soft_gold.append(gold)
            soft_pred.append(pred)
            soft_mismatches.append(item)
            counters["soft_wrong"] += 1
        else:
            hard_gold.append(gold)
            hard_pred.append(pred)
            hard_mismatches.append(item)
            counters["hard_wrong"] += 1

    write_jsonl(args.out_soft_gold, soft_gold)
    write_jsonl(args.out_hard_gold, hard_gold)

    if args.out_soft_pred:
        write_jsonl(args.out_soft_pred, soft_pred)
    if args.out_hard_pred:
        write_jsonl(args.out_hard_pred, hard_pred)
    if args.out_soft_mismatches:
        write_jsonl(args.out_soft_mismatches, soft_mismatches)
    if args.out_hard_mismatches:
        write_jsonl(args.out_hard_mismatches, hard_mismatches)

    summary = {
        **counters,
        "wrong_total_considered": counters["soft_wrong"] + counters["hard_wrong"],
        "soft_wrong_rate_among_wrong": (
            counters["soft_wrong"] / max(1, counters["soft_wrong"] + counters["hard_wrong"])
        ),
        "hard_wrong_rate_among_wrong": (
            counters["hard_wrong"] / max(1, counters["soft_wrong"] + counters["hard_wrong"])
        ),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "settings": {
            "rel_tol": args.rel_tol,
            "abs_tol": args.abs_tol,
            "unit_rel_tol": args.unit_rel_tol,
            "unit_scale_enabled": not args.no_unit_scale,
            "include_correct": args.include_correct,
            "skip_missing_pred": args.skip_missing_pred,
        },
        "outputs": {
            "out_soft_gold": args.out_soft_gold,
            "out_hard_gold": args.out_hard_gold,
            "out_soft_pred": args.out_soft_pred,
            "out_hard_pred": args.out_hard_pred,
            "out_soft_mismatches": args.out_soft_mismatches,
            "out_hard_mismatches": args.out_hard_mismatches,
        },
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
