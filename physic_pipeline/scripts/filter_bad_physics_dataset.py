#!/usr/bin/env python3
"""
Filter known-bad / internally inconsistent physics dataset rows and recompute accuracy
on the remaining clean subset.

Usage examples:
  python filter_bad_physics_dataset.py \
    --dataset physics_dataset.jsonl \
    --eval all_physics_solver_only_eval.json \
    --predictions all_physics_solver_only_predictions.jsonl \
    --out-dir cleaned_physics

  # Add your own reviewed bad IDs, one ID per line or JSON list/dict:
  python filter_bad_physics_dataset.py --bad-ids-file bad_ids.txt

Outputs:
  clean_dataset.jsonl          dataset with bad IDs removed
  removed_bad_data.jsonl       removed records with reason
  clean_eval_details.json      eval details after removing bad IDs
  clean_eval_summary.json      recomputed accuracy on clean subset
  review_candidates.csv        failed rows that may need manual review
  suggested_bad_ids.json       auto-detected suspicious IDs, NOT removed unless enabled

Design note:
  This script is conservative by default. It removes only DEFAULT_BAD_IDS and IDs
  you explicitly provide. Rule-based suspicious rows are exported for review so
  the filter does not become another form of answer memorization.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Conservative known-bad rows observed from manual physics review.
# Keep this small. Add more through --bad-ids-file after human review.
# ---------------------------------------------------------------------------
DEFAULT_BAD_ID_REASONS: dict[str, str] = {
    "TD364": "Gold/unit is inconsistent: Q=20 μC, U=5 V => C=4 μF, but gold is 0.100 nC.",
    "NL346": "Gold/unit is inconsistent: W=0.1 J, U=100 V => Q=0.002 C=2 mC, but gold is 0.002 mC.",
    "CH345": "CoT and formula give f0≈53.05 Hz for L=0.15 H, C=60 μF, but answer field says 51.05.",
    "DDT340": "Question gives C and f; Xc=1/(2πfC)≈33.16 Ω, while gold uses sqrt(Z²-R²)=38.16 Ω despite inconsistent inputs.",
    "THCB128": "Mean absolute error of 200.5, 200.3, 200.2 g is ≈0.111 g; answer says 0.133 while its own CoT says 0.11.",
    "CH241": "Prompt has no explicit target quantity; answer W=172.73 is under-specified from the question text.",
}

SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
})

UNIT_TO_SI = {
    "f": 1.0, "mf": 1e-3, "μf": 1e-6, "µf": 1e-6, "uf": 1e-6, "nf": 1e-9, "pf": 1e-12,
    "c": 1.0, "mc": 1e-3, "μc": 1e-6, "µc": 1e-6, "uc": 1e-6, "nc": 1e-9, "pc": 1e-12,
    "v": 1.0, "kv": 1e3, "mv": 1e-3,
    "h": 1.0, "mh": 1e-3, "μh": 1e-6, "µh": 1e-6, "uh": 1e-6,
    "hz": 1.0, "khz": 1e3,
    "ohm": 1.0, "ohms": 1.0, "ω": 1.0, "kohm": 1e3, "kω": 1e3,
    "j": 1.0, "mj": 1e-3, "μj": 1e-6, "µj": 1e-6, "uj": 1e-6,
    "g": 1.0, "kg": 1000.0,  # for measurement questions, keep output in original mass unit scale
}

NUMBER_RE = re.compile(
    r"""
    (?P<num>
        [-+]?\d+(?:\.\d+)?\s*(?:×|x|\*)\s*10\s*(?:\^)?\s*[-+]?\d+
      | [-+]?\d+(?:\.\d+)?\s*\.\s*10\s*(?:\^)?\s*[-+]?\d+
      | 10\s*(?:\^)?\s*[-+]?\d+
      | [-+]?\d+(?:\.\d+)?
      | [-+]?\.\d+
    )
    """,
    re.I | re.X,
)

@dataclass
class Flag:
    id: str
    reason: str
    severity: str = "review"  # "remove" or "review"
    computed: str | None = None
    gold: str | None = None


def norm_text(s: Any) -> str:
    s = str(s or "")
    s = s.translate(SUPERSCRIPT_MAP)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("µ", "μ")
    # Convert common TeX exponents: 10^{-6} -> 10^-6
    s = re.sub(r"10\s*\^\s*\{\s*([-+]?\d+)\s*\}", r"10^\1", s)
    return s


def parse_number(s: Any) -> float | None:
    t = norm_text(s).strip()
    m = NUMBER_RE.search(t)
    if not m:
        return None
    raw = m.group("num")
    raw = raw.replace(" ", "")
    raw = raw.replace("×", "x").replace("*", "x")
    # Handle school notation like 4.10^-9 meaning 4 * 10^-9, not 4.10.
    raw = re.sub(r"^([-+]?\d+(?:\.\d+)?)\.10\^?([-+]?\d+)$", r"\1x10^\2", raw)
    if "x10" in raw.lower():
        a, b = re.split(r"x10\^?", raw, flags=re.I)
        try:
            return float(a) * (10.0 ** int(b))
        except Exception:
            return None
    if raw.lower().startswith("10^"):
        try:
            return 10.0 ** int(raw[3:])
        except Exception:
            return None
    try:
        return float(raw)
    except Exception:
        return None


def numbers_in(s: Any) -> list[float]:
    t = norm_text(s)
    out: list[float] = []
    for m in NUMBER_RE.finditer(t):
        val = parse_number(m.group("num"))
        if val is not None and math.isfinite(val):
            out.append(val)
    return out


def close(a: float, b: float, rel: float = 0.025, abs_tol: float = 1e-12) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b), 1e-30))


def unit_factor(unit: str | None) -> float | None:
    if not unit:
        return 1.0
    u = norm_text(unit).strip().lower()
    u = u.replace(" ", "")
    u = u.split(";")[0]
    u = u.strip("().,[]")
    return UNIT_TO_SI.get(u)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_extra_bad_ids(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return {str(x): "manual bad id" for x in obj}
    except Exception:
        pass
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            k, v = line.split(",", 1)
            out[k.strip()] = v.strip() or "manual bad id"
        else:
            out[line] = "manual bad id"
    return out


def extract_symbol_value(text: str, symbol_patterns: list[str], unit_regex: str) -> tuple[float, str] | None:
    t = norm_text(text)
    syms = "|".join(symbol_patterns)
    pat = re.compile(rf"(?:\b(?:{syms})\b)\s*(?:=|is|of)?\s*(?P<v>{NUMBER_RE.pattern})\s*(?P<u>{unit_regex})\b", re.I | re.X)
    m = pat.search(t)
    if not m:
        return None
    val = parse_number(m.group("v"))
    if val is None:
        return None
    u = m.group("u")
    fac = unit_factor(u)
    return (val * (fac if fac is not None else 1.0), u)


def first_value_with_unit(text: str, unit_regex: str) -> tuple[float, str] | None:
    t = norm_text(text)
    pat = re.compile(rf"(?P<v>{NUMBER_RE.pattern})\s*(?P<u>{unit_regex})\b", re.I | re.X)
    m = pat.search(t)
    if not m:
        return None
    val = parse_number(m.group("v"))
    if val is None:
        return None
    u = m.group("u")
    fac = unit_factor(u)
    return (val * (fac if fac is not None else 1.0), u)


def answer_value_in_unit(record: dict[str, Any]) -> float | None:
    val = parse_number(record.get("answer"))
    if val is None:
        return None
    return val


def flag_capacitance_q_over_u(record: dict[str, Any]) -> Flag | None:
    q = norm_text(record.get("question", "")).lower()
    if "capacitance" not in q and "calculate c" not in q:
        return None
    if "charge" not in q and "stores q" not in q and " q =" not in q:
        return None
    Q = extract_symbol_value(record.get("question", ""), ["Q", "charge"], r"mC|μC|µC|uC|nC|pC|C")
    U = extract_symbol_value(record.get("question", ""), ["U", "V", "voltage", "potential difference"], r"kV|mV|V")
    if not Q:
        Q = first_value_with_unit(record.get("question", ""), r"mC|μC|µC|uC|nC|pC|C")
    if not U:
        U = first_value_with_unit(record.get("question", ""), r"kV|mV|V")
    if not Q or not U or U[0] == 0:
        return None
    C_si = Q[0] / U[0]
    out_unit = str(record.get("unit") or "").split(";")[0].strip()
    fac = unit_factor(out_unit)
    # If answer unit is not a capacitance unit, definitely suspicious.
    if out_unit and out_unit.lower().replace("µ", "μ") not in {"f", "mf", "μf", "uf", "nf", "pf"}:
        return Flag(str(record.get("id")), f"Capacitance question has non-capacitance gold unit '{out_unit}'.", computed=f"{C_si} F", gold=f"{record.get('answer')} {record.get('unit')}")
    if fac is None or fac == 0:
        return None
    computed_out = C_si / fac
    gold = answer_value_in_unit(record)
    if gold is not None and not close(computed_out, gold, rel=0.03):
        return Flag(str(record.get("id")), "Gold disagrees with C=Q/U.", computed=f"{computed_out:g} {out_unit}", gold=f"{record.get('answer')} {record.get('unit')}")
    return None


def flag_lc_resonance_frequency(record: dict[str, Any]) -> Flag | None:
    q = norm_text(record.get("question", "")).lower()
    if not ("resonant frequency" in q or "calculate the resonant frequency" in q):
        return None
    L = extract_symbol_value(record.get("question", ""), ["L", "inductance"], r"mH|μH|µH|uH|H")
    C = extract_symbol_value(record.get("question", ""), ["C", "capacitance"], r"mF|μF|µF|uF|nF|pF|F")
    if not L or not C or L[0] <= 0 or C[0] <= 0:
        return None
    f0 = 1.0 / (2.0 * math.pi * math.sqrt(L[0] * C[0]))
    gold = answer_value_in_unit(record)
    fac = unit_factor(record.get("unit")) or 1.0
    computed_out = f0 / fac
    if gold is not None and not close(computed_out, gold, rel=0.02):
        return Flag(str(record.get("id")), "Gold disagrees with LC resonance f0=1/(2π√LC).", computed=f"{computed_out:g} {record.get('unit')}", gold=f"{record.get('answer')} {record.get('unit')}")
    return None


def flag_charge_from_energy_voltage(record: dict[str, Any]) -> Flag | None:
    q = norm_text(record.get("question", "")).lower()
    if "charge" not in q or "energy" not in q or not ("voltage" in q or "potential difference" in q):
        return None
    W = first_value_with_unit(record.get("question", ""), r"mJ|μJ|µJ|uJ|J")
    U = first_value_with_unit(record.get("question", ""), r"kV|mV|V")
    if not W or not U or U[0] == 0:
        return None
    Q_si = 2.0 * W[0] / U[0]
    out_unit = str(record.get("unit") or "").split(";")[0].strip()
    fac = unit_factor(out_unit)
    if fac is None:
        return None
    computed_out = Q_si / fac
    gold = answer_value_in_unit(record)
    if gold is not None and not close(computed_out, gold, rel=0.03):
        return Flag(str(record.get("id")), "Gold disagrees with W=1/2 Q U => Q=2W/U.", computed=f"{computed_out:g} {out_unit}", gold=f"{record.get('answer')} {record.get('unit')}")
    return None


def flag_mean_abs_error(record: dict[str, Any]) -> Flag | None:
    q = norm_text(record.get("question", "")).lower()
    if not ("measurements" in q and ("mean absolute" in q or "average absolute" in q)):
        return None
    vals = []
    for m in re.finditer(r"(?P<v>[-+]?\d+(?:\.\d+)?)\s*(?:g|kg|a|v|cm|mm|m)\b", norm_text(record.get("question", "")), re.I):
        vals.append(float(m.group("v")))
    if len(vals) < 3:
        return None
    vals = vals[:3]
    avg = sum(vals) / len(vals)
    mad = sum(abs(x - avg) for x in vals) / len(vals)
    # Parse answer like "200.3; 0.133"
    nums = numbers_in(record.get("answer", ""))
    if len(nums) >= 2 and not close(nums[1], mad, rel=0.08, abs_tol=0.005):
        return Flag(str(record.get("id")), "Gold disagrees with mean absolute deviation.", computed=f"{avg:.6g}; {mad:.6g}", gold=str(record.get("answer")))
    return None


def flag_cot_answer_conflict(record: dict[str, Any]) -> Flag | None:
    """Weak heuristic: the CoT's final sentence contains a different number than answer."""
    cot = norm_text(record.get("cot") or record.get("explanation") or "")
    if not cot:
        return None
    tail = cot[-350:]
    ans_nums = numbers_in(record.get("answer", ""))
    tail_nums = numbers_in(tail)
    if not ans_nums or not tail_nums:
        return None
    # For multi-answer records, avoid over-flagging.
    if len(ans_nums) > 2:
        return None
    primary = ans_nums[-1]
    # If the answer number does not appear in the final calculation tail, and a different final-ish number does, review.
    if not any(close(primary, x, rel=0.015, abs_tol=1e-9) for x in tail_nums):
        last = tail_nums[-1]
        if not close(primary, last, rel=0.03, abs_tol=1e-9):
            return Flag(str(record.get("id")), "Answer field appears to disagree with the record's own CoT/explanation tail.", computed=f"tail_last_number={last:g}", gold=str(record.get("answer")))
    return None


def auto_review_flags(record: dict[str, Any]) -> list[Flag]:
    flags: list[Flag] = []
    for fn in [
        flag_capacitance_q_over_u,
        flag_lc_resonance_frequency,
        flag_charge_from_energy_voltage,
        flag_mean_abs_error,
        flag_cot_answer_conflict,
    ]:
        try:
            fl = fn(record)
            if fl:
                flags.append(fl)
        except Exception as e:
            flags.append(Flag(str(record.get("id")), f"validator_error:{fn.__name__}:{e}", severity="review"))
    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/mnt/data/physics_dataset(1).jsonl", help="Original dataset JSONL")
    ap.add_argument("--eval", default="/mnt/data/all_physics_solver_only_eval(2).json", help="Evaluation JSON with details")
    ap.add_argument("--predictions", default="/mnt/data/all_physics_solver_only_predictions(2).jsonl", help="Predictions JSONL")
    ap.add_argument("--analyze", default="/mnt/data/all_physics_solver_only_analyze(2).json", help="Optional analyze JSON")
    ap.add_argument("--out-dir", default="/mnt/data/cleaned_physics_dataset", help="Output directory")
    ap.add_argument("--bad-ids-file", default=None, help="Extra bad IDs to remove: txt, JSON list, or JSON dict id->reason")
    ap.add_argument("--no-default-bad-ids", action="store_true", help="Do not remove DEFAULT_BAD_IDS")
    ap.add_argument("--auto-remove-review-flags", action="store_true", help="Also remove rule-based suspicious rows. Use only after review.")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    eval_path = Path(args.eval)
    pred_path = Path(args.predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = read_jsonl(dataset_path)
    by_id = {str(r.get("id")): r for r in dataset_rows}
    eval_obj = json.loads(eval_path.read_text(encoding="utf-8"))
    details = eval_obj.get("details", [])
    predictions = read_jsonl(pred_path) if pred_path.exists() else []
    pred_by_id = {str(p.get("id")): p for p in predictions}

    remove_reasons: dict[str, str] = {}
    if not args.no_default_bad_ids:
        remove_reasons.update(DEFAULT_BAD_ID_REASONS)
    remove_reasons.update(load_extra_bad_ids(Path(args.bad_ids_file) if args.bad_ids_file else None))

    suggested: dict[str, list[dict[str, str | None]]] = {}
    for r in dataset_rows:
        rid = str(r.get("id"))
        flags = auto_review_flags(r)
        if flags:
            suggested[rid] = [flag.__dict__ for flag in flags]
            if args.auto_remove_review_flags:
                remove_reasons.setdefault(rid, "; ".join(f.reason for f in flags))

    remove_ids = set(remove_reasons)
    clean_rows = [r for r in dataset_rows if str(r.get("id")) not in remove_ids]
    removed_rows = []
    for r in dataset_rows:
        rid = str(r.get("id"))
        if rid in remove_ids:
            rr = dict(r)
            rr["bad_data_reason"] = remove_reasons[rid]
            if rid in pred_by_id:
                rr["latest_prediction"] = pred_by_id[rid].get("answer")
            removed_rows.append(rr)

    clean_details = [d for d in details if str(d.get("id")) not in remove_ids]
    removed_details = [d for d in details if str(d.get("id")) in remove_ids]
    clean_total = len(clean_details)
    clean_correct = sum(1 for d in clean_details if d.get("correct") is True)
    clean_accuracy = clean_correct / clean_total if clean_total else float("nan")

    summary = {
        "original_total_scored": eval_obj.get("total_scored"),
        "original_accuracy": eval_obj.get("accuracy"),
        "removed_count": len(remove_ids),
        "removed_ids": sorted(remove_ids),
        "clean_total_scored": clean_total,
        "clean_correct": clean_correct,
        "clean_failure_count": clean_total - clean_correct,
        "clean_accuracy": clean_accuracy,
        "note": "Rule-based suggested_bad_ids are not removed unless --auto-remove-review-flags is used.",
    }

    write_jsonl(out_dir / "clean_dataset.jsonl", clean_rows)
    write_jsonl(out_dir / "removed_bad_data.jsonl", removed_rows)
    (out_dir / "clean_eval_details.json").write_text(json.dumps(clean_details, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "removed_eval_details.json").write_text(json.dumps(removed_details, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "clean_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "suggested_bad_ids.json").write_text(json.dumps(suggested, ensure_ascii=False, indent=2), encoding="utf-8")

    # Review CSV: remaining failures + suggested reason if any.
    with (out_dir / "review_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "cluster", "gold", "pred", "unit", "suggested_reason", "question"])
        w.writeheader()
        # Build cluster map if analyze file exists.
        cluster_by_id: dict[str, str] = {}
        analyze_path = Path(args.analyze) if args.analyze else None
        if analyze_path and analyze_path.exists():
            try:
                analyze = json.loads(analyze_path.read_text(encoding="utf-8"))
                for cluster_name, items in analyze.get("clusters", {}).items():
                    for it in items:
                        cluster_by_id[str(it.get("id"))] = cluster_name
            except Exception:
                pass
        for d in clean_details:
            if d.get("correct") is True:
                continue
            rid = str(d.get("id"))
            rec = by_id.get(rid, {})
            reasons = []
            for fl in suggested.get(rid, []):
                reasons.append(str(fl.get("reason")))
            w.writerow({
                "id": rid,
                "cluster": cluster_by_id.get(rid, ""),
                "gold": d.get("gold"),
                "pred": d.get("pred"),
                "unit": rec.get("unit", ""),
                "suggested_reason": " | ".join(reasons),
                "question": rec.get("question", d.get("question", "")),
            })

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
