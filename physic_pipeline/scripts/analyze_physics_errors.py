#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_eval(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def _has_regex(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.I | re.U) for p in patterns)


def cluster_question(question: str, warnings: list[str]) -> str:
    """Cluster failed physics samples by likely formula/template family.

    This is heuristic on purpose: it is for debugging solver coverage, not for scoring.
    Put high-specificity templates before broad domains.
    """
    q = str(question or "").lower()

    if warnings and any("formula_error" in str(w).lower() for w in warnings):
        prefix = "formula_error"
    else:
        prefix = "wrong_numeric_or_format"

    # High-impact template families observed in current Physics failures.
    if _has_regex(q, [
        r"\bn\s+in\s+turns\s*/\s*m\b",
        r"\bturns\s*/\s*m\b",
        r"\bturns\s+per\s+(?:metre|meter|unit\s+length)\b",
        r"\buniformly\s+wound\b",
        r"\bcoil\s+has\s+\d+(?:\.\d+)?\s+turns\b",
    ]):
        return prefix + "/turn_density"

    if _has_any(q, [
        "mean absolute deviation",
        "mean and mean absolute deviation",
        "student repeats a measurement",
        "repeats a measurement",
    ]):
        return prefix + "/measurement_mean_mad"

    if _has_any(q, [
        "actually equals",
        "student measures",
        "but a student measures",
        "δ%",
        "delta x",
        "δx",
    ]) or _has_regex(q, [r"\bfind\s+[\u0394δ]x\s+and\s+[\u03b4δ]\s*%\b"]):
        return prefix + "/measurement_actual_vs_measured"

    if _has_regex(q, [
        r"t\s*=\s*2\s*(?:π|pi)\s*√?\s*\(?\s*l\s*c",
        r"2\s*(?:π|pi)\s*(?:√|sqrt)\s*\(?\s*l\s*c",
        r"\bcompute\s+t\b.*\bl\s*=.*\bc\s*=",
        r"\blc\b.*\bperiod\b",
        r"\bperiod\b.*\blc\b",
    ]):
        return prefix + "/lc_period"

    if _has_regex(q, [
        r"\bf\s*=\s*1\s*/\s*t\b",
        r"\blc\b.*\bfrequency\b",
        r"\bfrequency\b.*\blc\b",
    ]):
        return prefix + "/lc_frequency"

    if _has_regex(q, [
        r"\bl\s*=\s*1\s*/\s*\(?\s*c\s*(?:ω|omega|\w)\s*\^?\s*2",
        r"\brequired\s+inductance\b",
        r"\bfind\s+l\b.*\bresonan",
        r"\brlc\b.*\bl\b",
    ]):
        return prefix + "/rlc_required_inductance"

    if _has_regex(q, [
        r"\bc\s*=\s*1\s*/\s*\(?\s*l\s*(?:ω|omega|\w)\s*\^?\s*2",
        r"\brequired\s+capacitance\b",
        r"\bfind\s+c\b.*\bresonan",
        r"\brlc\b.*\bc\b",
    ]):
        return prefix + "/rlc_required_capacitance"

    if _has_any(q, ["rlc", "resonance", "resonant", "impedance", "reactance"]):
        return prefix + "/rlc_resonance"

    # Broader physics domains.
    if _has_any(q, [
        "capacitor",
        "capacitance",
        "electric field energy",
        "parallel-plate",
        "parallel plate",
    ]):
        return prefix + "/capacitor"

    if _has_any(q, [
        "electric charge",
        "point charge",
        "electric field",
        "coulomb",
        "semicircle",
        "equilateral triangle",
    ]):
        return prefix + "/electrostatics_geometry"

    if _has_any(q, [
        "self-induced",
        "self induced",
        "magnetic flux",
        "inductor",
        "inductance",
        "emf",
        "solenoid",
    ]):
        return prefix + "/solenoid_induction"

    if _has_any(q, [
        "relative error",
        "absolute error",
        "uncertainty",
        "measured value",
        "measurement error",
    ]):
        return prefix + "/measurement_error"

    if _has_any(q, [
        "lc circuit",
        "oscillation",
        "conservation of energy",
    ]):
        return prefix + "/lc_conceptual"

    return prefix + "/other"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster physics evaluation failures by formula/template family."
    )
    parser.add_argument("--eval", required=True, help="Path to evaluate.py JSON report.")
    parser.add_argument("--gold", required=True, help="Gold JSONL used for evaluation.")
    parser.add_argument("--out", default="", help="Optional JSON report output path.")
    parser.add_argument(
        "--max_examples_per_cluster",
        type=int,
        default=50,
        help="Limit examples stored per cluster. Use 0 for unlimited. Default: 50.",
    )
    args = parser.parse_args()

    ev = read_eval(args.eval)
    rows = {str(r.get("id")): r for r in read_jsonl(args.gold)}
    failures = [d for d in ev.get("details", []) if not d.get("correct")]

    cluster_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_gold_ids: list[str] = []

    for d in failures:
        rid = str(d.get("id"))
        row = rows.get(rid)
        if row is None:
            row = {}
            missing_gold_ids.append(rid)

        warnings = d.get("warnings") or []
        cluster = cluster_question(row.get("question", ""), warnings)

        cluster_counts[cluster] += 1
        if warnings:
            for w in warnings:
                warning_counts[str(w)] += 1
        else:
            warning_counts["NO_WARNING"] += 1

        if args.max_examples_per_cluster <= 0 or len(by_cluster[cluster]) < args.max_examples_per_cluster:
            by_cluster[cluster].append({
                "id": rid,
                "gold": d.get("gold"),
                "pred": d.get("pred"),
                "gold_norm": d.get("gold_norm"),
                "pred_norm": d.get("pred_norm"),
                "gold_unit": d.get("gold_unit") or row.get("unit"),
                "pred_unit": d.get("pred_unit"),
                "warnings": warnings,
                "question": row.get("question", ""),
            })

    report = {
        "accuracy": ev.get("accuracy"),
        "total_scored": ev.get("total_scored"),
        "failure_count": len(failures),
        "cluster_counts": dict(cluster_counts.most_common()),
        "warning_counts": dict(warning_counts.most_common()),
        "missing_gold_id_count": len(missing_gold_ids),
        "missing_gold_ids_sample": missing_gold_ids[:20],
        "clusters": dict(sorted(by_cluster.items())),
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
