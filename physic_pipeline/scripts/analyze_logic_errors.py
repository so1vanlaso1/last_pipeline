#!/usr/bin/env python
from __future__ import annotations

"""
Cluster logic evaluation failures.

This is the logic counterpart of scripts/analyze_physics_failures.py.

Basic usage:
python scripts/analyze_logic_failures.py \
  --eval artifacts/fixed_logic100_solver_only_eval.json \
  --gold data/eval/fixed_smoke_logic_100.jsonl \
  --out artifacts/fixed_logic100_logic_failure_clusters.json

With baseline comparison:
python scripts/analyze_logic_failures.py \
  --eval artifacts/fixed_logic100_new_eval.json \
  --gold data/eval/fixed_smoke_logic_100.jsonl \
  --baseline_eval artifacts/fixed_logic100_solver_only_eval.json \
  --out artifacts/fixed_logic100_new_logic_failure_clusters.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OPTION_RE = re.compile(r"\b([A-D])\.\s*(.*?)(?=\n\s*[A-D]\.\s*|\Z)", re.I | re.S)

YES_SET = {"yes", "true", "entailed", "correct", "1"}
NO_SET = {"no", "false", "contradiction", "contradicted", "incorrect", "0"}
UNKNOWN_SET = {
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


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} in {p} is not a JSON object")
            rows.append(obj)
    return rows


def read_eval(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def normalize_label(x: Any) -> str:
    text = norm_text(x).lower()
    text = text.strip(" .;:()[]{}")
    m = re.fullmatch(r"(?:option\s*)?([a-d])", text)
    if m:
        return m.group(1).upper()
    if text in YES_SET:
        return "Yes"
    if text in NO_SET:
        return "No"
    if text in UNKNOWN_SET:
        return "Unknown"
    return norm_text(x)


def label_family(label: Any) -> str:
    y = normalize_label(label)
    if y in {"A", "B", "C", "D"}:
        return "mcq_option"
    if y in {"Yes", "No"}:
        return "yes_no"
    if y == "Unknown":
        return "unknown"
    if not y:
        return "empty"
    return "other"


def question_type(row: dict[str, Any], detail: dict[str, Any] | None = None) -> str:
    q = str(row.get("question") or "")
    gold = normalize_label((detail or {}).get("gold") or row.get("answer"))
    if OPTION_RE.search(q) or gold in {"A", "B", "C", "D"}:
        return "mcq"
    if re.search(r"\b(is|are|does|do|did|can|could|should|would|will|has|have)\b", q.lower()):
        return "yes_no"
    return "open_or_unknown"


def warning_bucket(warnings: list[str]) -> str:
    if not warnings:
        return "no_warning"
    joined = " | ".join(str(w) for w in warnings or []).lower()
    if "output_schema_error" in joined or "schema" in joined or "crashed" in joined:
        return "schema_or_output_error"
    if "uncertain" in joined or "not entailed" in joined or "not contradicted" in joined:
        return "logic_uncertain"
    if "contradiction" in joined:
        return "logic_contradiction"
    if "parser" in joined or "parse" in joined:
        return "parser_warning"
    if "fol" in joined:
        return "fol_warning"
    return "other_warning"


def detect_option_count(question: str) -> int:
    return len(OPTION_RE.findall(question or ""))


def cluster_failure(*, gold: Any, pred: Any, warnings: list[str], row: dict[str, Any], detail: dict[str, Any]) -> str:
    g = normalize_label(gold)
    p = normalize_label(pred)
    qtype = question_type(row, detail)
    wb = warning_bucket(warnings)

    if wb == "schema_or_output_error":
        return f"system/{wb}"

    # Most important EXACT logic failure: solver proves too much.
    if g == "Unknown" and p in {"Yes", "No", "A", "B", "C", "D"}:
        if p in {"A", "B", "C", "D"}:
            return "over_entailment/unknown_to_option"
        return "over_entailment/unknown_to_yes_no"

    # Solver is too conservative.
    if g in {"Yes", "No", "A", "B", "C", "D"} and p == "Unknown":
        if wb == "logic_uncertain":
            return f"under_entailment/{qtype}_uncertain"
        return f"under_entailment/{qtype}_unknown"

    # Yes/No polarity errors.
    if {g, p} == {"Yes", "No"}:
        return "wrong_polarity/yes_no_flip"

    # MCQ option mismatch.
    if g in {"A", "B", "C", "D"} and p in {"A", "B", "C", "D"} and g != p:
        return "wrong_option/mcq_option_mismatch"

    # MCQ/yes-no cross-type confusion.
    if g in {"A", "B", "C", "D"} and p in {"Yes", "No"}:
        return "format_confusion/mcq_gold_yesno_pred"
    if g in {"Yes", "No"} and p in {"A", "B", "C", "D"}:
        return "format_confusion/yesno_gold_option_pred"

    if wb != "no_warning":
        return f"warning_driven/{wb}"

    gf = label_family(g)
    pf = label_family(p)
    return f"other/{gf}_to_{pf}"


def premise_stats(row: dict[str, Any]) -> dict[str, Any]:
    premises_nl = row.get("premises-NL") or row.get("premises_nl") or []
    premises_fol = row.get("premises-FOL") or row.get("premises_fol") or []
    if not isinstance(premises_nl, list):
        premises_nl = [premises_nl]
    if not isinstance(premises_fol, list):
        premises_fol = [premises_fol]
    q = str(row.get("question") or "")
    return {
        "premise_nl_count": len([p for p in premises_nl if str(p).strip()]),
        "premise_fol_count": len([p for p in premises_fol if str(p).strip()]),
        "option_count": detect_option_count(q),
        "has_fol": any(str(p).strip() for p in premises_fol),
    }


def compact_row(row: dict[str, Any], max_premises: int = 5) -> dict[str, Any]:
    premises_nl = row.get("premises-NL") or row.get("premises_nl") or []
    premises_fol = row.get("premises-FOL") or row.get("premises_fol") or []
    if not isinstance(premises_nl, list):
        premises_nl = [premises_nl]
    if not isinstance(premises_fol, list):
        premises_fol = [premises_fol]
    return {
        "question": row.get("question", ""),
        "source_record_id": row.get("source_record_id"),
        "question_index": row.get("question_index"),
        "premises_NL_preview": premises_nl[:max_premises],
        "premises_FOL_preview": premises_fol[:max_premises],
    }


def build_baseline_status(current_details: list[dict[str, Any]], baseline_eval: dict[str, Any] | None) -> tuple[dict[str, str], dict[str, Any]]:
    if not baseline_eval:
        return {}, {}

    base_by_id = {str(d.get("id")): d for d in baseline_eval.get("details", [])}
    status_by_id: dict[str, str] = {}
    counts: Counter[str] = Counter()

    for d in current_details:
        rid = str(d.get("id"))
        cur = bool(d.get("correct"))
        base = base_by_id.get(rid)
        if base is None:
            status = "no_baseline"
        else:
            old = bool(base.get("correct"))
            if old and cur:
                status = "same_correct"
            elif (not old) and (not cur):
                status = "same_wrong"
            elif (not old) and cur:
                status = "gain_wrong_to_correct"
            else:
                status = "regression_correct_to_wrong"
        status_by_id[rid] = status
        counts[status] += 1

    return status_by_id, dict(counts.most_common())


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster logic evaluation failures by answer type, warnings, and proof behavior.")
    parser.add_argument("--eval", required=True, help="Path to evaluate.py JSON report.")
    parser.add_argument("--gold", required=True, help="Gold JSONL used for evaluation.")
    parser.add_argument("--out", default="", help="Optional JSON report output path.")
    parser.add_argument("--baseline_eval", default="", help="Optional previous eval JSON to classify gains/regressions.")
    parser.add_argument("--include_correct", action="store_true", help="Include a compact preview of correct examples too.")
    parser.add_argument("--max_examples_per_cluster", type=int, default=200, help="Limit examples stored per cluster.")
    parser.add_argument("--max_premises", type=int, default=5, help="Number of premises to preview per example.")
    args = parser.parse_args()

    ev = read_eval(args.eval)
    gold_rows = {str(r.get("id")): r for r in read_jsonl(args.gold)}
    details = ev.get("details", [])
    if not isinstance(details, list):
        raise ValueError("Eval report does not contain a list field named 'details'.")

    baseline_eval = read_eval(args.baseline_eval) if args.baseline_eval else None
    baseline_status_by_id, baseline_summary = build_baseline_status(details, baseline_eval)

    failures = [d for d in details if not d.get("correct")]
    corrects = [d for d in details if d.get("correct")]

    cluster_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    question_type_counts: Counter[str] = Counter()
    premise_shape_counts: Counter[str] = Counter()
    source_record_failure_counts: Counter[str] = Counter()
    baseline_status_counts: Counter[str] = Counter()
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for d in failures:
        rid = str(d.get("id"))
        row = gold_rows.get(rid, {})
        warnings = d.get("warnings") or []
        if not isinstance(warnings, list):
            warnings = [str(warnings)]

        cluster = cluster_failure(gold=d.get("gold"), pred=d.get("pred"), warnings=warnings, row=row, detail=d)
        g = normalize_label(d.get("gold_norm") or d.get("gold"))
        p = normalize_label(d.get("pred_norm") or d.get("pred"))
        qtype = question_type(row, d)
        stats = premise_stats(row)
        shape = f"qtype={qtype}|nl={stats['premise_nl_count']}|fol={stats['premise_fol_count']}|options={stats['option_count']}"

        cluster_counts[cluster] += 1
        pair_counts[f"{g} -> {p}"] += 1
        question_type_counts[qtype] += 1
        premise_shape_counts[shape] += 1

        source_id = str(row.get("source_record_id") or rid.rsplit("-q", 1)[0])
        source_record_failure_counts[source_id] += 1

        if warnings:
            for w in warnings:
                warning_counts[str(w)] += 1
        else:
            warning_counts["NO_WARNING"] += 1

        baseline_status = baseline_status_by_id.get(rid, "")
        if baseline_status:
            baseline_status_counts[baseline_status] += 1

        if len(by_cluster[cluster]) < args.max_examples_per_cluster:
            by_cluster[cluster].append({
                "id": rid,
                "gold": d.get("gold"),
                "pred": d.get("pred"),
                "gold_norm": d.get("gold_norm"),
                "pred_norm": d.get("pred_norm"),
                "warnings": warnings,
                "question_type": qtype,
                "answer_pair": f"{g} -> {p}",
                "premise_stats": stats,
                "baseline_status": baseline_status or None,
                **compact_row(row, max_premises=args.max_premises),
            })

    report: dict[str, Any] = {
        "accuracy": ev.get("accuracy"),
        "total_scored": ev.get("total_scored"),
        "failure_count": len(failures),
        "correct_count": len(corrects),
        "cluster_counts": dict(cluster_counts.most_common()),
        "warning_counts": dict(warning_counts.most_common()),
        "answer_pair_counts": dict(pair_counts.most_common()),
        "question_type_failure_counts": dict(question_type_counts.most_common()),
        "premise_shape_failure_counts": dict(premise_shape_counts.most_common()),
        "source_record_failure_counts_top50": dict(source_record_failure_counts.most_common(50)),
        "baseline_comparison_summary": baseline_summary,
        "baseline_status_counts_on_failures": dict(baseline_status_counts.most_common()),
        "clusters": dict(sorted(by_cluster.items())),
    }

    if args.include_correct:
        report["correct_examples_preview"] = []
        for d in corrects[: args.max_examples_per_cluster]:
            rid = str(d.get("id"))
            row = gold_rows.get(rid, {})
            report["correct_examples_preview"].append({
                "id": rid,
                "gold": d.get("gold"),
                "pred": d.get("pred"),
                "warnings": d.get("warnings") or [],
                "question_type": question_type(row, d),
                **compact_row(row, max_premises=args.max_premises),
            })

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
