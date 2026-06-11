#!/usr/bin/env python
from __future__ import annotations

# Allow running scripts directly from the repository without installing first.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import csv
import json
import math
from collections import Counter
from typing import Any

from exact_fama.utils.jsonl import read_jsonl
from exact_fama.utils.answer_normalizer import answers_match


def norm_answer(x: Any) -> str:
    return str(x or "").strip().lower()


def numeric_equal(a: Any, b: Any, tol: float) -> bool:
    try:
        return math.isclose(float(str(a).strip()), float(str(b).strip()), rel_tol=tol, abs_tol=tol)
    except Exception:
        return False


def is_correct(gold: dict[str, Any], pred: dict[str, Any], tol: float) -> bool:
    return answers_match(gold.get("answer", ""), pred.get("answer", ""), tol)


def short(x: Any, n: int = 500) -> str:
    s = str(x or "").replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 3] + "..."


def warning_key(pred: dict[str, Any]) -> str:
    warnings = pred.get("warnings") or []
    if not warnings:
        return "NO_WARNING_BUT_WRONG"
    keys = [str(w).split(":")[0].strip() for w in warnings]
    return "|".join(sorted(set(keys))) or "NO_WARNING_BUT_WRONG"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create readable solver failure reports from gold/pred JSONL.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out_dir", default="artifacts/solver_failure_report")
    parser.add_argument("--numeric_tolerance", type=float, default=1e-5)
    parser.add_argument("--max_markdown_rows", type=int, default=200)
    args = parser.parse_args()

    gold_rows = read_jsonl(args.gold)
    pred_rows = read_jsonl(args.pred)
    pred_by_id = {str(r.get("id", i)): r for i, r in enumerate(pred_rows)}

    failures = []
    correct = 0
    total = 0
    for i, gold in enumerate(gold_rows):
        gid = str(gold.get("id", i))
        pred = pred_by_id.get(gid, pred_rows[i] if i < len(pred_rows) else {})
        if not str(gold.get("answer", "")).strip():
            continue
        total += 1
        ok = is_correct(gold, pred, args.numeric_tolerance)
        correct += int(ok)
        if not ok:
            failures.append({
                "id": gid,
                "type": gold.get("type", pred.get("task_type", "")),
                "warning_key": warning_key(pred),
                "question": gold.get("question", ""),
                "premises": gold.get("premises-NL", gold.get("premises_nl", [])),
                "gold_answer": gold.get("answer", ""),
                "gold_unit": gold.get("unit"),
                "pred_answer": pred.get("answer", ""),
                "pred_unit": pred.get("unit"),
                "pred_explanation": pred.get("explanation", ""),
                "warnings": pred.get("warnings", []),
                "used_modules": pred.get("used_modules", []),
                "debug": pred.get("debug", {}),
            })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "failures_detailed.jsonl").open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out_dir / "failures_compact.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "type", "warning_key", "gold_answer", "pred_answer", "gold_unit", "pred_unit", "question", "warnings"
        ])
        writer.writeheader()
        for r in failures:
            writer.writerow({
                "id": r["id"], "type": r["type"], "warning_key": r["warning_key"],
                "gold_answer": r["gold_answer"], "pred_answer": r["pred_answer"],
                "gold_unit": r["gold_unit"], "pred_unit": r["pred_unit"],
                "question": short(r["question"], 300), "warnings": short(r["warnings"], 300),
            })

    by_type = Counter(str(r["type"]) for r in failures)
    by_warning = Counter(str(r["warning_key"]) for r in failures)
    by_module = Counter()
    for r in failures:
        for m in r.get("used_modules") or []:
            by_module[str(m)] += 1

    summary = {
        "total_scored": total,
        "correct": correct,
        "wrong": len(failures),
        "accuracy": correct / total if total else 0.0,
        "failure_by_type": dict(by_type),
        "failure_by_warning": dict(by_warning.most_common()),
        "failure_by_used_module": dict(by_module.most_common()),
        "outputs": {
            "failures_detailed_jsonl": str(out_dir / "failures_detailed.jsonl"),
            "failures_compact_csv": str(out_dir / "failures_compact.csv"),
            "report_md": str(out_dir / "report.md"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Solver Failure Report")
    lines.append("")
    lines.append(f"- Total scored: **{total}**")
    lines.append(f"- Correct: **{correct}**")
    lines.append(f"- Wrong: **{len(failures)}**")
    lines.append(f"- Accuracy: **{summary['accuracy']:.4f}**")
    lines.append("")
    lines.append("## Failure by type")
    lines.append("")
    for k, v in by_type.most_common():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Failure by warning")
    lines.append("")
    for k, v in by_warning.most_common(30):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append(f"## First {min(args.max_markdown_rows, len(failures))} failures")
    lines.append("")
    for idx, r in enumerate(failures[: args.max_markdown_rows], 1):
        lines.append(f"### {idx}. `{r['id']}` — {r['type']} — {r['warning_key']}")
        lines.append("")
        lines.append(f"**Question:** {short(r['question'], 1000)}")
        lines.append("")
        premises = r.get("premises") or []
        if premises:
            lines.append("**Premises:**")
            if isinstance(premises, list):
                for p in premises[:8]:
                    lines.append(f"- {short(p, 250)}")
                if len(premises) > 8:
                    lines.append(f"- ... {len(premises) - 8} more")
            else:
                lines.append(f"- {short(premises, 800)}")
            lines.append("")
        lines.append(f"**Gold:** `{r['gold_answer']}` {r.get('gold_unit') or ''}")
        lines.append("")
        lines.append(f"**Pred:** `{r['pred_answer']}` {r.get('pred_unit') or ''}")
        lines.append("")
        lines.append(f"**Warnings:** `{short(r.get('warnings'), 800)}`")
        lines.append("")
        lines.append(f"**Explanation:** {short(r.get('pred_explanation'), 1000)}")
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
