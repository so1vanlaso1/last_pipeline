#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exact_fama.utils.jsonl import read_jsonl


BANNED = [
    "solver",
    "parser",
    "system parsed",
    "forward chaining",
    "derived conclusions",
    "debug",
    "warnings",
]

PREMISE_REF = re.compile(r"\bpremise\s+\d+\b", re.I)
FORMULA_REF = re.compile(r"\b(formula|using|apply|substitute|=|ohm|coulomb|capacitor|resistor)\b", re.I)


def is_mcq(question: str) -> bool:
    return bool(re.search(r"\bA\.\s+", question) and re.search(r"\bB\.\s+", question))


def has_selected_option(explanation: str, answer: str) -> bool:
    ans = str(answer or "").strip().upper().replace("OPTION ", "")
    if not ans:
        return False
    ans = ans[0]
    if ans not in {"A", "B", "C", "D"}:
        return True
    low = explanation.lower()
    return f"option {ans.lower()}" in low or f"đáp án {ans.lower()}" in low or f"{ans}." in explanation


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    explanation = str(row.get("explanation") or "")
    question = str(row.get("question") or row.get("input_question") or "")
    task_type = str(row.get("task_type") or row.get("type") or "")
    answer = str(row.get("answer") or "")

    low = explanation.lower()
    banned_hits = [b for b in BANNED if b in low]

    premise_ref = bool(PREMISE_REF.search(explanation))
    formula_ref = bool(FORMULA_REF.search(explanation))
    enough_length = 20 <= len(explanation.split()) <= 180
    no_banned = len(banned_hits) == 0
    selected_option_ok = has_selected_option(explanation, answer) if is_mcq(question) else True

    if task_type == "logic":
        evidence_ref = premise_ref or bool(row.get("premises"))
    elif task_type == "physics":
        evidence_ref = formula_ref or bool(row.get("cot"))
    else:
        evidence_ref = premise_ref or formula_ref or bool(row.get("premises") or row.get("cot"))

    good = enough_length and no_banned and selected_option_ok and evidence_ref

    return {
        "id": row.get("id"),
        "task_type": task_type,
        "good": good,
        "enough_length": enough_length,
        "no_banned": no_banned,
        "banned_hits": banned_hits,
        "premise_ref": premise_ref,
        "formula_ref": formula_ref,
        "selected_option_ok": selected_option_ok,
        "llm_rewrite_enabled": (row.get("debug") or {}).get("llm_rewrite_enabled"),
        "llm_rewrite_changed": (row.get("debug") or {}).get("llm_rewrite_changed"),
        "explanation": explanation,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.pred)
    details = [score_row(r) for r in rows]

    total = len(details) or 1
    report = {
        "total": len(details),
        "good_explanation_rate": sum(d["good"] for d in details) / total,
        "enough_length_rate": sum(d["enough_length"] for d in details) / total,
        "no_banned_rate": sum(d["no_banned"] for d in details) / total,
        "premise_ref_rate": sum(d["premise_ref"] for d in details) / total,
        "formula_ref_rate": sum(d["formula_ref"] for d in details) / total,
        "selected_option_ok_rate": sum(d["selected_option_ok"] for d in details) / total,
        "llm_rewrite_enabled_rate": sum(bool(d["llm_rewrite_enabled"]) for d in details) / total,
        "llm_rewrite_changed_rate": sum(bool(d["llm_rewrite_changed"]) for d in details) / total,
        "bad_examples": [d for d in details if not d["good"]][:30],
    }

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()