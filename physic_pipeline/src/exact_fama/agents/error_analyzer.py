from __future__ import annotations

from collections import Counter
from typing import Any

from .failure_taxonomy import FailureType
from exact_fama.utils.answer_normalizer import answers_match


def classify_failure(gold: dict[str, Any], pred: dict[str, Any]) -> list[FailureType]:
    failures: list[FailureType] = []
    warnings = " ".join(pred.get("warnings") or [])
    for ft in FailureType:
        if ft.value in warnings:
            failures.append(ft)

    gold_answer = str(gold.get("answer", "")).strip().lower()
    pred_answer = str(pred.get("answer", "")).strip().lower()
    if gold.get("answer", "") and pred.get("answer", ""):
        if not answers_match(gold.get("answer", ""), pred.get("answer", "")):
            failures.append(FailureType.ANSWER_WRONG)

    if not pred.get("explanation"):
        failures.append(FailureType.EXPLANATION_WEAK)
    if not pred.get("answer"):
        failures.append(FailureType.OUTPUT_SCHEMA_ERROR)
    return list(dict.fromkeys(failures))


def summarize_failures(failure_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    for row in failure_rows:
        for f in row.get("failure_types", []):
            counter[f] += 1
    total = sum(counter.values())
    denom = total or 1
    return {
        "total_failure_labels": total,
        "counts": dict(counter),
        "percentages": {k: round(v / denom, 4) for k, v in counter.items()},
        "top_failures": counter.most_common(),
    }
