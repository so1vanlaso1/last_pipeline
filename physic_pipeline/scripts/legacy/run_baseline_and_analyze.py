#!/usr/bin/env python
from __future__ import annotations

# Allow running scripts directly from the repository without installing the package first.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import argparse
import json
from pathlib import Path

from exact_fama.agents.error_analyzer import classify_failure, summarize_failures
from exact_fama.agents.mitigation import build_mitigation_config
from exact_fama.agents.orchestrator import select_dominant_failures
from exact_fama.pipeline import ExactFamaPipeline
from exact_fama.schemas import PredictRequest
from exact_fama.utils.jsonl import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ExactFamaPipeline()
    rows = read_jsonl(args.input)
    preds = []
    failures = []
    for i, row in enumerate(rows):
        req = PredictRequest.model_validate(row)
        pred = pipeline.predict(req).model_dump(mode="json")
        pred["id"] = row.get("id", str(i))
        preds.append(pred)
        failure_types = [ft.value for ft in classify_failure(row, pred)]
        if failure_types:
            failures.append({"id": pred["id"], "failure_types": failure_types, "gold": row, "pred": pred})

    report = summarize_failures(failures)
    dominant = select_dominant_failures(report)
    mitigation = build_mitigation_config(dominant)

    write_jsonl(out_dir / "predictions.jsonl", preds)
    write_jsonl(out_dir / "failures.jsonl", failures)
    (out_dir / "failure_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "mitigation_config.json").write_text(json.dumps(mitigation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Mitigation config:")
    print(json.dumps(mitigation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
