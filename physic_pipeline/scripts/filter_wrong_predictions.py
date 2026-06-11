# scripts/filter_wrong_predictions.py
from __future__ import annotations

import argparse
import json

from exact_fama.utils.jsonl import read_jsonl, write_jsonl

try:
    from evaluate import _answers_match_robust
except Exception:  # pragma: no cover
    from exact_fama.utils.answer_normalizer import answers_match as _project_answers_match

    def _answers_match_robust(gold, pred, numeric_tolerance=0.02):
        return _project_answers_match(gold, pred, numeric_tolerance)


def main():
    ap = argparse.ArgumentParser(description="Filter gold rows whose predicted answers are wrong.")
    ap.add_argument("--gold", required=True, help="Path to gold JSONL file.")
    ap.add_argument("--pred", required=True, help="Path to prediction JSONL file.")
    ap.add_argument("--out_gold", required=True, help="Output JSONL containing only wrong gold rows.")
    ap.add_argument("--out_pred", default=None, help="Optional output JSONL containing prediction rows for wrong cases.")
    ap.add_argument("--out_mismatches", default=None, help="Optional output JSONL with id/question/gold/pred/warnings for wrong cases.")
    ap.add_argument("--skip_missing_pred", action="store_true", help="If set, gold rows with no matching prediction are skipped instead of counted as wrong.")
    ap.add_argument("--numeric_tolerance", type=float, default=0.02)
    args = ap.parse_args()

    gold_rows = read_jsonl(args.gold)
    pred_rows = read_jsonl(args.pred)
    pred_by_id = {str(r.get("id", i)): r for i, r in enumerate(pred_rows)}

    wrong_gold = []
    wrong_pred = []
    mismatches = []
    missing_pred = 0

    for i, gold in enumerate(gold_rows):
        gid = str(gold.get("id", i))
        pred = pred_by_id.get(gid)

        if pred is None:
            missing_pred += 1
            if args.skip_missing_pred:
                continue
            wrong_gold.append(gold)
            mismatches.append({
                "id": gid,
                "question": gold.get("question", ""),
                "gold": gold.get("answer", ""),
                "pred": None,
                "warnings": ["MISSING_PREDICTION"],
            })
            continue

        gold_answer = gold.get("answer", "")
        pred_answer = pred.get("answer", "")

        if not _answers_match_robust(gold_answer, pred_answer, args.numeric_tolerance):
            wrong_gold.append(gold)
            wrong_pred.append(pred)
            mismatches.append({
                "id": gid,
                "question": gold.get("question", pred.get("question", "")),
                "gold": gold_answer,
                "pred": pred_answer,
                "warnings": pred.get("warnings", []),
            })

    write_jsonl(args.out_gold, wrong_gold)
    if args.out_pred:
        write_jsonl(args.out_pred, wrong_pred)
    if args.out_mismatches:
        write_jsonl(args.out_mismatches, mismatches)

    print(json.dumps({
        "input_gold": len(gold_rows),
        "input_pred": len(pred_rows),
        "wrong": len(wrong_gold),
        "missing_pred": missing_pred,
        "out_gold": args.out_gold,
        "out_pred": args.out_pred,
        "out_mismatches": args.out_mismatches,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
