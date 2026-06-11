# scripts/filter_correct_predictions.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from exact_fama.utils.jsonl import read_jsonl, write_jsonl

try:
    from evaluate import _answers_match_robust
except Exception:  # pragma: no cover
    from exact_fama.utils.answer_normalizer import answers_match as _project_answers_match

    def _answers_match_robust(gold, pred, numeric_tolerance=0.02):
        return _project_answers_match(gold, pred, numeric_tolerance)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out_gold", required=True)
    ap.add_argument("--numeric_tolerance", type=float, default=0.02)
    args = ap.parse_args()

    gold_rows = read_jsonl(args.gold)
    pred_rows = read_jsonl(args.pred)

    pred_by_id = {str(r.get("id", i)): r for i, r in enumerate(pred_rows)}

    kept = []
    for i, gold in enumerate(gold_rows):
        gid = str(gold.get("id", i))
        pred = pred_by_id.get(gid)
        if not pred:
            continue
        if _answers_match_robust(gold.get("answer", ""), pred.get("answer", ""), args.numeric_tolerance):
            kept.append(gold)

    write_jsonl(args.out_gold, kept)
    print(json.dumps({
        "input": len(gold_rows),
        "kept_correct": len(kept),
        "out_gold": args.out_gold,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
