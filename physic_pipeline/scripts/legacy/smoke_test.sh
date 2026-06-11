#!/usr/bin/env bash
set -euo pipefail
export LLM_BACKEND=none
pytest -q
python scripts/run_inference.py --input data/sample/dev.jsonl --output artifacts/predictions.jsonl
python scripts/evaluate.py --gold data/sample/dev.jsonl --pred artifacts/predictions.jsonl --report artifacts/eval_report.json
cat artifacts/eval_report.json
