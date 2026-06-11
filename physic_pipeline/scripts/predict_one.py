#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exact_fama.pipeline import ExactFamaPipeline
from exact_fama.schemas import PredictRequest
from exact_fama.utils.jsonl import read_jsonl


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)

    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"

    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60

    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"

    return f"{m:02d}:{s:02d}"


def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _json_print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False), flush=True)


def load_one(args: argparse.Namespace) -> dict[str, Any]:
    if args.json:
        return json.loads(args.json)

    if args.file:
        path = Path(args.file)

        if not path.exists():
            raise SystemExit(f"File not found: {path}")

        if path.suffix.lower() == ".jsonl":
            rows = read_jsonl(path)

            if not rows:
                raise SystemExit(f"No rows found in {path}")

            if args.id:
                for row in rows:
                    if str(row.get("id")) == args.id:
                        return row

                raise SystemExit(f"Could not find id={args.id} in {path}")

            if args.index < 0 or args.index >= len(rows):
                raise SystemExit(
                    f"Index out of range: index={args.index}, rows={len(rows)}"
                )

            return rows[args.index]

        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, list):
            if args.id:
                for row in data:
                    if isinstance(row, dict) and str(row.get("id")) == args.id:
                        return row

                raise SystemExit(f"Could not find id={args.id} in {path}")

            if args.index < 0 or args.index >= len(data):
                raise SystemExit(
                    f"Index out of range: index={args.index}, rows={len(data)}"
                )

            row = data[args.index]

            if not isinstance(row, dict):
                raise SystemExit(f"Selected row is not a JSON object: index={args.index}")

            return row

        if isinstance(data, dict):
            return data

        raise SystemExit(f"Unsupported JSON format: {path}")

    raise SystemExit("Use --json or --file")


def print_pipeline_info(pipeline: ExactFamaPipeline) -> None:
    print("=== PIPELINE SETTINGS ===", flush=True)
    print("EXACT_FAMA_CONFIG:", os.environ.get("EXACT_FAMA_CONFIG"), flush=True)
    print("LLM_BACKEND:", os.environ.get("LLM_BACKEND"), flush=True)
    print("MODEL_NAME:", os.environ.get("MODEL_NAME"), flush=True)
    print("LORA_ADAPTER_PATH:", os.environ.get("LORA_ADAPTER_PATH"), flush=True)
    print("LLM_TEMPERATURE:", os.environ.get("LLM_TEMPERATURE"), flush=True)
    print("LLM_MAX_NEW_TOKENS:", os.environ.get("LLM_MAX_NEW_TOKENS"), flush=True)
    print("EXACT_USE_PROVIDED_FOL:", os.environ.get("EXACT_USE_PROVIDED_FOL"), flush=True)

    llm = _safe_get(pipeline, "llm")
    parser = _safe_get(pipeline, "parser")
    explainer = _safe_get(pipeline, "explainer")

    print("Resolved backend:", _safe_get(llm, "backend"), flush=True)
    print("Resolved model:", _safe_get(llm, "model_name"), flush=True)
    print("Use LLM parse:", _safe_get(parser, "use_llm"), flush=True)
    print("Use LLM rewrite:", _safe_get(explainer, "use_llm"), flush=True)
    print("=========================", flush=True)


def print_reasoning_trace(pred: dict[str, Any]) -> None:
    print("\n=== PUBLIC REASONING TRACE ===", flush=True)

    print("\n--- answer/unit/confidence ---", flush=True)
    _json_print(
        {
            "answer": pred.get("answer"),
            "unit": pred.get("unit"),
            "confidence": pred.get("confidence"),
            "task_type": pred.get("task_type"),
        }
    )

    print("\n--- explanation ---", flush=True)
    print(pred.get("explanation") or "", flush=True)

    cot = pred.get("cot") or []
    print("\n--- cot / solver reasoning trace ---", flush=True)

    if cot:
        for i, step in enumerate(cot, start=1):
            print(f"{i}. {step}", flush=True)
    else:
        print("(empty)", flush=True)

    premises = pred.get("premises") or []
    print("\n--- premises used ---", flush=True)

    if premises:
        for i, premise in enumerate(premises, start=1):
            print(f"{i}. {premise}", flush=True)
    else:
        print("(empty)", flush=True)

    warnings = pred.get("warnings") or []
    print("\n--- warnings ---", flush=True)

    if warnings:
        for w in warnings:
            print(f"- {w}", flush=True)
    else:
        print("(none)", flush=True)

    debug = pred.get("debug") or {}

    llm_flags = {
        "llm_rewrite_enabled": debug.get("llm_rewrite_enabled"),
        "llm_rewrite_changed": debug.get("llm_rewrite_changed"),
        "llm_rewrite_failed": debug.get("llm_rewrite_failed"),
        "llm_parse_enabled": debug.get("llm_parse_enabled"),
    }

    print("\n--- llm status ---", flush=True)
    _json_print(llm_flags)

    proof_steps = debug.get("proof_steps") or []
    print("\n--- proof steps ---", flush=True)

    if proof_steps:
        _json_print(proof_steps)
    else:
        print("(empty)", flush=True)

    formula = debug.get("formula")
    quantities = debug.get("quantities")

    if formula or quantities:
        print("\n--- physics formula/quantities ---", flush=True)
        _json_print(
            {
                "formula": formula,
                "quantities": quantities,
            }
        )

    print("===============================", flush=True)


def make_crash_prediction(row: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "answer": "Uncertain",
        "unit": None,
        "explanation": "Prediction failed because the pipeline raised an exception.",
        "fol": None,
        "cot": [],
        "premises": row.get("premises-NL") or row.get("premises_nl") or [],
        "confidence": 0.0,
        "task_type": row.get("type", "unknown"),
        "used_modules": ["predict_one_exception_guard"],
        "warnings": [
            f"OUTPUT_SCHEMA_ERROR: prediction crashed: {type(exc).__name__}: {exc}"
        ],
        "debug": {
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run EXACT-FAMA prediction for one sample.")

    ap.add_argument("--json", default="", help="Inline JSON object request.")
    ap.add_argument("--file", default="", help="JSON/JSONL file containing one or more samples.")
    ap.add_argument("--index", type=int, default=0, help="Row index when reading a JSON array/JSONL file.")
    ap.add_argument("--id", default="", help="Row id when reading a JSON array/JSONL file.")
    ap.add_argument("--output", default="", help="Optional path to save prediction JSON.")

    ap.add_argument("--show_input", action="store_true", help="Print selected input row.")
    ap.add_argument("--show_pipeline", action="store_true", help="Print pipeline settings.")
    ap.add_argument("--show_trace", action="store_true", help="Print public reasoning trace: cot, proof_steps, warnings, LLM status.")
    ap.add_argument("--show_debug", action="store_true", help="Print full debug field.")
    ap.add_argument("--no_exception_guard", action="store_true", help="Raise exception instead of returning crash prediction.")

    args = ap.parse_args()

    total_start = time.perf_counter()

    load_start = time.perf_counter()
    row = load_one(args)
    load_time = time.perf_counter() - load_start

    row_id = row.get("id", f"index-{args.index}")

    print("=== PREDICT ONE ===", flush=True)
    print(f"Selected id: {row_id}", flush=True)
    print(f"Input load time: {_format_duration(load_time)}", flush=True)

    if args.show_input:
        print("\n=== INPUT ===", flush=True)
        _json_print(row)

    print("\nInitializing pipeline...", flush=True)
    pipeline_start = time.perf_counter()
    pipeline = ExactFamaPipeline()
    pipeline_time = time.perf_counter() - pipeline_start
    print(f"Pipeline init time: {_format_duration(pipeline_time)}", flush=True)

    if args.show_pipeline:
        print_pipeline_info(pipeline)

    print("\nRunning prediction...", flush=True)
    print("Note: if HF backend is enabled, the first prediction may load model weights here.", flush=True)

    predict_start = time.perf_counter()

    try:
        req = PredictRequest.model_validate(row)
        pred = pipeline.predict(req).model_dump(mode="json")

    except Exception as exc:
        if args.no_exception_guard:
            raise

        pred = make_crash_prediction(row, exc)

    predict_time = time.perf_counter() - predict_start

    if row.get("id"):
        pred["id"] = row["id"]

    print(f"Prediction time: {_format_duration(predict_time)}", flush=True)

    total_time = time.perf_counter() - total_start
    pred.setdefault("debug", {})
    pred["debug"]["predict_one_timing"] = {
        "input_load_seconds": load_time,
        "pipeline_init_seconds": pipeline_time,
        "prediction_seconds": predict_time,
        "total_seconds": total_time,
    }

    print("\n=== PREDICTION ===", flush=True)
    _json_print(pred)

    if args.show_trace:
        print_reasoning_trace(pred)

    if args.show_debug:
        print("\n=== FULL DEBUG ===", flush=True)
        _json_print(pred.get("debug") or {})

    if args.output:
        save_start = time.perf_counter()
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(pred, indent=2, ensure_ascii=False), encoding="utf-8")
        save_time = time.perf_counter() - save_start
        print(f"\nSaved to: {out}", flush=True)
        print(f"Save time: {_format_duration(save_time)}", flush=True)

    print("\n=== TIMING SUMMARY ===", flush=True)
    print(f"Input load:    {_format_duration(load_time)}", flush=True)
    print(f"Pipeline init: {_format_duration(pipeline_time)}", flush=True)
    print(f"Prediction:    {_format_duration(predict_time)}", flush=True)
    print(f"Total:         {_format_duration(total_time)}", flush=True)


if __name__ == "__main__":
    main()