#!/usr/bin/env python
from __future__ import annotations

# Allow running scripts directly from the repository without installing first.
import argparse
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

from exact_fama.config import load_settings
from exact_fama.llm.direct_answerer import DirectLLMAnswerer
from exact_fama.llm.qwen_client import QwenClient
from exact_fama.router import route_task
from exact_fama.schemas import PredictRequest
from exact_fama.utils.jsonl import read_jsonl, write_jsonl


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _make_crash_response(row: dict[str, Any], exc: Exception) -> dict[str, Any]:
    task_type = row.get("type") or row.get("task_type") or "unknown"
    answer = "Unknown" if task_type == "logic" else "Uncertain"

    return {
        "answer": answer,
        "unit": None,
        "explanation": "Direct LLM prediction failed because the model call or JSON parsing raised an exception.",
        "fol": None,
        "cot": [],
        "premises": row.get("premises-NL") or row.get("premises_nl") or [],
        "confidence": 0.0,
        "task_type": task_type,
        "used_modules": ["full_llm_direct_exception_guard"],
        "warnings": [
            f"FULL_LLM_EXCEPTION: prediction crashed: {type(exc).__name__}: {exc}"
        ],
        "debug": {
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def _print_progress(
    done: int,
    total: int,
    start_time: float,
    failures: int,
    current_id: str | None = None,
) -> None:
    now = time.perf_counter()
    elapsed = now - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = total - done
    eta = remaining / rate if rate > 0 else 0.0
    pct = (done / total * 100.0) if total > 0 else 100.0

    msg = (
        f"[{done}/{total}] {pct:6.2f}% | "
        f"elapsed={_format_duration(elapsed)} | "
        f"eta={_format_duration(eta)} | "
        f"speed={rate:.2f} sample/s | "
        f"failures={failures}"
    )

    if current_id:
        msg += f" | current_id={current_id}"

    print(msg, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full-LLM direct inference without parser, solver, verifier, or rewrite."
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output predictions JSONL")
    parser.add_argument("--log_every", type=int, default=10, help="Print progress every N samples")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N rows; 0 means all")
    parser.add_argument("--write_every", type=int, default=50, help="Write partial output every N samples")
    parser.add_argument("--no_exception_guard", action="store_true", help="Disable per-sample exception guard")
    parser.add_argument(
        "--no_raw_debug",
        action="store_true",
        help="Do not store raw LLM outputs in debug.raw_outputs",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=None,
        help="Override pipeline.full_llm_max_retries from config",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = read_jsonl(input_path)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    total = len(rows)

    print(f"Input: {input_path}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Rows: {total}", flush=True)

    settings = load_settings()
    llm = QwenClient(settings.model)

    config_retries = int(settings.pipeline.get("full_llm_max_retries", 1))
    max_retries = config_retries if args.max_retries is None else args.max_retries

    answerer = DirectLLMAnswerer(
        llm,
        max_retries=max_retries,
        include_debug_raw_output=not args.no_raw_debug,
    )

    print("=== Full-LLM direct settings ===", flush=True)
    print("EXACT_FAMA_CONFIG:", os.environ.get("EXACT_FAMA_CONFIG"), flush=True)
    print("LLM_BACKEND:", os.environ.get("LLM_BACKEND"), flush=True)
    print("MODEL_NAME:", os.environ.get("MODEL_NAME"), flush=True)
    print("LORA_ADAPTER_PATH:", os.environ.get("LORA_ADAPTER_PATH"), flush=True)
    print("Resolved backend:", _safe_get(llm, "backend"), flush=True)
    print("Resolved model:", _safe_get(llm, "model_name"), flush=True)
    print("Max retries:", max_retries, flush=True)
    print("No solver/parser/rewrite will be called.", flush=True)
    print("================================", flush=True)

    preds: list[dict[str, Any]] = []
    failures = 0
    start_time = time.perf_counter()

    _print_progress(0, total, start_time, failures)

    for i, row in enumerate(rows, start=1):
        row_id = str(row.get("id", i - 1))

        try:
            req = PredictRequest.model_validate(row)
            task_type = route_task(req)
            resp = answerer.predict(req, task_type=task_type).model_dump(mode="json")
        except Exception as exc:
            if args.no_exception_guard:
                raise
            failures += 1
            resp = _make_crash_response(row, exc)

        resp["id"] = row.get("id", str(i - 1))
        preds.append(resp)

        should_log = (
            i == 1
            or i == total
            or (args.log_every > 0 and i % args.log_every == 0)
        )
        if should_log:
            _print_progress(i, total, start_time, failures, current_id=row_id)

        should_write = (
            i == total
            or (args.write_every > 0 and i % args.write_every == 0)
        )
        if should_write:
            write_jsonl(output_path, preds)

    write_jsonl(output_path, preds)

    total_time = time.perf_counter() - start_time
    print("=== Finished full-LLM direct inference ===", flush=True)
    print(f"Wrote {len(preds)} predictions to {output_path}", flush=True)
    print(f"Total time: {_format_duration(total_time)}", flush=True)
    if total_time > 0:
        print(f"Average speed: {len(preds) / total_time:.2f} sample/s", flush=True)
    print(f"Failures: {failures}", flush=True)


if __name__ == "__main__":
    main()
