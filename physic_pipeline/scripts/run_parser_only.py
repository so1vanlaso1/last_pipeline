#!/usr/bin/env python
from __future__ import annotations

"""
Run only the zero-shot structured parser and dump its output.

This script does NOT call:
- logic solver
- physics solver
- verifier
- explanation rewriter
- evaluator

It is meant for inspecting whether the parser output is structurally and
semantically what you expect before letting it affect the solver path.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Allow running from repo root without installing first.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exact_fama.config import load_settings
from exact_fama.llm.qwen_client import QwenClient
from exact_fama.llm.structured_parser import StructuredParser
from exact_fama.router import route_task
from exact_fama.schemas import PredictRequest
from exact_fama.utils.jsonl import read_jsonl, write_jsonl


def _jsonable(obj: Any) -> Any:
    """Convert dataclasses / pydantic models / custom parse results into JSON."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}

    if hasattr(obj, "model_dump"):
        try:
            return _jsonable(obj.model_dump(mode="json"))
        except TypeError:
            return _jsonable(obj.model_dump())
        except Exception:
            pass

    if hasattr(obj, "__dataclass_fields__"):
        try:
            from dataclasses import asdict
            return _jsonable(asdict(obj))
        except Exception:
            pass

    attrs: dict[str, Any] = {}
    for name in [
        "ok", "accepted", "data", "warnings", "errors", "raw", "raw_text",
        "debug", "task_type", "validation_errors",
    ]:
        if hasattr(obj, name):
            try:
                attrs[name] = _jsonable(getattr(obj, name))
            except Exception:
                attrs[name] = f"<unserializable {name}>"
    return attrs if attrs else repr(obj)


def _data_payload(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    # Patched style may use {ok, accepted, data, warnings}; old style is direct dict.
    if isinstance(obj.get("data"), dict):
        return obj["data"]
    return obj


def _extract_parser_diagnostics(parsed_jsonable: Any) -> dict[str, Any]:
    """Best-effort summary that works with both old dict parser and patched parser."""
    out: dict[str, Any] = {
        "parser_ok": None,
        "parser_accepted": None,
        "warning_count": 0,
        "warnings": [],
        "fact_count": 0,
        "rule_count": 0,
        "query_count": 0,
        "option_count": 0,
    }

    obj = parsed_jsonable
    if not isinstance(obj, dict):
        return out

    if "ok" in obj:
        out["parser_ok"] = bool(obj.get("ok"))
    if "accepted" in obj:
        out["parser_accepted"] = bool(obj.get("accepted"))

    warnings = obj.get("warnings") or obj.get("errors") or []
    if isinstance(warnings, str):
        warnings = [warnings]
    if not isinstance(warnings, list):
        warnings = [repr(warnings)]
    out["warnings"] = [str(x) for x in warnings]
    out["warning_count"] = len(out["warnings"])

    data = _data_payload(obj)
    facts = data.get("facts") or []
    rules = data.get("rules") or []
    queries = data.get("queries", data.get("query", []))
    options = data.get("options") or []

    out["fact_count"] = len(facts) if isinstance(facts, list) else (1 if facts else 0)
    out["rule_count"] = len(rules) if isinstance(rules, list) else (1 if rules else 0)

    if isinstance(queries, dict):
        out["query_count"] = len(queries)
    elif isinstance(queries, list):
        out["query_count"] = len(queries)
    elif queries:
        out["query_count"] = 1

    if isinstance(options, dict):
        out["option_count"] = len(options)
    elif isinstance(options, list):
        out["option_count"] = len(options)
    elif options:
        out["option_count"] = 1

    return out


def _call_parser(parser: StructuredParser, request: PredictRequest, task_type: str) -> Any:
    """Call parser while staying compatible with old and patched signatures."""
    try:
        return parser.parse(request, task_type)  # current project signature
    except TypeError:
        return parser.parse(request)  # future-compatible fallback


def main() -> None:
    ap = argparse.ArgumentParser(description="Run only StructuredParser and dump raw parse output.")
    ap.add_argument("--input", required=True, help="Input JSONL / JSON array.")
    ap.add_argument("--output", required=True, help="Output parser-only JSONL.")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of rows; 0 means all.")
    ap.add_argument("--type", choices=["auto", "logic", "physics"], default="auto")
    ap.add_argument("--pretty", action="store_true", help="Also write pretty JSON next to JSONL.")
    ap.add_argument("--print", action="store_true", dest="print_rows", help="Print outputs to terminal.")
    ap.add_argument("--include_question", action="store_true")
    ap.add_argument("--include_premises", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    settings = load_settings()
    llm = QwenClient(settings.model)
    use_llm_parse = bool(settings.pipeline.get("use_llm_for_structured_parse", False))

    if not use_llm_parse:
        print(
            "WARNING: pipeline.use_llm_for_structured_parse is false. "
            "Use configs/eval_qwen_parse_rewrite.yaml for parser-only inspection.",
            file=sys.stderr,
        )

    parser = StructuredParser(llm, use_llm=use_llm_parse)

    print("=== Parser-only settings ===", flush=True)
    print("Input:", args.input, flush=True)
    print("Output:", args.output, flush=True)
    print("Rows:", len(rows), flush=True)
    print("EXACT_FAMA_CONFIG:", os.environ.get("EXACT_FAMA_CONFIG"), flush=True)
    print("LLM_BACKEND:", os.environ.get("LLM_BACKEND"), flush=True)
    print("MODEL_NAME:", os.environ.get("MODEL_NAME"), flush=True)
    print("Use LLM parse:", use_llm_parse, flush=True)
    print("============================", flush=True)

    out_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        req = PredictRequest.model_validate(row)
        task_type = args.type if args.type != "auto" else route_task(req)
        parsed = _call_parser(parser, req, task_type)
        parsed_jsonable = _jsonable(parsed)
        diagnostics = _extract_parser_diagnostics(parsed_jsonable)

        out: dict[str, Any] = {
            "id": row.get("id", str(idx - 1)),
            "task_type": task_type,
            "parser_diagnostics": diagnostics,
            "parser_output": parsed_jsonable,
        }
        if args.include_question:
            out["question"] = req.question
        if args.include_premises:
            out["premises-NL"] = req.premises_nl

        out_rows.append(out)

        if args.print_rows:
            print(json.dumps(out, ensure_ascii=False, indent=2))
            print("-" * 80)

    write_jsonl(args.output, out_rows)

    if args.pretty:
        pretty_path = Path(args.output).with_suffix(".pretty.json")
        pretty_path.parent.mkdir(parents=True, exist_ok=True)
        pretty_path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Pretty JSON:", pretty_path, flush=True)

    accepted = rejected = warning_total = total_facts = total_rules = 0
    for r in out_rows:
        d = r["parser_diagnostics"]
        if d.get("parser_accepted") is True or d.get("parser_ok") is True:
            accepted += 1
        elif d.get("parser_accepted") is False or d.get("parser_ok") is False:
            rejected += 1
        warning_total += int(d.get("warning_count") or 0)
        total_facts += int(d.get("fact_count") or 0)
        total_rules += int(d.get("rule_count") or 0)

    summary = {
        "rows": len(out_rows),
        "accepted_or_ok": accepted,
        "rejected_or_not_ok": rejected,
        "warning_total": warning_total,
        "total_facts": total_facts,
        "total_rules": total_rules,
        "avg_facts_per_row": round(total_facts / max(1, len(out_rows)), 3),
        "avg_rules_per_row": round(total_rules / max(1, len(out_rows)), 3),
    }

    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Parser-only summary ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("Summary:", summary_path, flush=True)


if __name__ == "__main__":
    main()
