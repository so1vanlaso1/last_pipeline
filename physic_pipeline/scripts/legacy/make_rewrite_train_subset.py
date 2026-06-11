#!/usr/bin/env python
from __future__ import annotations

# Run from project root.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
from collections import Counter
from typing import Any

from exact_fama.utils.jsonl import read_jsonl, write_jsonl
from make_solver_eval_split import (
    expand_records,
    infer_type,
    select_rows,
    signature,
    summarize_split,
    load_exclusion_signatures,
)


def unique_usable(rows: list[dict[str, Any]], excluded: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in rows:
        if not r.get("question") or not str(r.get("answer", "")).strip():
            continue
        rr = dict(r)
        rr["type"] = infer_type(rr)
        sig = signature(rr)
        if sig in excluded or sig in seen:
            continue
        seen.add(sig)
        out.append(rr)
    return out


def sigs_from_file(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        print(f"[warn] split file not found for overlap report: {p}")
        return set()
    rows = expand_records(read_jsonl(p))
    sigs = set()
    for r in rows:
        if isinstance(r, dict) and r.get("question"):
            rr = dict(r)
            rr["type"] = infer_type(rr)
            sigs.add(signature(rr))
    return sigs


def take_by_select(rows: list[dict[str, Any]], target: int, seed: int, source_cap_ratio: float) -> list[dict[str, Any]]:
    if target <= 0 or not rows:
        return []
    return select_rows(rows, target=target, seed=seed, source_cap_ratio=source_cap_ratio)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Create a 60k LoRA rewrite training subset. By default it excludes blind only, "
            "so diagnostic may overlap if you use diagnostic strictly for solver debugging. "
            "Use --exclude_diagnostic if diagnostic will be used to evaluate LLM explanation quality."
        )
    )
    ap.add_argument("--input", default="data/raw/train.jsonl")
    ap.add_argument("--output", default="data/raw/train_rewrite_60k.jsonl")
    ap.add_argument("--report", default="artifacts/train_rewrite_60k_report.json")
    ap.add_argument("--target_size", type=int, default=60000)
    ap.add_argument("--blind", default="data/raw/solver_blind_10k.jsonl", help="Always exclude this final blind set.")
    ap.add_argument("--diagnostic", default="data/raw/solver_diagnostic_20k.jsonl", help="Only excluded when --exclude_diagnostic is set; otherwise only used for overlap report.")
    ap.add_argument("--exclude_diagnostic", action="store_true")
    ap.add_argument("--exclude", action="append", default=[], help="Additional JSONL/JSON file to exclude by signature; can repeat.")
    ap.add_argument("--physics_policy", choices=["all_available", "fraction", "none"], default="all_available")
    ap.add_argument("--physics_fraction", type=float, default=1.0, help="Used only when --physics_policy fraction.")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--source_cap_ratio", type=float, default=1.0)
    args = ap.parse_args()

    rows = expand_records(read_jsonl(args.input))

    excluded_files: list[str] = []
    if args.blind:
        excluded_files.append(args.blind)
    if args.exclude_diagnostic and args.diagnostic:
        excluded_files.append(args.diagnostic)
    excluded_files.extend(args.exclude or [])

    excluded = load_exclusion_signatures(excluded_files)
    usable = unique_usable(rows, excluded)

    physics = [r for r in usable if infer_type(r) == "physics"]
    logic = [r for r in usable if infer_type(r) == "logic"]

    if args.physics_policy == "none":
        train_physics = []
    elif args.physics_policy == "fraction":
        n = min(len(physics), max(0, round(len(physics) * args.physics_fraction)))
        train_physics = take_by_select(physics, n, args.seed + 11, args.source_cap_ratio)
    else:
        train_physics = take_by_select(physics, min(len(physics), args.target_size), args.seed + 11, args.source_cap_ratio)

    used = {signature(r) for r in train_physics}
    remaining_logic = [r for r in logic if signature(r) not in used]
    train_logic = take_by_select(remaining_logic, args.target_size - len(train_physics), args.seed + 22, args.source_cap_ratio)
    train = (train_physics + train_logic)[: args.target_size]

    train_sigs = {signature(r) for r in train}
    blind_sigs = sigs_from_file(args.blind)
    diag_sigs = sigs_from_file(args.diagnostic)

    write_jsonl(args.output, train)

    report = {
        "policy": {
            "blind_is_always_excluded": bool(args.blind),
            "diagnostic_excluded": bool(args.exclude_diagnostic),
            "interpretation": (
                "If diagnostic_excluded=false, do not use diagnostic to judge LLM explanation generalization; "
                "use it only for solver debugging. Blind remains clean for final testing."
            ),
            "physics_policy": args.physics_policy,
            "physics_fraction": args.physics_fraction,
        },
        "input": args.input,
        "output": args.output,
        "excluded_files": excluded_files,
        "excluded_signature_count": len(excluded),
        "available_after_exclusion": summarize_split("available_after_exclusion", usable),
        "train": summarize_split("train_rewrite", train),
        "overlap_report": {
            "train_overlap_with_blind": len(train_sigs & blind_sigs),
            "train_overlap_with_diagnostic": len(train_sigs & diag_sigs),
        },
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
