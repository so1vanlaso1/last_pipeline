#!/usr/bin/env python
from __future__ import annotations

"""
Create stable logic-only and physics-only smoke benchmarks.

Why this script exists
----------------------
scripts/prepare_official_data.py shuffles rows and takes the first N examples
for smoke_100. If the source dataset changes after preprocessing, the smoke set
also changes. That makes old/new accuracy numbers incomparable.

This script samples by a stable hash of each row id, so the chosen examples do
not depend on input row order. Use it after prepare_official_data.py has created
all_official.jsonl.

Typical usage
-------------
python scripts/make_fixed_smoke_benchmarks.py ^
  --input data/raw/all_official.jsonl ^
  --out_dir data/eval ^
  --logic_size 100 ^
  --physics_size 100 ^
  --seed 2026

Then evaluate modes on the same fixed files:
  data/eval/fixed_smoke_logic_100.jsonl
  data/eval/fixed_smoke_physics_100.jsonl
  data/eval/fixed_smoke_mixed_200.jsonl
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Accept JSON array as a convenience.
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array or JSONL in {p}")
        return [row for row in data if isinstance(row, dict)]

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"Line {line_no} in {p} is not a JSON object")
        rows.append(obj)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_id(row: dict[str, Any], index: int) -> str:
    rid = str(row.get("id") or "").strip()
    if rid:
        return rid
    # Fallback should almost never be used. It keeps the script from crashing
    # on ad-hoc files, but official prepared rows should always have id.
    return f"__row_{index:06d}"


def infer_task_type(row: dict[str, Any]) -> str:
    explicit = str(row.get("type") or row.get("task_type") or "").strip().lower()
    if explicit in {"logic", "physics"}:
        return explicit

    rid = str(row.get("id") or "").strip().lower()
    if rid.startswith("logic-"):
        return "logic"

    # Prepared logic rows normally contain premises-NL. Physics rows normally do not.
    premises = row.get("premises-NL") or row.get("premises_nl") or []
    if premises:
        return "logic"

    return "physics"


def has_nonempty_answer(row: dict[str, Any]) -> bool:
    return str(row.get("answer") or row.get("gold") or "").strip() != ""


def stable_hash_int(seed: int, key: str) -> int:
    payload = f"{seed}|{key}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def stable_sample(
    rows: list[dict[str, Any]],
    *,
    size: int,
    seed: int,
    label: str,
    require_answer: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: set[str] = set()
    kept: list[tuple[int, str, dict[str, Any]]] = []
    duplicate_ids: list[str] = []
    answer_missing_ids: list[str] = []

    for i, row in enumerate(rows):
        rid = row_id(row, i)
        if rid in seen:
            duplicate_ids.append(rid)
            continue
        seen.add(rid)

        if require_answer and not has_nonempty_answer(row):
            answer_missing_ids.append(rid)
            continue

        h = stable_hash_int(seed, rid)
        kept.append((h, rid, row))

    kept.sort(key=lambda x: (x[0], x[1]))

    requested_size = size
    if size <= 0 or size > len(kept):
        size = len(kept)

    sampled = [row for _, _, row in kept[:size]]

    report = {
        "label": label,
        "requested_size": requested_size,
        "actual_size": len(sampled),
        "candidate_rows_after_filters": len(kept),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids_preview": duplicate_ids[:20],
        "answer_missing_count": len(answer_missing_ids),
        "answer_missing_ids_preview": answer_missing_ids[:20],
        "first_ids": [str(r.get("id", "")) for r in sampled[:10]],
        "last_ids": [str(r.get("id", "")) for r in sampled[-10:]],
        "checksum_sha256": rows_checksum(sampled),
    }

    return sampled, report


def rows_checksum(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        # Include full row content, not only id. This catches accidental data changes.
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def id_set(rows: list[dict[str, Any]]) -> set[str]:
    return {str(r.get("id") or "").strip() for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create stable fixed smoke benchmarks split by task type."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Prepared JSONL, usually data/raw/all_official.jsonl",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory, e.g. data/eval",
    )
    parser.add_argument(
        "--logic_size",
        type=int,
        default=100,
        help="Number of logic rows. Use 0 or negative to keep all logic rows.",
    )
    parser.add_argument(
        "--physics_size",
        type=int,
        default=100,
        help="Number of physics rows. Use 0 or negative to keep all physics rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Stable hash seed. Same seed + same ids => same benchmark.",
    )
    parser.add_argument(
        "--allow_missing_answer",
        action="store_true",
        help="Allow rows with empty answer. Default rejects them for scored benchmarks.",
    )
    parser.add_argument(
        "--prefix",
        default="fixed_smoke",
        help="Output filename prefix. Default: fixed_smoke",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    groups: dict[str, list[dict[str, Any]]] = {"logic": [], "physics": []}
    unknown_rows: list[str] = []

    for i, row in enumerate(rows):
        t = infer_task_type(row)
        if t in groups:
            groups[t].append(row)
        else:
            unknown_rows.append(row_id(row, i))

    require_answer = not args.allow_missing_answer

    logic_rows, logic_report = stable_sample(
        groups["logic"],
        size=args.logic_size,
        seed=args.seed,
        label="logic",
        require_answer=require_answer,
    )
    physics_rows, physics_report = stable_sample(
        groups["physics"],
        size=args.physics_size,
        seed=args.seed,
        label="physics",
        require_answer=require_answer,
    )

    # Mixed benchmark keeps task groups separate but combines them in a deterministic order.
    # This avoids random interleaving changes when one group changes.
    mixed_rows = logic_rows + physics_rows

    logic_path = out_dir / f"{args.prefix}_logic_{len(logic_rows)}.jsonl"
    physics_path = out_dir / f"{args.prefix}_physics_{len(physics_rows)}.jsonl"
    mixed_path = out_dir / f"{args.prefix}_mixed_{len(mixed_rows)}.jsonl"
    report_path = out_dir / f"{args.prefix}_benchmark_report.json"

    write_jsonl(logic_path, logic_rows)
    write_jsonl(physics_path, physics_rows)
    write_jsonl(mixed_path, mixed_rows)

    overlap = sorted(id_set(logic_rows) & id_set(physics_rows))

    source_counter = Counter(infer_task_type(r) for r in rows)

    report = {
        "input": str(input_path),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "require_answer": require_answer,
        "source_total_rows": len(rows),
        "source_task_counts": dict(source_counter),
        "unknown_row_count": len(unknown_rows),
        "unknown_ids_preview": unknown_rows[:20],
        "outputs": {
            "logic": str(logic_path),
            "physics": str(physics_path),
            "mixed": str(mixed_path),
            "report": str(report_path),
        },
        "logic": logic_report,
        "physics": physics_report,
        "mixed": {
            "actual_size": len(mixed_rows),
            "logic_size": len(logic_rows),
            "physics_size": len(physics_rows),
            "checksum_sha256": rows_checksum(mixed_rows),
            "first_ids": [str(r.get("id", "")) for r in mixed_rows[:10]],
            "last_ids": [str(r.get("id", "")) for r in mixed_rows[-10:]],
        },
        "logic_physics_id_overlap_count": len(overlap),
        "logic_physics_id_overlap_preview": overlap[:20],
    }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== Fixed benchmark files created ===")
    print(f"Logic:   {logic_path} ({len(logic_rows)} rows)")
    print(f"Physics: {physics_path} ({len(physics_rows)} rows)")
    print(f"Mixed:   {mixed_path} ({len(mixed_rows)} rows)")
    print(f"Report:  {report_path}")
    print("")
    print("First logic ids:", logic_report["first_ids"][:5])
    print("First physics ids:", physics_report["first_ids"][:5])
    print("")
    print("Use these fixed files for solver-only, rewrite-only, and parse+rewrite comparisons.")


if __name__ == "__main__":
    main()
