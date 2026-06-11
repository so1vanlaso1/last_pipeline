#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def read_json_array(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    data = json.loads(text)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")

    return [row for row in data if isinstance(row, dict)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick(row: dict[str, Any], names: list[str], default: Any = "") -> Any:
    normalized = {
        str(k).lower().strip().replace(" ", "_").replace("-", "_"): k
        for k in row.keys()
    }

    for name in names:
        key = name.lower().strip().replace(" ", "_").replace("-", "_")

        if key in normalized:
            value = row[normalized[key]]

            if value not in (None, ""):
                return value

    return default


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def stringify_or_empty(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return json.dumps(value, ensure_ascii=False)


def split_logic_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split official Type-1 logic records into one row per question.

    Important:
    - Do NOT use the official `idx` field as the row id.
    - In this dataset, `idx` may be a list of premise indices and is not unique.
    - We preserve it as `original_idx` only for metadata/debugging.
    """

    out: list[dict[str, Any]] = []

    for ridx, rec in enumerate(records):
        source_record_id = f"logic-{ridx:04d}"
        original_idx = rec.get("idx")

        premises_nl = (
            rec.get("premises-NL")
            or rec.get("premises_nl")
            or rec.get("premises")
            or []
        )

        premises_fol = (
            rec.get("premises-FOL")
            or rec.get("premises_fol")
            or rec.get("fol")
            or []
        )

        questions = as_list(rec.get("questions") or rec.get("question") or [])
        answers = as_list(rec.get("answers") or rec.get("answer") or [])
        explanations = as_list(rec.get("explanation") or rec.get("explanations") or [])

        premises_nl_list = [
            stringify_or_empty(p).strip()
            for p in as_list(premises_nl)
            if stringify_or_empty(p).strip()
        ]

        premises_fol_list = [
            stringify_or_empty(p).strip()
            for p in as_list(premises_fol)
            if stringify_or_empty(p).strip()
        ]

        for qidx, question in enumerate(questions):
            question_text = stringify_or_empty(question).strip()

            if not question_text:
                continue

            answer = stringify_or_empty(answers[qidx]).strip() if qidx < len(answers) else ""

            explanation = (
                stringify_or_empty(explanations[qidx]).strip()
                if qidx < len(explanations)
                else ""
            )

            item = {
                "id": f"{source_record_id}-q{qidx}",
                "source_record_id": source_record_id,
                "original_idx": original_idx,
                "question_index": qidx,
                "type": "logic",
                "premises-NL": premises_nl_list,
                "premises-FOL": premises_fol_list,
                "question": question_text,
                "answer": answer,
                "explanation": explanation,
            }

            out.append(item)

    return out


def read_physics_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            question = pick(
                row,
                [
                    "question",
                    "Question",
                    "problem",
                    "Problem",
                    "text",
                    "prompt",
                    "stem",
                    "body",
                ],
                "",
            )

            answer = pick(
                row,
                [
                    "answer",
                    "Answer",
                    "final_answer",
                    "Final Answer",
                    "target",
                    "output",
                    "gold_answer",
                    "ground_truth",
                ],
                "",
            )

            unit = pick(
                row,
                [
                    "unit",
                    "Unit",
                    "units",
                    "answer_unit",
                    "unit_text",
                ],
                None,
            )

            cot = pick(
                row,
                [
                    "cot",
                    "CoT",
                    "solution",
                    "Solution",
                    "reasoning",
                    "Reasoning",
                    "explanation",
                    "Explanation",
                    "derivation",
                ],
                "",
            )

            rid = pick(
                row,
                [
                    "id",
                    "ID",
                    "idx",
                    "sample_id",
                ],
                f"physics-{i:05d}",
            )

            question_text = stringify_or_empty(question).strip()

            if not question_text:
                continue

            # Physics IDs such as TD401 are expected to be unique, but we still
            # prefix only when missing to preserve official IDs.
            source_record_id = stringify_or_empty(rid).strip() or f"physics-{i:05d}"

            item = {
                "id": source_record_id,
                "source_record_id": source_record_id,
                "type": "physics",
                "question": question_text,
                "answer": stringify_or_empty(answer).strip(),
                "unit": stringify_or_empty(unit).strip() if unit not in (None, "") else None,
                "cot": stringify_or_empty(cot).strip(),
                "explanation": stringify_or_empty(cot).strip(),
            }

            rows.append(item)

    return rows


def group_split_logic(
    logic_rows: list[dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split logic rows by source_record_id to avoid premise leakage.

    All questions from the same original logic record must stay in the same split.
    """

    groups: dict[str, list[dict[str, Any]]] = {}

    for row in logic_rows:
        groups.setdefault(str(row["source_record_id"]), []).append(row)

    group_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n = len(group_ids)
    n_train = round(n * train_ratio)
    n_dev = round(n * dev_ratio)

    train_ids = set(group_ids[:n_train])
    dev_ids = set(group_ids[n_train:n_train + n_dev])
    blind_ids = set(group_ids[n_train + n_dev:])

    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    blind: list[dict[str, Any]] = []

    for gid, items in groups.items():
        if gid in train_ids:
            train.extend(items)
        elif gid in dev_ids:
            dev.extend(items)
        elif gid in blind_ids:
            blind.extend(items)
        else:
            raise RuntimeError(f"Unexpected group id during split: {gid}")

    return train, dev, blind


def split_rows(
    rows: list[dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows)

    rng = random.Random(seed)
    rng.shuffle(rows)

    n = len(rows)
    n_train = round(n * train_ratio)
    n_dev = round(n * dev_ratio)

    train = rows[:n_train]
    dev = rows[n_train:n_train + n_dev]
    blind = rows[n_train + n_dev:]

    return train, dev, blind


def make_smoke(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Create a small mixed smoke set from dev rows."""

    rng = random.Random(seed)

    logic = [r for r in rows if r.get("type") == "logic"]
    physics = [r for r in rows if r.get("type") == "physics"]

    selected: list[dict[str, Any]] = []

    target_logic = n // 2
    target_physics = n - target_logic

    if logic:
        selected.extend(rng.sample(logic, min(target_logic, len(logic))))

    if physics:
        selected.extend(rng.sample(physics, min(target_physics, len(physics))))

    selected_ids = {str(r["id"]) for r in selected}
    remaining = [r for r in rows if str(r["id"]) not in selected_ids]

    if len(selected) < n and remaining:
        selected.extend(rng.sample(remaining, min(n - len(selected), len(remaining))))

    rng.shuffle(selected)

    return selected


def duplicate_ids(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row.get("id")) for row in rows)
    return {k: v for k, v in counter.items() if v > 1}


def assert_no_duplicate_ids(named_splits: dict[str, list[dict[str, Any]]]) -> None:
    duplicates = {
        name: duplicate_ids(rows)
        for name, rows in named_splits.items()
    }

    duplicates = {
        name: dups
        for name, dups in duplicates.items()
        if dups
    }

    if duplicates:
        preview = json.dumps(duplicates, ensure_ascii=False, indent=2)
        raise RuntimeError(
            "Duplicate ids detected. This will break evaluation because predictions are matched by id.\n"
            f"{preview[:4000]}"
        )


def summarize_type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row.get("type", "unknown")) for row in rows)
    return dict(counter)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare official EXACT logic + physics datasets into project JSONL splits."
    )

    parser.add_argument(
        "--logic",
        required=True,
        help="Path to Logic_Based_Educational_Queries.json",
    )

    parser.add_argument(
        "--physics",
        required=True,
        help="Path to Physics_Problems_Text_Only.csv",
    )

    parser.add_argument(
        "--out_dir",
        default="data/raw",
        help="Output directory for train/dev/blind/smoke JSONL files.",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--smoke_size",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    if args.train_ratio <= 0 or args.dev_ratio < 0:
        raise ValueError("train_ratio must be > 0 and dev_ratio must be >= 0.")

    if args.train_ratio + args.dev_ratio >= 1.0:
        raise ValueError("train_ratio + dev_ratio must be < 1.0 so blind split is non-empty.")

    logic_path = Path(args.logic)
    physics_path = Path(args.physics)
    out_dir = Path(args.out_dir)

    if not logic_path.exists():
        raise FileNotFoundError(f"Logic file not found: {logic_path}")

    if not physics_path.exists():
        raise FileNotFoundError(f"Physics file not found: {physics_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    logic_raw = read_json_array(logic_path)
    logic_rows = split_logic_records(logic_raw)

    physics_rows = read_physics_csv(physics_path)

    logic_train, logic_dev, logic_blind = group_split_logic(
        logic_rows,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    physics_train, physics_dev, physics_blind = split_rows(
        physics_rows,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed + 1,
    )

    train = logic_train + physics_train
    dev = logic_dev + physics_dev
    blind = logic_blind + physics_blind

    rng_train = random.Random(args.seed + 10)
    rng_dev = random.Random(args.seed + 11)
    rng_blind = random.Random(args.seed + 12)

    rng_train.shuffle(train)
    rng_dev.shuffle(dev)
    rng_blind.shuffle(blind)

    all_official = train + dev + blind
    smoke = make_smoke(dev, args.smoke_size, args.seed + 20)

    assert_no_duplicate_ids(
        {
            "train": train,
            "dev": dev,
            "blind": blind,
            "smoke": smoke,
            "all_official": all_official,
        }
    )

    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"
    blind_path = out_dir / "blind.jsonl"
    smoke_path = out_dir / "smoke_100.jsonl"
    all_path = out_dir / "all_official.jsonl"
    report_path = out_dir / "official_split_report.json"

    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    write_jsonl(blind_path, blind)
    write_jsonl(smoke_path, smoke)
    write_jsonl(all_path, all_official)

    report = {
        "logic_records_raw": len(logic_raw),
        "logic_questions_expanded": len(logic_rows),
        "physics_rows": len(physics_rows),
        "total_rows": len(all_official),
        "train": {
            "rows": len(train),
            "type_counts": summarize_type_counts(train),
        },
        "dev": {
            "rows": len(dev),
            "type_counts": summarize_type_counts(dev),
        },
        "blind": {
            "rows": len(blind),
            "type_counts": summarize_type_counts(blind),
        },
        "smoke": {
            "rows": len(smoke),
            "type_counts": summarize_type_counts(smoke),
        },
        "split_config": {
            "train_ratio": args.train_ratio,
            "dev_ratio": args.dev_ratio,
            "blind_ratio_approx": 1.0 - args.train_ratio - args.dev_ratio,
            "seed": args.seed,
            "smoke_size": args.smoke_size,
            "logic_split_policy": "grouped_by_source_record_id",
            "physics_split_policy": "row_level_random",
        },
        "outputs": {
            "train": str(train_path),
            "dev": str(dev_path),
            "blind": str(blind_path),
            "smoke_100": str(smoke_path),
            "all_official": str(all_path),
            "report": str(report_path),
        },
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()