#!/usr/bin/env python
from __future__ import annotations

"""
Split the official dataset into logic-only and physics-only datasets.

Typical usage
-------------
python scripts/split_datasets.py \
  --input data/raw/all_official.jsonl \
  --out_dir data/split
"""

import argparse
import json
from pathlib import Path
from typing import Any

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Accept JSON array as a convenience (giữ nguyên logic từ script cũ).
    if text.startswith("["):
        data = json.loads(text)
        return data

    # Process standard JSONL
    return [json.loads(line) for line in text.splitlines() if line.strip()]

def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Split all_official.jsonl into physics and logic datasets.")
    parser.add_argument("--input", type=str, required=True, help="Path to input file (e.g., all_official.jsonl)")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the separated output files")
    
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading dataset from: {in_path}...")
    rows = read_jsonl(in_path)

    logic_rows = []
    physics_rows = []

    for row in rows:
        # Cách 1: Phân loại dựa trên trường 'type' nếu ban tổ chức có khai báo
        row_type = row.get("type")
        
        if row_type == "physics":
            physics_rows.append(row)
        elif row_type == "logic":
            logic_rows.append(row)
        else:
            # Cách 2 (Fallback): Phân loại dựa trên cấu trúc key đặc trưng của EXACT 2026
            if "premises-NL" in row or "premises-FOL" in row:
                logic_rows.append(row)
            else:
                physics_rows.append(row)

    # Đường dẫn output
    logic_path = out_dir / "logic_dataset.jsonl"
    physics_path = out_dir / "physics_dataset.jsonl"

    # Ghi file
    write_jsonl(logic_path, logic_rows)
    write_jsonl(physics_path, physics_rows)

    # In báo cáo
    print("\n=== Split Completed ===")
    print(f"Total rows processed : {len(rows)}")
    print(f"Logic rows saved   : {len(logic_rows)} -> {logic_path}")
    print(f"Physics rows saved : {len(physics_rows)} -> {physics_path}")

if __name__ == "__main__":
    main()