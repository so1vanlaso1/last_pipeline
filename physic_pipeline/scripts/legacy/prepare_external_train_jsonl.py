from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

random.seed(42)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stringify(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return "\n".join(str(i) for i in x)
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def pick(row: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    lower_map = {str(k).lower().replace(" ", "_").replace("-", "_"): k for k in row.keys()}
    for key in keys:
        norm = key.lower().replace(" ", "_").replace("-", "_")
        if norm in lower_map:
            v = row[lower_map[norm]]
            if v not in (None, ""):
                return v
    return default


def auto_pick_by_keywords(row: dict[str, Any], include: list[str], exclude: list[str] | None = None) -> Any:
    exclude = exclude or []
    for k, v in row.items():
        lk = str(k).lower()
        if any(word in lk for word in include) and not any(word in lk for word in exclude):
            if v not in (None, ""):
                return v
    return ""


def load_rows(path: Path) -> list[dict[str, Any]]:
    ds = load_from_disk(str(path))
    rows: list[dict[str, Any]] = []
    if hasattr(ds, "keys"):
        for split in ds.keys():
            part = ds[split]
            for i in range(len(part)):
                r = dict(part[i])
                r["_split"] = split
                rows.append(r)
    else:
        for i in range(len(ds)):
            rows.append(dict(ds[i]))
    return rows


def convert_logic(row: dict[str, Any], idx: int, prefix: str) -> dict[str, Any] | None:
    premises = pick(row, ["premises", "context", "theory", "facts", "rules", "premises-NL", "premises_nl"], "")
    question = pick(row, ["question", "hypothesis", "query", "conclusion"], "")
    answer = pick(row, ["answer", "label", "gold_label"], "")
    explanation = pick(row, ["explanation", "reasoning", "rationale", "proof", "proofs"], "")

    if not question or answer == "":
        return None

    if isinstance(premises, str):
        premises_list = [p.strip() for p in premises.replace(". ", ".\n").split("\n") if p.strip()]
    elif isinstance(premises, list):
        premises_list = [stringify(p) for p in premises]
    else:
        premises_list = [stringify(premises)]

    return {
        "id": f"{prefix}-{idx}",
        "type": "logic",
        "premises-NL": premises_list,
        "question": stringify(question),
        "answer": stringify(answer),
        "explanation": stringify(explanation) or f"The answer follows from the given premises. Label: {answer}.",
    }


def convert_physics(row: dict[str, Any], idx: int, prefix: str) -> dict[str, Any] | None:
    question = pick(row, [
        "question", "Question", "problem", "Problem", "input", "prompt",
        "problem_text", "question_text", "query", "text", "stem",
        "question_stem", "body"
    ], "")

    answer = pick(row, [
        "answer", "Answer", "final_answer", "Final Answer", "target", "output",
        "gold_answer", "correct_answer", "solution_answer", "final",
        "label", "gt_answer", "gt", "ground_truth"
    ], "")

    unit = pick(row, ["unit", "Unit", "answer_unit", "units", "Units", "unit_text"], None)

    cot = pick(row, [
        "cot", "CoT", "solution", "Solution", "rationale", "explanation",
        "Explanation", "reasoning", "Reasoning", "analysis", "derivation",
        "solution_text", "answer_explanation"
    ], "")

    if not question:
        question = auto_pick_by_keywords(row, ["question", "problem", "prompt", "stem", "text"], ["image", "figure"])
    if answer == "":
        answer = auto_pick_by_keywords(row, ["answer", "target", "label", "ground", "truth", "final"], ["image"])
    if not cot:
        cot = auto_pick_by_keywords(row, ["solution", "explanation", "rationale", "reason", "analysis"], ["image"])

    choices = pick(row, ["choices", "options", "Options"], "")
    if choices and question:
        question = stringify(question) + "\nChoices/options:\n" + stringify(choices)

    if not question or answer == "":
        return None

    q_str = stringify(question)
    a_str = stringify(answer)

    # Bỏ sample quá ngắn hoặc chỉ có metadata.
    if len(q_str.strip()) < 10:
        return None

    # Bỏ sample mà answer là ảnh/file rỗng.
    if "PIL." in a_str or "<PIL" in a_str:
        return None

    return {
        "id": f"{prefix}-{idx}",
        "type": "physics",
        "question": q_str,
        "answer": a_str,
        "unit": stringify(unit) if unit is not None else None,
        "cot": stringify(cot),
        "explanation": stringify(cot) or f"Solve the physics problem step by step and report the final answer: {a_str}.",
    }


def main() -> None:
    datasets = {
        "tasksource__folio": "logic",
        "tasksource__proofwriter": "logic",
        "lmms-lab__SciBench_Physics": "physics",
        "TIGER-Lab__TheoremQA": "physics",
        "xw27__scibench": "physics",
        "afdsafas__EEE-Bench": "physics",
    }

    all_items: list[dict[str, Any]] = []

    for folder, kind in datasets.items():
        path = EXTERNAL / folder
        if not path.exists():
            print("Skip missing:", path)
            continue

        print("Loading:", path)
        rows = load_rows(path)

        if rows:
            print("columns:", list(rows[0].keys()))

        converted = []
        for i, row in enumerate(rows):
            if kind == "logic":
                item = convert_logic(row, i, folder)
            else:
                item = convert_physics(row, i, folder)
            if item:
                converted.append(item)

        print(f"{folder}: {len(converted)} usable rows")
        if converted:
            print("sample converted:", json.dumps(converted[0], ensure_ascii=False)[:500])
        all_items.extend(converted)

    random.shuffle(all_items)

    debug = all_items[:2000]
    split = int(len(all_items) * 0.95)
    train = all_items[:split]
    dev = all_items[split:]

    write_jsonl(RAW / "train_external_debug_2000.jsonl", debug)
    write_jsonl(RAW / "train.jsonl", train)
    write_jsonl(RAW / "dev.jsonl", dev)

    print("Done")
    print("debug:", len(debug))
    print("train:", len(train))
    print("dev:", len(dev))


if __name__ == "__main__":
    main()