#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def read_json_array(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
    return [row for row in data if isinstance(row, dict)]


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def clean_spaces(text: Any) -> str:
    return re.sub(r"\s+", " ", stringify(text)).strip()


def normalized_key(row: dict[str, Any]) -> dict[str, str]:
    return {
        str(k).lower().strip().replace(" ", "_").replace("-", "_"): k
        for k in row.keys()
    }


def pick(row: dict[str, Any], names: list[str], default: Any = "") -> Any:
    norm = normalized_key(row)
    for name in names:
        key = name.lower().strip().replace(" ", "_").replace("-", "_")
        if key in norm:
            value = row[norm[key]]
            if value not in (None, ""):
                return value
    return default


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Logic data audit/cleaning
# ---------------------------------------------------------------------------

_OPTION_SUPPORT_RE = re.compile(
    r"\b(?:support(?:s|ing)?|making|makes|therefore|hence|so)\s+(?:option\s+)?([A-D])\b|\boption\s+([A-D])\s+(?:is|would be)\s+(?:correct|valid|true)",
    flags=re.I,
)


def selected_options_from_explanation(explanation: str) -> set[str]:
    out: set[str] = set()
    for m in _OPTION_SUPPORT_RE.finditer(explanation or ""):
        for group in m.groups():
            if group:
                out.add(group.upper())
    return out


def answer_kind(answer: str) -> str:
    a = clean_spaces(answer).lower()
    if a in {"yes", "true", "entailed"}:
        return "yes"
    if a in {"no", "false", "contradicted", "contradiction"}:
        return "no"
    if a in {"unknown", "uncertain", "undetermined", "cannot be determined"}:
        return "unknown"
    if re.fullmatch(r"(?:option\s*)?[a-d]", a):
        return a[-1].upper()
    return "other"


def logic_question_issues(
    *,
    source_record_id: str,
    question_index: int,
    question: str,
    answer: str,
    explanation: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    ak = answer_kind(answer)
    expl = clean_spaces(explanation)
    expl_low = expl.lower()
    selected = selected_options_from_explanation(expl)

    if not clean_spaces(question):
        issues.append({"severity": "reject", "reason": "empty_logic_question"})
    if not clean_spaces(answer):
        issues.append({"severity": "reject", "reason": "empty_logic_answer"})

    if ak in {"unknown", "no"} and selected:
        issues.append({
            "severity": "suspect",
            "reason": "logic_answer_explanation_option_conflict",
            "details": {"answer": answer, "explanation_selects": sorted(selected)},
        })

    if ak in {"A", "B", "C", "D"} and selected and ak not in selected:
        issues.append({
            "severity": "suspect",
            "reason": "logic_answer_explanation_different_option",
            "details": {"answer": answer, "explanation_selects": sorted(selected)},
        })

    if ak == "no" and re.search(
        r"\b(yes|statement follows|logically follows|is true|can therefore|so .* can|satisfying all conditions|meets all requirements)\b",
        expl_low,
    ):
        issues.append({"severity": "suspect", "reason": "logic_no_but_explanation_affirms"})

    if ak == "yes" and re.search(
        r"\b(does not follow|not enough information|cannot be determined|does not meet|fails? to|cannot)\b",
        expl_low,
    ):
        issues.append({"severity": "suspect", "reason": "logic_yes_but_explanation_negates"})

    return issues


def fol_issues(premises_fol: list[Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for i, item in enumerate(premises_fol, 1):
        text = clean_spaces(item)
        if "�" in text:
            issues.append({
                "severity": "suspect",
                "reason": "malformed_fol_replacement_character",
                "details": {"premise_fol_index": i, "text": text},
            })
        if text.count("(") != text.count(")"):
            issues.append({
                "severity": "suspect",
                "reason": "malformed_fol_unbalanced_parentheses",
                "details": {"premise_fol_index": i, "text": text},
            })
    return issues


def clean_logic_records(
    records: list[dict[str, Any]],
    *,
    drop_suspect_logic: bool,
    drop_exact_duplicate_records: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    cleaned: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    suspicious: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_record_hashes: set[str] = set()

    for ridx, rec in enumerate(records):
        source_record_id = f"logic-{ridx:04d}"
        premises_nl = as_list(rec.get("premises-NL") or rec.get("premises_nl") or rec.get("premises") or [])
        premises_fol = as_list(rec.get("premises-FOL") or rec.get("premises_fol") or rec.get("fol") or [])
        questions = as_list(rec.get("questions") or rec.get("question") or [])
        answers = as_list(rec.get("answers") or rec.get("answer") or [])
        explanations = as_list(rec.get("explanation") or rec.get("explanations") or [])
        idx_values = as_list(rec.get("idx")) if "idx" in rec else []

        record_hash = stable_hash({
            "premises-NL": premises_nl,
            "questions": questions,
            "answers": answers,
        })
        if drop_exact_duplicate_records and record_hash in seen_record_hashes:
            counts["logic_exact_duplicate_record_rejected"] += 1
            rejected.append({
                "source": "logic",
                "source_record_id": source_record_id,
                "question_index": None,
                "reason": "logic_exact_duplicate_record",
                "row": rec,
            })
            continue
        seen_record_hashes.add(record_hash)

        record_level_issues = fol_issues(premises_fol)
        for issue in record_level_issues:
            counts[issue["reason"]] += 1
            suspicious.append({
                "source": "logic",
                "source_record_id": source_record_id,
                "question_index": None,
                **issue,
            })

        keep_questions: list[Any] = []
        keep_answers: list[Any] = []
        keep_explanations: list[Any] = []
        keep_idx: list[Any] = []

        for qidx, question in enumerate(questions):
            answer = answers[qidx] if qidx < len(answers) else ""
            explanation = explanations[qidx] if qidx < len(explanations) else ""
            issues = logic_question_issues(
                source_record_id=source_record_id,
                question_index=qidx,
                question=stringify(question),
                answer=stringify(answer),
                explanation=stringify(explanation),
            )

            reject = any(i["severity"] == "reject" for i in issues)
            suspect = any(i["severity"] == "suspect" for i in issues)

            for issue in issues:
                counts[issue["reason"]] += 1
                entry = {
                    "source": "logic",
                    "source_record_id": source_record_id,
                    "question_index": qidx,
                    "question": question,
                    "answer": answer,
                    "explanation": explanation,
                    **issue,
                }
                if issue["severity"] == "reject" or (drop_suspect_logic and issue["severity"] == "suspect"):
                    rejected.append(entry)
                else:
                    suspicious.append(entry)

            if reject or (drop_suspect_logic and suspect):
                counts["logic_question_dropped"] += 1
                continue

            keep_questions.append(question)
            keep_answers.append(answer)
            keep_explanations.append(explanation)
            if idx_values:
                keep_idx.append(idx_values[qidx] if qidx < len(idx_values) else None)

        if not keep_questions:
            counts["logic_record_dropped_empty_after_cleaning"] += 1
            rejected.append({
                "source": "logic",
                "source_record_id": source_record_id,
                "question_index": None,
                "reason": "logic_record_empty_after_cleaning",
                "row": rec,
            })
            continue

        new_rec = dict(rec)
        new_rec["questions"] = keep_questions
        new_rec["answers"] = keep_answers
        new_rec["explanation"] = keep_explanations
        if idx_values:
            new_rec["idx"] = keep_idx
        cleaned.append(new_rec)

    return cleaned, rejected, suspicious, counts


# ---------------------------------------------------------------------------
# Physics data audit/cleaning
# ---------------------------------------------------------------------------

_TRANSLATION_META_PATTERNS = [
    r"here are a few ways to translate",
    r"all options convey",
    r"option\s+1\s*:",
    r"option\s+2\s*:",
    r"translation of the question",
    r"translate (?:this|the) question",
    r"a good translation would be",
]


def read_csv_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve all original fields and append any new fields at the end.
    fields = list(fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def physics_row_issues(row: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    rid = clean_spaces(pick(row, ["id", "ID", "idx", "sample_id"], ""))
    question = clean_spaces(pick(row, ["question", "Question", "problem", "Problem", "text", "prompt", "stem", "body"], ""))
    answer = clean_spaces(pick(row, ["answer", "Answer", "final_answer", "Final Answer", "target", "output", "gold_answer", "ground_truth"], ""))
    unit = clean_spaces(pick(row, ["unit", "Unit", "units", "answer_unit", "unit_text"], ""))

    if not question:
        issues.append({"severity": "reject", "reason": "empty_physics_question"})
    if not answer:
        issues.append({"severity": "reject", "reason": "missing_physics_answer"})
    if rid.upper().startswith("QA") and not answer:
        issues.append({"severity": "reject", "reason": "qa_physics_missing_answer"})
    if rid.upper().startswith("QA") and not unit:
        issues.append({"severity": "suspect", "reason": "qa_physics_missing_unit"})

    qlow = question.lower()
    if any(re.search(pattern, qlow) for pattern in _TRANSLATION_META_PATTERNS):
        issues.append({"severity": "reject", "reason": "physics_translation_meta_text"})

    return issues


def clean_physics_rows(
    rows: list[dict[str, Any]],
    *,
    keep_missing_answers: bool,
    keep_qa_rows: bool,
    keep_translation_meta: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    cleaned: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    suspicious: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for i, row in enumerate(rows):
        rid = clean_spaces(pick(row, ["id", "ID", "idx", "sample_id"], f"physics-{i:05d}")) or f"physics-{i:05d}"
        issues = physics_row_issues(row)

        if rid in seen_ids:
            issues.append({"severity": "reject", "reason": "duplicate_physics_id"})
        seen_ids.add(rid)

        final_issues: list[dict[str, Any]] = []
        for issue in issues:
            reason = issue["reason"]
            if keep_missing_answers and reason in {"missing_physics_answer", "qa_physics_missing_answer"}:
                issue = {**issue, "severity": "suspect"}
            if keep_qa_rows and reason in {"qa_physics_missing_answer", "qa_physics_missing_unit"}:
                issue = {**issue, "severity": "suspect"}
            if keep_translation_meta and reason == "physics_translation_meta_text":
                issue = {**issue, "severity": "suspect"}
            final_issues.append(issue)

        reject = any(issue["severity"] == "reject" for issue in final_issues)

        for issue in final_issues:
            counts[issue["reason"]] += 1
            entry = {
                "source": "physics",
                "row_index": i,
                "source_record_id": rid,
                "reason": issue["reason"],
                "severity": issue["severity"],
                "row": row,
            }
            if issue["severity"] == "reject":
                rejected.append(entry)
            else:
                suspicious.append(entry)

        if reject:
            counts["physics_row_dropped"] += 1
            continue
        cleaned.append(row)

    return cleaned, rejected, suspicious, counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and conservatively clean official EXACT-FAMA raw data before prepare_official_data.py."
    )
    parser.add_argument("--logic", "--logic_in", dest="logic", required=True, help="Official logic JSON array file")
    parser.add_argument("--physics", "--physics_in", dest="physics", required=True, help="Official physics CSV file")
    parser.add_argument("--out_dir", required=True, help="Directory for cleaned raw files and reports")

    parser.add_argument(
        "--drop_suspect_logic",
        action="store_true",
        help="Drop logic questions with high-confidence answer/explanation conflicts. Default only audits them.",
    )
    parser.add_argument(
        "--keep_missing_physics_answers",
        action="store_true",
        help="Keep physics rows with missing answers as suspicious instead of rejecting. Not recommended for supervised eval.",
    )
    parser.add_argument(
        "--keep_qa_rows",
        action="store_true",
        help="Keep QA-prefixed physics rows as suspicious instead of rejecting.",
    )
    parser.add_argument(
        "--keep_translation_meta",
        action="store_true",
        help="Keep physics rows whose question looks like translation meta-text.",
    )
    parser.add_argument(
        "--no_drop_duplicate_logic_records",
        action="store_true",
        help="Do not drop exact duplicate logic records.",
    )

    args = parser.parse_args()

    logic_path = Path(args.logic)
    physics_path = Path(args.physics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logic_raw = read_json_array(logic_path)
    physics_raw, physics_fields = read_csv_rows(physics_path)

    logic_clean, logic_rejected, logic_suspicious, logic_counts = clean_logic_records(
        logic_raw,
        drop_suspect_logic=args.drop_suspect_logic,
        drop_exact_duplicate_records=not args.no_drop_duplicate_logic_records,
    )
    physics_clean, physics_rejected, physics_suspicious, physics_counts = clean_physics_rows(
        physics_raw,
        keep_missing_answers=args.keep_missing_physics_answers,
        keep_qa_rows=args.keep_qa_rows,
        keep_translation_meta=args.keep_translation_meta,
    )

    logic_out = out_dir / "Logic_Based_Educational_Queries.clean.json"
    physics_out = out_dir / "Physics_Problems_Text_Only.clean.csv"
    rejected_out = out_dir / "preprocess_rejected.jsonl"
    suspicious_out = out_dir / "preprocess_suspicious.jsonl"
    report_out = out_dir / "preprocess_report.json"

    write_json_array(logic_out, logic_clean)
    write_csv_rows(physics_out, physics_clean, physics_fields)

    rejected = logic_rejected + physics_rejected
    suspicious = logic_suspicious + physics_suspicious
    write_jsonl(rejected_out, rejected)
    write_jsonl(suspicious_out, suspicious)

    report = {
        "inputs": {
            "logic": str(logic_path),
            "physics": str(physics_path),
        },
        "outputs": {
            "logic_clean": str(logic_out),
            "physics_clean": str(physics_out),
            "rejected": str(rejected_out),
            "suspicious": str(suspicious_out),
            "report": str(report_out),
        },
        "options": {
            "drop_suspect_logic": bool(args.drop_suspect_logic),
            "keep_missing_physics_answers": bool(args.keep_missing_physics_answers),
            "keep_qa_rows": bool(args.keep_qa_rows),
            "keep_translation_meta": bool(args.keep_translation_meta),
            "drop_exact_duplicate_logic_records": not args.no_drop_duplicate_logic_records,
        },
        "summary": {
            "logic_raw_records": len(logic_raw),
            "logic_clean_records": len(logic_clean),
            "physics_raw_rows": len(physics_raw),
            "physics_clean_rows": len(physics_clean),
            "rejected_total": len(rejected),
            "suspicious_total": len(suspicious),
            "logic_issue_counts": dict(logic_counts),
            "physics_issue_counts": dict(physics_counts),
        },
        "next_step": {
            "prepare_command": (
                "python scripts/prepare_official_data.py "
                f"--logic {logic_out} --physics {physics_out} --out_dir data/raw "
                "--train_ratio 0.80 --dev_ratio 0.10 --seed 2026 --smoke_size 100"
            )
        },
    }

    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
