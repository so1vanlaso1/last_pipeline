#!/usr/bin/env python
from __future__ import annotations

# Allow running scripts directly from the repository without installing first.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from typing import Any

from exact_fama.utils.jsonl import read_jsonl, write_jsonl


PHYSICS_TERMS = {
    "resistance", "resistor", "voltage", "current", "power", "capacitor", "capacitance",
    "electric field", "charge", "energy", "coulomb", "ohm", "circuit", "series", "parallel",
    "điện trở", "hiệu điện thế", "dòng điện", "công suất", "tụ điện", "điện trường",
}

NEGATION_TERMS = {
    "not", "cannot", "can't", "do not", "does not", "unless", "except", "absent", "miss",
    "failed", "fail", "ineligible", "không", "vắng", "rớt", "trừ khi",
}

ELECTROSTATIC_TERMS = {
    "coulomb", "electric field", "electric potential", "charge", "force", "distance",
    "điện trường", "điện tích", "lực điện", "điện thế",
}

UNIT_TERMS = {
    "μ", "µ", "micro", "milli", "kilo", "uf", "μf", "ma", "ka", "kohm", "kω",
    "mc", "μc", "nc", "cm", "mm", "km", "ev",
}

KNOWN_SOURCE_PREFIXES = [
    "tasksource__proofwriter",
    "tasksource__folio",
    "lmms-lab__SciBench_Physics",
    "TIGER-Lab__TheoremQA",
    "xw27__scibench",
    "afdsafas__EEE-Bench",
    "official",
    "sample",
]


def normalize_text(x: Any) -> str:
    s = str(x or "").lower().replace("µ", "μ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9μωạ-ỹ\s=+\-*/().]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def get_source(row: dict[str, Any]) -> str:
    rid = str(row.get("id", "unknown"))

    for prefix in KNOWN_SOURCE_PREFIXES:
        if rid.startswith(prefix):
            return prefix

    if "-" in rid:
        return rid.rsplit("-", 1)[0]

    return rid or "unknown"


def expand_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split official Type-1 rows with questions/answers arrays into single-question rows."""
    out: list[dict[str, Any]] = []

    for row in rows:
        if isinstance(row.get("questions"), list):
            questions = row.get("questions") or []
            answers = row.get("answers") or []
            explanations = row.get("explanation") or []

            for i, q in enumerate(questions):
                item = dict(row)
                item.pop("questions", None)
                item.pop("answers", None)

                item["id"] = f"{row.get('id', 'logic')}-{i}"
                item["type"] = row.get("type") or "logic"
                item["question"] = q
                item["answer"] = answers[i] if i < len(answers) else ""

                if isinstance(explanations, list):
                    item["explanation"] = explanations[i] if i < len(explanations) else ""
                else:
                    item["explanation"] = explanations

                out.append(item)
        else:
            out.append(row)

    return out


def infer_type(row: dict[str, Any]) -> str:
    typ = str(row.get("type") or "").lower()

    if typ in {"logic", "physics"}:
        return typ

    if row.get("premises-NL") or row.get("premises_nl"):
        return "logic"

    q = normalize_text(row.get("question", ""))
    p_score = sum(1 for t in PHYSICS_TERMS if t in q)

    if re.search(r"\b[crvipuqef]\d*\s*=\s*[-+]?\d", q):
        p_score += 2

    return "physics" if p_score > 0 else "logic"


def signature(row: dict[str, Any]) -> str:
    premises = row.get("premises-NL") or row.get("premises_nl") or []

    if isinstance(premises, list):
        premises_text = " ".join(map(str, premises))
    else:
        premises_text = str(premises or "")

    q = normalize_text(row.get("question", ""))
    p = normalize_text(premises_text)

    return stable_hash(f"{infer_type(row)}|{p[:1200]}|{q[:1200]}")


def sft_signature(row: dict[str, Any]) -> str | None:
    """Best-effort signature for SFT chat rows.

    This supports excluding an SFT file when the raw curated file is unavailable.
    Exact exclusion is best when using the raw curated file.
    """
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None

    user_text = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_text = str(msg.get("content", ""))
            break

    if not user_text:
        return None

    premises: list[str] = []
    question = ""

    if "Question:" in user_text:
        before, after = user_text.rsplit("Question:", 1)
        question = after.strip()

        if "Premises:" in before:
            prem_blob = before.split("Premises:", 1)[1]
            premises = [
                line.strip().lstrip("-").strip()
                for line in prem_blob.splitlines()
                if line.strip().startswith("-")
            ]
    else:
        question = user_text.strip()

    if not question:
        return None

    synthetic = {
        "type": "logic" if premises else None,
        "premises-NL": premises,
        "question": question,
    }

    return signature(synthetic)


def row_blob(row: dict[str, Any]) -> str:
    premises = row.get("premises-NL") or row.get("premises_nl") or []

    if isinstance(premises, list):
        premises_text = " ".join(map(str, premises))
    else:
        premises_text = str(premises or "")

    return normalize_text(" ".join([
        str(row.get("type", "")),
        premises_text,
        str(row.get("question", "")),
        str(row.get("answer", "")),
        str(row.get("unit", "")),
        str(row.get("cot", "")),
        str(row.get("explanation", "")),
    ]))


def answer_label(row: dict[str, Any]) -> str:
    ans = normalize_text(row.get("answer", ""))

    if ans in {"true", "yes", "entailed", "correct", "1"}:
        return "true"

    if ans in {"false", "no", "contradiction", "contradicted", "incorrect", "0"}:
        return "false"

    if ans in {
        "unknown",
        "uncertain",
        "undetermined",
        "cannot be determined",
        "cant be determined",
        "not enough information",
        "not enough info",
        "unanswerable",
    }:
        return "unknown"

    if re.fullmatch(r"[a-d]", ans):
        return "mcq"

    if re.fullmatch(r"option [a-d]", ans):
        return "mcq"

    raw_ans = str(row.get("answer", "") or "").strip().replace(",", "")
    if re.fullmatch(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", raw_ans):
        return "numeric"

    return "other"


def is_mcq(row: dict[str, Any]) -> bool:
    q = str(row.get("question", ""))
    opts = row.get("options")

    if isinstance(opts, list) and len(opts) >= 2:
        return True

    if re.search(r"\bA[\).]\s+", q) and re.search(r"\bB[\).]\s+", q):
        return True

    if "choices/options" in q.lower():
        return True

    return False


def is_electric_physics(row: dict[str, Any]) -> bool:
    if infer_type(row) != "physics":
        return False

    blob = " " + row_blob(row) + " "

    electric_terms = {
        "resistance", "resistor", "voltage", "current", "power",
        "capacitor", "capacitance", "electric field", "charge",
        "coulomb", "ohm", "circuit", "series", "parallel",
        "điện trở", "hiệu điện thế", "dòng điện", "công suất",
        "tụ điện", "điện trường", "điện tích",
        " v ", " a ", " ohm", "ω", "uf", "μf", "ma", "mc", "μc",
    }

    return any(t in blob for t in electric_terms)


def split_bucket_key(row: dict[str, Any]) -> str:
    typ = infer_type(row)

    if typ == "physics":
        return "physics_electric" if is_electric_physics(row) else "physics_other"

    label = answer_label(row)

    if is_mcq(row):
        return f"logic_mcq_{label}"

    if label in {"true", "false", "unknown"}:
        return f"logic_{label}"

    if any(t in row_blob(row) for t in NEGATION_TERMS):
        return "logic_negation_other"

    premises = row.get("premises-NL") or row.get("premises_nl") or []
    if isinstance(premises, list) and len(premises) >= 4:
        return "logic_multi_premise_other"

    return "logic_other"


def bucket_row(row: dict[str, Any]) -> str:
    """Old coarse bucket, kept only for report compatibility."""
    typ = infer_type(row)
    q = normalize_text(row.get("question", ""))
    blob = row_blob(row)
    premises = row.get("premises-NL") or row.get("premises_nl") or []
    premise_count = len(premises) if isinstance(premises, list) else (1 if premises else 0)

    if typ == "logic":
        ans = normalize_text(row.get("answer", ""))
        if "uncertain" in ans or "unknown" in ans:
            return "logic_uncertain"
        if re.search(r"\b[a-d]\s*[.)]", q) or "choices/options" in q:
            return "logic_mcq"
        if any(t in blob for t in NEGATION_TERMS):
            return "logic_negation"
        if premise_count >= 4:
            return "logic_multi_premise"
        return "logic_general"

    if any(t in q for t in [
        "equivalent resistance", "total resistance", "series", "parallel",
        "r_eq", "req", "điện trở tương đương",
    ]):
        return "physics_circuit"

    if any(t in q for t in ["capacitor", "capacitance", "tụ điện", "μf", "uf"]):
        return "physics_capacitor"

    if any(t in q for t in ELECTROSTATIC_TERMS):
        return "physics_electrostatics"

    if any(t in q for t in ["power", "watt", "công suất"]):
        return "physics_power"

    if any(t in q for t in UNIT_TERMS):
        return "physics_unit_conversion"

    return "physics_general"


def quality_score(row: dict[str, Any]) -> float:
    q = str(row.get("question", "") or "")
    a = str(row.get("answer", "") or "")
    blob = row_blob(row)

    score = 0.0

    if len(q.strip()) >= 20:
        score += 2
    if len(q.strip()) >= 60:
        score += 1
    if a.strip():
        score += 3
    if row.get("explanation") or row.get("cot"):
        score += 2
    if row.get("unit"):
        score += 1
    if row.get("premises-NL") or row.get("premises_nl"):
        score += 2
    if any(t in blob for t in NEGATION_TERMS):
        score += 1
    if re.search(r"\b[a-d]\s*[.)]", blob):
        score += 1
    if re.search(r"\b[crvipuqef]\d*\s*=\s*[-+]?\d", blob):
        score += 2
    if any(u in blob for u in UNIT_TERMS):
        score += 1
    if is_electric_physics(row):
        score += 3

    if len(q.strip()) < 10:
        score -= 5
    if not a.strip():
        score -= 5
    if "<pil" in blob:
        score -= 3
    if len(q) > 4000:
        score -= 2

    return score


def load_exclusion_signatures(paths: list[str]) -> set[str]:
    excluded: set[str] = set()

    for p in paths:
        if not p:
            continue

        path = Path(p)

        if not path.exists():
            print(f"[warn] exclude path not found: {path}")
            continue

        try:
            rows = read_jsonl(path)
        except Exception as exc:
            print(f"[warn] could not read exclude file {path}: {exc}")
            continue

        expanded = expand_records(rows)
        before = len(excluded)

        for row in expanded:
            if isinstance(row, dict) and "messages" in row:
                sig = sft_signature(row)
                if sig:
                    excluded.add(sig)
            elif isinstance(row, dict) and row.get("question"):
                excluded.add(signature(row))

        print(f"[exclude] {path}: added {len(excluded) - before} signatures")

    return excluded


def proportional_quotas(buckets: dict[str, list[dict[str, Any]]], target: int) -> dict[str, int]:
    """Create fixed target quotas.

    The goal is to avoid ProofWriter unknown cases dominating the split.
    """
    if target >= 20000:
        preferred = {
            "logic_true": 5500,
            "logic_false": 5500,
            "logic_unknown": 5500,

            "logic_mcq_mcq": 1000,

            "logic_negation_other": 700,
            "logic_multi_premise_other": 700,
            "logic_other": 600,

            "physics_electric": 400,
            "physics_other": 100,
        }
    else:
        preferred = {
            "logic_true": 2700,
            "logic_false": 2700,
            "logic_unknown": 2700,

            "logic_mcq_mcq": 500,

            "logic_negation_other": 300,
            "logic_multi_premise_other": 300,
            "logic_other": 300,

            "physics_electric": 400,
            "physics_other": 100,
        }

    quotas = {k: min(preferred.get(k, 0), len(v)) for k, v in buckets.items()}
    current = sum(quotas.values())
    remaining = target - current

    fill_order = [
        "logic_true",
        "logic_false",
        "logic_unknown",
        "physics_electric",
        "logic_mcq_mcq",
        "logic_negation_other",
        "logic_multi_premise_other",
        "logic_other",
        "physics_other",
    ]

    while remaining > 0:
        progressed = False

        for k in fill_order:
            if remaining <= 0:
                break

            available = len(buckets.get(k, [])) - quotas.get(k, 0)
            if available <= 0:
                continue

            quotas[k] = quotas.get(k, 0) + 1
            remaining -= 1
            progressed = True

        if not progressed:
            break

    return quotas


def select_rows(
    candidates: list[dict[str, Any]],
    target: int,
    seed: int,
    source_cap_ratio: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()

    for row in candidates:
        sig = signature(row)

        if sig in seen:
            continue

        seen.add(sig)
        buckets[split_bucket_key(row)].append(row)

    for rows in buckets.values():
        rng.shuffle(rows)
        rows.sort(key=quality_score, reverse=True)

    quotas = proportional_quotas(buckets, target)

    selected: list[dict[str, Any]] = []
    selected_sigs: set[str] = set()
    source_counts: Counter[str] = Counter()

    source_cap = max(100, int(target * source_cap_ratio))

    def add_row(row: dict[str, Any], enforce_source_cap: bool) -> bool:
        if len(selected) >= target:
            return False

        sig = signature(row)
        if sig in selected_sigs:
            return False

        src = get_source(row)
        if enforce_source_cap and source_counts[src] >= source_cap:
            return False

        selected.append(row)
        selected_sigs.add(sig)
        source_counts[src] += 1
        return True

    # Pass 1: fill each quota with source cap.
    for bucket_name, quota in quotas.items():
        take = 0

        for row in buckets.get(bucket_name, []):
            if len(selected) >= target or take >= quota:
                break

            if add_row(row, enforce_source_cap=True):
                take += 1

    # Pass 2: fill missing quota from same bucket, ignoring source cap.
    # This prevents imbalanced source cap from destroying answer-label balance.
    for bucket_name, quota in quotas.items():
        current_in_bucket = sum(1 for r in selected if split_bucket_key(r) == bucket_name)
        need = quota - current_in_bucket

        if need <= 0:
            continue

        for row in buckets.get(bucket_name, []):
            if len(selected) >= target or need <= 0:
                break

            if add_row(row, enforce_source_cap=False):
                need -= 1

    # Pass 3: if still short because some buckets are unavailable, fill by priority.
    fill_order = [
        "logic_true",
        "logic_false",
        "logic_unknown",
        "physics_electric",
        "logic_mcq_mcq",
        "logic_negation_other",
        "logic_multi_premise_other",
        "logic_other",
        "physics_other",
    ]

    for bucket_name in fill_order:
        for row in buckets.get(bucket_name, []):
            if len(selected) >= target:
                break
            add_row(row, enforce_source_cap=False)

        if len(selected) >= target:
            break

    # Final emergency fill from all buckets.
    if len(selected) < target:
        leftovers = [
            row
            for rows in buckets.values()
            for row in rows
            if signature(row) not in selected_sigs
        ]
        leftovers.sort(key=quality_score, reverse=True)

        for row in leftovers:
            if len(selected) >= target:
                break
            add_row(row, enforce_source_cap=False)

    rng.shuffle(selected)
    return selected[:target]


def summarize_split(name: str, split: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": name,
        "count": len(split),
        "type_counts": dict(Counter(infer_type(r) for r in split)),
        "answer_counts": dict(Counter(answer_label(r) for r in split)),
        "split_bucket_counts": dict(Counter(split_bucket_key(r) for r in split)),
        "old_bucket_counts": dict(Counter(bucket_row(r) for r in split)),
        "source_counts_top20": dict(Counter(get_source(r) for r in split).most_common(20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create balanced solver diagnostic/blind eval splits from raw data."
    )
    parser.add_argument("--input", required=True, help="Raw JSONL/JSON file, usually data/raw/train.jsonl")
    parser.add_argument("--diagnostic_output", default="data/raw/solver_diagnostic_20k.jsonl")
    parser.add_argument("--blind_output", default="data/raw/solver_blind_10k.jsonl")
    parser.add_argument("--train_pool_output", default="data/raw/train_pool_no_solver_eval.jsonl")
    parser.add_argument("--diagnostic_size", type=int, default=20000)
    parser.add_argument("--blind_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source_cap_ratio",
        type=float,
        default=1.0,
        help=(
            "Soft cap for one source in each split. "
            "Use 1.0 for highly imbalanced datasets like ProofWriter-heavy training data."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Optional JSONL file to exclude by question/premise signature. "
            "Can be repeated. For solver-only testing, you can omit this."
        ),
    )
    args = parser.parse_args()

    rows = expand_records(read_jsonl(args.input))
    print(f"Loaded rows after expansion: {len(rows)}")

    excluded_sigs = load_exclusion_signatures(args.exclude)

    if excluded_sigs:
        print(f"Total exclusion signatures: {len(excluded_sigs)}")

    usable: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = 0
    skipped_excluded = 0

    for r in rows:
        if not r.get("question") or not str(r.get("answer", "")).strip():
            skipped += 1
            continue

        rr = dict(r)
        rr["type"] = infer_type(rr)

        sig = signature(rr)

        if sig in excluded_sigs:
            skipped += 1
            skipped_excluded += 1
            continue

        if sig in seen:
            skipped += 1
            continue

        seen.add(sig)
        usable.append(rr)

    print(
        f"Usable unique rows: {len(usable)}; "
        f"skipped: {skipped}; "
        f"excluded_by_signature: {skipped_excluded}"
    )

    available_summary = summarize_split("available_before_split", usable)
    print("Available before split:")
    print(json.dumps(available_summary, indent=2, ensure_ascii=False))

    # Diagnostic is selected first because it is used for solver debugging and should get rare buckets.
    diagnostic = select_rows(
        usable,
        target=args.diagnostic_size,
        seed=args.seed,
        source_cap_ratio=args.source_cap_ratio,
    )
    diag_sigs = {signature(r) for r in diagnostic}

    remaining = [r for r in usable if signature(r) not in diag_sigs]

    blind = select_rows(
        remaining,
        target=args.blind_size,
        seed=args.seed + 1000,
        source_cap_ratio=args.source_cap_ratio,
    )
    blind_sigs = {signature(r) for r in blind}

    train_pool = [
        r for r in rows
        if signature(r) not in blind_sigs
        and signature(r) not in diag_sigs
        and signature(r) not in excluded_sigs
    ]

    write_jsonl(args.diagnostic_output, diagnostic)
    write_jsonl(args.blind_output, blind)
    write_jsonl(args.train_pool_output, train_pool)

    report = {
        "strategy": (
            "Diagnostic is selected first for solver tuning. Blind is selected after diagnostic. "
            "Train pool excludes blind, diagnostic, and all --exclude signatures. "
            "Splits are balanced by answer label and task bucket."
        ),
        "input": args.input,
        "excluded_files": args.exclude,
        "excluded_signature_count": len(excluded_sigs),
        "excluded_by_signature_from_input": skipped_excluded,
        "outputs": {
            "diagnostic": args.diagnostic_output,
            "blind": args.blind_output,
            "train_pool": args.train_pool_output,
        },
        "splits": [
            summarize_split("available_before_split", usable),
            summarize_split("diagnostic", diagnostic),
            summarize_split("blind", blind),
            summarize_split("train_pool", train_pool),
        ],
    }

    report_path = Path(args.diagnostic_output).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()