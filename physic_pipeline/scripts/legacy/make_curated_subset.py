#!/usr/bin/env python
from __future__ import annotations

"""Create a curated EXACT-FAMA training subset.

This script can read either:
1. raw project JSONL/JSON rows, e.g. data/raw/train.jsonl; or
2. SFT chat JSONL rows, e.g. data/processed/sft_train.jsonl.

It does not sample uniformly at random. It scores records by relevance to EXACT:
- logic/regulation/premise reasoning;
- electric circuits/electrostatics physics;
- answer/explanation/CoT completeness;
- reasoning hardness signals such as negation, conditions, MCQ, unit conversion.

It then fills target quotas by bucket and source to avoid a 60k subset being dominated
by one huge external dataset such as ProofWriter.
"""

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

LOGIC_KEYWORDS = {
    "premise", "premises", "regulation", "policy", "rule", "scholarship",
    "course", "semester", "student", "curriculum", "faculty", "eligible",
    "requirement", "pass", "fail", "grade", "if", "then", "unless", "only if",
    "all", "some", "none", "not", "cannot", "uncertain", "yes", "no",
    "quy định", "quy chế", "học bổng", "môn học", "sinh viên",
}

PHYSICS_KEYWORDS = {
    "resistance", "resistor", "voltage", "current", "power", "capacitor",
    "capacitance", "electric field", "charge", "energy stored", "ohm", "circuit",
    "parallel", "series", "kirchhoff", "coulomb", "potential", "force", "joule",
    "điện trở", "hiệu điện thế", "dòng điện", "công suất", "tụ điện", "điện trường",
    "điện tích", "mạch", "song song", "nối tiếp",
}

HARD_LOGIC_KEYWORDS = {
    "unless", "only if", "except", "at least", "at most", "greater than", "less than",
    "not", "cannot", "neither", "either", "all", "some", "none", "exactly",
    "uncertain", "contradict", "A.", "B.", "C.", "D.",
}

HARD_PHYSICS_KEYWORDS = {
    "micro", "μ", "uF", "mA", "kΩ", "kohm", "parallel", "series", "equivalent",
    "energy", "electric field", "potential", "coulomb", "capacitor", "capacitance",
}

ELECTRIC_REGEX = re.compile(
    r"\b([RIVPUCQEF]\d*|voltage|current|resistance|capacitance|charge|power)\s*(=|is)\s*[-+]?\d",
    re.I,
)


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        return [r for r in data if isinstance(r, dict)]
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object: {path}")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)


def extract_sft_parts(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return system/user/assistant text for chat-format SFT rows."""
    messages = row.get("messages") or []
    system = user = assistant = ""
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", ""))
            content = normalize_text(m.get("content", ""))
            if role == "system" and not system:
                system = content
            elif role == "user" and not user:
                user = content
            elif role == "assistant" and not assistant:
                assistant = content
    return system, user, assistant


def expand_raw_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split official Type-1 rows that contain questions/answers arrays."""
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row.get("questions"), list):
            questions = row.get("questions") or []
            answers = row.get("answers") or []
            explanations = row.get("explanation") or []
            for idx, q in enumerate(questions):
                item = dict(row)
                item.pop("questions", None)
                item.pop("answers", None)
                item["id"] = f"{row.get('id', 'logic')}-{idx}"
                item["type"] = "logic"
                item["question"] = q
                item["answer"] = answers[idx] if idx < len(answers) else ""
                if isinstance(explanations, list):
                    item["explanation"] = explanations[idx] if idx < len(explanations) else ""
                out.append(item)
        else:
            out.append(row)
    return out


def detect_kind(rows: list[dict[str, Any]]) -> str:
    for row in rows[:20]:
        if isinstance(row.get("messages"), list):
            return "sft"
    return "raw"


def source_key(row: dict[str, Any], kind: str) -> str:
    if kind == "sft":
        _system, user, assistant = extract_sft_parts(row)
        blob = user + "\n" + assistant
        # Keep broad source unknown for SFT unless an id was preserved.
        rid = str(row.get("id") or row.get("source") or "")
    else:
        blob = normalize_text(row.get("question")) + "\n" + normalize_text(row.get("id"))
        rid = str(row.get("id") or row.get("source") or "")
    rid_low = rid.lower()
    for marker in [
        "proofwriter", "p-folio", "p_folio", "folio", "scibench", "theoremqa",
        "physreason", "eee-bench", "eee_bench", "official", "sample",
    ]:
        if marker in rid_low or marker in blob.lower():
            return marker
    return rid.split("-")[0][:40] if rid else "unknown"


def record_text(row: dict[str, Any], kind: str) -> str:
    if kind == "sft":
        system, user, assistant = extract_sft_parts(row)
        return f"{system}\n{user}\n{assistant}"
    premises = row.get("premises-NL") or row.get("premises_nl") or []
    if isinstance(premises, list):
        premises_text = "\n".join(normalize_text(p) for p in premises)
    else:
        premises_text = normalize_text(premises)
    return "\n".join([
        premises_text,
        normalize_text(row.get("question")),
        normalize_text(row.get("answer")),
        normalize_text(row.get("unit")),
        normalize_text(row.get("cot")),
        normalize_text(row.get("explanation")),
    ])


def classify_bucket(row: dict[str, Any], kind: str) -> str:
    text = record_text(row, kind)
    low = text.lower()
    explicit_type = str(row.get("type", "")).lower() if kind == "raw" else ""
    physics_score = sum(1 for k in PHYSICS_KEYWORDS if k in low)
    logic_score = sum(1 for k in LOGIC_KEYWORDS if k in low)
    if ELECTRIC_REGEX.search(text):
        physics_score += 3
    if "premises:" in low or "premises-nl" in low or "premises_nl" in low:
        logic_score += 3
    if re.search(r"\b[A-D]\.\s+", text):
        logic_score += 2
    if explicit_type == "physics":
        physics_score += 5
    if explicit_type == "logic":
        logic_score += 5

    if physics_score > logic_score:
        if any(k in low for k in ["parallel", "series", "equivalent", "circuit", "resistor", "mạch"]):
            return "physics_circuit"
        if any(k in low for k in ["capacitor", "capacitance", "tụ điện"]):
            return "physics_capacitor"
        if any(k in low for k in ["electric field", "coulomb", "charge", "potential", "điện trường", "điện tích"]):
            return "physics_electrostatics"
        return "physics_general"
    if logic_score > 0:
        if re.search(r"\b[A-D]\.\s+", text):
            return "logic_mcq"
        if any(k.lower() in low for k in HARD_LOGIC_KEYWORDS):
            return "logic_hard"
        return "logic_general"
    return "format_general"


def quality_score(row: dict[str, Any], kind: str) -> float:
    text = record_text(row, kind)
    low = text.lower()
    score = 0.0

    if kind == "raw":
        question = normalize_text(row.get("question"))
        answer = normalize_text(row.get("answer"))
        explanation = normalize_text(row.get("explanation"))
        cot = normalize_text(row.get("cot"))
        premises = row.get("premises-NL") or row.get("premises_nl") or []
        if question:
            score += 5
        if answer:
            score += 5
        if explanation and len(explanation) >= 20:
            score += 4
        if cot and len(cot) >= 20:
            score += 3
        if premises:
            score += min(4, len(premises) if isinstance(premises, list) else 2)
        q_len = len(question.split())
    else:
        _system, user, assistant = extract_sft_parts(row)
        score += 5 if user else 0
        score += 5 if assistant else 0
        if '"answer"' in assistant or "answer" in assistant.lower():
            score += 3
        if '"explanation"' in assistant or "explanation" in assistant.lower():
            score += 3
        if '"cot"' in assistant or "step" in assistant.lower():
            score += 2
        q_len = len(user.split())

    if 8 <= q_len <= 320:
        score += 3
    elif q_len > 600:
        score -= 5

    score += min(6, sum(1 for k in HARD_LOGIC_KEYWORDS if k.lower() in low))
    score += min(6, sum(1 for k in HARD_PHYSICS_KEYWORDS if k.lower() in low))
    if re.search(r"\d", text):
        score += 2
    if any(unit in text for unit in ["V", "A", "ohm", "Ω", "J", "F", "C", "W", "μ", "micro"]):
        score += 2
    if "{}" in text or "[]" in text or "null" == low.strip():
        score -= 5
    if len(text.strip()) < 30:
        score -= 10

    # Stable tiny jitter for deterministic tie-breaking without pure randomness.
    score += (stable_hash(text) % 1000) / 10000.0
    return score


def dedupe_rows(rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        text = record_text(row, kind)
        key_text = re.sub(r"\s+", " ", text.lower()).strip()[:1200]
        key = hashlib.sha1(key_text.encode("utf-8", errors="ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def allocate_quotas(target: int) -> dict[str, int]:
    # EXACT-oriented default: roughly balanced logic/physics, plus a little format/general.
    raw = {
        "logic_hard": 0.16,
        "logic_mcq": 0.10,
        "logic_general": 0.19,
        "physics_circuit": 0.18,
        "physics_capacitor": 0.10,
        "physics_electrostatics": 0.10,
        "physics_general": 0.12,
        "format_general": 0.05,
    }
    quotas = {k: int(target * v) for k, v in raw.items()}
    diff = target - sum(quotas.values())
    quotas["logic_hard"] += diff
    return quotas


def select_with_source_diversity(scored: list[tuple[float, dict[str, Any], str]], quota: int) -> list[dict[str, Any]]:
    if quota <= 0 or not scored:
        return []
    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    by_source: dict[str, list[tuple[float, dict[str, Any], str]]] = defaultdict(list)
    for item in scored:
        by_source[item[2]].append(item)

    # Soft cap prevents a massive source from filling an entire bucket.
    cap = max(100, math.ceil(quota * 0.45))
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    # First pass with cap.
    for _score, row, src in scored:
        if len(selected) >= quota:
            break
        if counts[src] >= cap:
            continue
        selected.append(row)
        counts[src] += 1

    # Fill remainder by quality if there are not enough diverse sources.
    if len(selected) < quota:
        selected_ids = {id(r) for r in selected}
        for _score, row, _src in scored:
            if len(selected) >= quota:
                break
            if id(row) not in selected_ids:
                selected.append(row)
                selected_ids.add(id(row))

    return selected[:quota]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create curated 60k subset for EXACT-FAMA training")
    parser.add_argument("--input", required=True, help="Input raw JSONL/JSON or SFT chat JSONL")
    parser.add_argument("--output", required=True, help="Output curated JSONL, same format as input")
    parser.add_argument("--target_size", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kind", choices=["auto", "raw", "sft"], default="auto")
    parser.add_argument("--report", default="", help="Optional JSON report path")
    args = parser.parse_args()

    random.seed(args.seed)
    in_path = Path(args.input)
    rows = read_json_or_jsonl(in_path)
    kind = detect_kind(rows) if args.kind == "auto" else args.kind
    if kind == "raw":
        rows = expand_raw_records(rows)
    rows = dedupe_rows(rows, kind)

    buckets: dict[str, list[tuple[float, dict[str, Any], str]]] = defaultdict(list)
    for row in rows:
        bucket = classify_bucket(row, kind)
        score = quality_score(row, kind)
        src = source_key(row, kind)
        buckets[bucket].append((score, row, src))

    quotas = allocate_quotas(args.target_size)
    selected: list[dict[str, Any]] = []
    selected_by_bucket: Counter[str] = Counter()

    for bucket, quota in quotas.items():
        chosen = select_with_source_diversity(buckets.get(bucket, []), quota)
        selected.extend(chosen)
        selected_by_bucket[bucket] += len(chosen)

    # Backfill from all remaining high-quality records if a bucket is undersupplied.
    if len(selected) < args.target_size:
        selected_ids = {id(r) for r in selected}
        all_scored: list[tuple[float, dict[str, Any], str, str]] = []
        for bucket, items in buckets.items():
            for score, row, src in items:
                if id(row) not in selected_ids:
                    all_scored.append((score, row, src, bucket))
        all_scored.sort(key=lambda x: x[0], reverse=True)
        for _score, row, _src, bucket in all_scored:
            if len(selected) >= args.target_size:
                break
            selected.append(row)
            selected_by_bucket[bucket] += 1

    # Deterministic shuffle for training order.
    random.shuffle(selected)
    selected = selected[: args.target_size]

    out_path = Path(args.output)
    write_jsonl(out_path, selected)

    report = {
        "input": str(in_path),
        "output": str(out_path),
        "kind": kind,
        "input_rows_after_expand_dedupe": len(rows),
        "target_size": args.target_size,
        "selected_rows": len(selected),
        "available_by_bucket": {k: len(v) for k, v in sorted(buckets.items())},
        "selected_by_bucket": dict(selected_by_bucket),
        "selected_by_source": dict(Counter(source_key(r, kind) for r in selected).most_common()),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
