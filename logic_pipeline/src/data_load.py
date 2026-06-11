"""EXACT-style dataset loader for the cascade pipeline.

The release JSON (`Logic_Based_Educational_Queries.json`) bundles ONE premise
set with N parallel (question, answer, explanation) tuples:

    {
      "idx": [[...], [...]],
      "premises-FOL": ["ForAll(x, ...)", ...],   # ignored — NL-only pipeline
      "premises-NL":  ["Students who ...", ...],
      "questions":    ["… which conclusion?\nA. …\nB. …", "Does Sophia …?"],
      "answers":      ["C", "Yes"],
      "explanation":  ["…", "…"],
    }

Each source record expands into N normalized `Record`s, one per question. MCQ
options are embedded in the question string and split out. Only `premises-NL`
and `questions` ever reach a model; gold `answers` are canonicalized and kept
aside for optional scoring.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from schema import AnswerType, Record

FIELD_ALIASES: dict[str, list[str]] = {
    "id": ["id", "qid", "question_id", "record_id", "idx"],
    "premises_nl": ["premises_nl", "premises-NL", "premises", "context", "context_nl", "facts_nl"],
    "questions": ["questions", "question_nl", "question", "queries"],
    "answers": ["answers", "answer", "label", "gold", "gold_answer"],
    "explanations": ["explanation", "explanations", "rationale", "reasoning"],
}

# Option line: "A. text", "A) text", "(A) text" at start of a line.
_MCQ_OPTION_LINE = re.compile(r"^\s*[\(\[]?([A-H])[\)\.\:\]\s]\s*(.+?)$", re.MULTILINE)


def parse_mcq_question(text: str) -> tuple[str, list[str]] | None:
    """Split 'stem\\nA. opt\\nB. opt…' into (stem, [opt, …]); None if not MCQ."""
    matches = list(_MCQ_OPTION_LINE.finditer(text))
    if len(matches) < 2:
        return None
    labels = [m.group(1) for m in matches]
    expected = [chr(ord("A") + i) for i in range(len(matches))]
    if labels != expected:
        return None
    stem = text[: matches[0].start()].rstrip()
    options = [m.group(2).strip() for m in matches]
    return stem, options


def _first_present(obj: dict[str, Any], aliases: list[str]) -> Any:
    for k in aliases:
        if k in obj and obj[k] not in (None, "", []):
            return obj[k]
    return None


def _coerce_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def _canon_gold(gold: str | None, atype: AnswerType, options: list[str] | None) -> str | None:
    """Canonicalize gold so it lines up with the model's canonical output:
    MCQ → an uppercase letter; YNN → Yes / No / Unknown."""
    if not gold:
        return None
    g = gold.strip().rstrip(".")
    if atype == AnswerType.MCQ:
        # An MCQ whose correct answer is "none of the options" — keep it canonical
        # as "Unknown" (matches what the model emits for that case).
        if g.lower() in ("unknown", "none", "none of the above", "not given", "n/a", "na"):
            return "Unknown"
        if len(g) == 1 and g.isalpha():
            return g.upper()
        # gold given as option text → map back to a letter
        if options:
            from prompts import _norm  # local import avoids a cycle at import time
            ng = _norm(g)
            for i, o in enumerate(options):
                if _norm(o) == ng:
                    return chr(ord("A") + i)
        return g.upper()
    if atype == AnswerType.YES_NO_UNKNOWN:
        # Reuse the marker-aware normalizer so "Not Given"/"no information"/… are
        # mapped to Unknown BEFORE the bare startswith("no")→No rule. Anything it
        # can't classify (shouldn't happen for valid gold) collapses to Unknown.
        from prompts import canonicalize_ynn
        return canonicalize_ynn(g) or "Unknown"
    return g


def _classify(gold: str | None, options: list[str] | None) -> AnswerType:
    if options:
        return AnswerType.MCQ
    if isinstance(gold, str):
        norm = gold.strip().lower().rstrip(".")
        if norm in {"yes", "no", "unknown", "uncertain", "not given", "notgiven", "not-given"}:
            return AnswerType.YES_NO_UNKNOWN
        if len(norm) == 1 and norm.isalpha() and norm.upper() <= "H":
            return AnswerType.MCQ
    # No gold (test set) and no MCQ options → assume the dataset's dominant
    # non-MCQ format rather than open-ended.
    return AnswerType.YES_NO_UNKNOWN


def _record_id(raw_idx: Any, record_pos: int, q_pos: int) -> str:
    # `idx` in this dataset is the list of premise indices a question uses, NOT a
    # unique key — the same idx recurs across source objects. So always prefix the
    # source object position to guarantee globally unique ids (otherwise records
    # from different objects would collide and share one prediction slot).
    base = ""
    if isinstance(raw_idx, list):
        flat: list[str] = []
        for sub in raw_idx:
            flat.extend(str(x) for x in sub) if isinstance(sub, list) else flat.append(str(sub))
        base = "-".join(flat)
    elif isinstance(raw_idx, (str, int)):
        base = str(raw_idx)
    return f"r{record_pos}_{base}_q{q_pos}" if base else f"r{record_pos}_q{q_pos}"


def expand_record(obj: dict[str, Any], record_pos: int) -> list[Record]:
    def pick(name: str) -> Any:
        return _first_present(obj, FIELD_ALIASES[name])

    premises_nl = _coerce_list(pick("premises_nl"))
    questions = _coerce_list(pick("questions"))
    answers = _coerce_list(pick("answers"))
    explanations = _coerce_list(pick("explanations"))
    raw_idx = pick("id")
    if not questions:
        return []
    while len(answers) < len(questions):
        answers.append("")

    out: list[Record] = []
    for q_i, q_text in enumerate(questions):
        mcq = parse_mcq_question(q_text)
        stem, options = mcq if mcq is not None else (q_text, None)
        gold_raw = answers[q_i] or None
        atype = _classify(gold_raw, options)
        out.append(
            Record(
                id=_record_id(raw_idx, record_pos, q_i),
                premises_nl=premises_nl,
                question_nl=stem,
                answer_type=atype,
                answer=_canon_gold(gold_raw, atype, options),
                options=options,
                raw={"original_record_pos": record_pos, "question_pos": q_i,
                     "gold_explanation": explanations[q_i] if q_i < len(explanations) else None,
                     "idx": raw_idx},
            )
        )
    return out


def load_records(path: str | Path) -> list[Record]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("data", "records", "items", "examples"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of records, got {type(data).__name__}")
    out: list[Record] = []
    for i, obj in enumerate(data):
        out.extend(expand_record(obj, record_pos=i))
    return out
