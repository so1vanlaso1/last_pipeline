from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..schemas import PredictRequest
from .qwen_client import QwenClient

TaskType = Literal["logic", "physics"]


# ---------------------------------------------------------------------------
# Strict parser schema
# ---------------------------------------------------------------------------

class ParsedFact(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: int | None = None
    premise_id: int | None = None
    original: str
    fact: str

    @field_validator("original", "fact")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("field must be non-empty")
        return value


class ParsedRule(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: int | None = None
    premise_id: int | None = None
    original: str
    if_: list[str] = Field(alias="if")
    then: list[str]

    @field_validator("original")
    @classmethod
    def _non_empty_original(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("original must be non-empty")
        return value

    @field_validator("if_", "then")
    @classmethod
    def _non_empty_list(cls, value: list[str]) -> list[str]:
        cleaned = [str(v or "").strip() for v in value if str(v or "").strip()]
        if not cleaned:
            raise ValueError("list must be non-empty")
        return cleaned


class PremiseMapItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    premise_id: int
    original: str
    kind: Literal["fact", "rule", "skip"]
    fact: str | None = None
    rule: ParsedRule | None = None
    skip_reason: str | None = None


class ParsedOption(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    text: str
    query_clauses: list[str] = Field(default_factory=list)


class LogicStructuredParse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    task_type: Literal["logic"] = "logic"
    question_type: Literal["mcq", "yes_no", "open", "unknown"] = "unknown"
    facts: list[ParsedFact] = Field(default_factory=list)
    rules: list[ParsedRule] = Field(default_factory=list)
    premise_map: list[PremiseMapItem] = Field(default_factory=list)
    query: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, ParsedOption] = Field(default_factory=dict)
    uncertainty_notes: list[str] = Field(default_factory=list)


@dataclass
class StructuredParseResult:
    data: dict[str, Any] = field(default_factory=dict)
    accepted: bool = False
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

_OPTION_RE = re.compile(r"\b([A-D])\.\s*(.*?)(?=\n\s*[A-D]\.\s*|\Z)", flags=re.S)


def _clean_spaces(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _norm(text: Any) -> str:
    text = _clean_spaces(text).lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _similar(a: Any, b: Any) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _contains_similar_long_fragment(haystack: str, needle: str, min_words: int = 5) -> bool:
    h = _norm(haystack)
    n = _norm(needle)
    if not h or not n:
        return False
    if len(n.split()) >= min_words and n in h:
        return True
    return False


def _extract_options(question: str) -> dict[str, str]:
    return {letter.upper(): _clean_spaces(body) for letter, body in _OPTION_RE.findall(question or "")}


def _strip_options(question: str) -> str:
    return re.split(r"\n\s*A\.\s*", str(question or ""), maxsplit=1)[0].strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _id_from_item(item: Any) -> int | None:
    if isinstance(item, BaseModel):
        raw = getattr(item, "premise_id", None) or getattr(item, "id", None)
    elif isinstance(item, dict):
        raw = item.get("premise_id") or item.get("id")
    else:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _item_original(item: Any) -> str:
    if isinstance(item, BaseModel):
        return _clean_spaces(getattr(item, "original", ""))
    if isinstance(item, dict):
        return _clean_spaces(item.get("original", ""))
    return ""


def _looks_like_question_or_option(text: str) -> bool:
    low = _norm(text)
    bad_fragments = [
        "which conclusion", "which statement", "which capabilities", "based on",
        "according to", "does it follow", "does the", "can dr", "can professor",
        "option a", "option b", "option c", "option d", "answer is",
        "strongest conclusion", "correct conclusion",
    ]
    return any(fragment in low for fragment in bad_fragments)


def _looks_like_rule_text(text: str) -> bool:
    low = _norm(text)
    return (
        low.startswith("if ")
        or " then " in f" {low} "
        or low.startswith("all ")
        or low.startswith("every ")
        or " who " in f" {low} "
        or " implies " in f" {low} "
        or " there exists " in f" {low} "
    )


def _validate_source_grounding(
    item: Any,
    premises: list[str],
    warnings: list[str],
    label: str,
) -> bool:
    pid = _id_from_item(item)
    original = _item_original(item)

    if pid is None:
        warnings.append(f"STRUCTURED_PARSE_REJECTED: {label} has no premise_id/id.")
        return False

    if pid < 1 or pid > len(premises):
        warnings.append(f"STRUCTURED_PARSE_REJECTED: {label} premise_id {pid} is out of range.")
        return False

    official = premises[pid - 1]
    sim = _similar(original, official)
    if sim < 0.74 and not _contains_similar_long_fragment(official, original, min_words=6):
        warnings.append(
            f"STRUCTURED_PARSE_REJECTED: {label} premise_id {pid} original text does not match the source premise."
        )
        return False

    return True


def _validate_no_question_contamination(
    clause: str,
    question: str,
    premises: list[str],
    warnings: list[str],
    label: str,
) -> bool:
    if _looks_like_question_or_option(clause):
        warnings.append(f"STRUCTURED_PARSE_REJECTED: {label} appears to contain question/option wording.")
        return False

    options = _extract_options(question)
    clause_norm = _norm(clause)

    for letter, option in options.items():
        option_norm = _norm(option)
        if not option_norm:
            continue
        # If a clause is almost exactly an option but not grounded in any premise,
        # it is likely query contamination.
        option_like = option_norm in clause_norm or _similar(clause, option) >= 0.82
        premise_grounded = any(_similar(clause, p) >= 0.74 for p in premises)
        if option_like and not premise_grounded:
            warnings.append(f"STRUCTURED_PARSE_REJECTED: {label} overlaps MCQ option {letter}.")
            return False

    question_stem = _strip_options(question)
    if _similar(clause, question_stem) >= 0.78:
        warnings.append(f"STRUCTURED_PARSE_REJECTED: {label} overlaps the question stem.")
        return False

    return True


def _sanitize_logic_parse(
    parsed: LogicStructuredParse,
    request: PredictRequest,
) -> StructuredParseResult:
    warnings: list[str] = []
    premises = [str(p or "").strip() for p in (request.premises_nl or []) if str(p or "").strip()]

    if not premises:
        return StructuredParseResult(
            data={},
            accepted=False,
            warnings=["STRUCTURED_PARSE_REJECTED: logic request has no premises."],
            raw=parsed.model_dump(mode="json", by_alias=True),
        )

    sanitized_facts: list[dict[str, Any]] = []
    sanitized_rules: list[dict[str, Any]] = []
    sanitized_premise_map: list[dict[str, Any]] = []

    for fact in parsed.facts:
        if not _validate_source_grounding(fact, premises, warnings, "fact"):
            continue
        if not _validate_no_question_contamination(fact.fact, request.question, premises, warnings, "fact"):
            continue
        if _looks_like_rule_text(fact.fact):
            warnings.append("STRUCTURED_PARSE_REJECTED: fact appears to be a rule or quantified statement.")
            continue
        sanitized_facts.append(fact.model_dump(mode="json", by_alias=True))

    for rule in parsed.rules:
        if not _validate_source_grounding(rule, premises, warnings, "rule"):
            continue
        bad_clause = False
        for clause in list(rule.if_) + list(rule.then):
            if not _validate_no_question_contamination(clause, request.question, premises, warnings, "rule clause"):
                bad_clause = True
                break
        if bad_clause:
            continue
        sanitized_rules.append(rule.model_dump(mode="json", by_alias=True))

    for item in parsed.premise_map:
        if item.kind == "skip":
            sanitized_premise_map.append(item.model_dump(mode="json", by_alias=True, exclude_none=True))
            continue
        if not _validate_source_grounding(item, premises, warnings, "premise_map item"):
            continue

        if item.kind == "fact":
            if not item.fact or _looks_like_rule_text(item.fact):
                warnings.append("STRUCTURED_PARSE_REJECTED: premise_map fact is empty or rule-like.")
                continue
            if not _validate_no_question_contamination(item.fact, request.question, premises, warnings, "premise_map fact"):
                continue
            # Convert premise_map facts into top-level facts because the current
            # solver bridge consumes top-level `fact`, but premise_map fact entries
            # would be re-parsed from `original` and can duplicate/noise the KB.
            sanitized_facts.append({
                "id": item.premise_id,
                "premise_id": item.premise_id,
                "original": item.original,
                "fact": item.fact,
            })
            continue

        if item.kind == "rule":
            if item.rule is None:
                warnings.append("STRUCTURED_PARSE_REJECTED: premise_map rule item has no rule object.")
                continue
            clauses = list(item.rule.if_) + list(item.rule.then)
            if any(not _validate_no_question_contamination(c, request.question, premises, warnings, "premise_map rule") for c in clauses):
                continue
            rule_dict = item.rule.model_dump(mode="json", by_alias=True)
            rule_dict["id"] = item.premise_id
            rule_dict["premise_id"] = item.premise_id
            rule_dict["original"] = item.original
            sanitized_rules.append(rule_dict)

    # Deduplicate items after merging premise_map into top-level facts/rules.
    def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for item in items:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    sanitized_facts = _dedupe(sanitized_facts)
    sanitized_rules = _dedupe(sanitized_rules)

    total_items = len(sanitized_facts) + len(sanitized_rules)

    if total_items == 0:
        warnings.append("STRUCTURED_PARSE_REJECTED: no safe grounded facts/rules survived validation.")
        return StructuredParseResult(
            data={},
            accepted=False,
            warnings=warnings,
            raw=parsed.model_dump(mode="json", by_alias=True),
        )

    if total_items > max(4, len(premises) * 3):
        warnings.append("STRUCTURED_PARSE_REJECTED: parse produced too many derived items for the number of premises.")
        return StructuredParseResult(
            data={},
            accepted=False,
            warnings=warnings,
            raw=parsed.model_dump(mode="json", by_alias=True),
        )

    data = parsed.model_dump(mode="json", by_alias=True)
    data["facts"] = sanitized_facts
    data["rules"] = sanitized_rules
    data["premise_map"] = sanitized_premise_map
    data["parser_validation"] = {
        "accepted": True,
        "safe_items": total_items,
        "warnings": warnings,
    }

    # Validation warnings are kept as diagnostics, but non-fatal if some safe
    # items survived. Fatal errors above return accepted=False.
    return StructuredParseResult(data=data, accepted=True, warnings=warnings, raw=parsed.model_dump(mode="json", by_alias=True))


# ---------------------------------------------------------------------------
# Parser implementation
# ---------------------------------------------------------------------------

class StructuredParser:
    """Zero-shot, schema-gated semantic parser.

    The parser is deliberately a helper, not an answer authority. It only parses
    grounded premise facts/rules that the deterministic solver may verify. If
    parsing fails or validation rejects the output, the pipeline must fall back
    to the baseline solver path.
    """

    def __init__(self, llm: QwenClient | None = None, use_llm: bool = False):
        self.llm = llm
        self.use_llm = use_llm

    def parse(self, request: PredictRequest, task_type: TaskType) -> dict[str, Any]:
        result = self.parse_with_diagnostics(request, task_type)
        return result.data if result.accepted else {}

    def parse_with_diagnostics(self, request: PredictRequest, task_type: TaskType) -> StructuredParseResult:
        if not self.use_llm or self.llm is None or self.llm.backend == "none":
            return StructuredParseResult(data={}, accepted=False, warnings=[])

        if task_type != "logic":
            # The current physics solver owns formula/quantity extraction. Passing
            # unverified LLM physics parses risks formula noise without a consumer
            # contract, so zero-shot parser is enabled for logic only.
            return StructuredParseResult(
                data={},
                accepted=False,
                warnings=["STRUCTURED_PARSE_SKIPPED: zero-shot structured parser is enabled for logic only."],
            )

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._user_prompt(request)},
        ]

        schema = LogicStructuredParse.model_json_schema()

        try:
            raw = self.llm.generate_json(messages, schema=schema, max_retries=1)
        except Exception as exc:
            return StructuredParseResult(
                data={},
                accepted=False,
                warnings=[f"STRUCTURED_PARSE_ERROR: {exc}"],
            )

        try:
            parsed = LogicStructuredParse.model_validate(raw)
        except ValidationError as exc:
            return StructuredParseResult(
                data={},
                accepted=False,
                raw=raw,
                warnings=[f"STRUCTURED_PARSE_ERROR: schema validation failed: {exc}"],
            )

        return _sanitize_logic_parse(parsed, request)

    def _system_prompt(self) -> str:
        return (
            "You are a strict semantic parser for educational logic QA. "
            "Return JSON only, conforming exactly to the provided schema. "
            "Your job is to parse PREMISES ONLY into grounded facts and Horn-style rules. "
            "Do not solve the question. Do not choose an answer. Do not add any fact, rule, or condition that is not explicitly stated in a premise. "
            "Never turn answer options or the question text into facts or rules. "
            "Use premise_id as a 1-based index into premises-NL. The original field must copy the corresponding premise text. "
            "For rules, put atomic natural-language condition clauses in `if` and atomic conclusion clauses in `then`; avoid FOL notation. "
            "For facts, write a short natural-language fact with the concrete subject preserved. "
            "If a premise is ambiguous, mark it as kind=skip in premise_map and explain in skip_reason. "
            "Preserve negation and missing requirements exactly. "
            "For the query field, include a query_clauses list containing the atomic claim(s) asked by the question when possible. "
            "For each MCQ option, fill options.<A/B/C/D>.query_clauses with the atomic claim(s) that option asserts. "
            "Option/query clauses are read-only targets for verification and must not be copied into facts/rules."
        )

    def _user_prompt(self, request: PredictRequest) -> str:
        options = _extract_options(request.question)
        payload = {
            "task_type_hint": "logic",
            "premises-NL": [
                {"premise_id": i + 1, "text": p}
                for i, p in enumerate(request.premises_nl or [])
            ],
            "question": request.question,
            "options_read_only_do_not_parse_as_facts": options,
            "required_output_contract": {
                "facts": "Only explicit premise facts, with valid premise_id.",
                "rules": "Only explicit premise rules, with valid premise_id.",
                "premise_map": "One item per premise when possible; use skip for unsafe premises.",
                "query/options": "Describe query/options as target claims only. For query use query.query_clauses. For options use options.A/B/C/D.query_clauses. Never place query/options in facts/rules/premise_map.",
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
