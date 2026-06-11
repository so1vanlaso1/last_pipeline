from __future__ import annotations

"""Dynamic LLM semantic head for logic QA.

This component is deliberately different from a static dataset head:
- it has no saved mapping, no signatures, and no access to gold labels;
- it reads only the current sample input plus solver/parser diagnostics;
- it can override the symbolic result only under explicit confidence/evidence gates.

The LLM is used as a test-time semantic interpreter for question/options, not as
public-set calibration.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from ..schemas import PredictRequest, SolverResult
from .qwen_client import QwenClient
from .structured_parser import StructuredParseResult


_VALID_LOGIC = {"Yes", "No", "Unknown", "A", "B", "C", "D"}


def _clean(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def _get_extra(request: PredictRequest) -> dict[str, Any]:
    return getattr(request, "model_extra", None) or {}


def _get_premises_fol(request: PredictRequest) -> list[str]:
    extra = _get_extra(request)
    vals = extra.get("premises-FOL") or extra.get("premises_fol") or getattr(request, "premises_fol", None) or []
    if isinstance(vals, str):
        vals = [vals]
    return [str(v) for v in vals if str(v).strip()]


def _extract_options(question: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.finditer(r"\b([A-D])[\.)]\s*(.*?)(?=\n\s*[A-D][\.)]\s*|\Z)", question or "", flags=re.S):
        out[m.group(1).upper()] = _clean(m.group(2))
    return out


def _is_mcq(question: str) -> bool:
    opts = _extract_options(question)
    return "A" in opts and "B" in opts


def _norm_answer(answer: Any, *, is_mcq: bool) -> str:
    text = str(answer or "").strip()
    low = text.lower().strip(" .:;()[]{}")
    m = re.fullmatch(r"(?:option\s*)?([a-d])", low, flags=re.I)
    if m:
        return m.group(1).upper()
    if low in {"yes", "true", "entailed", "correct"}:
        return "Yes"
    if low in {"no", "false", "contradicted", "contradiction", "incorrect"}:
        return "No"
    if low in {"unknown", "uncertain", "undetermined", "cannot be determined", "not enough information", "not enough info"}:
        return "Unknown"
    return "Unknown" if is_mcq or not text else text


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return default


def _as_int_list(x: Any) -> list[int]:
    if x is None:
        return []
    if not isinstance(x, list):
        x = [x]
    out: list[int] = []
    for v in x:
        try:
            iv = int(v)
            if iv > 0:
                out.append(iv)
        except Exception:
            continue
    return sorted(set(out))


def _answer_family(ans: str) -> str:
    if ans in {"A", "B", "C", "D"}:
        return "mcq"
    if ans in {"Yes", "No"}:
        return "yes_no"
    if ans == "Unknown":
        return "unknown"
    return "other"


@dataclass
class SemanticHeadDecision:
    answer: str
    confidence: float
    evidence_premise_ids: list[int]
    explanation: str
    accepted: bool
    reason: str
    raw: dict[str, Any]


class LogicSemanticHead:
    def __init__(self, llm: QwenClient | None, *, use_llm: bool = False, policy: str = "safe") -> None:
        self.llm = llm
        self.use_llm = bool(use_llm)
        self.policy = (policy or "safe").strip().lower()

    def refine(
        self,
        request: PredictRequest,
        result: SolverResult,
        *,
        baseline: SolverResult | None = None,
        candidate: SolverResult | None = None,
        parser_result: StructuredParseResult | None = None,
    ) -> SolverResult:
        if not self.use_llm or self.llm is None or self.llm.backend == "none":
            result.debug.setdefault("logic_semantic_head", {"enabled": False})
            return result

        try:
            decision = self._call(request, result, baseline=baseline, candidate=candidate, parser_result=parser_result)
        except Exception as exc:
            result.debug["logic_semantic_head"] = {
                "enabled": True,
                "accepted": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            result.warnings.append(f"LOGIC_LLM_HEAD_ERROR: {type(exc).__name__}: {exc}")
            return result

        out = result
        if decision.accepted:
            out = SolverResult(
                answer=decision.answer,
                unit=None,
                explanation=decision.explanation or result.explanation,
                fol=result.fol,
                cot=list(result.cot or []) + [
                    "LLM semantic head interpreted the current question/options at test time.",
                    f"LLM semantic head proposed {decision.answer} with confidence {decision.confidence:.2f}.",
                    f"Acceptance gate: {decision.reason}.",
                ],
                premises=list(result.premises or request.premises_nl or []),
                confidence=max(float(result.confidence or 0.0), min(0.92, decision.confidence)),
                warnings=list(result.warnings or []) + [f"LOGIC_INFO: dynamic LLM semantic head accepted override ({decision.reason})."],
                debug=dict(result.debug or {}),
            )
        else:
            out.debug = dict(out.debug or {})
            out.warnings = list(out.warnings or [])
            out.warnings.append(f"LOGIC_INFO: dynamic LLM semantic head rejected ({decision.reason}).")

        out.debug["logic_semantic_head"] = {
            "enabled": True,
            "policy": self.policy,
            "accepted": decision.accepted,
            "reason": decision.reason,
            "answer": decision.answer,
            "confidence": decision.confidence,
            "evidence_premise_ids": decision.evidence_premise_ids,
            "raw": decision.raw,
        }
        return out

    def _call(
        self,
        request: PredictRequest,
        result: SolverResult,
        *,
        baseline: SolverResult | None,
        candidate: SolverResult | None,
        parser_result: StructuredParseResult | None,
    ) -> SemanticHeadDecision:
        question = request.question or ""
        is_mcq = _is_mcq(question)
        messages = self._messages(request, result, baseline=baseline, candidate=candidate, parser_result=parser_result)
        data = self.llm.generate_json(messages, schema=self._schema(is_mcq), max_retries=1)

        answer = _norm_answer(data.get("answer"), is_mcq=is_mcq)
        confidence = _safe_float(data.get("confidence"), 0.0)
        evidence = _as_int_list(data.get("evidence_premise_ids"))
        explanation = _clean(data.get("explanation") or data.get("reasoning") or "")
        accepted, reason = self._accept(answer, confidence, evidence, result, is_mcq=is_mcq, data=data)

        return SemanticHeadDecision(
            answer=answer,
            confidence=confidence,
            evidence_premise_ids=evidence,
            explanation=explanation,
            accepted=accepted,
            reason=reason,
            raw=data,
        )

    def _accept(
        self,
        answer: str,
        confidence: float,
        evidence: list[int],
        symbolic: SolverResult,
        *,
        is_mcq: bool,
        data: dict[str, Any],
    ) -> tuple[bool, str]:
        sym = _norm_answer(symbolic.answer, is_mcq=is_mcq)
        if answer not in _VALID_LOGIC:
            return False, "invalid_answer"
        if is_mcq and answer in {"Yes", "No"}:
            return False, "wrong_answer_family_for_mcq"
        if (not is_mcq) and answer in {"A", "B", "C", "D"}:
            return False, "wrong_answer_family_for_yes_no"
        if answer == sym:
            return False, "same_as_symbolic"
        # Unknown is allowed to override over-entailment only when very confident.
        if answer == "Unknown":
            if confidence >= 0.82:
                return True, "high_confidence_unknown_veto"
            return False, "unknown_not_confident_enough"
        min_evidence = int(os.environ.get("EXACT_LOGIC_HEAD_MIN_EVIDENCE", "1"))
        if len(evidence) < min_evidence:
            return False, "insufficient_premise_evidence"

        # Policy thresholds. Aggressive is useful for local benchmarking; safe is
        # the recommended hidden-test default.
        if self.policy == "aggressive":
            threshold_unknown = 0.62
            threshold_disagree = 0.74
        elif self.policy == "strict":
            threshold_unknown = 0.78
            threshold_disagree = 0.90
        else:
            threshold_unknown = 0.68
            threshold_disagree = 0.84

        if sym in {"Unknown", "Uncertain"}:
            if confidence >= threshold_unknown:
                return True, "filled_symbolic_unknown"
            return False, "fill_unknown_below_threshold"

        # Let the LLM repair deterministic wrapper mistakes only when confidence is high.
        router = ((symbolic.debug or {}).get("hybrid_logic_router") or {})
        router_decision = str(router.get("decision") or "")
        if router_decision in {"conservative_truth_statement_yes_to_no", "legacy_default", "robust_vetoed_unproved_mcq_option"} and confidence >= threshold_disagree:
            return True, "high_confidence_disagreement_repair"
        return False, "disagreement_not_safe"

    def _schema(self, is_mcq: bool) -> dict[str, Any]:
        answer_enum = ["A", "B", "C", "D", "Unknown"] if is_mcq else ["Yes", "No", "Unknown"]
        return {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "answer": {"type": "string", "enum": answer_enum},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence_premise_ids": {"type": "array", "items": {"type": "integer"}},
                "explanation": {"type": "string"},
                "option_analysis": {"type": "object"},
                "query_interpretation": {"type": "string"},
            },
            "required": ["answer", "confidence", "evidence_premise_ids", "explanation"],
        }

    def _messages(
        self,
        request: PredictRequest,
        result: SolverResult,
        *,
        baseline: SolverResult | None,
        candidate: SolverResult | None,
        parser_result: StructuredParseResult | None,
    ) -> list[dict[str, str]]:
        premises_nl = request.premises_nl or []
        options = _extract_options(request.question)
        payload = {
            "question": request.question,
            "question_type": "mcq" if options else "yes_no_or_unknown",
            "options": options,
            "premises_NL_indexed": [{"id": i + 1, "text": p} for i, p in enumerate(premises_nl)],
            "premises_FOL_indexed": [{"id": i + 1, "text": p} for i, p in enumerate(_get_premises_fol(request))],
            "symbolic_result": {
                "answer": result.answer,
                "warnings": result.warnings,
                "decision": ((result.debug or {}).get("hybrid_logic_router") or {}).get("decision"),
            },
            "symbolic_candidates": {
                "baseline_answer": baseline.answer if baseline else None,
                "parser_candidate_answer": candidate.answer if candidate else None,
                "legacy_answer": (((result.debug or {}).get("hybrid_logic_router") or {}).get("legacy_answer")),
                "robust_answer": (((result.debug or {}).get("hybrid_logic_router") or {}).get("robust_answer")),
            },
            "structured_parser": {
                "accepted": bool(parser_result and parser_result.accepted),
                "data": parser_result.data if parser_result and parser_result.accepted else {},
                "warnings": parser_result.warnings if parser_result else [],
            },
            "instruction": {
                "do_not_use_gold": True,
                "do_not_memorize_dataset": True,
                "task": "Infer the answer from the provided premises and FOL only. If neither the statement nor its negation is supported, return Unknown. For MCQ, return A/B/C/D only if that option is best supported; otherwise Unknown.",
            },
        }
        system = (
            "You are a test-time semantic decision head for a logic QA solver. "
            "You do not have gold labels. You must use only the current question, options, premises-NL, premises-FOL, and solver diagnostics. "
            "Return exactly one JSON object. Do not include markdown. "
            "Cite supporting premises by their 1-based IDs in evidence_premise_ids. "
            "Prefer Unknown when the premises do not support a unique answer."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]
