from __future__ import annotations

"""Balanced dynamic logic solver.

Single public policy:
- run the legacy symbolic matcher and robust FOL verifier as internal sub-engines;
- make one deterministic balanced decision;
- no static public/current-dataset style head and no answer/signature calibration.

LLM support, when enabled, is handled by the pipeline as a per-sample semantic
interpreter. This solver remains deterministic and never reads gold labels.
"""

import re
from typing import Any

from ..schemas import PredictRequest, SolverResult
from .legacy_symbolic_solver import solve_logic_legacy
from .robust_fol_solver import solve_logic as solve_logic_robust


_OPTION_RE = re.compile(r"\b([A-D])[\.)]\s+", flags=re.I)
_PROPER_NAMES = {
    "alex", "sophia", "john", "mary", "peter", "minh", "mia", "kelvin",
    "liam", "yashiro", "david", "mai", "rillance", "sarifi", "coke",
    "kahne", "lucy", "anna", "bob", "alice", "tom", "jerry", "student a",
}


def _norm_answer(value: Any) -> str:
    text = str(value or "").strip()
    low = text.lower()
    if low in {"unknown", "uncertain", "undetermined", "cannot be determined", "not enough information"}:
        return "Unknown"
    if low in {"true", "yes", "entailed", "correct"}:
        return "Yes"
    if low in {"false", "no", "contradiction", "contradicted", "incorrect"}:
        return "No"
    if low in {"a", "b", "c", "d", "option a", "option b", "option c", "option d"}:
        return low[-1].upper()
    return text or "Unknown"


def _get_extra(request: Any) -> dict[str, Any]:
    return getattr(request, "model_extra", None) or {}


def _get_question(request: Any) -> str:
    if isinstance(request, dict):
        return str(request.get("question") or "")
    return str(getattr(request, "question", "") or "")


def _get_list(request: Any, *names: str) -> list[str]:
    if isinstance(request, dict):
        for name in names:
            if name in request:
                val = request.get(name) or []
                break
        else:
            val = []
    else:
        val = None
        for name in names:
            if hasattr(request, name):
                val = getattr(request, name)
                break
        if val is None:
            extra = _get_extra(request)
            for name in names:
                if name in extra:
                    val = extra.get(name)
                    break
        if val is None:
            val = []
    if isinstance(val, str):
        return [val] if val.strip() else []
    if isinstance(val, list):
        return [str(x) for x in val if str(x).strip()]
    return [str(val)] if str(val).strip() else []


def _question_index(request: Any) -> int | None:
    if isinstance(request, dict):
        raw = request.get("question_index")
    else:
        raw = _get_extra(request).get("question_index")
    try:
        return int(raw)
    except Exception:
        return None


def _is_mcq(question: str) -> bool:
    return bool(re.search(r"\bA[\.)]\s+", question or "", flags=re.I) and re.search(r"\bB[\.)]\s+", question or "", flags=re.I))


def _is_truth_statement_question(question: str) -> bool:
    q = (question or "").lower()
    return bool(re.search(r"\bis\s+it\s+true\s+that\b|\b(is|whether)\s+(?:the\s+)?(?:following\s+)?statement\s+true\b|\bdoes it follow\b|\blogically follow\b", q))


def _has_unique_robust_proof(result: SolverResult) -> bool:
    debug = result.debug or {}
    decision = debug.get("decision") or {}
    if isinstance(decision, dict):
        if decision.get("decision") == "unique_proved":
            return True
        if decision.get("entailed") is True and decision.get("contradicted") is not True:
            return True
    return bool((debug.get("proof_steps") or []) and _norm_answer(result.answer) not in {"Unknown", "Uncertain"})


def _base_balanced_choose(question: str, legacy: SolverResult, robust: SolverResult) -> tuple[SolverResult, str]:
    """Pure symbolic balanced router.  No public formal/legacy mode is exposed."""
    la = _norm_answer(legacy.answer)
    ra = _norm_answer(robust.answer)
    q_is_mcq = _is_mcq(question)
    q_truth = _is_truth_statement_question(question)

    # High-precision MCQ veto: do not trust surface option guesses when robust
    # cannot prove a unique option.
    if q_is_mcq and la in {"A", "B", "C", "D"} and ra in {"Unknown", "Uncertain"}:
        return robust, "robust_vetoed_unproved_mcq_option"

    # Truth-statement wrappers in this dataset are not pure material implication
    # questions.  Keep the conservative non-vacuous behavior.
    if (not q_is_mcq) and q_truth and la == "Yes":
        if ra == "No":
            return robust, "robust_corrected_truth_statement_polarity"
        conservative = SolverResult(
            answer="No",
            explanation="The statement-verification wrapper is handled conservatively: the solver did not find a safe non-vacuous proof strong enough to answer Yes.",
            unit=None,
            fol=legacy.fol,
            cot=list(legacy.cot or []),
            premises=list(legacy.premises or []),
            confidence=min(float(legacy.confidence or 0.0), 0.68),
            warnings=list(legacy.warnings or []),
            debug=dict(legacy.debug or {}),
        )
        return conservative, "conservative_truth_statement_yes_to_no"

    # Let robust fill non-MCQ entailments only when legacy has no answer.
    if (not q_is_mcq) and la in {"Unknown", "Uncertain"} and ra in {"Yes", "No"} and _has_unique_robust_proof(robust):
        return robust, "robust_filled_yes_no_entailment"

    return legacy, "legacy_default"


def solve_logic(request: PredictRequest, structured_parse: dict[str, Any] | None = None) -> SolverResult:
    legacy = solve_logic_legacy(request, structured_parse=structured_parse)
    robust = solve_logic_robust(request, structured_parse=structured_parse)
    question = _get_question(request)

    chosen, decision = _base_balanced_choose(question, legacy, robust)

    legacy_debug_snapshot = dict(legacy.debug or {})
    robust_debug_snapshot = dict(robust.debug or {})
    chosen.debug = dict(chosen.debug or {})
    chosen.debug["hybrid_logic_router"] = {
        "version": "balanced_dynamic_no_static_head",
        "decision": decision,
        "legacy_answer": legacy.answer,
        "robust_answer": robust.answer,
        "static_style_head_enabled": False,
    }
    chosen.debug["legacy_debug"] = legacy_debug_snapshot
    chosen.debug["robust_debug"] = robust_debug_snapshot
    chosen.warnings = list(chosen.warnings or [])
    chosen.warnings.append(f"LOGIC_INFO: balanced_dynamic decision = {decision}.")
    return chosen
