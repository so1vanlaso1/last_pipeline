"""Type 2 (physics) adapter.

Deterministic-first, LLM-adjudicated physics pipeline.

Flow used for competition serving:
  1) Run the exact_fama deterministic formula/template solver first.
  2) Ask an open-source <=8B LLM to solve the same problem independently.
  3) Give both solutions to an LLM adjudicator, which verifies formula choice,
     quantity extraction, arithmetic, and unit handling before returning the final
     answer.

The deterministic solver is treated as a candidate reference, not as ground
truth. Its internal confidence is deliberately hidden from the LLM adjudicator to
avoid anchoring small models. Python-side safety gates remain conservative: a
conflicting LLM/corrected answer must be parseable, unit-valid, and backed by an
explicit solver-side error diagnosis before it may replace the solver result.

Useful switches:
  PHYSICS_LLM_ADJUDICATION=0       disable the new solve+judge layer
  PHYSICS_ADJUDICATE_ALWAYS=0      adjudicate only when solver/LLM disagree or
                                   the solver is weak/uncertain
  PHYSICS_SOLVER_STRONG_CONF=0.995 confidence above which solver needs very strong
                                   evidence before override
  PHYSICS_MEDIUM_SOLVER_CONF=0.80  confidence above which LLM override must be
                                   explicit and self-consistent
  PHYSICS_ADJUDICATOR_MIN_CONF=0.72 minimum adjudicator confidence for override
  PHYSICS_LLM_FALLBACK=0           disable the old uncertain-only fallback
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import _paths  # noqa: F401  (side-effect: put physic_pipeline/src on sys.path)
from .io_log import model_labels
from .residency import get_manager
from .schema import PredictQuery, PredictResult, Reasoning
from .units import latex_to_ascii, to_ascii_answer, to_ascii_unit
from .vllm_client import LLMClient, extract_last_json_object

_UNCERTAIN = {"", "uncertain", "unknown", "none", "null", "n/a"}

_NUM_RE = re.compile(
    r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|"
    r"(?:\d+(?:\.\d*)?|\.\d+)\s*(?:x|\*)\s*10\s*\^?\s*\{?\s*[-+]?\d+\s*\}?)",
    re.I,
)
_X10_RE = re.compile(
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:x|\*)\s*10\s*\^?\s*\{?\s*([-+]?\d+)\s*\}?",
    re.I,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _build_settings(base_url: str, model: str):
    """Construct exact_fama Settings in-code. The deterministic solver remains the
    main workhorse; physics canonicalization stays rule-based by default."""
    from exact_fama.config import Settings  # type: ignore

    mode = os.environ.get("GATEWAY_LLM", "vllm").lower()
    use_llm = mode == "vllm"
    backend = "openai_compatible" if use_llm else "none"
    use_llm_expl = use_llm and os.environ.get("PHYSICS_LLM_EXPLANATION", "0") == "1"
    raw: Dict[str, Any] = {
        "model": {
            "backend": backend,
            "name": model,
            "vllm_base_url": base_url,
            "temperature": 0.0,
            "max_new_tokens": int(os.environ.get("PHYSICS_MAX_NEW_TOKENS", "768")),
        },
        "pipeline": {
            "use_llm_for_explanation": use_llm_expl,
            "use_llm_for_structured_parse": False,
            "use_physics_canonicalizer": True,
            "use_llm_for_physics_canonicalizer": False,
            "physics_canonicalizer": {
                "enabled": True,
                "use_llm": False,
                "policy": "fallback_only",
            },
            "numeric_tolerance": 1.0e-6,
        },
    }
    return Settings(raw=raw)


@dataclass
class PhysicsCandidate:
    source: str
    answer: str = ""
    unit: str = ""
    explanation: str = ""
    steps: List[str] = field(default_factory=list)
    confidence: float = 0.0
    formula: str = ""
    warnings: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_uncertain(self) -> bool:
        return _is_uncertain(self.answer)

    @property
    def number(self) -> Optional[float]:
        return _parse_number(self.answer)


def _is_uncertain(answer: Any) -> bool:
    return str(answer or "").strip().lower() in _UNCERTAIN


def _parse_number(value: Any) -> Optional[float]:
    """Best-effort parser for answer fields. Accepts plain floats, e-notation and
    'a x 10^b'. Returns None for symbolic/empty answers."""
    if value is None:
        return None
    s = to_ascii_answer(str(value)).strip()
    if not s or s.lower() in _UNCERTAIN:
        return None
    s = s.replace(",", "")

    def x10_sub(m: re.Match[str]) -> str:
        return str(float(m.group(1)) * (10.0 ** int(m.group(2))))

    s2 = _X10_RE.sub(x10_sub, s)
    try:
        return float(s2)
    except Exception:
        pass
    m = _NUM_RE.search(s2)
    if not m:
        return None
    token = _X10_RE.sub(x10_sub, m.group(0))
    try:
        return float(token)
    except Exception:
        return None


def _unit_key(unit: Any) -> str:
    u = to_ascii_unit(str(unit or "")).strip().lower()
    u = u.replace(" ", "")
    return u


def _units_compatible(a: Any, b: Any) -> bool:
    return _unit_key(a) == _unit_key(b)


def _numbers_close(a: Optional[float], b: Optional[float], *, rel: float = 1e-4, abs_tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return False
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


def _same_candidate_value(a: PhysicsCandidate, b: PhysicsCandidate) -> bool:
    return _numbers_close(a.number, b.number) and _units_compatible(a.unit, b.unit)


def _format_candidate(c: PhysicsCandidate) -> str:
    steps = c.steps[:6]
    debug_bits = []
    if c.formula:
        debug_bits.append(f"formula={c.formula}")
    solver_name = c.debug.get("solver") if isinstance(c.debug, dict) else None
    if solver_name:
        debug_bits.append(f"solver={solver_name}")
    if c.warnings:
        debug_bits.append("warnings=" + "; ".join(str(w) for w in c.warnings[:4]))
    return (
        f"Source: {c.source}\n"
        f"Answer: {c.answer or 'Uncertain'}\n"
        f"Unit: {c.unit or '(empty)'}\n"
        f"Explanation: {c.explanation or '(none)'}\n"
        f"Steps: " + (" | ".join(steps) if steps else "(none)") + "\n"
        f"Diagnostics: " + ("; ".join(debug_bits) if debug_bits else "(none)")
    )


def _coerce_steps(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()][:8]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _coerce_confidence(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if math.isnan(v) or math.isinf(v):
        return float(default)
    return max(0.0, min(1.0, v))


class PhysicsAdapter:
    def __init__(self, client: LLMClient, judges: Optional[List[Tuple[Any, ...]]] = None):
        self.client = client
        self.judges = list(judges or [])
        self.fallback_enabled = _env_bool("PHYSICS_LLM_FALLBACK", True) and client.mode == "vllm"
        self.adjudication_enabled = _env_bool("PHYSICS_LLM_ADJUDICATION", True) and client.mode == "vllm"
        self.adjudicate_always = _env_bool("PHYSICS_ADJUDICATE_ALWAYS", True)
        self.strong_solver_conf = _env_float("PHYSICS_SOLVER_STRONG_CONF", 0.995)
        self.medium_solver_conf = _env_float("PHYSICS_MEDIUM_SOLVER_CONF", 0.80)
        self.adjudicator_min_conf = _env_float("PHYSICS_ADJUDICATOR_MIN_CONF", 0.72)
        self.pipeline = None
        self.import_error: Optional[str] = None
        try:
            from exact_fama.pipeline import ExactFamaPipeline  # type: ignore
            self.pipeline = ExactFamaPipeline(_build_settings(client.base_url, client.model))
        except Exception as exc:  # pragma: no cover - exercised only if deps missing
            self.import_error = f"{type(exc).__name__}: {exc}"

    def answer(self, q: PredictQuery) -> PredictResult:
        question = latex_to_ascii(q.query or "")
        if self.pipeline is None:
            return self._fallback_or_uncertain(q, reason=f"physics pipeline unavailable ({self.import_error})")

        solver = self._run_solver(question)
        final = solver
        adjudicator_note = ""

        if self.adjudication_enabled:
            llm_candidate = self._llm_independent_solve(question, q.query_id)
            if llm_candidate is not None:
                should_adjudicate = self.adjudicate_always or solver.is_uncertain or not _same_candidate_value(solver, llm_candidate)
                if should_adjudicate:
                    judged, note = self._llm_adjudicate(question, solver, llm_candidate, q.query_id)
                    if judged is not None:
                        final, adjudicator_note = judged, note
                elif not solver.is_uncertain:
                    final = self._merge_agreement(solver, llm_candidate)
                    adjudicator_note = "The independent LLM solution agreed numerically with the deterministic solver."
                elif not llm_candidate.is_uncertain:
                    final = llm_candidate
                    adjudicator_note = "The deterministic solver abstained; the independent LLM solution supplied a value."
        elif solver.is_uncertain and self.fallback_enabled:
            fb = self._llm_fallback(question, q.query_id)
            if fb is not None:
                final = fb
                adjudicator_note = "Fallback computation supplied a value after the deterministic solver abstained."

        if final.is_uncertain and self.fallback_enabled:
            fb = self._llm_fallback(question, q.query_id)
            if fb is not None:
                final = fb
                adjudicator_note = "Fallback computation supplied a value after adjudication could not produce one."

        answer = to_ascii_answer(final.answer) or "Uncertain"
        unit = to_ascii_unit(final.unit)
        explanation = self._final_explanation(final, solver, adjudicator_note)
        steps = final.steps or solver.steps or [explanation]

        return PredictResult(
            query_id=q.query_id,
            answer=answer,
            unit=unit,
            explanation=explanation,
            premises_used=[],
            reasoning=Reasoning(type="cot", steps=steps),
        )

    # ── deterministic baseline ───────────────────────────────────────────────
    def _run_solver(self, question: str) -> PhysicsCandidate:
        from exact_fama.schemas import PredictRequest  # type: ignore

        req = PredictRequest(question=question, type="physics")
        resp = self.pipeline.predict(req)
        debug = dict(resp.debug or {})
        formula = str(debug.get("formula", "") or "")
        answer = to_ascii_answer(resp.answer)
        unit = to_ascii_unit(resp.unit)
        steps: List[str] = [str(s) for s in (resp.cot or [])]
        explanation = (resp.explanation or "").strip() or "Computed from the stated physical quantities."
        return PhysicsCandidate(
            source="deterministic_solver",
            answer=answer,
            unit=unit,
            explanation=explanation,
            steps=steps,
            confidence=_coerce_confidence(getattr(resp, "confidence", 0.0), 0.0),
            formula=formula,
            warnings=[str(w) for w in (resp.warnings or [])],
            debug=debug,
            raw=resp.model_dump(exclude_none=False) if hasattr(resp, "model_dump") else {},
        )

    # ── LLM stage 1: independent physics solution ────────────────────────────
    def _llm_independent_solve(self, question: str, query_id: str = "") -> Optional[PhysicsCandidate]:
        system = (
            "You are an expert physics solver for educational numerical problems. "
            "Solve the problem independently from first principles before writing the final JSON. "
            "Return ONE JSON object at the end with exactly these fields: "
            "{\"answer\": <numeric value only, no unit>, \"unit\": <ASCII unit>, "
            "\"formula\": <main formula>, \"confidence\": <0..1>, "
            "\"steps\": [<concise verification steps>]}. "
            "Use SI conversions carefully. Do not put the unit inside answer."
        )
        user = f"Problem:\n{question}\n\nSolve independently. End with the JSON object only."
        data = self._chat_object_thinking(
            self.client,
            system,
            user,
            query_id=query_id,
            stage="physics.independent_solve",
            loaded_clients=[self.client],
            max_tokens=int(os.environ.get("PHYSICS_SELF_SOLVE_TOKENS", "2048")),
        )
        if not isinstance(data, dict):
            return None
        answer = to_ascii_answer(data.get("answer"))
        if _is_uncertain(answer):
            return None
        return PhysicsCandidate(
            source="llm_independent",
            answer=answer,
            unit=to_ascii_unit(data.get("unit", "")),
            explanation=str(data.get("explanation", "") or "").strip(),
            steps=_coerce_steps(data.get("steps")),
            confidence=_coerce_confidence(data.get("confidence"), 0.55),
            formula=str(data.get("formula", "") or "").strip(),
            raw=data,
        )

    # ── LLM stage 2: adjudication / verification ─────────────────────────────
    def _llm_adjudicate(
        self,
        question: str,
        solver: PhysicsCandidate,
        llm_candidate: PhysicsCandidate,
        query_id: str = "",
    ) -> Tuple[Optional[PhysicsCandidate], str]:
        arbiter = self._judge_client() or self.client
        system = (
            "You are a senior physics verifier. You receive a physics problem and two candidate "
            "solutions. One candidate may come from a deterministic formula/template engine and "
            "the other from an independent LLM, but neither candidate is automatically trusted. "
            "First recompute the problem yourself from the original statement. Then compare both "
            "candidates against your recomputation. Check quantity extraction, unit conversions, "
            "formula choice, arithmetic, and requested output unit. Do not treat any candidate's "
            "internal confidence, source, or style as evidence. Choose the candidate that matches "
            "the verified derivation, or produce a corrected answer if both are wrong. "
            "Think carefully first, then end with ONE JSON object only: "
            "{\"winner\": \"solver\"|\"llm\"|\"corrected\", "
            "\"final_answer\": <numeric value only, no unit>, \"final_unit\": <ASCII unit>, "
            "\"confidence\": <0..1>, \"solver_error\": <true/false>, "
            "\"llm_error\": <true/false>, \"error_analysis\": <short>, "
            "\"explanation\": <1-2 sentences>, \"steps\": [<concise final derivation steps>]}."
        )
        user = (
            f"Problem:\n{question}\n\n"
            "Candidate A:\n"
            f"{_format_candidate(solver)}\n\n"
            "Candidate B:\n"
            f"{_format_candidate(llm_candidate)}\n\n"
            "Recompute from the original problem first. Do not trust either candidate by default. "
            "Choose the final answer only if it passes your verification."
        )

        if arbiter is not self.client:
            with get_manager().judge() as judge_ready:
                data = self._chat_object_thinking(
                    arbiter,
                    system,
                    user,
                    query_id=query_id,
                    stage="physics.adjudicator",
                    loaded_clients=[arbiter],
                    max_tokens=int(os.environ.get("PHYSICS_ADJUDICATE_TOKENS", "3072")),
                ) if judge_ready else None
        else:
            data = self._chat_object_thinking(
                arbiter,
                system,
                user,
                query_id=query_id,
                stage="physics.adjudicator",
                loaded_clients=[arbiter],
                max_tokens=int(os.environ.get("PHYSICS_ADJUDICATE_TOKENS", "3072")),
            )

        if not isinstance(data, dict):
            return None, "Adjudicator did not return parseable JSON; kept deterministic solver."

        chosen = self._choose_judged_candidate(data, solver, llm_candidate)
        if chosen is None:
            return solver, "Adjudicator verdict failed safety checks; kept deterministic solver."
        note = str(data.get("error_analysis") or data.get("explanation") or "").strip()
        return chosen, note

    def _choose_judged_candidate(
        self,
        data: Dict[str, Any],
        solver: PhysicsCandidate,
        llm_candidate: PhysicsCandidate,
    ) -> Optional[PhysicsCandidate]:
        final_answer = to_ascii_answer(data.get("final_answer") or data.get("answer"))
        final_unit = to_ascii_unit(data.get("final_unit") or data.get("unit") or "")
        if _is_uncertain(final_answer):
            return solver if not solver.is_uncertain else None

        verdict = str(data.get("winner") or data.get("chosen") or "").strip().lower()
        conf = _coerce_confidence(data.get("confidence"), 0.55)
        solver_error = bool(data.get("solver_error", False))
        explanation = str(data.get("explanation") or data.get("error_analysis") or "").strip()
        steps = _coerce_steps(data.get("steps"))
        final_num = _parse_number(final_answer)

        # If the judged value is just the deterministic value, keep the deterministic
        # formatting/rounding/unit exactly. This avoids unit-scale regressions.
        if _numbers_close(final_num, solver.number) and (_units_compatible(final_unit, solver.unit) or not final_unit):
            merged = self._merge_solver_with_judgment(solver, data)
            return merged

        # A direct 'solver' winner with a different value is internally inconsistent.
        if verdict == "solver" and not solver.is_uncertain:
            return solver

        # If the solver abstained or is weak, accept any parseable judged answer.
        if solver.is_uncertain or solver.confidence < 0.50:
            return PhysicsCandidate(
                source=f"llm_adjudicated_{verdict or 'corrected'}",
                answer=final_answer,
                unit=final_unit,
                explanation=explanation,
                steps=steps,
                confidence=conf,
                raw=data,
            )

        if final_num is None:
            return solver if not solver.is_uncertain else None

        agrees_llm = _numbers_close(final_num, llm_candidate.number) and (
            _units_compatible(final_unit, llm_candidate.unit) or not final_unit or not llm_candidate.unit
        )
        has_derivation = len(steps) >= 2 or len(explanation) >= 24
        valid_override_verdict = verdict in {"llm", "corrected"}

        # Conservative anti-hallucination gate. For any reasonably confident solver,
        # the adjudicator must explicitly diagnose a solver error, be confident, and
        # either agree with the independent solution or provide corrected derivation
        # evidence. This prevents a 4B judge from casually changing correct solver
        # answers while still allowing rescue of high-confidence solver mistakes.
        if solver.confidence >= self.medium_solver_conf:
            if not (solver_error and valid_override_verdict and conf >= self.adjudicator_min_conf and has_derivation):
                return solver
            if not agrees_llm and not (verdict == "corrected" and len(steps) >= 2):
                return solver

        # For a near-certain deterministic answer, demand even stronger evidence.
        if solver.confidence >= self.strong_solver_conf:
            if conf < max(0.80, self.adjudicator_min_conf):
                return solver
            if not solver_error:
                return solver

        # Low-confidence deterministic result: still require the final answer to be
        # backed by either an LLM agreement or a solver error diagnosis.
        if not solver_error and not agrees_llm:
            return solver if not solver.is_uncertain else None

        return PhysicsCandidate(
            source=f"llm_adjudicated_{verdict or 'corrected'}",
            answer=final_answer,
            unit=final_unit or llm_candidate.unit or solver.unit,
            explanation=explanation,
            steps=steps or llm_candidate.steps,
            confidence=conf,
            raw=data,
        )

    def _merge_solver_with_judgment(self, solver: PhysicsCandidate, data: Dict[str, Any]) -> PhysicsCandidate:
        explanation = str(data.get("explanation") or data.get("error_analysis") or solver.explanation).strip()
        steps = _coerce_steps(data.get("steps")) or solver.steps
        return PhysicsCandidate(
            source="deterministic_solver_verified_by_llm",
            answer=solver.answer,
            unit=solver.unit,
            explanation=explanation or solver.explanation,
            steps=steps,
            confidence=max(solver.confidence, _coerce_confidence(data.get("confidence"), solver.confidence)),
            formula=solver.formula,
            warnings=solver.warnings,
            debug=solver.debug,
            raw=data,
        )

    def _merge_agreement(self, solver: PhysicsCandidate, llm_candidate: PhysicsCandidate) -> PhysicsCandidate:
        steps = solver.steps or llm_candidate.steps
        explanation = solver.explanation or llm_candidate.explanation
        return PhysicsCandidate(
            source="deterministic_solver_llm_agreement",
            answer=solver.answer,
            unit=solver.unit,
            explanation=explanation,
            steps=steps,
            confidence=max(solver.confidence, llm_candidate.confidence),
            formula=solver.formula,
            warnings=solver.warnings,
            debug=solver.debug,
        )

    # ── model selection / chat helpers ───────────────────────────────────────
    def _judge_client(self) -> Optional[LLMClient]:
        for item in self.judges:
            try:
                role = str(item[3] if len(item) > 3 else "").strip().lower()
                if role == "judge":
                    return item[0]
            except Exception:
                continue
        return None

    def _chat_object_thinking(
        self,
        client: LLMClient,
        system: str,
        user: str,
        *,
        query_id: str,
        stage: str,
        loaded_clients: Iterable[LLMClient],
        max_tokens: int,
    ) -> Optional[dict]:
        try:
            text = client.chat(
                system,
                user,
                max_tokens=max_tokens,
                temperature=0.0,
                enable_thinking=True,
                log_context=f"type2 query_id={query_id or 'q'} stage={stage}",
                loaded_models=model_labels(list(loaded_clients)),
            )
        except Exception:
            return None
        return extract_last_json_object(text)

    # ── legacy guarded numeric fallback ──────────────────────────────────────
    def _llm_fallback(self, question: str, query_id: str = "") -> Optional[PhysicsCandidate]:
        system = (
            "You are a physics problem solver for circuits and electrostatics. "
            "Solve the problem step by step, then return JSON only with these fields: "
            "{\"answer\": <the numerical value only, as a plain number>, "
            "\"unit\": <the unit in ASCII, e.g. A, V, ohm, uF, V/m; empty string if none>, "
            "\"steps\": [<short reasoning steps>]}. "
            "Do not include the unit inside the answer field."
        )
        try:
            data = self.client.chat_json(
                system, f"Problem: {question}", max_tokens=768,
                log_context=f"type2 query_id={query_id or 'q'} stage=physics.fallback",
            )
        except Exception:
            data = None
        if not isinstance(data, dict):
            return None
        answer = to_ascii_answer(data.get("answer"))
        if _is_uncertain(answer):
            return None
        return PhysicsCandidate(
            source="llm_fallback",
            answer=answer,
            unit=to_ascii_unit(str(data.get("unit", ""))),
            steps=_coerce_steps(data.get("steps")),
            confidence=0.45,
            raw=data,
        )

    def _fallback_or_uncertain(self, q: PredictQuery, reason: str) -> PredictResult:
        if self.fallback_enabled:
            fb = self._llm_fallback(latex_to_ascii(q.query or ""), q.query_id)
            if fb is not None:
                explanation = self._final_explanation(fb, fb, "")
                return PredictResult(
                    query_id=q.query_id,
                    answer=fb.answer,
                    unit=fb.unit,
                    explanation=explanation,
                    premises_used=[],
                    reasoning=Reasoning(type="cot", steps=fb.steps or [explanation]),
                )
        return PredictResult(
            query_id=q.query_id, answer="Uncertain", unit="",
            explanation=f"Could not compute a physics answer ({reason}).",
            premises_used=[], reasoning=Reasoning(type="cot", steps=[reason]),
        )

    def _final_explanation(self, final: PhysicsCandidate, solver: PhysicsCandidate, note: str) -> str:
        base = (final.explanation or "").strip()
        if not base:
            value = f"{final.answer}{(' ' + final.unit) if final.unit else ''}"
            if final.source.startswith("deterministic"):
                base = f"Computed by the deterministic formula solver as {value}."
            else:
                base = f"Computed and verified as {value}."
        if note:
            base = f"{base} Verification: {note}".strip()
        if final.source != solver.source and not solver.is_uncertain:
            base += " The final value was selected after comparing the deterministic solution with an independent LLM solution."
        return base.strip()
