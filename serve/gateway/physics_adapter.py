"""Type 2 (physics) adapter.

Reuses the physics pipeline (`exact_fama.ExactFamaPipeline`) verbatim. The
deterministic formula/template solver decides the answer + unit; the shared vLLM
model (via the pipeline's own QwenClient) drives the optional physics
canonicalizer and the explanation rewrite — exactly as the pipeline intends.

The only addition is a guarded numeric fallback: when the pipeline abstains
("Uncertain"), there is nothing to preserve, so a single chain-of-thought vLLM
call may fill in a value. It never overrides a confident solver answer. Toggle
with PHYSICS_LLM_FALLBACK=0.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from . import _paths  # noqa: F401  (side-effect: put physic_pipeline/src on sys.path)
from .schema import PredictQuery, PredictResult, Reasoning
from .units import latex_to_ascii, to_ascii_answer, to_ascii_unit
from .vllm_client import LLMClient

_UNCERTAIN = {"", "uncertain", "unknown"}


def _build_settings(base_url: str, model: str):
    """Construct exact_fama Settings in-code (no YAML needed). vLLM mode wires the
    pipeline's LLM helpers to the primary shared server; stub mode runs solver-only."""
    from exact_fama.config import Settings  # type: ignore

    mode = os.environ.get("GATEWAY_LLM", "vllm").lower()
    use_llm = mode == "vllm"
    backend = "openai_compatible" if use_llm else "none"
    # Explanation rewrite is OFF by default: explanation is not scored this round and
    # a per-query LLM call hurts the speed bonus. Turn on with PHYSICS_LLM_EXPLANATION=1.
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
            "use_llm_for_structured_parse": False,   # logic-only; not used for physics
            # Keep the canonicalizer RULE-BASED only (deterministic, no external
            # prompt file). The guarded gateway-level LLM fallback (below) handles
            # the cases the solver can't parse, with a prompt we fully control.
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


class PhysicsAdapter:
    def __init__(self, client: LLMClient):
        self.client = client
        self.fallback_enabled = os.environ.get("PHYSICS_LLM_FALLBACK", "1") == "1" and client.mode == "vllm"
        self.pipeline = None
        self.import_error: Optional[str] = None
        try:
            from exact_fama.pipeline import ExactFamaPipeline  # type: ignore
            self.pipeline = ExactFamaPipeline(_build_settings(client.base_url, client.model))
        except Exception as exc:  # pragma: no cover - exercised only if deps missing
            self.import_error = f"{type(exc).__name__}: {exc}"

    def answer(self, q: PredictQuery) -> PredictResult:
        # Convert the committee's LaTeX into the ASCII the deterministic extractor
        # parses (e-notation, ohm, uF, R1) up front, so solving is robust even if the
        # notation-map regex missed something. See units.latex_to_ascii.
        question = latex_to_ascii(q.query or "")
        if self.pipeline is None:
            return self._fallback_or_uncertain(q, reason=f"physics pipeline unavailable ({self.import_error})")

        from exact_fama.schemas import PredictRequest  # type: ignore

        req = PredictRequest(question=question, type="physics")
        resp = self.pipeline.predict(req)

        answer = to_ascii_answer(resp.answer)
        unit = to_ascii_unit(resp.unit)
        steps: List[str] = [str(s) for s in (resp.cot or [])]
        explanation = (resp.explanation or "").strip() or "Computed from the stated physical quantities."

        if answer.lower() in _UNCERTAIN and self.fallback_enabled:
            fb = self._llm_fallback(question, q.query_id)
            if fb is not None:
                answer, unit, fb_steps = fb
                steps = fb_steps or steps
                explanation = (
                    f"{explanation} Fallback computation produced {answer}"
                    f"{(' ' + unit) if unit else ''}."
                ).strip()

        if not answer:
            answer = "Uncertain"

        return PredictResult(
            query_id=q.query_id,
            answer=answer,
            unit=unit,
            explanation=explanation,
            premises_used=[],                      # always [] for Type 2
            reasoning=Reasoning(type="cot", steps=steps or [explanation]),
        )

    # ── guarded numeric fallback ──────────────────────────────────────────────
    def _llm_fallback(self, question: str, query_id: str = ""):
        system = (
            "You are a physics problem solver for circuits and electrostatics. "
            "Solve the problem step by step, then return JSON only with these fields: "
            "{\"answer\": <the numerical value only, as a plain number>, "
            "\"unit\": <the unit in ASCII, e.g. A, V, ohm, uF, V/m; empty string if none>, "
            "\"steps\": [<short reasoning steps>]}. "
            "Do not include the unit inside the answer field."
        )
        try:
            # log_context routes this call into serve/logs/log.txt — without it the
            # Type 2 physics fallback was invisible in the model I/O log (the log
            # showed only Type 1 stages, making physics look like it never ran).
            data = self.client.chat_json(
                system, f"Problem: {question}", max_tokens=768,
                log_context=f"type2 query_id={query_id or 'q'} stage=physics.fallback",
            )
        except Exception:
            data = None
        if not isinstance(data, dict):
            return None
        answer = to_ascii_answer(data.get("answer"))
        if not answer or answer.lower() in _UNCERTAIN:
            return None
        unit = to_ascii_unit(str(data.get("unit", "")))
        steps = [str(s) for s in (data.get("steps") or [])]
        return answer, unit, steps

    def _fallback_or_uncertain(self, q: PredictQuery, reason: str) -> PredictResult:
        if self.fallback_enabled:
            fb = self._llm_fallback(latex_to_ascii(q.query or ""), q.query_id)
            if fb is not None:
                answer, unit, steps = fb
                return PredictResult(
                    query_id=q.query_id, answer=answer, unit=unit,
                    explanation=f"Computed via fallback. {answer}{(' ' + unit) if unit else ''}.",
                    premises_used=[], reasoning=Reasoning(type="cot", steps=steps),
                )
        return PredictResult(
            query_id=q.query_id, answer="Uncertain", unit="",
            explanation=f"Could not compute a physics answer ({reason}).",
            premises_used=[], reasoning=Reasoning(type="cot", steps=[reason]),
        )
