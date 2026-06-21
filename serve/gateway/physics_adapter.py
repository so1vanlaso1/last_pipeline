"""Type 2 (physics) adapter.

The gateway keeps the deterministic physics pipeline as the first candidate, then
uses the shared serve/vLLM line-up in the same generate-to-judge shape as Type 1:

    deterministic solver answer
        -> two 4B generator answers, with the solver answer as reference only
        -> 8B judge solves independently and arbitrates across all three

The public competition schema is unchanged. Model selection stays entirely in
serve/logic_config.yaml.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

from . import _paths  # noqa: F401  (side-effect: put physic_pipeline/src on sys.path)
from . import config as cfg
from .io_log import model_labels
from .residency import get_manager
from .schema import PredictQuery, PredictResult, Reasoning
from .units import latex_to_ascii, to_ascii_answer, to_ascii_unit, to_plain_numeric_answer
from .vllm_client import LLMClient, extract_last_json_object

Judge = Tuple[LLMClient, float, str, str]

_UNCERTAIN = {"", "uncertain", "unknown"}


def _build_settings(base_url: str, model: str, temperature: float = 0.0):
    """Construct exact_fama Settings in-code. The deterministic pipeline may still
    use the primary shared server for optional explanation rewrite only when that
    old knob is enabled."""
    from exact_fama.config import Settings  # type: ignore

    mode = "vllm"
    use_llm = mode == "vllm"
    backend = "openai_compatible" if use_llm else "none"
    settings = cfg.serve_settings()
    use_llm_expl = use_llm and bool(settings["physics_llm_explanation"])
    raw: Dict[str, Any] = {
        "model": {
            "backend": backend,
            "name": model,
            "vllm_base_url": base_url,
            "temperature": temperature,
            "max_new_tokens": int(settings["physics_max_new_tokens"]),
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


def _role(judge: Judge) -> str:
    return (judge[3] if len(judge) > 3 else "") or ""


def _primary_client(judges: List[Judge]) -> LLMClient:
    if not judges:
        return LLMClient()
    return judges[0][0]


def _split_lineup(judges: List[Judge]) -> Tuple[List[LLMClient], LLMClient]:
    gens = [j[0] for j in judges if _role(j) == "generator"]
    tagged = [j[0] for j in judges if _role(j) == "judge"]
    arbiter = tagged[0] if tagged else max(judges, key=lambda j: j[1])[0]
    if not gens:
        gens = [j[0] for j in judges]
    return gens, arbiter


def _physics_tokens() -> int:
    return int(cfg.serve_settings()["physics_think_tokens"])


def _valid_answer(answer: str) -> bool:
    return str(answer or "").strip().lower() not in _UNCERTAIN


def _coerce_steps(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _confidence(value: Any, default: float = 0.5) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    return max(0.0, min(1.0, out))


def _candidate(
    *,
    label: str,
    source: str,
    answer: Any,
    unit: Any = "",
    explanation: str = "",
    steps: Any = None,
    confidence: Any = 0.5,
) -> Dict[str, Any]:
    ans = to_plain_numeric_answer(answer) or to_ascii_answer(answer)
    return {
        "label": label,
        "source": source,
        "answer": ans,
        "unit": to_ascii_unit(unit),
        "explanation": str(explanation or "").strip(),
        "steps": _coerce_steps(steps),
        "confidence": _confidence(confidence),
    }


def _candidate_from_json(label: str, source: str, data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return _candidate(label=label, source=source, answer="", confidence=0.0)
    return _candidate(
        label=label,
        source=source,
        answer=data.get("answer"),
        unit=data.get("unit", ""),
        explanation=str(data.get("explanation") or "").strip(),
        steps=data.get("steps") or data.get("cot") or data.get("reasoning_steps"),
        confidence=data.get("confidence", 0.5),
    )


def _render_candidate(name: str, cand: Dict[str, Any]) -> str:
    unit = f" {cand['unit']}" if cand.get("unit") else ""
    steps = "; ".join(cand.get("steps") or [])
    return (
        f"{name} ({cand.get('label') or cand.get('source')}): "
        f"answer = {cand.get('answer') or 'Uncertain'}{unit}; "
        f"confidence = {cand.get('confidence', 0.0):.2f}; "
        f"explanation = {cand.get('explanation') or '(none)'}; "
        f"steps = {steps or '(none)'}"
    )


class PhysicsAdapter:
    def __init__(self, judges: List[Judge] | LLMClient):
        # Backward-compatible constructor for old direct tests/imports.
        if isinstance(judges, LLMClient):
            self.judges: List[Judge] = [(judges, 1.0, "4b", "generator")]
        else:
            self.judges = list(judges or [])
        self.client = _primary_client(self.judges)
        mode = self.client.mode
        settings = cfg.serve_settings()
        self.cascade_enabled = (
            bool(settings["physics_cascade"])
            and mode in {"vllm", "stub"}
            and len(self.judges) >= 1
        )
        self.fallback_enabled = (
            bool(settings["physics_llm_fallback"])
            and self.client.mode == "vllm"
        )
        self.pipeline = None
        self.import_error: Optional[str] = None
        try:
            from exact_fama.pipeline import ExactFamaPipeline  # type: ignore
            self.pipeline = ExactFamaPipeline(
                _build_settings(self.client.base_url, self.client.model, self.client.temperature)
            )
        except Exception as exc:  # pragma: no cover - exercised only if deps missing
            self.import_error = f"{type(exc).__name__}: {exc}"

    def answer(self, q: PredictQuery) -> PredictResult:
        question = latex_to_ascii(q.query or "")
        deterministic = self._deterministic_candidate(q, question)

        if self.cascade_enabled:
            result = self._cascade_answer(q, question, deterministic)
            if result is not None:
                return result

        # Legacy behavior when the cascade is disabled or unavailable.
        if _valid_answer(deterministic["answer"]):
            return self._result_from_candidate(q, deterministic, "deterministic solver")
        if self.fallback_enabled:
            fb = self._llm_fallback(question, q.query_id)
            if fb is not None:
                answer, unit, steps = fb
                return PredictResult(
                    query_id=q.query_id, answer=answer, unit=unit,
                    explanation=f"Computed via fallback. {answer}{(' ' + unit) if unit else ''}.",
                    premises_used=[], reasoning=Reasoning(type="cot", steps=steps),
                )
        return self._result_from_candidate(q, deterministic, "deterministic solver")

    def _deterministic_candidate(self, q: PredictQuery, question: str) -> Dict[str, Any]:
        if self.pipeline is None:
            reason = f"physics pipeline unavailable ({self.import_error})"
            return _candidate(
                label="deterministic",
                source="deterministic",
                answer="Uncertain",
                unit="",
                explanation=f"Could not compute a physics answer ({reason}).",
                steps=[reason],
                confidence=0.0,
            )

        from exact_fama.schemas import PredictRequest  # type: ignore

        req = PredictRequest(question=question, type="physics")
        resp = self.pipeline.predict(req)
        explanation = (resp.explanation or "").strip() or "Computed from the stated physical quantities."
        return _candidate(
            label="deterministic",
            source="deterministic",
            answer=resp.answer or "Uncertain",
            unit=resp.unit,
            explanation=explanation,
            steps=[str(s) for s in (resp.cot or [])],
            confidence=resp.confidence,
        )

    def _cascade_answer(
        self,
        q: PredictQuery,
        question: str,
        deterministic: Dict[str, Any],
    ) -> Optional[PredictResult]:
        try:
            gens, arbiter = _split_lineup(self.judges)
        except ValueError:
            return None
        if not gens:
            return None
        gen_clients = gens[:2] if len(gens) >= 2 else [gens[0], gens[0]]

        try:
            get_manager().ensure_generators()
            generator_candidates = self._run_generators(gen_clients, q, question, deterministic)
        except Exception:
            generator_candidates = []

        judge_candidate: Dict[str, Any] | None = None
        chosen: Dict[str, Any] | None = None
        try:
            with get_manager().judge() as judge_ready:
                if judge_ready:
                    judge_candidate, chosen = self._judge(
                        arbiter, q, question, deterministic, generator_candidates
                    )
        except Exception:
            judge_candidate = None
            chosen = None

        final = self._choose_final(deterministic, generator_candidates, judge_candidate, chosen)
        return self._result_from_candidate(q, final, "physics cascade")

    def _run_generators(
        self,
        clients: List[LLMClient],
        q: PredictQuery,
        question: str,
        deterministic: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        loaded = list(clients)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(clients)) as ex:
            return list(ex.map(
                lambda client: self._generate_candidate(client, q, question, deterministic, loaded),
                clients,
            ))

    def _generate_candidate(
        self,
        client: LLMClient,
        q: PredictQuery,
        question: str,
        deterministic: Dict[str, Any],
        loaded: List[LLMClient],
    ) -> Dict[str, Any]:
        system = (
            "You are a physics problem solver for educational QA. Solve the problem "
            "from the question. The deterministic solver answer is provided as a "
            "reference only: use it to recheck your work, but do not copy it blindly. "
            "Finish your reply with ONE JSON object and nothing after it: "
            '{"answer": <final numerical or symbolic value only>, '
            '"unit": <ASCII unit, or empty string>, '
            '"explanation": <2-4 concise public sentences>, '
            '"steps": [<short public calculation steps>], '
            '"confidence": <number from 0 to 1>}. '
            "For numerical answers, the answer field must be an unrounded plain "
            "ASCII decimal or integer only, e.g. 0.02304; never use LaTeX, "
            "fractions, radicals, currency symbols, commas, scientific notation, "
            "or units inside the answer field."
        )
        user = (
            f"Problem:\n{question}\n\n"
            "Deterministic reference answer:\n"
            f"{_render_candidate('Deterministic', deterministic)}"
        )
        try:
            response_format = None if client.thinking else {"type": "json_object"}
            text = client.chat(
                system,
                user,
                max_tokens=_physics_tokens(),
                response_format=response_format,
                log_context=f"type2 query_id={q.query_id or 'q'} stage=physics.generator",
                loaded_models=model_labels(loaded),
            )
            data = extract_last_json_object(text) or {}
        except Exception:
            data = {}
        return _candidate_from_json(client.model, "generator", data)

    def _judge(
        self,
        arbiter: LLMClient,
        q: PredictQuery,
        question: str,
        deterministic: Dict[str, Any],
        generators: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
        system = (
            "You are the SENIOR physics arbiter. First solve the problem yourself "
            "from the original question. Then recheck your solution against three "
            "candidate answers: the deterministic solver and two junior LLM answers. "
            "Choose the actually correct answer, or give your own corrected answer if "
            "all candidates are wrong. Finish with ONE JSON object and nothing after it: "
            '{"chosen": <"deterministic" or 1 or 2 or "self">, '
            '"answer": <final numerical or symbolic value only>, '
            '"unit": <ASCII unit, or empty string>, '
            '"explanation": <2-5 concise public sentences>, '
            '"steps": [<short public calculation/checking steps>]}. '
            "For numerical answers, the answer field must be an unrounded plain "
            "ASCII decimal or integer only, e.g. 0.02304; never use LaTeX, "
            "fractions, radicals, currency symbols, commas, scientific notation, "
            "or units inside the answer field. LaTeX is allowed in explanation "
            "and steps."
        )
        rendered = [_render_candidate("Deterministic", deterministic)]
        rendered.extend(
            _render_candidate(f"Junior {i}", cand)
            for i, cand in enumerate(generators, 1)
        )
        user = (
            f"Problem:\n{question}\n\n"
            "Candidate answers (reference only):\n"
            + "\n".join(rendered)
            + "\n\nSolve independently first, then compare these candidates."
        )
        text = arbiter.chat(
            system,
            user,
            max_tokens=_physics_tokens(),
            log_context=f"type2 query_id={q.query_id or 'q'} stage=physics.judge",
            loaded_models=[arbiter.model],
        )
        data = extract_last_json_object(text) or {}
        cand = _candidate_from_json(arbiter.model, "judge", data)
        return cand, self._chosen_candidate(data, deterministic, generators)

    def _chosen_candidate(
        self,
        data: Any,
        deterministic: Dict[str, Any],
        generators: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        if not isinstance(data, dict):
            return None
        raw = data.get("chosen")
        text = str(raw or "").strip().lower()
        if text in {"deterministic", "solver", "0"}:
            return deterministic
        if text == "self":
            return None
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            return None
        if 1 <= idx <= len(generators):
            return generators[idx - 1]
        return None

    def _choose_final(
        self,
        deterministic: Dict[str, Any],
        generators: List[Dict[str, Any]],
        judge: Dict[str, Any] | None,
        chosen: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        if judge is not None and _valid_answer(judge["answer"]):
            return judge
        if chosen is not None and _valid_answer(chosen["answer"]):
            return chosen
        if _valid_answer(deterministic["answer"]):
            return deterministic
        for cand in generators:
            if _valid_answer(cand["answer"]):
                return cand
        return _candidate(
            label="physics-cascade",
            source="fallback",
            answer="Uncertain",
            unit="",
            explanation="No candidate produced a usable physics answer.",
            steps=["No candidate produced a usable physics answer."],
            confidence=0.0,
        )

    def _result_from_candidate(
        self,
        q: PredictQuery,
        cand: Dict[str, Any],
        decider: str,
    ) -> PredictResult:
        answer = cand.get("answer") or "Uncertain"
        unit = cand.get("unit") or ""
        explanation = (cand.get("explanation") or "").strip()
        if not explanation:
            explanation = f"{decider} selected {answer}{(' ' + unit) if unit else ''}."
        steps = cand.get("steps") or []
        if not steps:
            steps = [explanation]
        return PredictResult(
            query_id=q.query_id,
            answer=answer,
            unit=unit,
            explanation=explanation,
            premises_used=[],
            reasoning=Reasoning(type="cot", steps=[str(s) for s in steps]),
        )

    # Legacy guarded numeric fallback.
    def _llm_fallback(self, question: str, query_id: str = ""):
        system = (
            "You are a physics problem solver for circuits and electrostatics. "
            "Solve the problem step by step, then return JSON only with these fields: "
            "{\"answer\": <the numerical value only, as a plain number>, "
            "\"unit\": <the unit in ASCII, e.g. A, V, ohm, uF, V/m; empty string if none>, "
            "\"steps\": [<short reasoning steps>]}. "
            "The answer field must be an unrounded plain ASCII decimal or integer "
            "only; do not use LaTeX, fractions, radicals, commas, scientific "
            "notation, or units inside the answer field."
        )
        try:
            data = self.client.chat_json(
                system,
                f"Problem: {question}",
                max_tokens=768,
                log_context=f"type2 query_id={query_id or 'q'} stage=physics.fallback",
            )
        except Exception:
            data = None
        if not isinstance(data, dict):
            return None
        answer = to_plain_numeric_answer(data.get("answer")) or to_ascii_answer(data.get("answer"))
        if not answer or answer.lower() in _UNCERTAIN:
            return None
        unit = to_ascii_unit(str(data.get("unit", "")))
        steps = [str(s) for s in (data.get("steps") or [])]
        return answer, unit, steps
