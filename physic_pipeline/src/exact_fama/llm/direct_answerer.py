from __future__ import annotations

import json
import re
from typing import Any

from ..schemas import PredictRequest, PredictResponse
from .qwen_client import QwenClient


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _strip_code_fence(text: str) -> str:
    text = str(text or "").strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


def _extract_first_json_object(text: str) -> str:
    """Extract the first balanced JSON object from a raw LLM response."""
    s = _strip_code_fence(text)
    if not s:
        raise ValueError("empty LLM output")

    # Fast path.
    if s.startswith("{") and s.endswith("}"):
        return s

    start = s.find("{")
    if start < 0:
        raise ValueError(f"LLM output did not contain a JSON object. Raw output:\n{s}")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]

    raise ValueError(f"Could not find a balanced JSON object. Raw output:\n{s}")


def _loads_json_object(text: str) -> dict[str, Any]:
    payload = _extract_first_json_object(text)
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    return data


def _clip_text(text: str, max_chars: int = 12000) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED]"


def _normalize_answer(answer: Any, task_type: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return "Unknown" if task_type == "logic" else "Uncertain"

    # Keep MCQ labels clean.
    m = re.fullmatch(r"(?:option\s*)?[\(\[]?([A-Da-d])[\)\].:]?", text)
    if m:
        return m.group(1).upper()

    low = text.lower()
    if task_type == "logic":
        if low in {"true", "entailed", "correct"}:
            return "Yes"
        if low in {"false", "contradicted", "contradiction", "incorrect"}:
            return "No"
        if low in {
            "unknown",
            "uncertain",
            "undetermined",
            "cannot be determined",
            "not enough information",
            "not enough info",
            "nei",
        }:
            return "Unknown"

    return text


def _normalize_unit(unit: Any) -> str | None:
    if unit is None:
        return None
    text = str(unit).strip()
    if not text or text.lower() in {"none", "null", "n/a", "na"}:
        return None
    return text


def _normalize_cot(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _extract_options_from_question(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for label, value in re.findall(r"\b([A-D])\.\s*([^\n]+)", question or ""):
        options[label.upper()] = value.strip()
    return options


class DirectLLMAnswerer:
    """Full-LLM answerer.

    This component intentionally bypasses parser, solver, verifier, and rewrite.
    It asks the LLM to produce the final answer directly in the project's
    PredictResponse-compatible JSON format.

    It is intended for experiments and comparison only. It should not replace
    the solver-first path unless empirical evaluation justifies it.
    """

    def __init__(
        self,
        llm: QwenClient,
        *,
        max_retries: int = 1,
        include_debug_raw_output: bool = True,
    ) -> None:
        self.llm = llm
        self.max_retries = max(0, int(max_retries))
        self.include_debug_raw_output = include_debug_raw_output

    def predict(self, request: PredictRequest, task_type: str) -> PredictResponse:
        messages = self._build_messages(request, task_type)
        warnings: list[str] = []
        raw_outputs: list[str] = []

        last_exc: Exception | None = None
        data: dict[str, Any] | None = None

        for attempt in range(self.max_retries + 1):
            try:
                raw = self.llm.generate(messages)
                raw_outputs.append(raw)
                data = _loads_json_object(raw)
                break
            except Exception as exc:
                last_exc = exc
                warnings.append(
                    f"FULL_LLM_OUTPUT_SCHEMA_ERROR: attempt {attempt + 1} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                messages = self._build_retry_messages(messages, raw_outputs, exc)

        if data is None:
            answer = "Unknown" if task_type == "logic" else "Uncertain"
            explanation = "The direct LLM flow failed to return valid JSON."
            confidence = 0.0
            cot: list[str] = []
            unit = None
        else:
            answer = _normalize_answer(data.get("answer"), task_type)
            unit = _normalize_unit(data.get("unit"))
            explanation = str(data.get("explanation") or "").strip()
            if not explanation:
                explanation = "The direct LLM flow returned an answer without a usable explanation."
                warnings.append("FULL_LLM_OUTPUT_SCHEMA_ERROR: missing explanation.")

            cot = _normalize_cot(data.get("cot") or data.get("reasoning_steps"))
            try:
                confidence = float(data.get("confidence", 0.5))
            except Exception:
                confidence = 0.5
                warnings.append("FULL_LLM_OUTPUT_SCHEMA_ERROR: invalid confidence; defaulted to 0.5.")

            if not answer:
                answer = "Unknown" if task_type == "logic" else "Uncertain"
                warnings.append("FULL_LLM_OUTPUT_SCHEMA_ERROR: empty answer replaced.")

        confidence = max(0.0, min(1.0, float(confidence)))

        debug: dict[str, Any] = {
            "mode": "full_llm_direct",
            "llm_backend": getattr(self.llm, "backend", None),
            "model_name": getattr(self.llm, "model_name", None),
            "attempts": len(raw_outputs),
        }

        if last_exc is not None and data is None:
            debug["last_exception_type"] = type(last_exc).__name__
            debug["last_exception"] = str(last_exc)

        if self.include_debug_raw_output:
            debug["raw_outputs"] = raw_outputs

        return PredictResponse(
            answer=answer,
            unit=unit,
            explanation=explanation,
            fol=None,
            cot=cot,
            premises=request.premises_nl if task_type == "logic" else [],
            confidence=confidence,
            task_type=task_type if task_type in {"logic", "physics"} else "unknown",
            used_modules=[
                "input_normalizer",
                "task_router",
                "full_llm_direct_answerer",
                "full_llm_output_validator",
            ],
            warnings=warnings,
            debug=debug,
        )

    def _build_messages(self, request: PredictRequest, task_type: str) -> list[dict[str, str]]:
        if task_type == "logic":
            system = self._logic_system_prompt()
        elif task_type == "physics":
            system = self._physics_system_prompt()
        else:
            system = self._general_system_prompt()

        user_payload = self._user_payload(request, task_type)

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": _safe_json_dumps(user_payload)},
        ]

    def _build_retry_messages(
        self,
        original_messages: list[dict[str, str]],
        raw_outputs: list[str],
        exc: Exception,
    ) -> list[dict[str, str]]:
        retry_note = {
            "role": "user",
            "content": (
                "Your previous response was invalid JSON for this task. "
                "Return exactly one JSON object and no markdown. "
                "Required keys: answer, unit, explanation, cot, confidence. "
                f"Parser error: {type(exc).__name__}: {exc}\n"
                f"Previous output:\n{_clip_text(raw_outputs[-1] if raw_outputs else '')}"
            ),
        }
        return list(original_messages) + [retry_note]

    def _user_payload(self, request: PredictRequest, task_type: str) -> dict[str, Any]:
        options = _extract_options_from_question(request.question)
        payload: dict[str, Any] = {
            "id": request.id,
            "task_type": task_type,
            "question": request.question,
            "instructions": [
                "Solve the problem directly.",
                "Do not use hidden chain-of-thought.",
                "Return only the final JSON object.",
            ],
            "output_schema": {
                "answer": "string; final answer only",
                "unit": "string or null",
                "explanation": "concise public explanation with premises/formulas used",
                "cot": "list of short public reasoning steps, no hidden chain-of-thought",
                "confidence": "number from 0 to 1",
            },
        }

        if request.premises_nl:
            payload["premises-NL"] = request.premises_nl

        if options:
            payload["options"] = options

        return payload

    def _general_system_prompt(self) -> str:
        return (
            "You are a direct educational QA model. "
            "Answer the user problem directly. "
            "Return JSON only with keys: answer, unit, explanation, cot, confidence. "
            "Do not include markdown or extra text."
        )

    def _logic_system_prompt(self) -> str:
        return (
            "You are a direct logic/regulation QA model for an educational QA benchmark. "
            "Use only the provided question and premises. Do not invent premises. "
            "For multiple-choice questions, answer exactly one of A, B, C, D, or Unknown. "
            "For yes/no questions, answer exactly Yes, No, or Unknown. "
            "If the conclusion is not entailed and not contradicted by the premises, answer Unknown. "
            "For 'Does it follow...' questions, answer Yes only if the claim is entailed by the premises; "
            "answer No only if the premises contradict it or a required condition is explicitly missing; "
            "otherwise answer Unknown. "
            "Return JSON only with keys: answer, unit, explanation, cot, confidence. "
            "Set unit to null for logic. "
            "The explanation should cite the relevant premise numbers when possible. "
            "Do not include markdown, comments, or extra text."
        )

    def _physics_system_prompt(self) -> str:
        return (
            "You are a direct physics QA model for electric circuits, capacitance, electrostatics, "
            "and related calculation problems. "
            "Solve the problem directly from the question. "
            "Return JSON only with keys: answer, unit, explanation, cot, confidence. "
            "Put the numerical or symbolic final value in answer. Put the physical unit in unit when clear; "
            "use null if the expected answer is unitless, symbolic, qualitative, or the unit is unclear. "
            "Preserve requested rounding when specified. "
            "If the problem cannot be solved from the given information, answer Uncertain. "
            "The explanation should mention the formula or physical principle used. "
            "Do not include markdown, comments, or extra text."
        )
