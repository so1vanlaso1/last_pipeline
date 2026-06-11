from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from .llm.qwen_client import QwenClient
from .schemas import PredictRequest, SolverResult


BANNED_EXPLANATION_PHRASES = [
    "the system parsed",
    "system parsed",
    "solver",
    "forward chaining",
    "derived conclusions",
    "logic_rule_parser",
    "physics_quantity_extractor",
    "debug",
    "warnings",
]


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _extract_mc_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    matches = re.findall(r"\b([A-D])\.\s*([^\n]+)", question)
    for letter, text in matches:
        options[letter.upper()] = text.strip()
    return options


def _selected_option(question: str, answer: str) -> dict[str, str] | None:
    ans = str(answer or "").strip().upper().replace("OPTION ", "")
    if not ans:
        return None
    ans = ans[0]
    options = _extract_mc_options(question)
    if ans in options:
        return {"label": ans, "text": options[ans]}
    return None


def _looks_bad(text: str) -> bool:
    low = text.lower()
    if len(text.split()) < 18:
        return True
    return any(p in low for p in BANNED_EXPLANATION_PHRASES)


def _too_similar(a: str, b: str, threshold: float = 0.86) -> bool:
    if not a.strip() or not b.strip():
        return False
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio() >= threshold


def _indexed_premises(premises: list[str]) -> list[dict[str, Any]]:
    return [{"id": i + 1, "text": p} for i, p in enumerate(premises or [])]


class ExplanationGenerator:
    def __init__(self, llm: QwenClient | None = None, use_llm: bool = False):
        self.llm = llm
        self.use_llm = use_llm

    def rewrite(self, request: PredictRequest, result: SolverResult) -> SolverResult:
        if not self.use_llm or self.llm is None or self.llm.backend == "none":
            result.debug["llm_rewrite_enabled"] = False
            return result

        old_explanation = result.explanation or ""

        payload = {
            "question": request.question,
            "task_hint": request.type,
            "fixed_answer": result.answer,
            "fixed_unit": result.unit,
            "selected_option": _selected_option(request.question, result.answer),
            "premises_indexed": _indexed_premises(result.premises),
            "fol": result.fol,
            "verified_cot": result.cot,
            "proof_steps": result.debug.get("proof_steps", []),
            "formula": result.debug.get("formula"),
            "quantities": result.debug.get("quantities"),
        }

        messages = self._messages(payload, strict_retry=False)

        result.debug["llm_rewrite_enabled"] = True

        try:
            data = self.llm.generate_json(messages)
            explanation = self._extract_explanation(data)

            if (
                not explanation
                or _looks_bad(explanation)
                or _too_similar(explanation, old_explanation)
            ):
                retry_payload = dict(payload)
                retry_payload["bad_draft_to_avoid"] = old_explanation
                retry_payload["rewrite_instruction"] = (
                    "The previous explanation was too mechanical or too close to the solver draft. "
                    "Rewrite it in a clearer educational style while preserving all facts."
                )
                data = self.llm.generate_json(self._messages(retry_payload, strict_retry=True))
                explanation = self._extract_explanation(data)

            if explanation:
                result.explanation = explanation.strip()
                result.debug["llm_rewrite_changed"] = not _too_similar(result.explanation, old_explanation, threshold=0.94)

                if _looks_bad(result.explanation):
                    result.warnings.append("EXPLANATION_WEAK: LLM rewrite still looks mechanical.")

                if not result.debug["llm_rewrite_changed"]:
                    result.warnings.append("EXPLANATION_WEAK: LLM rewrite was too similar to solver draft.")
            else:
                result.warnings.append("EXPLANATION_WEAK: LLM rewrite returned no usable explanation.")

        except Exception as exc:
            result.debug["llm_rewrite_failed"] = str(exc)
            result.warnings.append(f"EXPLANATION_WEAK: LLM rewrite failed: {exc}")

        return result

    def _extract_explanation(self, data: Any) -> str | None:
        if not isinstance(data, dict):
            return None
        explanation = data.get("explanation")
        if isinstance(explanation, str) and explanation.strip():
            return explanation.strip()
        return None

    def _messages(self, payload: dict[str, Any], strict_retry: bool) -> list[dict[str, str]]:
        system = (
            "You are an explanation writer for the EXACT educational QA challenge.\n"
            "Your job is to rewrite verified solver evidence into a concise, human-readable explanation.\n\n"
            "Hard rules:\n"
            "- Return JSON only.\n"
            "- Output exactly these fields: answer, unit, explanation.\n"
            "- Copy fixed_answer exactly into answer.\n"
            "- Copy fixed_unit exactly into unit. If fixed_unit is null, return null.\n"
            "- Never change the final answer.\n"
            "- Never change the unit.\n"
            "- Use only the provided premises, formulas, quantities, FOL, proof steps, and verified CoT.\n"
            "- Do not invent new premises, formulas, numbers, units, or assumptions.\n"
            "- Do not mention internal implementation terms such as solver, parser, system, debug, warnings, or forward chaining.\n"
            "- For logic questions, prefer premise-numbered reasoning: 'Premise 4 confirms ..., premise 3 implies ..., therefore option B is supported.'\n"
            "- For physics questions, explain the formula, substitution, computation, and unit clearly.\n"
            "- For MCQ questions, explicitly connect the selected option to the derived conclusion.\n"
            "- Keep the explanation concise: usually 2-5 sentences.\n"
        )

        if strict_retry:
            system += (
                "\nThe previous explanation was too mechanical. Rewrite more naturally, but still stay verifiable. "
                "Do not copy the bad draft wording.\n"
            )

        user = {
            "fixed_answer": payload.get("fixed_answer"),
            "fixed_unit": payload.get("fixed_unit"),
            "evidence": payload,
            "required_output_example": {
                "answer": payload.get("fixed_answer"),
                "unit": payload.get("fixed_unit"),
                "explanation": "Premise 4 confirms the key fact, and premise 3 applies the rule to derive the intermediate conclusion. Premise 5 then satisfies the remaining condition, so premise 1 supports the final answer."
            },
        }

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": _safe_json_dumps(user)},
        ]