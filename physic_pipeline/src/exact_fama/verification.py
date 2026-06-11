from __future__ import annotations

from .schemas import PredictRequest, SolverResult


def verify_result(request: PredictRequest, result: SolverResult, task_type: str) -> SolverResult:
    """Lightweight post-solver verifier.

    This verifier is intentionally conservative: it does not invent a new answer.
    It only adds warnings/confidence adjustments when the answer lacks support.
    More specialized checks should live inside the logic/physics solvers.
    """
    answer = str(result.answer or "").strip()
    if not answer:
        result.answer = "Uncertain"
        result.warnings.append("OUTPUT_SCHEMA_ERROR: verifier replaced empty answer with Uncertain.")
        result.confidence = min(result.confidence, 0.2)

    if task_type == "physics":
        qlow = request.question.lower()
        expects_unit = any(k in qlow for k in [
            "calculate", "find", "determine", "what is", "compute", "tính", "xác định",
        ])
        if expects_unit and result.answer != "Uncertain" and not result.unit:
            result.warnings.append("PHYSICS_UNIT_ERROR: numerical physics answer has no unit.")
            result.confidence = min(result.confidence, 0.55)
        if "division by zero" in " ".join(result.warnings).lower():
            result.answer = "Uncertain"
            result.confidence = min(result.confidence, 0.2)

    if task_type == "logic" and result.answer == "Uncertain":
        result.confidence = min(result.confidence, 0.4)

    if not result.premises and not result.cot:
        result.warnings.append("EXPLANATION_WEAK: no premises or CoT evidence attached.")
        result.confidence = min(result.confidence, 0.6)

    result.confidence = max(0.0, min(1.0, float(result.confidence)))
    return result
