from __future__ import annotations

from .schemas import PredictResponse, SolverResult


def validate_result(result: SolverResult, task_type: str, used_modules: list[str]) -> PredictResponse:
    warnings = list(result.warnings)
    answer = str(result.answer).strip() if result.answer is not None else "Uncertain"
    explanation = str(result.explanation).strip() if result.explanation else "No explanation was generated."
    if not answer:
        answer = "Uncertain"
        warnings.append("OUTPUT_SCHEMA_ERROR: empty answer replaced with Uncertain.")
    if not explanation or explanation == "No explanation was generated.":
        warnings.append("OUTPUT_SCHEMA_ERROR: explanation is missing or weak.")
    confidence = max(0.0, min(1.0, float(result.confidence)))
    return PredictResponse(
        answer=answer,
        explanation=explanation,
        unit=result.unit,
        fol=result.fol,
        cot=result.cot,
        premises=result.premises,
        confidence=confidence,
        task_type=task_type if task_type in {"logic", "physics"} else "unknown",
        used_modules=used_modules,
        warnings=warnings,
        debug=result.debug,
    )
