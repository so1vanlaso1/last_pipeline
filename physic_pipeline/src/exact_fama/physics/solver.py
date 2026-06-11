from __future__ import annotations
from typing import Any
try:
    from .physics_solvers.common import PredictRequest, SolverResult, _uncertain
    from .physics_solvers.registry import solve_with_registered_solvers
except Exception:
    from physics_solvers.common import PredictRequest, SolverResult, _uncertain
    from physics_solvers.registry import solve_with_registered_solvers


def solve_physics(request: PredictRequest, structured_parse: dict[str, Any] | None = None) -> SolverResult:
    """Formula/template-only physics solver.

    This entrypoint deliberately performs no sample-id lookup, no gold-answer lookup,
    and no exact question-text answer lookup.  All answers must come from registered
    formula/template solvers that parse quantities from the question text.
    """
    question = str(getattr(request, "question", "") or "")
    requested_unit = getattr(request, "unit", None)
    try:
        requested_unit = requested_unit or getattr(request, "model_extra", {}).get("unit")
    except Exception:
        pass
    if requested_unit and str(requested_unit).lower() != "none":
        question = f"{question} [expected_unit: {requested_unit}]"
    result = solve_with_registered_solvers(question)
    if result is not None:
        return result
    return _uncertain(question, debug={"structured_parse_used": bool(structured_parse)})
