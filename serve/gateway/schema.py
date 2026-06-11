"""Competition I/O schema (EXACT 2026 Submission Guide, sections 3 & 4).

One unified input schema for both query types; one unified output schema. Every
field is always present; fields that do not apply are empty ("" or []).
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictQuery(BaseModel):
    """A single competition query (Section 3.1). The evaluation server posts one
    of these as the request body (we also accept a list of them)."""

    model_config = ConfigDict(extra="allow")

    query_id: str = ""
    type: str = ""                                   # "type1" | "type2"
    query: str = ""
    premises: List[str] = Field(default_factory=list)
    options: List[str] = Field(default_factory=list)


class Reasoning(BaseModel):
    """Optional structured reasoning evidence (Section 4.4)."""

    type: str = "cot"                                # "fol" | "cot" | "proof"
    steps: List[str] = Field(default_factory=list)


class PredictResult(BaseModel):
    """A single result object (Section 4.1). The endpoint returns a JSON *list*
    of these — one per query — even for a single query."""

    query_id: str
    answer: str
    unit: str = ""
    explanation: str
    premises_used: List[int] = Field(default_factory=list)
    reasoning: Optional[Reasoning] = None


def empty_result(query_id: str, message: str) -> PredictResult:
    """A schema-valid placeholder for a query we could not answer at all. Keeps
    the endpoint from ever returning a malformed object (a missing answer would be
    scored wrong anyway; this guarantees the shape is right)."""
    return PredictResult(
        query_id=query_id or "",
        answer="Uncertain",
        unit="",
        explanation=message or "No answer could be produced for this query.",
        premises_used=[],
        reasoning=None,
    )
