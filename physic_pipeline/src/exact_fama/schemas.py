from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str | None = None
    type: Literal["logic", "physics"] | None = None
    question: str
    premises_nl: list[str] = Field(default_factory=list, alias="premises-NL")
    premises_fol: list[str] = Field(default_factory=list, alias="premises-FOL")
    options: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    answer: str
    explanation: str
    unit: str | None = None
    fol: str | None = None
    cot: list[str] = Field(default_factory=list)
    premises: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    task_type: Literal["logic", "physics", "unknown"] = "unknown"
    used_modules: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class SolverResult(BaseModel):
    answer: str
    explanation: str
    unit: str | None = None
    fol: str | None = None
    cot: list[str] = Field(default_factory=list)
    premises: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)
