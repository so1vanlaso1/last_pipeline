from __future__ import annotations

import re
from typing import Literal

from .schemas import PredictRequest

TaskType = Literal["logic", "physics"]

PHYSICS_KEYWORDS = {
    "resistance", "resistor", "voltage", "current", "power", "capacitor",
    "capacitance", "electric field", "charge", "energy stored", "ohm", "circuit",
    "điện trở", "hiệu điện thế", "dòng điện", "công suất", "tụ điện", "điện trường",
}
LOGIC_KEYWORDS = {
    "premise", "premises", "regulation", "policy", "scholarship", "course",
    "semester", "student", "curriculum", "faculty", "if", "then",
    "quy định", "quy chế", "học bổng", "môn học", "sinh viên",
}


def route_task(request: PredictRequest) -> TaskType:
    if request.type in {"logic", "physics"}:
        return request.type
    if request.premises_nl:
        return "logic"

    text = request.question.lower()
    physics_score = sum(1 for kw in PHYSICS_KEYWORDS if kw in text)
    logic_score = sum(1 for kw in LOGIC_KEYWORDS if kw in text)

    if re.search(r"\b[crvipuqe]\s*=\s*[-+]?\d", text):
        physics_score += 2
    if re.search(r"\b(a|b|c|d)\.", text) or "yes/no" in text:
        logic_score += 1

    return "physics" if physics_score >= logic_score else "logic"
