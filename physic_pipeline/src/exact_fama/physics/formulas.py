from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
@dataclass(frozen=True)
class Formula:
    name: str
    expression: str
    required: tuple[str, ...]
    target: str
    unit: str
    compute: Callable[[dict[str, float]], float]
    keywords: tuple[str, ...]
def _safe_div(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("division by zero in physics formula")
    return a / b
FORMULAS = [
    Formula(
        name="capacitor_energy_cv",
        expression="E = 1/2 * C * V^2",
        required=("C", "V"),
        target="E",
        unit="J",
        compute=lambda q: 0.5 * q["C"] * q["V"] ** 2,
        keywords=("energy stored", "stored energy", "capacitor", "capacitance", "năng lượng", "tụ điện"),
    ),
    Formula(
        name="capacitor_charge",
        expression="Q = C * V",
        required=("C", "V"),
        target="Q",
        unit="C",
        compute=lambda q: q["C"] * q["V"],
        keywords=("charge", "capacitor", "capacitance", "điện tích", "tụ điện"),
    ),
    Formula(
        name="capacitor_voltage",
        expression="V = Q / C",
        required=("Q", "C"),
        target="V",
        unit="V",
        compute=lambda q: _safe_div(q["Q"], q["C"]),
        keywords=("voltage", "potential difference", "capacitor", "hiệu điện thế"),
    ),
    Formula(
        name="ohm_voltage",
        expression="V = I * R",
        required=("I", "R"),
        target="V",
        unit="V",
        compute=lambda q: q["I"] * q["R"],
        keywords=("voltage", "potential difference", "hiệu điện thế"),
    ),
    Formula(
        name="ohm_current",
        expression="I = V / R",
        required=("V", "R"),
        target="I",
        unit="A",
        compute=lambda q: _safe_div(q["V"], q["R"]),
        keywords=("current", "dòng điện"),
    ),
    Formula(
        name="ohm_resistance",
        expression="R = V / I",
        required=("V", "I"),
        target="R",
        unit="ohm",
        compute=lambda q: _safe_div(q["V"], q["I"]),
        keywords=("resistance", "điện trở"),
    ),
    Formula(
        name="electric_power_vi",
        expression="P = V * I",
        required=("V", "I"),
        target="P",
        unit="W",
        compute=lambda q: q["V"] * q["I"],
        keywords=("power", "công suất"),
    ),
    Formula(
        name="electric_power_i2r",
        expression="P = I^2 * R",
        required=("I", "R"),
        target="P",
        unit="W",
        compute=lambda q: q["I"] ** 2 * q["R"],
        keywords=("power", "công suất", "heat", "dissipated"),
    ),
    Formula(
        name="electric_power_v2r",
        expression="P = V^2 / R",
        required=("V", "R"),
        target="P",
        unit="W",
        compute=lambda q: _safe_div(q["V"] ** 2, q["R"]),
        keywords=("power", "công suất", "dissipated"),
    ),
    Formula(
        name="coulomb_force",
        expression="F = k * |Q1 * Q2| / r^2",
        required=("Q1", "Q2", "D", "K"),
        target="Fforce",
        unit="N",
        compute=lambda q: q["K"] * abs(q["Q1"] * q["Q2"]) / (q["D"] ** 2),
        keywords=("force", "coulomb", "định luật coulomb", "lực"),
    ),
    Formula(
        name="electric_field_force_per_charge",
        expression="E = F / Q",
        required=("Fforce", "Q"),
        target="Efield",
        unit="N/C",
        compute=lambda q: _safe_div(q["Fforce"], q["Q"]),
        keywords=("electric field", "field strength", "điện trường"),
    ),
    Formula(
        name="electric_field_point_charge",
        expression="E = k * |Q| / r^2",
        required=("Q", "D", "K"),
        target="Efield",
        unit="N/C",
        compute=lambda q: q["K"] * abs(q["Q"]) / (q["D"] ** 2),
        keywords=("electric field", "point charge", "điện trường"),
    ),
    Formula(
        name="electric_potential_point_charge",
        expression="V = k * Q / r",
        required=("Q", "D", "K"),
        target="V",
        unit="V",
        compute=lambda q: q["K"] * q["Q"] / q["D"],
        keywords=("potential", "electric potential", "điện thế"),
    ),
]
TARGET_HINTS = {
    "E": ("energy", "năng lượng", "stored"),
    "Q": ("charge", "điện tích"),
    "V": ("voltage", "potential", "hiệu điện thế", "điện thế"),
    "I": ("current", "dòng điện"),
    "R": ("resistance", "điện trở"),
    "P": ("power", "công suất"),
    "Fforce": ("force", "lực", "coulomb"),
    "Efield": ("electric field", "field strength", "điện trường"),
}
def choose_formula(question: str, values: dict[str, float], target_hint: str | None = None) -> Formula | None:
    q = question.lower()
    candidates: list[tuple[int, Formula]] = []
    for formula in FORMULAS:
        if not all(k in values for k in formula.required):
            continue
        score = sum(3 for kw in formula.keywords if kw.lower() in q)
        if target_hint and target_hint.lower() == formula.target.lower():
            score += 8
        for hint in TARGET_HINTS.get(formula.target, ()):                                     
            if hint in q:
                score += 2
        if formula.target.lower() in q:
            score += 1
        if formula.target == "Q" and any(k in q for k in ["energy", "power", "current", "resistance"]):
            score -= 4
        if formula.target == "E" and "electric field" in q:
            score -= 5
        if formula.name == "capacitor_energy_cv" and "capacitor" in q and "energy" in q:
            score += 7
        candidates.append((score, formula))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]
