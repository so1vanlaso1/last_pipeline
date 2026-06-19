from __future__ import annotations

import math
import re
from typing import Any, Callable

from ..common import SolverResult, _make_result, _uncertain

# Deterministic formula/template solver for fluid mechanics.
# No ID lookup and no answer lookup: every result is computed from parsed quantities.

_SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+", "₀": "0", "₁": "1", "₂": "2",
    "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7",
    "₈": "8", "₉": "9",
})
_DEC = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_EXP_PART = r"(?:\^\s*\{?\s*[-+]?\d+\s*\}?|[-+]\d+)"
NUM = (
    rf"(?:{_DEC}\s*(?:×|x|\*|·)\s*10\s*{_EXP_PART})|"
    rf"(?:{_DEC}[eE][-+]?\d+)|"
    rf"(?:{_DEC})"
)

_UNIT_ALIASES = {
    "pa": "Pa", "kpa": "kPa", "mpa": "MPa", "gpa": "GPa",
    "n": "N", "w": "W", "j": "J",
    "m": "m", "cm": "cm", "mm": "mm", "km": "km",
    "m/s": "m/s", "ms^-1": "m/s",
    "m^2": "m^2", "m2": "m^2", "m²": "m^2",
    "cm^2": "cm^2", "cm2": "cm^2", "cm²": "cm^2",
    "mm^2": "mm^2", "mm2": "mm^2", "mm²": "mm^2",
    "m^3": "m^3", "m3": "m^3", "m³": "m^3",
    "m^3/s": "m^3/s", "m3/s": "m^3/s", "m³/s": "m^3/s",
    "l": "L", "liter": "L", "litre": "L", "liters": "L", "litres": "L",
    "l/min": "L/min", "liter/min": "L/min", "litre/min": "L/min", "liters/min": "L/min", "litres/min": "L/min",
    "kg/s": "kg/s", "kg/m^3": "kg/m^3", "kg/m3": "kg/m^3", "kg/m³": "kg/m^3",
    "pa.s": "Pa·s", "pa·s": "Pa·s", "pa*s": "Pa·s", "pas": "Pa·s",
    "n/m": "N/m", "n/m^3": "N/m^3", "n/m3": "N/m^3", "n/m³": "N/m^3",
    "m^2/s": "m^2/s", "m2/s": "m^2/s", "m²/s": "m^2/s",
    "j/(kg.k)": "J/(kg·K)", "j/(kg·k)": "J/(kg·K)",
    "dimensionless": "dimensionless",
}

# Accept every unit used in the dataset plus common variants.
UNIT = (
    r"J/\(kg[·.]?K\)|J/\(kg\s*·\s*K\)|"
    r"kg/m\^3|kg/m3|kg/m³|N/m\^3|N/m3|N/m³|m\^2/s|m2/s|m²/s|m\^3/s|m3/s|m³/s|"
    r"L/min|liters?/min|litres?/min|Pa\s*[·*\.]\s*s|Pa·s|Pa\*s|"
    r"GPa|MPa|kPa|Pa|N/m|kg/s|m/s|cm\^2|cm²|mm\^2|mm²|m\^2|m²|"
    r"cm2|mm2|m2|L|liters?|litres?|kg|g|mm|cm|km|m|N|W|K|dimensionless"
)


def _norm(text: str) -> str:
    s = str(text or "").translate(_SUPERSCRIPT_MAP)
    repl = {
        "−": "-", "–": "-", "—": "-", "µ": "μ",
        "ρ": "rho", "μ": "mu", "ν": "nu", "τ": "tau", "γ": "gamma",
        "σ": "sigma", "η": "eta", "Δ": "Delta", "∞": "inf", "π": "pi",
        "·": "·",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _lower(text: str) -> str:
    return _norm(text).lower()


def _parse_num(raw: str) -> float:
    s = _norm(raw)
    s = re.sub(r"\s+", "", s)
    s = s.replace("×", "x").replace("*", "x").replace("·", "x")
    s = re.sub(r"10\^?\{\s*([-+]?\d+)\s*\}", r"10^\1", s, flags=re.I)
    if "," in s:
        if "." in s:
            s = s.replace(",", "")
        elif re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:[eE][-+]?\d+)?", s):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    if re.fullmatch(rf"{_DEC}(?:[eE][-+]?\d+)?", s):
        return float(s)
    m = re.fullmatch(rf"({_DEC})x10(?:\^)?([-+]?\d+)", s, flags=re.I)
    if m:
        return float(m.group(1)) * (10 ** int(m.group(2)))
    return float(s)


def _canon_unit(unit: str | None) -> str:
    if not unit:
        return ""
    u = _norm(unit).strip()
    key = re.sub(r"\s+", "", u).lower().replace("µ", "μ")
    return _UNIT_ALIASES.get(key, u)


def _to_si(value: float, unit: str | None) -> float:
    u = _canon_unit(unit)
    mult = {
        "": 1.0,
        "Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9,
        "N": 1.0, "W": 1.0, "J": 1.0,
        "m": 1.0, "cm": 1e-2, "mm": 1e-3, "km": 1e3,
        "m/s": 1.0,
        "m^2": 1.0, "cm^2": 1e-4, "mm^2": 1e-6,
        "m^3": 1.0, "L": 1e-3,
        "m^3/s": 1.0, "L/min": 1e-3 / 60.0,
        "kg": 1.0, "g": 1e-3,
        "kg/s": 1.0, "kg/m^3": 1.0,
        "Pa·s": 1.0, "N/m": 1.0, "N/m^3": 1.0,
        "m^2/s": 1.0, "K": 1.0,
        "J/(kg·K)": 1.0,
        "dimensionless": 1.0,
    }.get(u, 1.0)
    return value * mult


def _from_si(value: float, unit: str | None) -> float:
    u = _canon_unit(unit)
    scale = {
        "Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9,
        "N": 1.0, "W": 1.0,
        "m": 1.0, "cm": 1e-2, "mm": 1e-3,
        "m/s": 1.0,
        "m^2/s": 1.0, "m^3/s": 1.0,
        "kg/s": 1.0, "kg/m^3": 1.0,
        "N/m^3": 1.0,
        "dimensionless": 1.0,
    }.get(u or "", 1.0)
    return value / scale


def _expected_unit(question: str, default: str | None = None) -> str | None:
    q0 = str(question or "")
    m = re.search(r"\[expected_unit:\s*([^\]]+)\]", q0, flags=re.I)
    if m:
        u = _canon_unit(m.group(1).strip())
        return None if u.lower() in {"none", ""} else u
    q = _lower(q0)
    # Conservative phrase-based unit inference. Formula branches also provide defaults.
    patterns = [
        rf"(?:answer|return|express|report|give)\s+(?:the\s+answer\s+)?(?:in|as)\s+(?P<u>{UNIT})\b",
        rf"(?:unit|units)\s*[:=]\s*(?P<u>{UNIT})\b",
        rf"\((?:unit\s*[:=]\s*)?(?P<u>{UNIT})\)",
    ]
    for pat in patterns:
        mm = re.search(pat, q, flags=re.I)
        if mm:
            return _canon_unit(mm.group("u"))
    return default


def _fmt(x: float) -> str:
    if not math.isfinite(x):
        return "Uncertain"
    # Dataset-compatible numeric style: six significant figures, compact exponent.
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.6g}"
    s = re.sub(r"e([+-])0+(\d+)", r"e\1\2", s)
    return s


def _qty(text: str, symbol: str, unit_regex: str | None = None, occurrence: int = 1) -> float | None:
    t = _norm(text)
    # Symbol variants: p1, p_1, rho_fluid, Delta p, etc.
    sym = re.escape(symbol)
    sym = sym.replace(r"\_", r"_?")
    unit_part = unit_regex or UNIT
    patterns = [
        rf"(?<![A-Za-z0-9_]){sym}\s*=\s*(?P<v>{NUM})\s*(?P<u>{unit_part})?\b",
        rf"(?<![A-Za-z0-9_]){sym}\s+(?:is|of)\s+(?P<v>{NUM})\s*(?P<u>{unit_part})?\b",
    ]
    vals: list[float] = []
    for pat in patterns:
        for m in re.finditer(pat, t, flags=re.I):
            try:
                vals.append(_to_si(_parse_num(m.group("v")), m.group("u") or ""))
            except Exception:
                pass
    if len(vals) >= occurrence:
        return vals[occurrence - 1]
    return None


def _all_symbol(text: str, symbols: list[str], unit_regex: str | None = None) -> list[float]:
    vals: list[float] = []
    for sym in symbols:
        k = 1
        while True:
            v = _qty(text, sym, unit_regex, k)
            if v is None:
                break
            vals.append(v)
            k += 1
    # stable de-duplication
    out: list[float] = []
    for v in vals:
        if not any(abs(v - x) <= max(1e-12, abs(v) * 1e-12) for x in out):
            out.append(v)
    return out


def _number_before(text: str, phrase_regex: str, unit_regex: str | None = None, occurrence: int = 1) -> float | None:
    t = _norm(text)
    unit_part = unit_regex or UNIT
    pat = rf"{phrase_regex}[^.?,;:]*?(?P<v>{NUM})\s*(?P<u>{unit_part})\b"
    vals = []
    for m in re.finditer(pat, t, flags=re.I):
        try:
            vals.append(_to_si(_parse_num(m.group("v")), m.group("u")))
        except Exception:
            pass
    if len(vals) >= occurrence:
        return vals[occurrence - 1]
    return None


def _all_units(text: str, unit_regex: str) -> list[float]:
    t = _norm(text)
    vals: list[float] = []
    for m in re.finditer(rf"(?P<v>{NUM})\s*(?P<u>{unit_regex})\b", t, flags=re.I):
        try:
            vals.append(_to_si(_parse_num(m.group("v")), m.group("u")))
        except Exception:
            pass
    return vals


def _g(text: str) -> float:
    return _qty(text, "g", r"m/s\^2|m/s2|m/s²") or 9.81


def _rho(text: str, which: str = "rho") -> float | None:
    syms = {
        "rho": ["rho"],
        "fluid": ["rho_fluid", "rho_f", "rho"],
        "object": ["rho_object", "rho_obj"],
        "sphere": ["rho_sphere", "rho_s"],
    }.get(which, [which])
    vals = _all_symbol(text, syms, r"kg/m\^3|kg/m3|kg/m³")
    return vals[0] if vals else None


def _result(question: str, value_si: float, default_unit: str, formula: str, quantities: dict[str, Any], confidence: float = 0.97) -> SolverResult:
    unit = _expected_unit(question, default_unit) or default_unit
    out = _from_si(value_si, unit)
    return _make_result(_fmt(out), unit, "Fluid-mechanics formula solver computed the requested quantity.", formula, quantities, confidence=confidence)


def _is_fluid_question(question: str) -> bool:
    q = _lower(question)
    keys = [
        "fluid", "liquid", "water", "pipe", "flow", "hydraulic", "manometer", "buoyant",
        "bernoulli", "torricelli", "orifice", "venturi", "poiseuille", "viscosity",
        "reynolds", "stokes", "darcy", "weisbach", "head loss", "pump", "turbine",
        "froude", "open channel", "manning", "capillary", "surface tension", "droplet",
        "soap bubble", "specific gravity", "specific weight", "mach number", "speed of sound", "ideal gas", "pitot",
        "jet", "bulk modulus", "pressure coefficient", "hatch", "gate", "piston", "center of pressure", "free surface", "rectangular plate"
    ]
    return any(k in q for k in keys)


# Individual formula branches. They deliberately key on physical wording and variables,
# not IDs, so unseen numbers and close paraphrases still solve.

def _solve_hydrostatic_pressure(q: str) -> SolverResult | None:
    l = _lower(q)
    rho = _rho(q)
    g = _g(q)
    h = _qty(q, "h", r"m|cm|mm")
    if rho is None or h is None:
        return None
    if "gauge pressure" in l and "at what depth" not in l:
        return _result(q, rho * g * h, "kPa", "p_g = rho*g*h", {"rho": rho, "g": g, "h": h})
    if ("hydrostatic pressure" in l or "pressure at depth" in l or "pressure at a depth" in l) and "at what depth" not in l:
        return _result(q, rho * g * h, "Pa", "p = rho*g*h", {"rho": rho, "g": g, "h": h})
    if "absolute pressure" in l:
        patm = _number_before(q, r"atmospheric pressure(?:\s+is)?", r"kPa|Pa|MPa") or _qty(q, "p_atm", r"kPa|Pa|MPa") or 101325.0
        return _result(q, patm + rho * g * h, "kPa", "p_abs = p_atm + rho*g*h", {"p_atm": patm, "rho": rho, "g": g, "h": h})
    return None


def _solve_depth_from_pressure(q: str) -> SolverResult | None:
    l = _lower(q)
    if "at what depth" not in l and "find the depth" not in l:
        return None
    rho = _rho(q)
    g = _g(q)
    p = _number_before(q, r"gauge pressure(?:\s+of)?", r"kPa|Pa|MPa") or _qty(q, "p", r"kPa|Pa|MPa")
    if rho is not None and p is not None:
        return _result(q, p / (rho * g), "m", "h = p_g/(rho*g)", {"p": p, "rho": rho, "g": g})
    return None


def _solve_pressure_difference(q: str) -> SolverResult | None:
    l = _lower(q)
    if "pressure difference" not in l or "static liquid" not in l:
        return None
    rho = _rho(q)
    g = _g(q)
    h1 = _qty(q, "h1", r"m|cm|mm") or _qty(q, "h_1", r"m|cm|mm")
    h2 = _qty(q, "h2", r"m|cm|mm") or _qty(q, "h_2", r"m|cm|mm")
    if rho is not None and h1 is not None and h2 is not None:
        return _result(q, rho * g * (h2 - h1), "kPa", "Delta p = rho*g*(h2-h1)", {"rho": rho, "g": g, "h1": h1, "h2": h2})
    return None


def _solve_hydrostatic_force(q: str) -> SolverResult | None:
    l = _lower(q)
    rho = _rho(q)
    g = _g(q)
    if rho is None:
        return None
    if "horizontal hatch" in l or ("hydrostatic force" in l and "area" in l and "depth" in l):
        A = _qty(q, "A", r"m\^2|m2|m²|cm\^2|cm2|cm²|mm\^2|mm2|mm²") or _number_before(q, r"area(?:\s+A\s*=)?", r"m\^2|m2|m²|cm\^2|cm2|cm²")
        h = _qty(q, "h", r"m|cm|mm")
        if A is not None and h is not None:
            return _result(q, rho * g * h * A, "N", "F = rho*g*h*A", {"rho": rho, "g": g, "h": h, "A": A})
    if "vertical rectangular gate" in l or ("resultant hydrostatic force" in l and "rectangular" in l):
        b = _qty(q, "b", r"m|cm|mm") or _number_before(q, r"width(?:\s+b\s*=)?", r"m|cm|mm")
        H = _qty(q, "H", r"m|cm|mm") or _number_before(q, r"height(?:\s+H\s*=)?", r"m|cm|mm")
        if b is not None and H is not None:
            return _result(q, 0.5 * rho * g * b * H * H, "N", "F = rho*g*b*H^2/2", {"rho": rho, "g": g, "b": b, "H": H})
    return None


def _solve_center_pressure(q: str) -> SolverResult | None:
    l = _lower(q)
    if "center of pressure" not in l:
        return None
    H = _qty(q, "H", r"m|cm|mm") or _number_before(q, r"height(?:\s+H\s*=)?", r"m|cm|mm")
    if H is not None:
        return _result(q, 2.0 * H / 3.0, "m", "h_cp = 2H/3", {"H": H})
    return None


def _solve_hydraulic_press(q: str) -> SolverResult | None:
    l = _lower(q)
    if "hydraulic press" in l:
        A1 = _qty(q, "A1", r"m\^2|m2|m²|cm\^2|cm2|cm²|mm\^2|mm2|mm²") or _qty(q, "A_1", r"m\^2|m2|m²|cm\^2|cm2|cm²|mm\^2|mm2|mm²")
        A2 = _qty(q, "A2", r"m\^2|m2|m²|cm\^2|cm2|cm²|mm\^2|mm2|mm²") or _qty(q, "A_2", r"m\^2|m2|m²|cm\^2|cm2|cm²|mm\^2|mm2|mm²")
        F1 = _qty(q, "F1", r"N") or _qty(q, "F_1", r"N")
        if A1 and A2 and F1 is not None:
            return _result(q, F1 * A2 / A1, "N", "F2 = F1*A2/A1", {"F1": F1, "A1": A1, "A2": A2})
    if "transmitted pressure" in l and "piston" in l:
        F = _qty(q, "F", r"N")
        A = _qty(q, "A", r"m\^2|m2|m²|cm\^2|cm2|cm²|mm\^2|mm2|mm²")
        if F is not None and A:
            return _result(q, F / A, "kPa", "p = F/A", {"F": F, "A": A})
    return None


def _solve_manometer(q: str) -> SolverResult | None:
    l = _lower(q)
    if "manometer" not in l:
        return None
    rho = _rho(q)
    dh = _qty(q, "Deltah", r"m|cm|mm") or _qty(q, "Delta h", r"m|cm|mm") or _number_before(q, r"height difference(?:\s+Deltah\s*=)?", r"m|cm|mm")
    g = _g(q)
    if rho is not None and dh is not None:
        return _result(q, rho * g * dh, "kPa", "Delta p = rho*g*Delta h", {"rho": rho, "g": g, "Delta h": dh})
    return None


def _solve_buoyancy(q: str) -> SolverResult | None:
    l = _lower(q)
    rho = _rho(q, "fluid")
    g = _g(q)
    if "buoyant force" in l or "displaces" in l:
        V = _qty(q, "V", r"m\^3|m3|m³|L|liters?|litres?")
        if rho is not None and V is not None:
            return _result(q, rho * g * V, "N", "F_B = rho*g*V", {"rho": rho, "g": g, "V": V})
    if "apparent weight" in l:
        W = _qty(q, "W", r"N") or _number_before(q, r"weighs(?:\s+W\s*=)?", r"N")
        V = _qty(q, "V", r"m\^3|m3|m³|L|liters?|litres?")
        if rho is not None and V is not None and W is not None:
            return _result(q, W - rho * g * V, "N", "W_app = W - rho*g*V", {"W": W, "rho": rho, "g": g, "V": V})
    if "fraction" in l and "submerged" in l and "calculate" in l:
        ro = _rho(q, "object")
        rf = _rho(q, "fluid")
        if ro is not None and rf is not None:
            return _result(q, ro / rf, "dimensionless", "fraction = rho_object/rho_fluid", {"rho_object": ro, "rho_fluid": rf})
    if "object's density" in l or "object density" in l:
        rf = _rho(q, "fluid")
        m = re.search(rf"fraction\s+(?P<v>{NUM})\b", _norm(q), flags=re.I)
        frac = _parse_num(m.group("v")) if m else _qty(q, "fraction")
        if rf is not None and frac is not None:
            return _result(q, frac * rf, "kg/m^3", "rho_object = fraction*rho_fluid", {"fraction": frac, "rho_fluid": rf})
    return None


def _solve_continuity(q: str) -> SolverResult | None:
    l = _lower(q)
    if "incompressible flow" in l and "diameter" in l and "velocity" in l:
        d1 = _qty(q, "d1", r"m|cm|mm") or _qty(q, "d_1", r"m|cm|mm")
        d2 = _qty(q, "d2", r"m|cm|mm") or _qty(q, "d_2", r"m|cm|mm")
        v1 = _qty(q, "v1", r"m/s") or _qty(q, "v_1", r"m/s")
        if d1 and d2 and v1 is not None:
            return _result(q, v1 * (d1 / d2) ** 2, "m/s", "v2 = v1*(d1/d2)^2", {"d1": d1, "d2": d2, "v1": v1})
    if "volume flow rate" in l and "circular pipe" in l and "diameter" in l and "mean velocity" in l and "required" not in l:
        d = _qty(q, "d", r"m|cm|mm")
        v = _qty(q, "v", r"m/s")
        if d and v is not None:
            return _result(q, math.pi * d * d * v / 4.0, "m^3/s", "Q = (pi*d^2/4)*v", {"d": d, "v": v})
    if "required circular pipe diameter" in l or ("find" in l and "pipe diameter" in l and "flow rate" in l):
        Q = _qty(q, "Q", r"m\^3/s|m3/s|m³/s|L/min")
        v = _qty(q, "v", r"m/s")
        if Q is not None and v:
            return _result(q, math.sqrt(4.0 * Q / (math.pi * v)), "m", "d = sqrt(4Q/(pi*v))", {"Q": Q, "v": v})
    if "mass flow rate" in l:
        rho = _rho(q)
        Q = _qty(q, "Q", r"m\^3/s|m3/s|m³/s|L/min")
        if rho is not None and Q is not None:
            return _result(q, rho * Q, "kg/s", "m_dot = rho*Q", {"rho": rho, "Q": Q})
    return None


def _solve_bernoulli(q: str) -> SolverResult | None:
    l = _lower(q)
    if "bernoulli" not in l and "horizontal pipe" not in l:
        return None
    rho = _rho(q)
    if rho is None:
        return None
    p1 = _qty(q, "p1", r"kPa|Pa|MPa") or _qty(q, "p_1", r"kPa|Pa|MPa")
    v1 = _qty(q, "v1", r"m/s") or _qty(q, "v_1", r"m/s")
    v2 = _qty(q, "v2", r"m/s") or _qty(q, "v_2", r"m/s")
    z1 = _qty(q, "z1", r"m|cm|mm") or _qty(q, "z_1", r"m|cm|mm") or 0.0
    z2 = _qty(q, "z2", r"m|cm|mm") or _qty(q, "z_2", r"m|cm|mm") or 0.0
    g = _g(q)
    if p1 is not None and v1 is not None and v2 is not None:
        p2 = p1 + 0.5 * rho * (v1 * v1 - v2 * v2) + rho * g * (z1 - z2)
        return _result(q, p2, "kPa", "p2 = p1 + 0.5*rho*(v1^2-v2^2) + rho*g*(z1-z2)", {"p1": p1, "rho": rho, "v1": v1, "v2": v2, "z1": z1, "z2": z2, "g": g})
    return None


def _solve_orifice_venturi_pitot(q: str) -> SolverResult | None:
    l = _lower(q)
    g = _g(q)
    if "torricelli" in l or "efflux speed" in l:
        h = _qty(q, "h", r"m|cm|mm")
        if h is not None:
            return _result(q, math.sqrt(2.0 * g * h), "m/s", "v = sqrt(2gh)", {"g": g, "h": h})
    if "orifice" in l and "discharge" in l:
        d = _qty(q, "d", r"m|cm|mm")
        h = _qty(q, "h", r"m|cm|mm")
        Cd = _qty(q, "Cd") or _qty(q, "C_d")
        if d and h is not None and Cd is not None:
            A = math.pi * d * d / 4.0
            return _result(q, Cd * A * math.sqrt(2.0 * g * h), "m^3/s", "Q = Cd*A*sqrt(2gh)", {"Cd": Cd, "A": A, "d": d, "g": g, "h": h})
    if "venturi" in l:
        rho = _rho(q)
        d1 = _qty(q, "d1", r"m|cm|mm") or _qty(q, "d_1", r"m|cm|mm")
        d2 = _qty(q, "d2", r"m|cm|mm") or _qty(q, "d_2", r"m|cm|mm")
        Q = _qty(q, "Q", r"m\^3/s|m3/s|m³/s|L/min")
        if rho is not None and d1 and d2 and Q is not None:
            A1 = math.pi * d1 * d1 / 4.0
            A2 = math.pi * d2 * d2 / 4.0
            v1, v2 = Q / A1, Q / A2
            return _result(q, 0.5 * rho * (v2 * v2 - v1 * v1), "kPa", "p1-p2 = 0.5*rho*(v2^2-v1^2)", {"rho": rho, "Q": Q, "d1": d1, "d2": d2, "v1": v1, "v2": v2})
    if "pitot" in l:
        rho = _rho(q)
        dp = _qty(q, "Deltap", r"kPa|Pa|MPa") or _qty(q, "Delta p", r"kPa|Pa|MPa") or _number_before(q, r"dynamic pressure(?:\s+Deltap\s*=)?", r"kPa|Pa|MPa")
        if rho is not None and dp is not None:
            return _result(q, math.sqrt(2.0 * dp / rho), "m/s", "v = sqrt(2*Delta p/rho)", {"Delta p": dp, "rho": rho})
    return None


def _solve_dynamic_pressure(q: str) -> SolverResult | None:
    l = _lower(q)
    if "dynamic pressure" not in l or "pitot" in l:
        return None
    rho = _rho(q)
    v = _qty(q, "v", r"m/s") or _qty(q, "V", r"m/s")
    if rho is not None and v is not None:
        return _result(q, 0.5 * rho * v * v, "kPa", "q = 0.5*rho*v^2", {"rho": rho, "v": v})
    return None


def _solve_viscous(q: str) -> SolverResult | None:
    l = _lower(q)
    if "poiseuille" in l and "flow rate" in l:
        r = _qty(q, "r", r"m|cm|mm")
        dp = _qty(q, "Deltap", r"kPa|Pa|MPa") or _qty(q, "Delta p", r"kPa|Pa|MPa")
        mu = _qty(q, "mu", r"Pa\s*[·*\.]\s*s|Pa·s|Pa\*s")
        L = _qty(q, "L", r"m|cm|mm")
        if r and dp is not None and mu and L:
            return _result(q, math.pi * r**4 * dp / (8.0 * mu * L), "m^3/s", "Q = pi*r^4*Delta p/(8*mu*L)", {"r": r, "Delta p": dp, "mu": mu, "L": L})
    if "poiseuille" in l and "pressure drop" in l:
        r = _qty(q, "r", r"m|cm|mm")
        L = _qty(q, "L", r"m|cm|mm")
        Q = _qty(q, "Q", r"m\^3/s|m3/s|m³/s|L/min")
        mu = _qty(q, "mu", r"Pa\s*[·*\.]\s*s|Pa·s|Pa\*s")
        if r and L and Q is not None and mu:
            return _result(q, 8.0 * mu * L * Q / (math.pi * r**4), "kPa", "Delta p = 8*mu*L*Q/(pi*r^4)", {"mu": mu, "L": L, "Q": Q, "r": r})
    if "shear stress" in l:
        mu = _qty(q, "mu", r"Pa\s*[·*\.]\s*s|Pa·s|Pa\*s")
        du = _qty(q, "Deltau", r"m/s") or _qty(q, "Delta u", r"m/s")
        dy = _qty(q, "Deltay", r"m|cm|mm") or _qty(q, "Delta y", r"m|cm|mm")
        if mu is not None and du is not None and dy:
            return _result(q, mu * du / dy, "Pa", "tau = mu*Delta u/Delta y", {"mu": mu, "Delta u": du, "Delta y": dy})
    if "kinematic viscosity" in l:
        mu = _qty(q, "mu", r"Pa\s*[·*\.]\s*s|Pa·s|Pa\*s")
        rho = _rho(q)
        if mu is not None and rho:
            return _result(q, mu / rho, "m^2/s", "nu = mu/rho", {"mu": mu, "rho": rho})
    if "reynolds number" in l and "desired" not in l:
        rho = _rho(q)
        v = _qty(q, "v", r"m/s")
        D = _qty(q, "D", r"m|cm|mm")
        mu = _qty(q, "mu", r"Pa\s*[·*\.]\s*s|Pa·s|Pa\*s")
        if rho is not None and v is not None and D and mu:
            return _result(q, rho * v * D / mu, "dimensionless", "Re = rho*v*D/mu", {"rho": rho, "v": v, "D": D, "mu": mu})
    if "desired reynolds" in l or "for a desired reynolds" in l:
        Re = _qty(q, "Re")
        D = _qty(q, "D", r"m|cm|mm")
        rho = _rho(q)
        mu = _qty(q, "mu", r"Pa\s*[·*\.]\s*s|Pa·s|Pa\*s")
        if Re is not None and D and rho and mu:
            return _result(q, Re * mu / (rho * D), "m/s", "v = Re*mu/(rho*D)", {"Re": Re, "mu": mu, "rho": rho, "D": D})
    if "stokes" in l or "terminal speed" in l:
        r = _qty(q, "r", r"m|cm|mm")
        rs = _rho(q, "sphere")
        rf = _rho(q, "fluid")
        mu = _qty(q, "mu", r"Pa\s*[·*\.]\s*s|Pa·s|Pa\*s")
        g = _g(q)
        if r and rs is not None and rf is not None and mu:
            return _result(q, 2.0 * r * r * g * (rs - rf) / (9.0 * mu), "m/s", "v_t = 2*r^2*g*(rho_s-rho_f)/(9*mu)", {"r": r, "rho_s": rs, "rho_f": rf, "mu": mu, "g": g})
    return None


def _solve_drag_losses_power(q: str) -> SolverResult | None:
    l = _lower(q)
    g = _g(q)
    if "drag force" in l:
        Cd = _qty(q, "Cd") or _qty(q, "C_d")
        A = _qty(q, "A", r"m\^2|m2|m²|cm\^2|cm2|cm²")
        rho = _rho(q)
        v = _qty(q, "v", r"m/s") or _number_before(q, r"speed", r"m/s")
        if Cd is not None and A and rho is not None and v is not None:
            return _result(q, 0.5 * Cd * rho * A * v * v, "N", "F_D = 0.5*Cd*rho*A*v^2", {"Cd": Cd, "rho": rho, "A": A, "v": v})
    if "darcy" in l or "weisbach" in l:
        f = _qty(q, "f")
        L = _qty(q, "L", r"m|cm|mm")
        D = _qty(q, "D", r"m|cm|mm")
        v = _qty(q, "v", r"m/s")
        if f is not None and L and D and v is not None:
            return _result(q, f * (L / D) * (v * v / (2.0 * g)), "m", "h_f = f*(L/D)*v^2/(2g)", {"f": f, "L": L, "D": D, "v": v, "g": g})
    if "pressure drop" in l and "head loss" in l:
        hf = _qty(q, "hf", r"m|cm|mm") or _qty(q, "h_f", r"m|cm|mm")
        rho = _rho(q)
        if hf is not None and rho is not None:
            return _result(q, rho * g * hf, "kPa", "Delta p = rho*g*h_f", {"rho": rho, "g": g, "h_f": hf})
    if "minor head loss" in l or "minor-loss" in l:
        K = _qty(q, "K")
        v = _qty(q, "v", r"m/s")
        if K is not None and v is not None:
            return _result(q, K * v * v / (2.0 * g), "m", "h_m = K*v^2/(2g)", {"K": K, "v": v, "g": g})
    if "pump" in l and "shaft power" in l:
        Q = _qty(q, "Q", r"m\^3/s|m3/s|m³/s|L/min")
        H = _qty(q, "H", r"m|cm|mm")
        rho = _rho(q)
        eta = _qty(q, "eta")
        if Q is not None and H and rho is not None and eta:
            return _result(q, rho * g * Q * H / eta, "W", "P_shaft = rho*g*Q*H/eta", {"rho": rho, "g": g, "Q": Q, "H": H, "eta": eta})
    if "turbine" in l and "output power" in l:
        Q = _qty(q, "Q", r"m\^3/s|m3/s|m³/s|L/min")
        H = _qty(q, "H", r"m|cm|mm")
        rho = _rho(q)
        eta = _qty(q, "eta")
        if Q is not None and H and rho is not None and eta:
            return _result(q, eta * rho * g * Q * H, "W", "P_out = eta*rho*g*Q*H", {"eta": eta, "rho": rho, "g": g, "Q": Q, "H": H})
    return None


def _solve_open_channel(q: str) -> SolverResult | None:
    l = _lower(q)
    g = _g(q)
    if "froude number" in l:
        v = _qty(q, "v", r"m/s")
        y = _qty(q, "y", r"m|cm|mm")
        if v is not None and y:
            return _result(q, v / math.sqrt(g * y), "dimensionless", "Fr = v/sqrt(g*y)", {"v": v, "g": g, "y": y})
    if "critical depth" in l:
        # lower-case q can be confused with normalized full text; use symbol extraction.
        qq = _qty(q, "q", r"m\^2/s|m2/s|m²/s")
        if qq is not None:
            return _result(q, (qq * qq / g) ** (1.0 / 3.0), "m", "y_c = (q^2/g)^(1/3)", {"q": qq, "g": g})
    if "hydraulic radius" in l and "rectangular" in l:
        b = _qty(q, "b", r"m|cm|mm")
        y = _qty(q, "y", r"m|cm|mm")
        if b and y:
            A = b * y
            P = b + 2.0 * y
            return _result(q, A / P, "m", "R = A/P = b*y/(b+2y)", {"b": b, "y": y, "A": A, "P": P})
    if "manning" in l and "mean velocity" in l:
        n = _qty(q, "n")
        R = _qty(q, "R", r"m|cm|mm")
        S = _qty(q, "S")
        if n and R and S is not None:
            return _result(q, (1.0 / n) * (R ** (2.0 / 3.0)) * math.sqrt(S), "m/s", "v = (1/n)*R^(2/3)*S^(1/2)", {"n": n, "R": R, "S": S})
    if "manning" in l and "discharge" in l and "rectangular" in l:
        b = _qty(q, "b", r"m|cm|mm")
        y = _qty(q, "y", r"m|cm|mm")
        n = _qty(q, "n")
        S = _qty(q, "S")
        if b and y and n and S is not None:
            A = b * y
            R = A / (b + 2.0 * y)
            return _result(q, (1.0 / n) * A * (R ** (2.0 / 3.0)) * math.sqrt(S), "m^3/s", "Q = (1/n)*A*R^(2/3)*S^(1/2)", {"b": b, "y": y, "A": A, "R": R, "n": n, "S": S})
    return None


def _solve_surface_bulk_properties(q: str) -> SolverResult | None:
    l = _lower(q)
    g = _g(q)
    if "capillary rise" in l:
        r = _qty(q, "r", r"m|cm|mm")
        sigma = _qty(q, "sigma", r"N/m")
        rho = _rho(q)
        if r and sigma is not None and rho is not None:
            return _result(q, 2.0 * sigma / (rho * g * r), "m", "h = 2*sigma/(rho*g*r)", {"sigma": sigma, "rho": rho, "g": g, "r": r})
    if "droplet" in l and "pressure" in l:
        r = _qty(q, "r", r"m|cm|mm")
        sigma = _qty(q, "sigma", r"N/m")
        if r and sigma is not None:
            return _result(q, 2.0 * sigma / r, "Pa", "Delta p = 2*sigma/r", {"sigma": sigma, "r": r})
    if "soap bubble" in l and "pressure" in l:
        r = _qty(q, "r", r"m|cm|mm")
        sigma = _qty(q, "sigma", r"N/m")
        if r and sigma is not None:
            return _result(q, 4.0 * sigma / r, "Pa", "Delta p = 4*sigma/r", {"sigma": sigma, "r": r})
    if "bulk modulus" in l or "fractional volume decrease" in l:
        K = _qty(q, "K", r"GPa|MPa|kPa|Pa")
        m = re.search(rf"(?:DeltaV/V|Delta V/V|fractional volume decrease)\s*(?:=|of)?\s*(?P<v>{NUM})", _norm(q), flags=re.I)
        frac = _parse_num(m.group("v")) if m else None
        if K is not None and frac is not None:
            return _result(q, K * frac, "kPa", "Delta p = K*(Delta V/V)", {"K": K, "fractional_decrease": frac})
    if "specific gravity" in l:
        rho = _rho(q)
        vals = _all_units(q, r"kg/m\^3|kg/m3|kg/m³")
        rho_w = vals[-1] if len(vals) >= 2 else 1000.0
        if rho is not None:
            return _result(q, rho / rho_w, "dimensionless", "SG = rho/rho_water", {"rho": rho, "rho_water": rho_w})
    if "specific gravity sg" in l or "has specific gravity" in l:
        SG = _qty(q, "SG")
        vals = _all_units(q, r"kg/m\^3|kg/m3|kg/m³")
        rho_w = vals[-1] if vals else 1000.0
        if SG is not None:
            return _result(q, SG * rho_w, "kg/m^3", "rho = SG*rho_water", {"SG": SG, "rho_water": rho_w})
    if "specific weight" in l:
        rho = _rho(q)
        if rho is not None:
            return _result(q, rho * g, "N/m^3", "gamma = rho*g", {"rho": rho, "g": g})
    return None


def _solve_compressible_aero_jet(q: str) -> SolverResult | None:
    l = _lower(q)
    if "mach number" in l:
        V = _qty(q, "V", r"m/s") or _number_before(q, r"speed(?:\s+V\s*=)?", r"m/s")
        c = _qty(q, "c", r"m/s") or _number_before(q, r"speed of sound(?:\s+c\s*=)?", r"m/s")
        if V is not None and c:
            return _result(q, V / c, "dimensionless", "M = V/c", {"V": V, "c": c})
    if "speed of sound" in l and "ideal gas" in l:
        gamma = _qty(q, "gamma")
        R = _qty(q, "R", r"J/\(kg[·.]?K\)|J/\(kg\s*·\s*K\)")
        if R is None:
            mR = re.search(rf"(?<![A-Za-z0-9_])R\s*=\s*(?P<v>{NUM})\s*J/\(kg\s*[·.]?\s*K\)", _norm(q), flags=re.I)
            if mR:
                R = _parse_num(mR.group("v"))
        T = _qty(q, "T", r"K")
        if gamma and R and T:
            return _result(q, math.sqrt(gamma * R * T), "m/s", "c = sqrt(gamma*R*T)", {"gamma": gamma, "R": R, "T": T})
    if "pressure coefficient" in l:
        p = _qty(q, "p", r"kPa|Pa|MPa")
        pinf = _qty(q, "pinf", r"kPa|Pa|MPa") or _qty(q, "p_inf", r"kPa|Pa|MPa")
        rho = _rho(q)
        Vinf = _qty(q, "Vinf", r"m/s") or _qty(q, "V_inf", r"m/s")
        if p is not None and pinf is not None and rho is not None and Vinf:
            return _result(q, (p - pinf) / (0.5 * rho * Vinf * Vinf), "dimensionless", "Cp = (p-p_inf)/(0.5*rho*V_inf^2)", {"p": p, "p_inf": pinf, "rho": rho, "V_inf": Vinf})
    if "kinetic power" in l and "jet" in l:
        rho = _rho(q)
        Q = _qty(q, "Q", r"m\^3/s|m3/s|m³/s|L/min")
        v = _qty(q, "v", r"m/s")
        if rho is not None and Q is not None and v is not None:
            return _result(q, 0.5 * rho * Q * v * v, "W", "P = 0.5*rho*Q*v^2", {"rho": rho, "Q": Q, "v": v})
    if "water jet" in l and "force" in l:
        rho = _rho(q)
        Q = _qty(q, "Q", r"m\^3/s|m3/s|m³/s|L/min") or _number_before(q, r"flow rate", r"m\^3/s|m3/s|m³/s|L/min")
        v = _qty(q, "v", r"m/s") or _number_before(q, r"speed", r"m/s")
        if rho is not None and Q is not None and v is not None:
            return _result(q, rho * Q * v, "N", "F = rho*Q*v", {"rho": rho, "Q": Q, "v": v})
    if "hydraulic power" in l or "deltapq" in l:
        dp = _qty(q, "Deltap", r"kPa|Pa|MPa") or _qty(q, "Delta p", r"kPa|Pa|MPa")
        Q = _qty(q, "Q", r"m\^3/s|m3/s|m³/s|L/min")
        if dp is not None and Q is not None:
            return _result(q, dp * Q, "W", "P = Delta p * Q", {"Delta p": dp, "Q": Q})
    return None


_FLUID_SOLVERS: tuple[Callable[[str], SolverResult | None], ...] = (
    _solve_depth_from_pressure,
    _solve_hydrostatic_pressure,
    _solve_pressure_difference,
    _solve_hydrostatic_force,
    _solve_center_pressure,
    _solve_hydraulic_press,
    _solve_manometer,
    _solve_buoyancy,
    _solve_continuity,
    _solve_bernoulli,
    _solve_orifice_venturi_pitot,
    _solve_dynamic_pressure,
    _solve_viscous,
    _solve_drag_losses_power,
    _solve_open_channel,
    _solve_surface_bulk_properties,
    _solve_compressible_aero_jet,
)


def solve_fluid_mechanics(question: str) -> SolverResult | None:
    if not _is_fluid_question(question):
        return None
    l = _lower(question)
    # Fast keyword dispatch keeps large dataset evaluation practical while the
    # branch functions remain formula-based and unit-aware.
    if "at what depth" in l or "find the depth" in l:
        route = (_solve_depth_from_pressure,)
    elif "gauge pressure" in l or "absolute pressure" in l or "hydrostatic pressure" in l or "pressure at depth" in l or "pressure at a depth" in l:
        route = (_solve_hydrostatic_pressure,)
    elif "static liquid" in l and "pressure difference" in l:
        route = (_solve_pressure_difference,)
    elif "horizontal hatch" in l or "vertical rectangular gate" in l or "hydrostatic force" in l or "resultant hydrostatic force" in l:
        route = (_solve_hydrostatic_force,)
    elif "center of pressure" in l:
        route = (_solve_center_pressure,)
    elif "hydraulic press" in l or "transmitted pressure" in l:
        route = (_solve_hydraulic_press,)
    elif "manometer" in l:
        route = (_solve_manometer,)
    elif "buoyant" in l or "apparent weight" in l or "floats" in l or "volume submerged" in l:
        route = (_solve_buoyancy,)
    elif "incompressible flow" in l or "mass flow rate" in l or "required circular pipe diameter" in l or ("volume flow rate" in l and "circular pipe" in l):
        route = (_solve_continuity,)
    elif "bernoulli" in l or "horizontal pipe" in l:
        route = (_solve_bernoulli,)
    elif "torricelli" in l or "efflux" in l or "orifice" in l or "venturi" in l or "pitot" in l:
        route = (_solve_orifice_venturi_pitot,)
    elif "dynamic pressure" in l:
        route = (_solve_dynamic_pressure,)
    elif "poiseuille" in l or "shear stress" in l or "kinematic viscosity" in l or "reynolds" in l or "stokes" in l or "terminal speed" in l:
        route = (_solve_viscous,)
    elif "drag force" in l or "darcy" in l or "weisbach" in l or "head loss" in l or "minor-loss" in l or "minor head" in l or "pump" in l or "turbine" in l:
        route = (_solve_drag_losses_power,)
    elif "froude" in l or "open channel" in l or "manning" in l or "hydraulic radius" in l or "critical depth" in l:
        route = (_solve_open_channel,)
    elif "capillary" in l or "droplet" in l or "soap bubble" in l or "bulk modulus" in l or "specific gravity" in l or "specific weight" in l:
        route = (_solve_surface_bulk_properties,)
    elif "mach number" in l or "speed of sound" in l or "pressure coefficient" in l or "kinetic power" in l or "water jet" in l or "hydraulic power" in l:
        route = (_solve_compressible_aero_jet,)
    else:
        route = _FLUID_SOLVERS
    for solver in route:
        try:
            ans = solver(question)
        except (ZeroDivisionError, ValueError, OverflowError):
            ans = None
        if ans is not None:
            try:
                ans.confidence = max(float(getattr(ans, "confidence", 0.0)), 0.97)
            except Exception:
                pass
            return ans
    return None
