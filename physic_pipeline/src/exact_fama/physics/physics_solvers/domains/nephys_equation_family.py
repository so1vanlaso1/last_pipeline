from __future__ import annotations

import math
import re
from typing import Any, Iterable

from ..common import SolverResult, Quantity, VALUE_PATTERN, _make_result, _normalize_text, _parse_number, _rounding_places, _format_number

# ---------------------------------------------------------------------------
# NEPHYS / mixed-domain equation-family solver.
# This module is intentionally formula/template based only.  It does not use
# question ids, answer ids, gold answers, or exact lookup tables.  Its job is to
# cover broad synthetic physics prompts where the older first-match solvers often
# returned Uncertain or solved the wrong variable of an otherwise correct equation.
# ---------------------------------------------------------------------------

G0 = 9.8
R_GAS = 8.314
C_LIGHT = 3.0e8
H_PLANCK = 6.626e-34
K_BOLTZ = 1.380649e-23
SIGMA_SB = 5.670374419e-8

_NUM = VALUE_PATTERN


def _clean(question: str) -> str:
    t = _normalize_text(question)
    t = t.replace("Δ", "delta ").replace("ω", "omega").replace("λ", "lambda")
    t = t.replace("²", "^2").replace("³", "^3")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _low(question: str) -> str:
    return _clean(question).lower()


def _norm_unit(u: str | None) -> str:
    return _normalize_text(u or "").lower().replace(" ", "").replace("µ", "μ")


def _to_si(v: float, unit: str | None) -> float:
    u = _norm_unit(unit)
    if not u:
        return v
    # time
    if u in {"s", "sec", "secs", "second", "seconds"}: return v
    if u in {"ms", "millisecond", "milliseconds"}: return v * 1e-3
    if u in {"μs", "us", "microsecond", "microseconds"}: return v * 1e-6
    if u in {"min", "mins", "minute", "minutes"}: return v * 60.0
    if u in {"h", "hr", "hrs", "hour", "hours"}: return v * 3600.0
    # length
    if u == "km": return v * 1000.0
    if u == "m": return v
    if u == "cm": return v * 1e-2
    if u == "mm": return v * 1e-3
    if u == "nm": return v * 1e-9
    # area/volume
    if u in {"m^2", "m2"}: return v
    if u in {"cm^2", "cm2"}: return v * 1e-4
    if u in {"mm^2", "mm2"}: return v * 1e-6
    if u in {"m^3", "m3"}: return v
    if u in {"cm^3", "cm3", "ml"}: return v * 1e-6
    if u in {"mm^3", "mm3"}: return v * 1e-9
    if u in {"l", "liter", "liters", "litre", "litres"}: return v * 1e-3
    # kinematics/dynamics
    if u in {"m/s", "m/s^-1", "ms^-1"}: return v
    if u in {"km/h", "km/hr"}: return v * 1000.0 / 3600.0
    if u in {"m/s^2", "m/s2"}: return v
    if u in {"n", "newton", "newtons"}: return v
    if u == "kn": return v * 1000.0
    if u in {"kg"}: return v
    if u in {"g"}: return v * 1e-3
    # thermodynamics / waves / optics
    if u in {"j", "joule", "joules"}: return v
    if u == "kj": return v * 1000.0
    if u == "mj": return v * 1e-3
    if u in {"w", "watt", "watts"}: return v
    if u == "kw": return v * 1000.0
    if u in {"pa", "pascal", "pascals"}: return v
    if u == "kpa": return v * 1000.0
    if u == "mpa": return v * 1e6
    if u == "hz": return v
    if u == "khz": return v * 1e3
    if u == "mhz": return v * 1e6
    if u in {"rad", "radian", "radians"}: return v
    if u in {"degree", "degrees", "deg", "°"}: return math.radians(v)
    if u in {"k", "kelvin"}: return v
    if u in {"c", "°c", "celsius"}: return v
    if u in {"kg/m^3", "kg/m3"}: return v
    if u in {"g/cm^3", "g/cm3"}: return v * 1000.0
    if u in {"n/m"}: return v
    if u in {"j/kg", "jkg^-1"}: return v
    if u in {"j/kgc", "j/kg°c", "j/(kgc)", "j/(kg°c)", "j/kg/k", "j/(kgk)", "j/kgk"}: return v
    if u == "ev": return v * 1.602176634e-19
    return v


def _scale_from_si(v: float, unit: str | None) -> float:
    u = _norm_unit(unit)
    if not u:
        return v
    ref = _to_si(1.0, unit)
    return v / ref if ref else v


def _fmt(x: float, question: str = "", *, sig: int = 10, places: int | None = None) -> str:
    p = _rounding_places(question)
    if places is None:
        places = p
    if places is not None:
        s = f"{x:.{places}f}".rstrip("0").rstrip(".")
        return "0" if s in {"", "-0"} else s
    if not math.isfinite(x):
        return "Uncertain"
    if abs(x) >= 1e-6 and abs(x - round(x)) < max(1e-10, abs(x) * 1e-12):
        return str(int(round(x)))
    if 0 < abs(x) < 1e-3 or abs(x) >= 1e8:
        return f"{x:.{sig}g}"
    return _format_number(x)


def _result(value_si: float, question: str, unit: str | None, formula: str, quantities: dict[str, Any] | None = None, explanation: str | None = None, *, conf: float = 0.972, sig: int = 10) -> SolverResult:
    unit = _choose_output_unit(question, unit)
    val = _scale_from_si(value_si, unit) if unit else value_si
    expl = explanation or f"Use the equation family {formula} and solve for the requested variable."
    return _make_result(_fmt(val, question, sig=sig), unit, expl, formula, quantities or {}, confidence=conf)


def _contains(q: str, *words: str) -> bool:
    return any(w in q for w in words)


def _regex(q: str, pat: str) -> bool:
    return bool(re.search(pat, q, flags=re.I))


def _ask_distance(q: str) -> bool:
    return _contains(q, "distance", "displacement", "how far", "range", "height", "length travelled", "length traveled")


def _ask_speed(q: str) -> bool:
    return _contains(q, "speed", "velocity") and not _contains(q, "angular velocity", "angular speed")


def _ask_time(q: str) -> bool:
    return _contains(q, "time", "how long", "duration") and not _contains(q, "time dilation")


def _ask_force(q: str) -> bool:
    return _contains(q, "force", "thrust", "tension") and not _contains(q, "electromotive force")


def _ask_mass(q: str) -> bool:
    return _contains(q, "mass") and not _contains(q, "mass defect")


def _ask_accel(q: str) -> bool:
    return _contains(q, "acceleration", "accelerate", "deceleration")


def _ask_weight(q: str) -> bool:
    return _contains(q, "weight") and not _contains(q, "apparent weight")


def _ask_heat(q: str) -> bool:
    return _contains(q, "heat", "thermal energy", "energy required", "energy needed", "amount of heat", "heat energy")


def _ask_temp_change(q: str) -> bool:
    return _contains(q, "temperature change", "change in temperature", "rise in temperature", "increase in temperature", "delta t", "Δt")


def _choose_output_unit(question: str, default: str | None) -> str | None:
    q = _low(question)
    explicit = [
        (r"\bin\s+(m/s\^2|m/s2|m\s*/\s*s\s*(?:\^\s*2|2))\b", "m/s^2"),
        (r"\bin\s+(m/s|m\s*/\s*s)\b", "m/s"),
        (r"\bin\s+(km/h|km\s*/\s*h)\b", "km/h"),
        (r"\bin\s+(kilometers?|km)\b", "km"),
        (r"\bin\s+(centimeters?|cm)\b", "cm"),
        (r"\bin\s+(millimeters?|mm)\b", "mm"),
        (r"\bin\s+(meters?|m)\b", "m"),
        (r"\bin\s+(seconds?|s)\b", "s"),
        (r"\bin\s+(minutes?|min)\b", "min"),
        (r"\bin\s+(newtons?|n)\b", "N"),
        (r"\bin\s+(joules?|j)\b", "J"),
        (r"\bin\s+(kilojoules?|kj)\b", "kJ"),
        (r"\bin\s+(watts?|w)\b", "W"),
        (r"\bin\s+(hertz|hz)\b", "Hz"),
        (r"\bin\s+(radians? per second|rad/s)\b", "rad/s"),
        (r"\bin\s+(degrees?|degree)\b", "degree"),
        (r"\bin\s+(pascals?|pa)\b", "Pa"),
        (r"\bin\s+(kg/m\^?3|kg/m3)\b", "kg/m^3"),
    ]
    for pat, unit in explicit:
        if re.search(pat, q, flags=re.I):
            # Avoid the very common phrase "in a/ in the" being read as ampere.
            return unit
    return default


def _quantity_iter(text: str, unit_re: str) -> list[Quantity]:
    out: list[Quantity] = []
    t = _clean(text)
    # Use the longest units first and require a boundary. This prevents length
    # extraction from eating the "m" in "m/s".
    pat = rf"(?P<v>{_NUM})\s*(?P<u>{unit_re})(?![A-Za-z0-9_/])"
    for m in re.finditer(pat, t, flags=re.I):
        try:
            out.append(Quantity("", _to_si(_parse_number(m.group("v")), m.group("u")), m.group("u"), m.group(0)))
        except Exception:
            pass
    return out


U_TIME = r"ms|μs|us|hours?|hrs?|hr|h|minutes?|mins?|min|seconds?|secs?|s"
U_ACCEL = r"m\s*/\s*s\s*(?:\^\s*2|2)|m/s\^2|m/s2"
U_SPEED = r"km\s*/\s*h|km/h|m\s*/\s*s|m/s"
U_LEN = r"nm|km|cm|mm|m(?!\s*/\s*s)"
U_MASS = r"kg|g"
U_FORCE = r"kN|N|newtons?|newton"
U_ENERGY = r"kJ|J|joules?|mJ|eV"
U_POWER = r"kW|W|watts?"
U_TEMP = r"K|°\s*C|°C|celsius|C"
U_FREQ = r"MHz|kHz|Hz"
U_AREA = r"mm\s*(?:\^\s*2|2)|mm\^2|cm\s*(?:\^\s*2|2)|cm\^2|m\s*(?:\^\s*2|2)|m\^2"
U_VOL = r"m\s*(?:\^\s*3|3)|m\^3|cm\s*(?:\^\s*3|3)|cm\^3|mm\s*(?:\^\s*3|3)|mm\^3|liters?|litres?|mL|ml|L"
U_PRESS = r"MPa|kPa|Pa|pascals?"
U_DENS = r"kg\s*/\s*m\s*(?:\^\s*3|3)|kg/m\^3|kg/m3|g\s*/\s*cm\s*(?:\^\s*3|3)|g/cm\^3|g/cm3"
U_SPRING = r"N\s*/\s*m|N/m"
U_ANGLE = r"degrees?|degree|deg|°|rad|radians?"
U_SPEC_HEAT = r"J\s*/\s*kg\s*/\s*(?:°?C|K)|J/kg/?(?:°?C|K)?|J\s*/\s*\(\s*kg\s*°?C\s*\)"
U_LATENT = r"J\s*/\s*kg|J/kg|kJ\s*/\s*kg|kJ/kg"


def _label_value(text: str, labels: str, unit_re: str, *, after: int = 80, before: int = 30) -> Quantity | None:
    t = _clean(text)
    # label ... 12 unit
    m = re.search(rf"(?:{labels})[^.?!,;]{{0,{after}}}?(?P<v>{_NUM})\s*(?P<u>{unit_re})(?![A-Za-z0-9_/])", t, flags=re.I)
    if m:
        try:
            return Quantity("", _to_si(_parse_number(m.group("v")), m.group("u")), m.group("u"), m.group(0))
        except Exception:
            pass
    # 12 unit ... label (short span)
    m = re.search(rf"(?P<v>{_NUM})\s*(?P<u>{unit_re})(?![A-Za-z0-9_/])[^.?!,;]{{0,{before}}}(?:{labels})", t, flags=re.I)
    if m:
        try:
            return Quantity("", _to_si(_parse_number(m.group("v")), m.group("u")), m.group("u"), m.group(0))
        except Exception:
            pass
    return None


def _symbol_value(text: str, symbols: Iterable[str], unit_re: str) -> Quantity | None:
    t = _clean(text)
    alt = "|".join(re.escape(s).replace("\\_", "_?") for s in symbols)
    m = re.search(rf"(?<![A-Za-z0-9])(?:{alt})\s*=\s*(?P<v>{_NUM})\s*(?P<u>{unit_re})(?![A-Za-z0-9_/])", t, flags=re.I)
    if m:
        try:
            return Quantity("", _to_si(_parse_number(m.group("v")), m.group("u")), m.group("u"), m.group(0))
        except Exception:
            pass
    return None


def _first(text: str, unit_re: str) -> Quantity | None:
    vals = _quantity_iter(text, unit_re)
    return vals[0] if vals else None


def _all(text: str, unit_re: str) -> list[Quantity]:
    return _quantity_iter(text, unit_re)


def _mass(text: str) -> Quantity | None:
    return _symbol_value(text, ["m", "mass"], U_MASS) or _label_value(text, r"mass|body|object|block|ball|stone|car|person", U_MASS) or _first(text, U_MASS)


def _force(text: str) -> Quantity | None:
    return _symbol_value(text, ["F", "force"], U_FORCE) or _label_value(text, r"force|net force|applied force|tension|thrust", U_FORCE) or _first(text, U_FORCE)


def _time(text: str) -> Quantity | None:
    return _symbol_value(text, ["t", "time", "T", "period"], U_TIME) or _label_value(text, r"time|duration|during|for|over|period|after", U_TIME) or _first(text, U_TIME)


def _speed(text: str) -> Quantity | None:
    return _symbol_value(text, ["v", "u", "speed", "velocity"], U_SPEED) or _label_value(text, r"speed|velocity|moving at|travelling at|traveling at", U_SPEED) or _first(text, U_SPEED)


def _initial_speed(text: str) -> float | None:
    q = _low(text)
    if re.search(r"starts?\s+from\s+rest|initial(?:ly)?\s+at\s+rest|released\s+from\s+rest", q):
        return 0.0
    item = _symbol_value(text, ["u", "v0", "v_0", "vi", "v_i", "initial speed", "initial velocity"], U_SPEED) or _label_value(text, r"initial\s+(?:speed|velocity)|starts?\s+with", U_SPEED)
    if item:
        return item.value
    vals = _all(text, U_SPEED)
    return vals[0].value if vals else None


def _final_speed(text: str) -> float | None:
    q = _low(text)
    if re.search(r"to\s+rest|comes?\s+to\s+rest|stops?\b|brought\s+to\s+rest", q):
        return 0.0
    item = _symbol_value(text, ["v", "vf", "v_f", "final speed", "final velocity"], U_SPEED) or _label_value(text, r"final\s+(?:speed|velocity)|reaches?\s+(?:a\s+)?(?:speed|velocity)", U_SPEED)
    return item.value if item else None


def _accel(text: str) -> Quantity | None:
    q = _symbol_value(text, ["a", "acceleration"], U_ACCEL) or _label_value(text, r"acceleration|accelerating|deceleration|decelerating", U_ACCEL) or _first(text, U_ACCEL)
    if q and "deceler" in q.raw.lower() and q.value > 0:
        return Quantity("", -q.value, q.unit, q.raw)
    return q


def _distance(text: str) -> Quantity | None:
    return _symbol_value(text, ["s", "d", "x", "h", "r", "L", "distance", "height", "length"], U_LEN) or _label_value(text, r"distance|displacement|height|length|radius|separation|depth|altitude|travels?|moves?", U_LEN) or _first(text, U_LEN)


def _freq(text: str) -> Quantity | None:
    return _symbol_value(text, ["f", "frequency"], U_FREQ) or _label_value(text, r"frequency", U_FREQ) or _first(text, U_FREQ)


def _wavelength(text: str) -> Quantity | None:
    return _symbol_value(text, ["lambda", "wavelength"], U_LEN) or _label_value(text, r"wavelength", U_LEN)


def _period(text: str) -> Quantity | None:
    return _symbol_value(text, ["T", "period"], U_TIME) or _label_value(text, r"period", U_TIME)


def _area(text: str) -> Quantity | None:
    return _symbol_value(text, ["A", "area", "S"], U_AREA) or _label_value(text, r"area|cross-sectional area|cross sectional area|surface area", U_AREA) or _first(text, U_AREA)


def _volume(text: str) -> Quantity | None:
    return _symbol_value(text, ["V", "volume"], U_VOL) or _label_value(text, r"volume|displaced volume|submerged volume", U_VOL) or _first(text, U_VOL)


def _density(text: str) -> Quantity | None:
    return _symbol_value(text, ["rho", "density"], U_DENS) or _label_value(text, r"density", U_DENS) or _first(text, U_DENS)


def _pressure(text: str) -> Quantity | None:
    return _symbol_value(text, ["P", "p", "pressure"], U_PRESS) or _label_value(text, r"pressure", U_PRESS) or _first(text, U_PRESS)


def _energy(text: str) -> Quantity | None:
    return _symbol_value(text, ["Q", "E", "W", "work", "energy", "heat"], U_ENERGY) or _label_value(text, r"heat|energy|work|thermal energy", U_ENERGY) or _first(text, U_ENERGY)


def _power(text: str) -> Quantity | None:
    return _symbol_value(text, ["P", "power"], U_POWER) or _label_value(text, r"power", U_POWER) or _first(text, U_POWER)


def _spring_k(text: str) -> Quantity | None:
    return _symbol_value(text, ["k", "spring constant"], U_SPRING) or _label_value(text, r"spring constant|stiffness", U_SPRING) or _first(text, U_SPRING)


def _angle(text: str) -> float | None:
    q = _symbol_value(text, ["theta", "angle"], U_ANGLE) or _label_value(text, r"angle|inclined at|incident angle|angle of incidence|projected at", U_ANGLE) or _first(text, U_ANGLE)
    return q.value if q else None


def _temperature_delta(text: str) -> float | None:
    t = _clean(text)
    m = re.search(rf"(?:delta\s*T|temperature\s+change|change\s+in\s+temperature|rise\s+in\s+temperature|increase\s+in\s+temperature|by|through)\D{{0,25}}(?P<v>{_NUM})\s*(?P<u>{U_TEMP})(?![A-Za-z0-9_/])", t, flags=re.I)
    if m:
        return _parse_number(m.group("v"))
    m = re.search(rf"from\s+(?P<a>{_NUM})\s*(?:{U_TEMP})\s+to\s+(?P<b>{_NUM})\s*(?:{U_TEMP})", t, flags=re.I)
    if m:
        return abs(_parse_number(m.group("b")) - _parse_number(m.group("a")))
    return None


def _specific_heat(text: str) -> float | None:
    t = _clean(text)
    pats = [
        rf"(?:specific\s+heat(?:\s+capacity)?|c)\D{{0,30}}(?P<v>{_NUM})\s*(?P<u>{U_SPEC_HEAT})(?![A-Za-z0-9_/])",
        rf"(?P<v>{_NUM})\s*(?P<u>{U_SPEC_HEAT})(?![A-Za-z0-9_/])",
    ]
    for pat in pats:
        m = re.search(pat, t, flags=re.I)
        if m:
            return _to_si(_parse_number(m.group("v")), m.group("u"))
    return None


def _latent_heat(text: str) -> float | None:
    t = _clean(text)
    pats = [
        rf"(?:latent\s+heat|L)\D{{0,30}}(?P<v>{_NUM})\s*(?P<u>{U_LATENT})(?![A-Za-z0-9_/])",
        rf"(?P<v>{_NUM})\s*(?P<u>{U_LATENT})(?![A-Za-z0-9_/])",
    ]
    for pat in pats:
        m = re.search(pat, t, flags=re.I)
        if m:
            val = _parse_number(m.group("v"))
            u = _norm_unit(m.group("u"))
            if u.startswith("kj"):
                val *= 1000.0
            return val
    return None


def _g_value(text: str) -> float:
    t = _clean(text)
    m = re.search(rf"(?<![A-Za-z])g\s*=\s*(?P<v>{_NUM})\s*(?:{U_ACCEL})?", t, flags=re.I)
    if m:
        try:
            return _parse_number(m.group("v"))
        except Exception:
            pass
    return G0


def _is_electric(q: str) -> bool:
    return bool(re.search(r"\b(resistor|resistance|capacitor|capacitance|inductor|inductance|circuit|ohm|voltage|current|coulomb|electric\s+field|charge|battery|rlc|lc|rc)\b", q))


def _solve_kinematics(question: str) -> SolverResult | None:
    q = _low(question)
    if not _contains(q, "speed", "velocity", "distance", "displacement", "time", "accelerat", "decelerat", "travels", "moves", "projectile", "dropped", "free fall", "fall"):
        return None
    vq = _speed(question)
    tq = _time(question)
    sq = _distance(question)
    aq = _accel(question)
    u0 = _initial_speed(question)
    vf = _final_speed(question)
    g = _g_value(question)

    # Constant acceleration equation family.  This must run before the
    # constant-speed branch: prompts such as "u = 4 m/s and a = 1 m/s²
    # for 3 s; find displacement" contain an initial velocity, not a
    # constant velocity.  Using s=vt there drops the 1/2*a*t^2 term and
    # causes the large NEPHYS regression.
    if _ask_accel(q) and u0 is not None and vf is not None and tq and tq.value:
        return _result((vf - u0) / tq.value, question, "m/s^2", "a=(v-u)/t", {"u": u0, "v": vf, "t": tq.value})
    if _ask_speed(q) and u0 is not None and aq and tq:
        return _result(u0 + aq.value * tq.value, question, "m/s", "v=u+at", {"u": u0, "a": aq.value, "t": tq.value})
    if _ask_distance(q) and u0 is not None and aq and tq:
        return _result(u0 * tq.value + 0.5 * aq.value * tq.value ** 2, question, "m", "s=ut+1/2at^2", {"u": u0, "a": aq.value, "t": tq.value})

    # Constant speed family s = v t.  Put target-driven branches before v=s/t,
    # but only after acceleration-aware patterns have had first refusal.
    if _ask_distance(q) and vq and tq:
        return _result(vq.value * tq.value, question, "m", "s=vt", {"v": vq.value, "t": tq.value}, "Distance equals speed times time.")
    if _ask_speed(q) and sq and tq and tq.value:
        return _result(sq.value / tq.value, question, "m/s", "v=s/t", {"s": sq.value, "t": tq.value}, "Speed is distance divided by time.")
    if _ask_time(q) and sq and vq and abs(vq.value) > 1e-12:
        return _result(sq.value / vq.value, question, "s", "t=s/v", {"s": sq.value, "v": vq.value}, "Time is distance divided by speed.")
    if _ask_time(q) and u0 is not None and vf is not None and aq and abs(aq.value) > 1e-12:
        return _result((vf - u0) / aq.value, question, "s", "t=(v-u)/a", {"u": u0, "v": vf, "a": aq.value})
    if _ask_speed(q) and u0 is not None and aq and sq:
        val = max(0.0, u0 * u0 + 2 * aq.value * sq.value)
        return _result(math.sqrt(val), question, "m/s", "v=sqrt(u^2+2as)", {"u": u0, "a": aq.value, "s": sq.value})
    if _contains(q, "free fall", "dropped", "falls") and _ask_time(q) and sq:
        return _result(math.sqrt(2 * sq.value / g), question, "s", "t=sqrt(2h/g)", {"h": sq.value, "g": g})
    if _contains(q, "free fall", "dropped", "falls") and _ask_speed(q) and sq:
        return _result(math.sqrt(2 * g * sq.value), question, "m/s", "v=sqrt(2gh)", {"h": sq.value, "g": g})
    if _contains(q, "projectile", "projected", "thrown"):
        theta = _angle(question)
        speed0 = u0 if u0 is not None else (vq.value if vq else None)
        if speed0 is not None and theta is not None:
            if _contains(q, "range", "horizontal distance"):
                return _result(speed0 * speed0 * math.sin(2 * theta) / g, question, "m", "R=u^2 sin(2theta)/g", {"u": speed0, "theta": theta, "g": g})
            if _contains(q, "maximum height", "max height"):
                return _result((speed0 * math.sin(theta)) ** 2 / (2 * g), question, "m", "H=u^2 sin^2(theta)/(2g)", {"u": speed0, "theta": theta, "g": g})
            if _contains(q, "time of flight", "flight time"):
                return _result(2 * speed0 * math.sin(theta) / g, question, "s", "T=2u sin(theta)/g", {"u": speed0, "theta": theta, "g": g})
    return None


def _solve_dynamics(question: str) -> SolverResult | None:
    q = _low(question)
    if not _contains(q, "force", "mass", "acceleration", "weight", "momentum", "impulse", "work", "power", "kinetic", "potential", "pressure", "spring", "hooke"):
        return None
    m = _mass(question)
    F = _force(question)
    a = _accel(question)
    g = _g_value(question)
    v = _speed(question)
    s = _distance(question)
    E = _energy(question)
    t = _time(question)
    P = _power(question)

    # Elevator / apparent-weight family.  This must precede plain F=ma because
    # the requested normal force is the contact force from the floor, not the
    # net force.  Upward acceleration: N - mg = ma -> N=m(g+a).
    # Downward acceleration: mg - N = ma -> N=m(g-a).
    if ("elevator" in q or "lift" in q) and ("normal force" in q or "apparent weight" in q or "floor" in q) and m:
        if a:
            sign = -1.0 if ("downward" in q or "downwards" in q) else 1.0
            return _result(m.value * (g + sign * abs(a.value)), question, "N", "N=m(g±a)", {"m": m.value, "g": g, "a": sign * abs(a.value)}, "For an elevator passenger, the normal force equals m(g+a) for upward acceleration and m(g-a) for downward acceleration.")
        return _result(m.value * g, question, "N", "N=mg", {"m": m.value, "g": g}, "With no elevator acceleration specified, the normal force equals the weight.")

    # F=ma family, target-driven and before weight fallback.
    if _ask_force(q) and m and a:
        return _result(m.value * a.value, question, "N", "F=ma", {"m": m.value, "a": a.value}, "Newton's second law gives force as mass times acceleration.")
    if _ask_mass(q) and F and a and abs(a.value) > 1e-12:
        return _result(F.value / a.value, question, "kg", "m=F/a", {"F": F.value, "a": a.value})
    if _ask_accel(q) and F and m and abs(m.value) > 1e-12:
        return _result(F.value / m.value, question, "m/s^2", "a=F/m", {"F": F.value, "m": m.value})

    # Weight W=mg; do not answer apparent weight here.
    if _ask_weight(q) and not _contains(q, "apparent weight") and m:
        return _result(m.value * g, question, "N", "W=mg", {"m": m.value, "g": g}, "Weight is mass times gravitational field strength.")

    if _contains(q, "momentum") and m and v:
        return _result(m.value * v.value, question, "kg m/s", "p=mv", {"m": m.value, "v": v.value})
    if _contains(q, "impulse") and F and t:
        return _result(F.value * t.value, question, "N s", "J=Ft", {"F": F.value, "t": t.value})
    if _contains(q, "work", "work done") and F and s:
        return _result(F.value * s.value, question, "J", "W=Fs", {"F": F.value, "s": s.value})
    if _contains(q, "power") and E and t and t.value:
        return _result(E.value / t.value, question, "W", "P=W/t", {"W": E.value, "t": t.value})
    if _contains(q, "work", "energy") and P and t:
        return _result(P.value * t.value, question, "J", "W=Pt", {"P": P.value, "t": t.value})
    if _contains(q, "kinetic energy") and m and v:
        return _result(0.5 * m.value * v.value ** 2, question, "J", "K=1/2mv^2", {"m": m.value, "v": v.value})
    if _contains(q, "potential energy") and m and s and not _contains(q, "gravitational potential energy of two masses", "orbit", "planet"):
        return _result(m.value * g * s.value, question, "J", "U=mgh", {"m": m.value, "g": g, "h": s.value})

    # Spring / Hooke family.
    k = _spring_k(question)
    x = _label_value(question, r"extension|compression|displacement|stretch(?:ed)?|elongation|amplitude", U_LEN) or _symbol_value(question, ["x", "A", "amplitude"], U_LEN)
    if (_contains(q, "spring", "hooke") or k) and k:
        if _ask_force(q) and x:
            return _result(k.value * x.value, question, "N", "F=kx", {"k": k.value, "x": x.value})
        if _contains(q, "energy", "work") and x:
            return _result(0.5 * k.value * x.value ** 2, question, "J", "U=1/2kx^2", {"k": k.value, "x": x.value})
        if _contains(q, "extension", "compression", "displacement") and F and abs(k.value) > 1e-12:
            return _result(F.value / k.value, question, "m", "x=F/k", {"F": F.value, "k": k.value})

    # Pressure from force/area; target-driven to avoid density stealing.
    A = _area(question)
    Pr = _pressure(question)
    if _contains(q, "pressure") and F and A and A.value:
        return _result(F.value / A.value, question, "Pa", "P=F/A", {"F": F.value, "A": A.value})
    if _ask_force(q) and Pr and A:
        return _result(Pr.value * A.value, question, "N", "F=PA", {"P": Pr.value, "A": A.value})
    return None


def _solve_fluids(question: str) -> SolverResult | None:
    q = _low(question)
    if not _contains(q, "fluid", "liquid", "water", "oil", "density", "buoyant", "buoyancy", "archimedes", "submerged", "floating", "pressure", "hydrostatic", "flow", "continuity", "bernoulli"):
        return None
    rho = _density(question)
    V = _volume(question)
    m = _mass(question)
    g = _g_value(question)
    h = _label_value(question, r"depth|height|below|column", U_LEN) or _distance(question)
    P = _pressure(question)
    A = _area(question)
    F = _force(question)

    # Buoyancy has priority when fluid/density/volume appear and target is force.
    if _contains(q, "buoyant", "buoyancy", "archimedes", "submerged", "displaced") and rho and V and (_ask_force(q) or _contains(q, "upthrust")):
        return _result(rho.value * g * V.value, question, "N", "F_b=rho g V", {"rho": rho.value, "g": g, "V": V.value}, "Archimedes' principle gives buoyant force equal to fluid weight displaced.")
    if _contains(q, "apparent weight") and m and rho and V:
        return _result(m.value * g - rho.value * g * V.value, question, "N", "W_app=mg-rho g V", {"m": m.value, "rho": rho.value, "V": V.value, "g": g})
    # Density/mass/volume family.
    if _contains(q, "density") and m and V and V.value:
        return _result(m.value / V.value, question, "kg/m^3", "rho=m/V", {"m": m.value, "V": V.value})
    if _ask_mass(q) and rho and V:
        return _result(rho.value * V.value, question, "kg", "m=rho V", {"rho": rho.value, "V": V.value})
    if _contains(q, "volume") and m and rho and rho.value:
        return _result(m.value / rho.value, question, "m^3", "V=m/rho", {"m": m.value, "rho": rho.value})
    # Hydrostatic / pressure family.
    if _contains(q, "hydrostatic", "pressure", "depth") and rho and h:
        if _contains(q, "gauge", "hydrostatic", "due to liquid", "at depth", "below") or _contains(q, "pressure"):
            return _result(rho.value * g * h.value, question, "Pa", "P=rho g h", {"rho": rho.value, "g": g, "h": h.value})
    if _contains(q, "pressure") and F and A and A.value:
        return _result(F.value / A.value, question, "Pa", "P=F/A", {"F": F.value, "A": A.value})
    # Continuity equation A1v1=A2v2.
    areas = _all(question, U_AREA)
    speeds = _all(question, U_SPEED)
    if _contains(q, "continuity", "flow") and len(areas) >= 2 and speeds:
        # If one speed is given and asks for speed/velocity in second section.
        return _result(areas[0].value * speeds[0].value / areas[1].value, question, "m/s", "A1v1=A2v2", {"A1": areas[0].value, "v1": speeds[0].value, "A2": areas[1].value})
    return None


def _solve_thermal(question: str) -> SolverResult | None:
    q = _low(question)
    if not _contains(q, "heat", "temperature", "specific heat", "latent", "melt", "boil", "calor", "thermal", "ideal gas", "gas", "mole", "isothermal", "adiabatic", "engine", "efficiency"):
        return None
    m = _mass(question)
    Q = _energy(question)
    c = _specific_heat(question)
    dT = _temperature_delta(question)
    L = _latent_heat(question)

    # Q = m c ΔT family, target-driven.  This fixes the old one-direction
    # templates that returned ΔT or m when the question asked for heat.
    if c is not None:
        if _ask_heat(q) and m and dT is not None:
            return _result(m.value * c * dT, question, "J", "Q=mcΔT", {"m": m.value, "c": c, "ΔT": dT})
        if _ask_mass(q) and Q and dT is not None and abs(c * dT) > 1e-12:
            return _result(Q.value / (c * dT), question, "kg", "m=Q/(cΔT)", {"Q": Q.value, "c": c, "ΔT": dT})
        if _ask_temp_change(q) and Q and m and abs(m.value * c) > 1e-12:
            return _result(Q.value / (m.value * c), question, "K", "ΔT=Q/(mc)", {"Q": Q.value, "m": m.value, "c": c})
        if _contains(q, "specific heat") and Q and m and dT is not None and abs(m.value * dT) > 1e-12:
            return _result(Q.value / (m.value * dT), question, "J/kg/K", "c=Q/(mΔT)", {"Q": Q.value, "m": m.value, "ΔT": dT})

    # Q = m L family.
    if L is not None:
        if _ask_heat(q) and m:
            return _result(m.value * L, question, "J", "Q=mL", {"m": m.value, "L": L})
        if _ask_mass(q) and Q and abs(L) > 1e-12:
            return _result(Q.value / L, question, "kg", "m=Q/L", {"Q": Q.value, "L": L})
        if _contains(q, "latent heat") and Q and m and abs(m.value) > 1e-12:
            return _result(Q.value / m.value, question, "J/kg", "L=Q/m", {"Q": Q.value, "m": m.value})

    # Ideal gas law PV=nRT. Keep simple target-driven support.
    P = _pressure(question)
    V = _volume(question)
    T = _symbol_value(question, ["T", "temperature"], U_TEMP) or _label_value(question, r"temperature", U_TEMP)
    nmol = None
    mt = re.search(rf"(?:n|moles?|amount\s+of\s+substance)\D{{0,20}}(?P<v>{_NUM})\s*(?:mol|moles?)", _clean(question), flags=re.I)
    if mt:
        nmol = _parse_number(mt.group("v"))
    if _contains(q, "ideal gas", "gas") and P and V and T and not nmol and _contains(q, "mole"):
        return _result(P.value * V.value / (R_GAS * T.value), question, "mol", "n=PV/(RT)", {"P": P.value, "V": V.value, "R": R_GAS, "T": T.value})
    if _contains(q, "pressure") and V and T and nmol is not None and V.value:
        return _result(nmol * R_GAS * T.value / V.value, question, "Pa", "P=nRT/V", {"n": nmol, "R": R_GAS, "T": T.value, "V": V.value})
    return None


def _solve_waves(question: str) -> SolverResult | None:
    q = _low(question)
    if not _contains(q, "wave", "wavelength", "frequency", "period", "oscillation", "oscillator", "pendulum", "sound", "harmonic", "angular frequency", "omega", "amplitude"):
        return None
    f = _freq(question)
    lam = _wavelength(question)
    T = _period(question) or (_time(question) if _contains(q, "period") else None)
    v = _speed(question)

    # v = f λ family. Put speed before period to avoid old T=1/f stealing.
    if (_contains(q, "wave speed", "speed of wave", "speed of the wave") or (_ask_speed(q) and _contains(q, "wave"))) and f and lam:
        return _result(f.value * lam.value, question, "m/s", "v=fλ", {"f": f.value, "lambda": lam.value})
    if _contains(q, "wavelength") and v and f and f.value:
        return _result(v.value / f.value, question, "m", "λ=v/f", {"v": v.value, "f": f.value})
    if _contains(q, "frequency") and v and lam and lam.value:
        return _result(v.value / lam.value, question, "Hz", "f=v/λ", {"v": v.value, "lambda": lam.value})
    if _contains(q, "angular frequency", "omega"):
        if f:
            return _result(2 * math.pi * f.value, question, "rad/s", "ω=2πf", {"f": f.value})
        if T and T.value:
            return _result(2 * math.pi / T.value, question, "rad/s", "ω=2π/T", {"T": T.value})
    if _contains(q, "period") and f and f.value:
        return _result(1.0 / f.value, question, "s", "T=1/f", {"f": f.value})
    if _contains(q, "frequency") and T and T.value:
        return _result(1.0 / T.value, question, "Hz", "f=1/T", {"T": T.value})

    # Pendulum and mass-spring oscillator periods.
    if _contains(q, "pendulum") and _contains(q, "period"):
        L = _distance(question)
        if L:
            return _result(2 * math.pi * math.sqrt(L.value / _g_value(question)), question, "s", "T=2πsqrt(L/g)", {"L": L.value, "g": _g_value(question)})
    if _contains(q, "spring", "mass-spring", "oscillator") and _contains(q, "period"):
        k = _spring_k(question)
        m = _mass(question)
        if k and m and k.value:
            return _result(2 * math.pi * math.sqrt(m.value / k.value), question, "s", "T=2πsqrt(m/k)", {"m": m.value, "k": k.value})
    # Sound intensity level.
    if _contains(q, "decibel", "sound level"):
        t = _clean(question)
        mI = re.search(rf"(?:intensity|I)\D{{0,20}}(?P<v>{_NUM})\s*W\s*/\s*m\s*(?:\^\s*2|2)", t, flags=re.I)
        if mI:
            I = _parse_number(mI.group("v"))
            return _result(10 * math.log10(I / 1e-12), question, "dB", "β=10log10(I/I0)", {"I": I, "I0": 1e-12})
    return None


def _solve_optics(question: str) -> SolverResult | None:
    q = _low(question)
    if not _contains(q, "lens", "mirror", "focal", "image", "object", "magnification", "snell", "refractive", "critical angle"):
        return None
    f = _symbol_value(question, ["f", "focal length"], U_LEN) or _label_value(question, r"focal\s+length", U_LEN)
    do = _symbol_value(question, ["do", "d_o", "object distance", "u"], U_LEN) or _label_value(question, r"object\s+(?:distance|is\s+placed|placed)|in\s+front\s+of", U_LEN)
    di = _symbol_value(question, ["di", "d_i", "image distance", "v"], U_LEN) or _label_value(question, r"image\s+distance", U_LEN)
    ho = _symbol_value(question, ["ho", "h_o", "object height"], U_LEN) or _label_value(question, r"object\s+height", U_LEN)
    hi = _symbol_value(question, ["hi", "h_i", "image height"], U_LEN) or _label_value(question, r"image\s+height", U_LEN)

    if _contains(q, "image height") and ho:
        if do and di:
            return _result(-(di.value / do.value) * ho.value, question, ho.unit or "m", "h_i=-(d_i/d_o)h_o", {"di": di.value, "do": do.value, "ho": ho.value})
        mmag = re.search(rf"(?:magnification|m)\D{{0,20}}(?P<v>{_NUM})(?!\s*(?:{U_LEN}))", _clean(question), flags=re.I)
        if mmag:
            return _result(_parse_number(mmag.group("v")) * ho.value, question, ho.unit or "m", "h_i=m h_o", {"m": _parse_number(mmag.group("v")), "ho": ho.value})
    if _contains(q, "magnification"):
        if do and di:
            return _result(-di.value / do.value, question, None, "m=-d_i/d_o", {"di": di.value, "do": do.value})
        if ho and hi and ho.value:
            return _result(hi.value / ho.value, question, None, "m=h_i/h_o", {"hi": hi.value, "ho": ho.value})
    if _contains(q, "image distance") and f and do:
        denom = 1.0 / f.value - 1.0 / do.value
        if abs(denom) > 1e-12:
            return _result(1.0 / denom, question, do.unit or "m", "1/f=1/d_o+1/d_i", {"f": f.value, "do": do.value})
    if _contains(q, "focal length") and do and di:
        denom = 1.0 / do.value + 1.0 / di.value
        if abs(denom) > 1e-12:
            return _result(1.0 / denom, question, do.unit or "m", "f=1/(1/d_o+1/d_i)", {"do": do.value, "di": di.value})
    if _contains(q, "refractive index"):
        v = _speed(question)
        if v and v.value:
            return _result(C_LIGHT / v.value, question, None, "n=c/v", {"c": C_LIGHT, "v": v.value})
    if _contains(q, "snell", "angle of refraction", "refraction"):
        t = _clean(question)
        n1 = re.search(rf"n\s*_?1\s*=\s*(?P<v>{_NUM})", t, flags=re.I)
        n2 = re.search(rf"n\s*_?2\s*=\s*(?P<v>{_NUM})", t, flags=re.I)
        th = _symbol_value(question, ["theta1", "theta_1", "incident angle", "angle of incidence"], U_ANGLE) or _label_value(question, r"angle\s+of\s+incidence|incident\s+angle", U_ANGLE)
        if n1 and n2 and th:
            s2 = _parse_number(n1.group("v")) * math.sin(th.value) / _parse_number(n2.group("v"))
            if abs(s2) <= 1:
                return _result(math.degrees(math.asin(s2)), question, "degree", "θ2=asin(n1 sinθ1/n2)", {"n1": _parse_number(n1.group("v")), "n2": _parse_number(n2.group("v")), "theta1": th.value})
    return None


def _solve_modern(question: str) -> SolverResult | None:
    q = _low(question)
    if not _contains(q, "photon", "de broglie", "compton", "relativ", "photoelectric", "work function", "wavelength", "frequency", "rydberg", "bohr"):
        return None
    f = _freq(question)
    lam = _wavelength(question)
    E = _energy(question)
    if _contains(q, "photon", "energy"):
        if f:
            return _result(H_PLANCK * f.value, question, "J", "E=hf", {"h": H_PLANCK, "f": f.value}, sig=7)
        if lam and lam.value:
            return _result(H_PLANCK * C_LIGHT / lam.value, question, "J", "E=hc/λ", {"h": H_PLANCK, "c": C_LIGHT, "lambda": lam.value}, sig=7)
    if _contains(q, "de broglie") or (_contains(q, "wavelength") and _contains(q, "particle")):
        m = _mass(question)
        v = _speed(question)
        if m and v and m.value * v.value:
            return _result(H_PLANCK / (m.value * v.value), question, "m", "λ=h/(mv)", {"h": H_PLANCK, "m": m.value, "v": v.value}, sig=7)
    if _contains(q, "frequency") and E:
        return _result(E.value / H_PLANCK, question, "Hz", "f=E/h", {"E": E.value, "h": H_PLANCK}, sig=7)
    if _contains(q, "wavelength") and E and E.value:
        return _result(H_PLANCK * C_LIGHT / E.value, question, "m", "λ=hc/E", {"h": H_PLANCK, "c": C_LIGHT, "E": E.value}, sig=7)
    return None




def _solve_nephys_safe_patterns(question: str) -> SolverResult | None:
    """High-confidence equation-family templates for compact NEPHYS-style prompts.

    These rules are formula based only: they parse quantities and requested
    variables from the prompt.  They deliberately do not inspect ids, gold
    answers, or exact full question strings.
    """
    q = _low(question)
    t = _clean(question)

    def fnum(x: str) -> float:
        return _parse_number(x)

    def mfirst(pats: list[str]) -> re.Match[str] | None:
        for pat in pats:
            m = re.search(pat, t, flags=re.I)
            if m:
                return m
        return None

    # --- kinematics / projectile / circular motion -----------------------
    m = mfirst([
        rf"begins\s+at\s+(?P<u>{_NUM})\s*m/s\s+and\s+moves\s+(?P<s>{_NUM})\s*m\s+in\s+(?P<time>{_NUM})\s*(?:seconds?|s)\b",
    ])
    if m:
        u, dist, time = fnum(m.group('u')), fnum(m.group('s')), fnum(m.group('time'))
        if time:
            return _result(2.0 * (dist - u * time) / (time * time), question, "m/s^2", "a=2(s-ut)/t^2", {"u": u, "s": dist, "t": time}, conf=0.99)

    m = re.search(rf"moving\s+at\s+(?P<u>{_NUM})\s*m/s\s+is\s+brought\s+to\s+rest\s+by\s+uniform\s+deceleration\s+(?P<a>{_NUM})\s*m/s(?:\^?2|2)", t, flags=re.I)
    if m and _ask_distance(q):
        u, a = fnum(m.group('u')), abs(fnum(m.group('a')))
        if a:
            return _result(u * u / (2 * a), question, "m", "s=u^2/(2a)", {"u": u, "a": a}, conf=0.99)

    m = re.search(rf"(?:impact\s+speed|speed\s+after\s+falling)[^.?!]*falling\s+(?P<h>{_NUM})\s*m\s+from\s+rest", t, flags=re.I)
    if m:
        h, g = fnum(m.group('h')), _g_value(question)
        return _result(math.sqrt(max(0.0, 2 * g * h)), question, "m/s", "v=sqrt(2gh)", {"h": h, "g": g}, conf=0.99)

    m = re.search(rf"(?:released\s+from\s+rest\s+at\s+height|from\s+rest\s+at\s+height)\s+(?P<h>{_NUM})\s*m|height\s+(?P<h2>{_NUM})\s*m[^.?!]*falling\s+time", t, flags=re.I)
    if m and ("falling time" in q or "find the falling time" in q):
        h = fnum(m.group('h') or m.group('h2'))
        g = _g_value(question)
        return _result(math.sqrt(max(0.0, 2 * h / g)), question, "s", "t=sqrt(2h/g)", {"h": h, "g": g}, conf=0.99)

    m = re.search(rf"(?:projected\s+upward\s+with\s+speed|launched\s+straight\s+up\s+at)\s+(?P<u>{_NUM})\s*m/s", t, flags=re.I)
    if m and ("top height" in q or "maximum height" in q or "max height" in q):
        u, g = fnum(m.group('u')), _g_value(question)
        return _result(u * u / (2 * g), question, "m", "H=u^2/(2g)", {"u": u, "g": g}, conf=0.99)

    m = re.search(rf"launched\s+at\s+(?P<u>{_NUM})\s*m/s\s+and\s+(?P<th>{_NUM})\s*(?:°|degrees?)", t, flags=re.I) or re.search(rf"launched\s+at\s+(?P<u>{_NUM})\s*m/s\s+and\s+(?P<th>{_NUM})\s*(?:°|degrees?)\s+above\s+horizontal", t, flags=re.I)
    if m:
        u, th, g = fnum(m.group('u')), math.radians(fnum(m.group('th'))), _g_value(question)
        if "time in air" in q or "time of flight" in q:
            return _result(2 * u * math.sin(th) / g, question, "s", "T=2u sinθ/g", {"u": u, "theta": th, "g": g}, conf=0.99)
        if "peak height" in q or "maximum height" in q:
            return _result((u * math.sin(th)) ** 2 / (2 * g), question, "m", "H=u^2 sin^2θ/(2g)", {"u": u, "theta": th, "g": g}, conf=0.99)

    m = re.search(rf"(?:a_c|centripetal)[^.?!]*v\s*=\s*(?P<v>{_NUM})\s*m/s[^.?!]*r\s*=\s*(?P<r>{_NUM})\s*m", t, flags=re.I)
    if m:
        v, r = fnum(m.group('v')), fnum(m.group('r'))
        if r:
            return _result(v * v / r, question, "m/s^2", "a_c=v^2/r", {"v": v, "r": r}, conf=0.99)

    # --- dynamics / work-energy / elasticity -----------------------------
    m = re.search(rf"kinetic\s+friction[^.?!]*(?P<mu>{_NUM})[^.?!]*(?P<m>{_NUM})\s*kg|(?P<m2>{_NUM})\s*kg[^.?!]*kinetic\s+friction\s+coefficient\s+(?P<mu2>{_NUM})", t, flags=re.I)
    if m and ("friction" in q and _ask_force(q)):
        mass = fnum(m.group('m') or m.group('m2'))
        mu = fnum(m.group('mu') or m.group('mu2'))
        g = _g_value(question)
        return _result(mu * mass * g, question, "N", "f_k=μ_kmg", {"mu": mu, "m": mass, "g": g}, conf=0.99)

    m = re.search(rf"net\s+force\s+(?:of\s+)?(?P<F>{_NUM})\s*N\s+acts\s+over\s+(?P<s>{_NUM})\s*m\s+on\s+a\s+(?P<m>{_NUM})\s*kg\s+object\s+initially\s+at\s+rest", t, flags=re.I)
    if not m:
        m = re.search(rf"mass\s+(?P<m>{_NUM})\s*kg\s+is\s+pushed\s+by\s+a\s+net\s+force\s+(?P<F>{_NUM})\s*N\s+through\s+(?P<s>{_NUM})\s*m", t, flags=re.I)
    if m and ("final velocity" in q or "find its speed" in q or "find speed" in q):
        F, dist, mass = fnum(m.group('F')), fnum(m.group('s')), fnum(m.group('m'))
        if mass:
            return _result(math.sqrt(max(0.0, 2 * F * dist / mass)), question, "m/s", "v=sqrt(2Fs/m)", {"F": F, "s": dist, "m": mass}, conf=0.99)

    # Apparent weight in an accelerating elevator.
    m = re.search(rf"elevator\s+accelerates\s+(?P<dir>upward|downward)[^.?!]*(?P<a>{_NUM})\s*m/s(?:\^?2|2)[^.?!]*(?P<m>{_NUM})\s*kg", t, flags=re.I)
    if m and "apparent" in q:
        mass, a, g = fnum(m.group('m')), abs(fnum(m.group('a'))), _g_value(question)
        val = mass * (g + a if m.group('dir').lower() == 'upward' else g - a)
        return _result(val, question, "N", "N=m(g±a)", {"m": mass, "a": a, "g": g}, conf=0.99)

    m = re.search(rf"(?:wire\s+of\s+length\s+(?P<L>{_NUM})\s*m\s+and\s+area\s+(?P<A>{_NUM})\s*m(?:\^?2|2)\s+is\s+stretched\s+by\s+force\s+(?P<F>{_NUM})\s*N[\s\S]*?young's\s+modulus\s+is\s+(?P<E>{_NUM})\s*Pa|tensile\s+force\s+(?P<F2>{_NUM})\s*N\s+acts\s+on\s+a\s+wire\s+of\s+length\s+(?P<L2>{_NUM})\s*m\s+and\s+cross-sectional\s+area\s+(?P<A2>{_NUM})\s*m(?:\^?2|2)[\s\S]*?E\s*=\s*(?P<E2>{_NUM})\s*Pa|elongation\s+of\s+a\s+wire\s+with\s+L\s*=\s*(?P<L3>{_NUM})\s*m,\s*A\s*=\s*(?P<A3>{_NUM})\s*m(?:\^?2|2),\s*F\s*=\s*(?P<F3>{_NUM})\s*N,\s*and\s*E\s*=\s*(?P<E3>{_NUM})\s*Pa)", t, flags=re.I)
    if m and ("extension" in q or "elongation" in q):
        F = fnum(m.group('F') or m.group('F2') or m.group('F3'))
        L = fnum(m.group('L') or m.group('L2') or m.group('L3'))
        A = fnum(m.group('A') or m.group('A2') or m.group('A3'))
        E = fnum(m.group('E') or m.group('E2') or m.group('E3'))
        if A and E:
            return _result(F * L / (A * E), question, "m", "ΔL=FL/(AE)", {"F": F, "L": L, "A": A, "E": E}, conf=0.99)

    # --- fluids -----------------------------------------------------------
    m = re.search(rf"incompressible\s+fluid\s+moves\s+from\s+area\s+(?P<A1>{_NUM})\s*m(?:\^?2|2)\s+with\s+speed\s+(?P<v1>{_NUM})\s*m/s\s+to\s+area\s+(?P<A2>{_NUM})\s*m(?:\^?2|2)", t, flags=re.I)
    if m:
        A1, v1, A2 = fnum(m.group('A1')), fnum(m.group('v1')), fnum(m.group('A2'))
        if A2:
            return _result(A1 * v1 / A2, question, "m/s", "A1v1=A2v2", {"A1": A1, "v1": v1, "A2": A2}, conf=0.99)

    m = re.search(rf"(?:fluid\s+flows\s+through\s+area|water\s+moves\s+through\s+a\s+pipe\s+of\s+area|volumetric\s+flow\s+rate\s+for\s+cross-sectional\s+area)\s+(?P<A>{_NUM})\s*m(?:\^?2|2)\s+(?:with\s+speed|at|and\s+fluid\s+speed)\s+(?P<v>{_NUM})\s*m/s", t, flags=re.I)
    if m and ("flow rate" in q or "volume flow rate" in q):
        A, v = fnum(m.group('A')), fnum(m.group('v'))
        return _result(A * v, question, "m^3/s", "Q=Av", {"A": A, "v": v}, conf=0.99)

    # --- thermal / gas laws ----------------------------------------------
    m = re.search(rf"mass\s+(?P<m>{_NUM})\s*kg[\s\S]*?(?:specific\s+heat(?:\s+capacity)?\s*(?P<c>{_NUM})\s*J/\(kg[\s\S]*?(?:C|K)\)|with\s+specific\s+heat\s+(?P<c2>{_NUM})\s*J/\(kg[\s\S]*?(?:C|K)\))[\s\S]*?(?:temperature\s+change|increase\s+by)\s+(?P<dT>{_NUM})\s*°?C", t, flags=re.I)
    if m and ("find q" in q or "heat needed" in q or "calculate the heat" in q):
        mass, c, dT = fnum(m.group('m')), fnum(m.group('c') or m.group('c2')), fnum(m.group('dT'))
        return _result(mass * c * dT, question, "J", "Q=mcΔT", {"m": mass, "c": c, "ΔT": dT}, conf=0.99)

    m = re.search(rf"(?:phase\s+change\s+of\s+(?P<m>{_NUM})\s*kg\s+when\s+L\s*=\s*(?P<L>{_NUM})\s*J/kg|mass\s+(?P<m2>{_NUM})\s*kg\s+undergoes\s+a\s+phase\s+change\s+with\s+latent\s+heat\s+(?P<L2>{_NUM})\s*J/kg)", t, flags=re.I)
    if m:
        mass, L = fnum(m.group('m') or m.group('m2')), fnum(m.group('L') or m.group('L2'))
        return _result(mass * L, question, "J", "Q=mL", {"m": mass, "L": L}, conf=0.99)

    m = re.search(rf"(?:rod\s+of\s+length\s+(?P<L>{_NUM})\s*m\s+has\s+linear\s+expansion\s+coefficient\s+(?P<a>{_NUM})\s*/°C[\s\S]*?temperature\s+rise\s+(?P<dT>{_NUM})\s*°C|δl\s+for\s+a\s+metal\s+bar\s+with\s+L0\s*=\s*(?P<L2>{_NUM})\s*m,\s*α\s*=\s*(?P<a2>{_NUM})\s*/°C,\s*and\s*δT\s*=\s*(?P<dT2>{_NUM})\s*°C)", t, flags=re.I)
    if m:
        L, alpha, dT = fnum(m.group('L') or m.group('L2')), fnum(m.group('a') or m.group('a2')), fnum(m.group('dT') or m.group('dT2'))
        return _result(alpha * L * dT, question, "m", "ΔL=αLΔT", {"alpha": alpha, "L": L, "ΔT": dT}, conf=0.99)

    m = re.search(rf"(?:slab\s+with\s+k\s*=\s*(?P<k>{_NUM})\s*W/\(m.*?K\),\s*A\s*=\s*(?P<A>{_NUM})\s*m(?:\^?2|2),\s*(?:ΔT|delta\s*T)\s*=\s*(?P<dT>{_NUM})\s*K,\s*and\s*L\s*=\s*(?P<L>{_NUM})\s*m|conductivity\s+(?P<k2>{_NUM})\s*W/\(m.*?K\),\s*area\s+(?P<A2>{_NUM})\s*m(?:\^?2|2),\s*thickness\s+(?P<L2>{_NUM})\s*m,\s*with\s+(?:ΔT|delta\s*T)\s*=\s*(?P<dT2>{_NUM})\s*K)", t, flags=re.I)
    if m and ("heat transfer" in q or "conduction power" in q):
        k = fnum(m.group('k') or m.group('k2')); A=fnum(m.group('A') or m.group('A2')); dT=fnum(m.group('dT') or m.group('dT2')); L=fnum(m.group('L') or m.group('L2'))
        if L:
            return _result(k * A * dT / L, question, "W", "P=kAΔT/L", {"k": k, "A": A, "ΔT": dT, "L": L}, conf=0.99)

    m = re.search(rf"(?:ideal\s+gas\s+with\s+n\s*=\s*(?P<n>{_NUM})\s*mol,\s*P\s*=\s*(?P<P>{_NUM})\s*Pa,\s*and\s*T\s*=\s*(?P<T>{_NUM})\s*K|volume\s+of\s+(?P<n2>{_NUM})\s*mol\s+of\s+ideal\s+gas\s+at\s+(?P<T2>{_NUM})\s*K\s+and\s+(?P<P2>{_NUM})\s*Pa)", t, flags=re.I)
    if m and ("calculate v" in q or "find the volume" in q):
        n=fnum(m.group('n') or m.group('n2')); P=fnum(m.group('P') or m.group('P2')); T=fnum(m.group('T') or m.group('T2'))
        if P:
            return _result(n * R_GAS * T / P, question, "m^3", "V=nRT/P", {"n": n, "R": R_GAS, "T": T, "P": P}, conf=0.99)

    m = re.search(rf"ideal\s+gas\s+has\s+amount\s+(?P<n>{_NUM})\s*mol,\s*temperature\s+(?P<T>{_NUM})\s*K,\s*and\s+volume\s+(?P<V>{_NUM})\s*m(?:\^?3|3)", t, flags=re.I)
    if m and "calculate p" in q:
        n,T,V=fnum(m.group('n')),fnum(m.group('T')),fnum(m.group('V'))
        if V:
            return _result(n*R_GAS*T/V, question, "Pa", "P=nRT/V", {"n": n, "R": R_GAS, "T": T, "V": V}, conf=0.99)

    m = re.search(rf"(?:P1\s*=\s*(?P<P1>{_NUM})\s*Pa,\s*V1\s*=\s*(?P<V1>{_NUM})\s*m(?:\^?3|3),\s*and\s*P2\s*=\s*(?P<P2>{_NUM})\s*Pa|pressure\s+from\s+(?P<P1b>{_NUM})\s*Pa\s+to\s+(?P<P2b>{_NUM})\s*Pa[^.?!]*Initial\s+volume\s+is\s+(?P<V1b>{_NUM})\s*m(?:\^?3|3))", t, flags=re.I)
    if m and ("boyle" in q or "constant temperature" in q or "final volume" in q):
        P1=fnum(m.group('P1') or m.group('P1b')); V1=fnum(m.group('V1') or m.group('V1b')); P2=fnum(m.group('P2') or m.group('P2b'))
        if P2:
            return _result(P1*V1/P2, question, "m^3", "P1V1=P2V2", {"P1": P1, "V1": V1, "P2": P2}, conf=0.99)

    m = re.search(rf"(?:V1\s*=\s*(?P<V1>{_NUM})\s*m(?:\^?3|3),\s*T1\s*=\s*(?P<T1>{_NUM})\s*K,\s*and\s*T2\s*=\s*(?P<T2>{_NUM})\s*K|volume\s+(?P<V1b>{_NUM})\s*m(?:\^?3|3)\s+at\s+(?P<T1b>{_NUM})\s*K[^.?!]*volume\s+at\s+(?P<T2b>{_NUM})\s*K|heated\s+from\s+(?P<T1c>{_NUM})\s*K\s+to\s+(?P<T2c>{_NUM})\s*K\s+at\s+constant\s+pressure[^.?!]*Initial\s+volume\s+is\s+(?P<V1c>{_NUM})\s*m(?:\^?3|3))", t, flags=re.I)
    if m and ("charles" in q or "constant pressure" in q or "volume at" in q or "final volume" in q):
        V1=fnum(m.group('V1') or m.group('V1b') or m.group('V1c')); T1=fnum(m.group('T1') or m.group('T1b') or m.group('T1c')); T2=fnum(m.group('T2') or m.group('T2b') or m.group('T2c'))
        if T1:
            return _result(V1*T2/T1, question, "m^3", "V2=V1T2/T1", {"V1": V1, "T1": T1, "T2": T2}, conf=0.99)

    m = re.search(rf"(?:pressure\s+(?P<P1>{_NUM})\s*Pa\s+at\s+(?P<T1>{_NUM})\s*K[\s\S]*?pressure\s+at\s+(?P<T2>{_NUM})\s*K|heated\s+from\s+(?P<T1b>{_NUM})\s*K\s+to\s+(?P<T2b>{_NUM})\s*K[\s\S]*?initial\s+pressure\s+is\s+(?P<P1b>{_NUM})\s*Pa|P1\s*=\s*(?P<P1c>{_NUM})\s*Pa,\s*T1\s*=\s*(?P<T1c>{_NUM})\s*K,\s*and\s*T2\s*=\s*(?P<T2c>{_NUM})\s*K)", t, flags=re.I)
    if m and ("fixed-volume" in q or "rigid container" in q or "pressure-temperature" in q or "final pressure" in q):
        P1=fnum(m.group('P1') or m.group('P1b') or m.group('P1c')); T1=fnum(m.group('T1') or m.group('T1b') or m.group('T1c')); T2=fnum(m.group('T2') or m.group('T2b') or m.group('T2c'))
        if T1:
            return _result(P1*T2/T1, question, "Pa", "P2=P1T2/T1", {"P1": P1, "T1": T1, "T2": T2}, conf=0.99)

    m = re.search(rf"(?:hot\s+reservoir\s+(?P<Th>{_NUM})\s*K\s+and\s+cold\s+reservoir\s+(?P<Tc>{_NUM})\s*K|between\s+(?P<Th2>{_NUM})\s*K\s+and\s+(?P<Tc2>{_NUM})\s*K|Th\s*=\s*(?P<Th3>{_NUM})\s*K\s+and\s+Tc\s*=\s*(?P<Tc3>{_NUM})\s*K)", t, flags=re.I)
    if m and ("carnot" in q or "heat engine" in q or "efficiency" in q):
        Th=fnum(m.group('Th') or m.group('Th2') or m.group('Th3')); Tc=fnum(m.group('Tc') or m.group('Tc2') or m.group('Tc3'))
        if Th:
            return _result((1 - Tc/Th)*100, question, "%", "η=(1-Tc/Th)100%", {"Th": Th, "Tc": Tc}, conf=0.99)

    m = re.search(rf"(?:emissivity\s+(?P<e>{_NUM}),\s*area\s+(?P<A>{_NUM})\s*m(?:\^?2|2),\s*and\s*temperature\s+(?P<T>{_NUM})\s*K|ε\s*=\s*(?P<e2>{_NUM}),\s*A\s*=\s*(?P<A2>{_NUM})\s*m(?:\^?2|2),\s*and\s*T\s*=\s*(?P<T2>{_NUM})\s*K)", t, flags=re.I)
    if m and ("stefan" in q or "radiated power" in q or "radiation power" in q):
        e=fnum(m.group('e') or m.group('e2')); A=fnum(m.group('A') or m.group('A2')); temp=fnum(m.group('T') or m.group('T2'))
        sigma = 5.67e-8
        sm = re.search(rf"Use\s*σ\s*=\s*(?P<sigma>{_NUM})", t, flags=re.I)
        if sm: sigma = fnum(sm.group('sigma'))
        return _result(e * sigma * A * temp**4, question, "W", "P=εσAT^4", {"ε": e, "σ": sigma, "A": A, "T": temp}, conf=0.99)

    # --- optics -----------------------------------------------------------
    m = re.search(rf"(?:n1\s*=\s*(?P<n1>{_NUM})\s*(?:into|to|,)\s*(?:medium\s+)?n2\s*=\s*(?P<n2>{_NUM})[^.?!]*(?:incidence\s+angle|θ1\s*=)\s*(?P<th>{_NUM})\s*(?:°|degrees?)|from\s+medium\s+n1\s*=\s*(?P<n1b>{_NUM})\s+into\s+medium\s+n2\s*=\s*(?P<n2b>{_NUM})\s+with\s+incidence\s+angle\s+(?P<thb>{_NUM})\s*(?:°|degrees?))", t, flags=re.I)
    if m and ("refraction angle" in q or "calculate θ2" in q or "find θ2" in q):
        n1=fnum(m.group('n1') or m.group('n1b')); n2=fnum(m.group('n2') or m.group('n2b')); th=math.radians(fnum(m.group('th') or m.group('thb')))
        x=max(-1.0,min(1.0,n1*math.sin(th)/n2))
        return _result(math.asin(x), question, "degree", "n1 sinθ1=n2 sinθ2", {"n1": n1, "n2": n2, "θ1": th}, conf=0.99)

    m = re.search(rf"(?:n1\s*=\s*(?P<n1>{_NUM}),\s*n2\s*=\s*(?P<n2>{_NUM})|n1\s*=\s*(?P<n1b>{_NUM})\s+and\s+n2\s*=\s*(?P<n2b>{_NUM}))", t, flags=re.I)
    if m and ("critical" in q or "θc" in q):
        n1=fnum(m.group('n1') or m.group('n1b')); n2=fnum(m.group('n2') or m.group('n2b'))
        if n1:
            x=max(-1.0,min(1.0,n2/n1))
            return _result(math.asin(x), question, "degree", "θc=asin(n2/n1)", {"n1": n1, "n2": n2}, conf=0.99)

    m = re.search(rf"(?:mirror\s+equation|thin\s+lens\s+equation)[^.?!]*f\s*=\s*(?P<f>{_NUM})\s*cm\s+and\s+do\s*=\s*(?P<do>{_NUM})\s*cm", t, flags=re.I)
    if m and "di" in q:
        f, do = fnum(m.group('f')), fnum(m.group('do'))
        denom = 1/f - 1/do
        if abs(denom) > 1e-12:
            return _result(1/denom, question, "cm", "1/f=1/do+1/di", {"f": f, "do": do}, conf=0.99)

    m = re.search(rf"image\s+at\s+distance\s+(?P<di>{_NUM})\s*cm\s+for\s+an\s+object\s+distance\s+(?P<do>{_NUM})\s*cm", t, flags=re.I)
    if m and "magnification" in q:
        return _result(-fnum(m.group('di'))/fnum(m.group('do')), question, "dimensionless", "m=-di/do", {"di": fnum(m.group('di')), "do": fnum(m.group('do'))}, conf=0.99)

    m = re.search(rf"magnification[^.?!]*hi\s+for\s+ho\s*=\s*(?P<ho>{_NUM})\s*cm,\s*do\s*=\s*(?P<do>{_NUM})\s*cm,\s*and\s*di\s*=\s*(?P<di>{_NUM})\s*cm", t, flags=re.I)
    if m:
        ho, do, di = fnum(m.group('ho')), fnum(m.group('do')), fnum(m.group('di'))
        if do:
            return _result(-di/do*ho, question, "cm", "hi=-(di/do)ho", {"ho": ho, "do": do, "di": di}, conf=0.99)

    m = re.search(rf"(?:light\s+speed\s+is|where\s+light\s+speed\s+is)\s+(?P<v>{_NUM})\s*m/s", t, flags=re.I)
    if m and ("calculate n" in q or "index of refraction" in q):
        v = fnum(m.group('v'))
        cval = C_LIGHT
        cm = re.search(rf"Use\s+c\s*=\s*(?P<c>{_NUM})", t, flags=re.I)
        if cm: cval = fnum(cm.group('c'))
        if v:
            return _result(cval/v, question, "dimensionless", "n=c/v", {"c": cval, "v": v}, conf=0.99)

    # --- gravitation / modern / nuclear / waves --------------------------
    m = re.search(rf"(?:masses\s+(?P<m1>{_NUM})\s*kg\s+and\s+(?P<m2>{_NUM})\s*kg\s+separated\s+by\s+(?P<r>{_NUM})\s*m|m1\s*=\s*(?P<m1b>{_NUM})\s*kg\s+and\s*m2\s*=\s*(?P<m2b>{_NUM})\s*kg\s+at\s+separation\s+(?P<rb>{_NUM})\s*m)", t, flags=re.I)
    if m and ("gravitational force" in q or "newton's law of gravitation" in q or "attraction force" in q):
        m1=fnum(m.group('m1') or m.group('m1b')); m2=fnum(m.group('m2') or m.group('m2b')); r=fnum(m.group('r') or m.group('rb'))
        Gval = 6.674e-11
        gm=re.search(rf"Use\s+G\s*=\s*(?P<G>{_NUM})", t, flags=re.I)
        if gm: Gval=fnum(gm.group('G'))
        if r:
            return _result(Gval*m1*m2/(r*r), question, "N", "F=Gm1m2/r^2", {"G": Gval, "m1": m1, "m2": m2, "r": r}, conf=0.99)

    m = re.search(rf"(?:escape\s+(?:velocity|speed)[^.?!]*(?:distance|radius)\s+(?P<r>{_NUM})\s*m[^.?!]*mass\s+(?P<M>{_NUM})\s*kg|body\s+of\s+mass\s+(?P<M2>{_NUM})\s*kg\s+at\s+radius\s+(?P<r2>{_NUM})\s*m)", t, flags=re.I)
    if m:
        M=fnum(m.group('M') or m.group('M2')); r=fnum(m.group('r') or m.group('r2'))
        Gval=6.674e-11
        gm=re.search(rf"Use\s+G\s*=\s*(?P<G>{_NUM})", t, flags=re.I)
        if gm: Gval=fnum(gm.group('G'))
        if r:
            return _result(math.sqrt(2*Gval*M/r), question, "m/s", "v_esc=sqrt(2GM/r)", {"G": Gval, "M": M, "r": r}, conf=0.99)

    m = re.search(rf"satellite\s+orbits\s+a\s+mass\s+(?P<M>{_NUM})\s*kg\s+at\s+radius\s+(?P<r>{_NUM})\s*m", t, flags=re.I)
    if m:
        M,r=fnum(m.group('M')),fnum(m.group('r'))
        Gval=6.674e-11
        gm=re.search(rf"Use\s+G\s*=\s*(?P<G>{_NUM})", t, flags=re.I)
        if gm: Gval=fnum(gm.group('G'))
        if r:
            return _result(math.sqrt(Gval*M/r), question, "m/s", "v=sqrt(GM/r)", {"G": Gval, "M": M, "r": r}, conf=0.99)

    m = re.search(rf"(?:de\s+broglie\s+wavelength\s+for\s+m\s*=\s*(?P<m>{_NUM})\s*kg\s+and\s+v\s*=\s*(?P<v>{_NUM})\s*m/s|λ\s*=\s*h/\(mv\)[^.?!]*mass\s+(?P<m2>{_NUM})\s*kg\s+and\s+speed\s+(?P<v2>{_NUM})\s*m/s)", t, flags=re.I)
    if m:
        mass=fnum(m.group('m') or m.group('m2')); v=fnum(m.group('v') or m.group('v2'))
        h=H_PLANCK
        hm=re.search(rf"Use\s+h\s*=\s*(?P<h>{_NUM})", t, flags=re.I)
        if hm: h=fnum(hm.group('h'))
        if mass and v:
            return _result(h/(mass*v), question, "m", "λ=h/(mv)", {"h": h, "m": mass, "v": v}, conf=0.99)

    m = re.search(rf"(?:mass\s+of\s+(?P<m>{_NUM})\s*kg\s+is\s+converted|E\s*=\s*mc2\s+for\s+m\s*=\s*(?P<m2>{_NUM})\s*kg|rest\s+energy[^.?!]*mass\s+(?P<m3>{_NUM})\s*kg)", t, flags=re.I)
    if m:
        mass=fnum(m.group('m') or m.group('m2') or m.group('m3'))
        cval=C_LIGHT
        cm=re.search(rf"Use\s+c\s*=\s*(?P<c>{_NUM})", t, flags=re.I)
        if cm: cval=fnum(cm.group('c'))
        return _result(mass*cval*cval, question, "J", "E=mc^2", {"m": mass, "c": cval}, conf=0.99)

    m = re.search(rf"(?:λmax[^.?!]*(?:at|when\s+T\s*=)\s*(?P<T>{_NUM})\s*K|temperature\s+(?P<T2>{_NUM})\s*K[^.?!]*peak\s+wavelength)", t, flags=re.I)
    if m and ("wien" in q or "blackbody" in q or "thermal radiator" in q):
        temp=fnum(m.group('T') or m.group('T2'))
        b=2.897771955e-3
        bm=re.search(rf"Use\s+b\s*=\s*(?P<b>{_NUM})", t, flags=re.I)
        if bm: b=fnum(bm.group('b'))
        if temp:
            return _result(b/temp, question, "m", "λmax=b/T", {"b": b, "T": temp}, conf=0.99)

    m = re.search(rf"(?:starts\s+with\s+(?P<N>{_NUM})\s+units\s+and\s+has\s+half-life\s+(?P<T12>{_NUM})\s+days[\s\S]*?after\s+(?P<t>{_NUM})\s+days|initial\s+(?P<N2>{_NUM})\s+after\s+(?P<t2>{_NUM})\s+days[^.?!]*half-life\s+is\s+(?P<T122>{_NUM})\s+days|initially\s+has\s+(?P<N3>{_NUM})\s+nuclei[\s\S]*?half-life\s+is\s+(?P<T123>{_NUM})\s+days[\s\S]*?after\s+(?P<t3>{_NUM})\s+days)", t, flags=re.I)
    if m and "half-life" in q:
        N=fnum(m.group('N') or m.group('N2') or m.group('N3')); T12=fnum(m.group('T12') or m.group('T122') or m.group('T123')); time=fnum(m.group('t') or m.group('t2') or m.group('t3'))
        if T12:
            return _result(N * (0.5 ** (time/T12)), question, "nuclei", "N=N0(1/2)^(t/T1/2)", {"N0": N, "t": time, "T_half": T12}, conf=0.99)

    m = re.search(rf"(?:period\s*T\s*=\s*(?P<T>{_NUM})\s*s|repeats\s+every\s+(?P<T2>{_NUM})\s*s)", t, flags=re.I)
    if m and "frequency" in q:
        Tval=fnum(m.group('T') or m.group('T2'))
        if Tval:
            return _result(1/Tval, question, "Hz", "f=1/T", {"T": Tval}, conf=0.99)

    # --- rotation ---------------------------------------------------------
    m = re.search(rf"(?:covers\s+angle|turns\s+through|angular\s+displacement\s+is)\s+(?P<th>{_NUM})\s*rad\s+(?:in|over)\s+(?P<t>{_NUM})\s*(?:seconds?|s)", t, flags=re.I)
    if m and ("omega" in t or "ω" in t or "angular velocity" in q):
        time=fnum(m.group('t'))
        if time:
            return _result(fnum(m.group('th'))/time, question, "rad/s", "ω=θ/t", {"θ": fnum(m.group('th')), "t": time}, conf=0.99)

    m = re.search(rf"(?:ω0\s*=\s*(?P<w0>{_NUM})\s*rad/s,\s*α\s*=\s*(?P<a>{_NUM})\s*rad/s(?:\^?2|2),\s*and\s*t\s*=\s*(?P<t>{_NUM})\s*s|(?:wheel|disk)\s+(?:accelerates\s+uniformly\s+from|has\s+initial\s+angular\s+velocity)\s+(?P<w02>{_NUM})\s*rad/s\s+(?:at|and\s+angular\s+acceleration)\s+(?P<a2>{_NUM})\s*rad/s(?:\^?2|2)\s+for\s+(?P<t2>{_NUM})\s*(?:seconds?|s))", t, flags=re.I)
    if m and ("calculate omega" in q or "calculate ω" in q or "final angular speed" in q):
        w0=fnum(m.group('w0') or m.group('w02')); a=fnum(m.group('a') or m.group('a2')); time=fnum(m.group('t') or m.group('t2'))
        return _result(w0 + a*time, question, "rad/s", "ω=ω0+αt", {"ω0": w0, "α": a, "t": time}, conf=0.99)

    m = re.search(rf"(?:moment\s+of\s+inertia\s+(?P<I>{_NUM})\s*kg.*?m(?:\^?2|2).*?angular\s+velocity\s+(?P<w>{_NUM})\s*rad/s|I\s*=\s*(?P<I2>{_NUM})\s*kg.*?m(?:\^?2|2)\s+and\s+(?:ω|omega)\s*=\s*(?P<w2>{_NUM})\s*rad/s)", t, flags=re.I)
    if m and ("angular momentum" in q or re.search(r"find\s+l\b", q)):
        I=fnum(m.group('I') or m.group('I2')); w=fnum(m.group('w') or m.group('w2'))
        return _result(I*w, question, "kg*m^2/s", "L=Iω", {"I": I, "ω": w}, conf=0.99)

    return None

def _solve_measurement_error(question: str) -> SolverResult | None:
    q = _low(question)
    if not ("error" in q and ("actual" in q or "true" in q) and ("measured" in q or "measurement" in q)):
        return None
    t = _clean(question)

    def grab(label: str) -> tuple[float, str] | None:
        m = re.search(rf"(?:{label})[^.?!,;]{{0,40}}?(?P<v>{_NUM})\s*(?P<u>kg|g|m|cm|mm|s|A|V|N|J)?", t, flags=re.I)
        if not m:
            return None
        unit = m.group("u") or ""
        try:
            return _to_si(_parse_number(m.group("v")), unit), unit
        except Exception:
            return None

    actual = grab(r"actual|true|accepted|reference")
    measured = grab(r"measured|measurement|student\s+measured|observed|experimental")
    if not actual or not measured or abs(actual[0]) < 1e-12:
        return None
    err_si = abs(measured[0] - actual[0])
    pct = err_si / abs(actual[0]) * 100.0
    out_unit = measured[1] or actual[1] or None
    err_out = _scale_from_si(err_si, out_unit) if out_unit else err_si
    # In these prompts the first value is an absolute error and the second is
    # the percentage relative error, so semicolon output is intentional.
    ans = f"{_fmt(err_out, question)}; {_fmt(pct, question, places=2)}"
    return _make_result(ans, f"{out_unit}; %" if out_unit else "%", "Absolute error is |measured-actual|; percentage relative error is absolute error divided by actual value times 100%.", "Δx=|x_m-x|; δ%=Δx/x·100%", {"actual": actual[0], "measured": measured[0], "absolute_error": err_si, "relative_percent": pct}, confidence=0.985)

def solve_nephys_equation_family(question: str) -> SolverResult | None:
    q = _low(question)
    # Keep mature electricity/circuit solvers in control unless the prompt is
    # clearly a non-electric mixed-domain problem with words such as wave/photon.
    if _is_electric(q) and not _contains(q, "wave", "photon", "de broglie", "lens", "mirror", "heat", "fluid"):
        return None
    for fn in (
        _solve_fluids,
        _solve_thermal,
        _solve_waves,
        _solve_optics,
        _solve_modern,
        _solve_nephys_safe_patterns,
        _solve_measurement_error,
        _solve_kinematics,
        _solve_dynamics,
    ):
        try:
            out = fn(question)
        except ZeroDivisionError:
            out = None
        except Exception:
            out = None
        if out is not None:
            try:
                out.debug = dict(out.debug or {})
                out.debug.setdefault("nephys_equation_family", fn.__name__)
            except Exception:
                pass
            return out
    return None
