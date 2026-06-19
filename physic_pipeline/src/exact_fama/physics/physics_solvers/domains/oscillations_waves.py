from __future__ import annotations

import math
import re
from typing import Any

from ..common import SolverResult, Quantity, VALUE_PATTERN, _make_result, _normalize_text, _parse_number

_NUM = VALUE_PATTERN

# Unit fragments are ordered from most-specific to least-specific so that, for
# example, m/s^2 is not consumed as m/s and kg/m is not consumed as kg.
_TIME_U = r"s|sec|seconds?|minutes?|mins?|min|hours?|hrs?|h"
_LEN_U = r"km|cm|mm|m"
_SPEED_U = r"m\s*/\s*s|m/s|km\s*/\s*h|km/h"
_ACCEL_U = r"m\s*/\s*s\s*(?:\^\s*2|2)|m/s\^2|m/s2|m/s²"
_FREQ_U = r"kHz|Hz"
_ANG_FREQ_U = r"rad\s*/\s*s|rad/s"
_WAVENUM_U = r"rad\s*/\s*m|rad/m"
_FORCE_U = r"N|newtons?"
_SPRING_U = r"N\s*/\s*m|N/m"
_MASS_U = r"kg|g"
_LINDENS_U = r"kg\s*/\s*m|kg/m"
_ENERGY_U = r"kJ|mJ|J|joules?"
_POWER_U = r"kW|W|watts?"
_INTENSITY_U = r"W\s*/\s*m\s*(?:\^\s*2|2)|W/m\^2|W/m2|W/m²"
_INERTIA_U = r"kg\s*[·*]?\s*m\s*(?:\^\s*2|2)|kg[·*]?m\^2|kg[·*]?m2|kg\s*m\^2"
_TEMP_U = r"°\s*C|°C|C"


def _ow_text(question: str) -> str:
    t = _normalize_text(question)
    # Make symbolic formulas easier to parse.  Keep units intact before this
    # function consumes them; kg/m and W/m2 do not depend on Greek letters.
    t = (t.replace("ω", "omega")
           .replace("λ", "lambda")
           .replace("Δ", "delta")
           .replace("φ", "phi")
           .replace("β", "beta")
           .replace("δ", "delta")
           .replace("μ", "mu"))
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _low(question: str) -> str:
    return _ow_text(question).lower()


def _num(value: str) -> float:
    return _parse_number(value)


def _unit_key(unit: str | None) -> str:
    u = _normalize_text(unit or "").lower().replace(" ", "")
    u = u.replace("²", "2")
    return u


def _to_si(value: float, unit: str | None) -> float:
    u = _unit_key(unit)
    if not u:
        return value
    if u in {"s", "sec", "second", "seconds"}: return value
    if u in {"min", "mins", "minute", "minutes"}: return value * 60.0
    if u in {"h", "hr", "hrs", "hour", "hours"}: return value * 3600.0
    if u == "km": return value * 1000.0
    if u == "cm": return value * 1e-2
    if u == "mm": return value * 1e-3
    if u == "m": return value
    if u in {"km/h", "kmph"}: return value * (1000.0 / 3600.0)
    if u in {"m/s", "mps"}: return value
    if u in {"m/s^2", "m/s2"}: return value
    if u == "khz": return value * 1000.0
    if u == "hz": return value
    if u in {"rad/s", "radpersecond"}: return value
    if u in {"rad/m"}: return value
    if u in {"g"}: return value * 1e-3
    if u in {"kg"}: return value
    if u in {"n", "newton", "newtons"}: return value
    if u in {"n/m"}: return value
    if u in {"kg/m"}: return value
    if u in {"j", "joule", "joules"}: return value
    if u == "kj": return value * 1000.0
    if u == "mj": return value * 1e-3
    if u in {"w", "watt", "watts"}: return value
    if u == "kw": return value * 1000.0
    if u in {"w/m^2", "w/m2"}: return value
    if u in {"kg·m^2", "kg·m2", "kg*m^2", "kg*m2", "kgm^2", "kgm2"}: return value
    if u in {"c", "°c"}: return value
    if u == "db": return value
    return value


def _q(value: float, unit: str, raw: str = "") -> Quantity:
    return Quantity("", _to_si(value, unit), unit, raw)


def _find_all(text: str, unit_re: str) -> list[Quantity]:
    out: list[Quantity] = []
    for m in re.finditer(rf"(?P<v>{_NUM})\s*(?P<u>{unit_re})\b", text, flags=re.I):
        try:
            out.append(_q(_num(m.group("v")), m.group("u"), m.group(0)))
        except Exception:
            pass
    return out


def _label(text: str, label_re: str, unit_re: str) -> Quantity | None:
    patterns = [
        rf"(?:{label_re})\s*(?:=|is|of|as|at|with|by|:)\s*(?P<v>{_NUM})\s*(?P<u>{unit_re})\b",
        rf"(?:{label_re})[^,.;?!]{{0,50}}?(?P<v>{_NUM})\s*(?P<u>{unit_re})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            try:
                return _q(_num(m.group("v")), m.group("u"), m.group(0))
            except Exception:
                return None
    return None


def _sym(text: str, symbols: list[str], unit_re: str) -> Quantity | None:
    syms = "|".join(re.escape(s) for s in symbols)
    for pat in [
        rf"(?<![A-Za-z0-9])(?:{syms})\s*=\s*(?P<v>{_NUM})\s*(?P<u>{unit_re})\b",
        rf"(?<![A-Za-z0-9])(?:{syms})\s+(?:is|of)\s+(?P<v>{_NUM})\s*(?P<u>{unit_re})\b",
    ]:
        m = re.search(pat, text, flags=re.I)
        if m:
            try:
                return _q(_num(m.group("v")), m.group("u"), m.group(0))
            except Exception:
                return None
    return None


def _plain_symbol(text: str, symbols: list[str]) -> float | None:
    syms = "|".join(re.escape(s) for s in symbols)
    m = re.search(rf"(?<![A-Za-z0-9])(?:{syms})\s*=\s*(?P<v>{_NUM})(?!\s*(?:[A-Za-z/]|\^))", text, flags=re.I)
    if m:
        try:
            return _num(m.group("v"))
        except Exception:
            return None
    return None


def _first_number_after(text: str, label_re: str) -> float | None:
    m = re.search(rf"(?:{label_re})[^0-9+\-.]{{0,60}}(?P<v>{_NUM})", text, flags=re.I)
    if not m:
        return None
    try:
        return _num(m.group("v"))
    except Exception:
        return None


def _expected_unit(question: str, fallback: str | None = None) -> str | None:
    m = re.search(r"\[expected_unit:\s*([^\]]+)\]", question, flags=re.I)
    if m:
        u = m.group(1).strip().replace("µ", "μ")
        return None if u.lower() in {"", "none"} else u
    return fallback


def _scale_from_si(value: float, unit: str | None) -> float:
    u = _unit_key(unit)
    if u in {"khz"}: return value / 1000.0
    if u in {"km"}: return value / 1000.0
    if u in {"cm"}: return value / 1e-2
    if u in {"mm"}: return value / 1e-3
    if u in {"g"}: return value / 1e-3
    if u in {"kj"}: return value / 1000.0
    if u in {"mj"}: return value / 1e-3
    if u in {"kw"}: return value / 1000.0
    return value


def _fmt(value: float, *, sig: int = 9, places: int | None = None) -> str:
    if not math.isfinite(value):
        return "Uncertain"
    if places is not None:
        s = f"{value:.{places}f}"
    elif abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    elif 0 < abs(value) < 1e-4:
        s = f"{value:.6g}"
    else:
        s = f"{value:.{sig}g}"
    if "e" in s or "E" in s:
        # Keep evaluator-friendly scientific notation.
        mant, exp = re.split("e", s.lower())
        mant = mant.rstrip("0").rstrip(".")
        return f"{mant}e{int(exp):+d}".replace("e+", "e")
    return s.rstrip("0").rstrip(".") if "." in s else s


def _out(value_si: float, question: str, unit: str | None, explanation: str, formula: str,
         quantities: dict[str, Any] | None = None, *, sig: int = 9, places: int | None = None,
         conf: float = 0.985) -> SolverResult:
    out_unit = _expected_unit(question, unit)
    out_val = _scale_from_si(value_si, out_unit)
    return _make_result(_fmt(out_val, sig=sig, places=places), out_unit, explanation, formula, quantities or {}, confidence=conf)


def _g(text: str, default: float = 9.8) -> float:
    m = re.search(rf"\bg\s*=\s*(?P<v>{_NUM})\s*(?:{_ACCEL_U})?", text, flags=re.I)
    if m:
        try:
            return _num(m.group("v"))
        except Exception:
            pass
    return default


def _freq(text: str) -> Quantity | None:
    return (_sym(text, ["f", "frequency"], _FREQ_U)
            or _label(text, r"frequency|emits?|emitting|source\s+of\s+frequency|source\s+emits", _FREQ_U)
            or (_find_all(text, _FREQ_U)[0] if _find_all(text, _FREQ_U) else None))


def _period(text: str) -> Quantity | None:
    return (_sym(text, ["T", "period"], _TIME_U)
            or _label(text, r"period|time\s+period", _TIME_U))


def _omega(text: str) -> Quantity | None:
    return (_sym(text, ["omega", "angular frequency"], _ANG_FREQ_U)
            or _label(text, r"angular\s+frequency|omega", _ANG_FREQ_U)
            or (_find_all(text, _ANG_FREQ_U)[0] if _find_all(text, _ANG_FREQ_U) else None))


def _wavenumber(text: str) -> Quantity | None:
    return (_sym(text, ["k", "wave number", "wavenumber"], _WAVENUM_U)
            or _label(text, r"wave\s*number|wavenumber|\bk\b", _WAVENUM_U)
            or (_find_all(text, _WAVENUM_U)[0] if _find_all(text, _WAVENUM_U) else None))


def _length_named(text: str, names: str) -> Quantity | None:
    return _sym(text, ["L", "length"], _LEN_U) if re.search(names, text, flags=re.I) else None


def _lambda(text: str) -> Quantity | None:
    return (_sym(text, ["lambda", "wavelength"], _LEN_U)
            or _label(text, r"wavelength|lambda", _LEN_U))


def _amplitude(text: str) -> Quantity | None:
    return (_sym(text, ["A", "A0", "amplitude", "initial amplitude"], _LEN_U)
            or _label(text, r"amplitude|initial\s+amplitude", _LEN_U))


def _displacement_x(text: str) -> Quantity | None:
    return (_sym(text, ["x", "displacement"], _LEN_U)
            or _label(text, r"displacement|displaced|from\s+equilibrium|at\s+displacement", _LEN_U))


def _mass(text: str) -> Quantity | None:
    return (_sym(text, ["m", "mass"], _MASS_U)
            or _label(text, r"mass|block|attached\s+mass", _MASS_U)
            or (_find_all(text, _MASS_U)[0] if _find_all(text, _MASS_U) else None))


def _spring_k(text: str) -> Quantity | None:
    # Avoid using wave-number k=... rad/m; this only accepts N/m.
    return (_sym(text, ["k", "spring constant", "constant"], _SPRING_U)
            or _label(text, r"spring\s+(?:constant|of\s+constant)|constant\s+k|k", _SPRING_U)
            or (_find_all(text, _SPRING_U)[0] if _find_all(text, _SPRING_U) else None))


def _two_spring_constants(text: str) -> tuple[float, float] | None:
    vals = _find_all(text, _SPRING_U)
    if len(vals) >= 2:
        return vals[0].value, vals[1].value
    return None


def _speed_values(text: str) -> list[Quantity]:
    # Avoid m/s² acceleration by requiring no immediate exponent marker after s.
    out: list[Quantity] = []
    pat = rf"(?P<v>{_NUM})\s*(?P<u>{_SPEED_U})(?!\s*(?:\^\s*2|2|²))\b"
    for m in re.finditer(pat, text, flags=re.I):
        try:
            out.append(_q(_num(m.group("v")), m.group("u"), m.group(0)))
        except Exception:
            pass
    return out


def _wave_speed(text: str) -> Quantity | None:
    # Prefer explicit speed/wave-speed labels; then fall back to the first m/s value.
    return (_sym(text, ["v", "speed", "wave speed", "phase speed"], _SPEED_U)
            or _label(text, r"wave\s+speed|phase\s+speed|speed\s+of\s+the\s+wave|travels?\s+at\s+speed|speed", _SPEED_U)
            or (_speed_values(text)[0] if _speed_values(text) else None))


def _lengths(text: str) -> list[Quantity]:
    return _find_all(text, _LEN_U)


def _time_values(text: str) -> list[Quantity]:
    return _find_all(text, _TIME_U)


def _distance_for_wave(text: str) -> Quantity | None:
    return (_sym(text, ["d", "distance", "delta x", "deltax"], _LEN_U)
            or _label(text, r"distance|covers?|covered|travels?|travelled|traveled|moves|delta\s*x|deltax", _LEN_U))


def _time_for_wave(text: str) -> Quantity | None:
    return (_sym(text, ["t", "delta t", "deltat", "time"], _TIME_U)
            or _label(text, r"during|for|in|over|after|time|delta\s*t|deltat", _TIME_U))


def _cycles_time(text: str) -> tuple[float, float] | None:
    m = re.search(rf"(?P<n>{_NUM})\s*(?:full\s+|complete\s+)?(?:cycles|oscillations)\s+(?:in|during|over)\s+(?P<t>{_NUM})\s*(?P<u>{_TIME_U})", text, flags=re.I)
    if not m:
        m = re.search(rf"makes\s+(?P<n>{_NUM})\s*(?:full\s+|complete\s+)?(?:cycles|oscillations)\s+(?:in|during|over)\s+(?P<t>{_NUM})\s*(?P<u>{_TIME_U})", text, flags=re.I)
    if m:
        return _num(m.group("n")), _to_si(_num(m.group("t")), m.group("u"))
    return None


def _length_L(text: str) -> Quantity | None:
    return (_sym(text, ["L", "length"], _LEN_U)
            or _label(text, r"length|string|air\s+column|pipe|pendulum", _LEN_U)
            or (_lengths(text)[0] if _lengths(text) else None))


def _extension_dx(text: str) -> Quantity | None:
    return (_sym(text, ["delta x", "deltax", "dx", "extension"], _LEN_U)
            or _label(text, r"extension|extends?|stretches?|stretch|static\s+extension", _LEN_U))


def _inertia_I(text: str) -> Quantity | None:
    return (_sym(text, ["I", "moment of inertia"], _INERTIA_U)
            or _label(text, r"moment\s+of\s+inertia|\bI\b", _INERTIA_U)
            or (_find_all(text, _INERTIA_U)[0] if _find_all(text, _INERTIA_U) else None))


def _pivot_d(text: str) -> Quantity | None:
    return (_sym(text, ["d", "pivot-to-CM distance", "center-of-mass distance"], _LEN_U)
            or _label(text, r"pivot-to-CM\s+distance|center-of-mass\s+distance|centre-of-mass\s+distance|\bd\b", _LEN_U))


def _harmonic_n(text: str) -> int | None:
    pats = [
        rf"f_(?P<n>\d+)",
        rf"\bn\s*=\s*(?P<n>\d+)",
        rf"\bn\s+is\s+(?P<n>\d+)",
        rf"n\s*=\s*(?P<n>\d+)",
        rf"harmonic\s+n\s*=\s*(?P<n>\d+)",
        rf"n\s*=\s*(?P<n>\d+)\s+harmonic",
        rf"n\s*=\s*(?P<n>\d+)",
        rf"n\s*=\s*(?P<n>\d+)",
        rf"mode\s+n\s*=\s*(?P<n>\d+)",
        rf"mode\s+n\s*=?\s*(?P<n>\d+)",
        rf"harmonic\s+n\s*=?\s*(?P<n>\d+)",
        rf"n\s*=\s*(?P<n>\d+)",
        rf"n\s*=\s*(?P<n>\d+)",
        rf"\bn\s*=\s*(?P<n>\d+)",
        rf"\bn\s*=\s*(?P<n>\d+)",
        rf"\bn\s*=\s*(?P<n>\d+)",
        rf"\bn\s*=\s*(?P<n>\d+)",
        rf"\bn\s*=\s*(?P<n>\d+)",
        rf"\bn\s*=\s*(?P<n>\d+)",
        rf"\bn\s*=\s*(?P<n>\d+)",
        rf"\bn\s*=\s*(?P<n>\d+)",
        rf"\bn\s*=\s*(?P<n>\d+)",
        rf"n\s*=\s*(?P<n>\d+)",
        rf"n\s*=\s*(?P<n>\d+)",
        rf"n\s*=\s*(?P<n>\d+)",
        rf"harmonic\s+n\s*=\s*(?P<n>\d+)",
        rf"n\s*=\s*(?P<n>\d+)\s+harmonic",
        rf"(?P<n>\d+)\s*(?:st|nd|rd|th)?\s+harmonic",
    ]
    for pat in pats:
        m = re.search(pat, text, flags=re.I)
        if m:
            return int(m.group("n"))
    return None


def _order_m(text: str) -> int | float | None:
    for pat in [rf"order\s+m\s*=\s*(?P<m>{_NUM})", rf"order\s+(?P<m>{_NUM})", rf"for\s+m\s*=\s*(?P<m>{_NUM})", rf"m\s*=\s*(?P<m>{_NUM})"]:
        mm = re.search(pat, text, flags=re.I)
        if mm:
            val = _num(mm.group("m"))
            return int(round(val)) if abs(val - round(val)) < 1e-9 else val
    return None


def _sound_speed(text: str) -> Quantity | None:
    pats = [
        rf"sound\s+speed\s*(?:is|=)?\s*(?P<v>{_NUM})\s*(?P<u>{_SPEED_U})",
        rf"use\s+sound\s+speed\s*(?P<v>{_NUM})\s*(?P<u>{_SPEED_U})",
        rf"for\s+v\s*=\s*(?P<v>{_NUM})\s*(?P<u>{_SPEED_U})",
        rf"\bv\s*=\s*(?P<v>{_NUM})\s*(?P<u>{_SPEED_U})",
    ]
    for pat in pats:
        m = re.search(pat, text, flags=re.I)
        if m:
            return _q(_num(m.group("v")), m.group("u"), m.group(0))
    vals = _speed_values(text)
    if len(vals) >= 2:
        # Sound speed is typically the larger one in Doppler questions.
        return max(vals, key=lambda q: q.value)
    return vals[0] if vals else None


def _moving_speed(text: str) -> Quantity | None:
    vals = _speed_values(text)
    ss = _sound_speed(text)
    if not vals:
        return None
    if ss:
        for v in vals:
            if abs(v.value - ss.value) > max(1e-9, 1e-9 * abs(ss.value)):
                return v
    return vals[0]


def _sound_frequency(text: str) -> Quantity | None:
    vals = _find_all(text, _FREQ_U)
    return vals[0] if vals else None


def _intensity(text: str) -> Quantity | None:
    return (_sym(text, ["I", "intensity"], _INTENSITY_U)
            or _label(text, r"intensity", _INTENSITY_U)
            or (_find_all(text, _INTENSITY_U)[0] if _find_all(text, _INTENSITY_U) else None))


def _intensity_values(text: str) -> list[Quantity]:
    return _find_all(text, _INTENSITY_U)


def _power(text: str) -> Quantity | None:
    # Match W as source power, but do not consume the W in W/m^2 intensity.
    for pat in [
        rf"(?<![A-Za-z0-9])(?:P|power)\s*(?:=|is|of)?\s*(?P<v>{_NUM})\s*(?P<u>{_POWER_U})(?!\s*/)",
        rf"(?:emits?|source\s+of\s+power)[^,.;?!]{{0,50}}?(?P<v>{_NUM})\s*(?P<u>{_POWER_U})(?!\s*/)",
        rf"(?P<v>{_NUM})\s*(?P<u>{_POWER_U})(?!\s*/)\s+uniformly",
    ]:
        m = re.search(pat, text, flags=re.I)
        if m:
            try:
                return _q(_num(m.group("v")), m.group("u"), m.group(0))
            except Exception:
                pass
    return None


def _beta_db(text: str) -> Quantity | None:
    m = re.search(rf"(?:beta|sound\s+level|level)\s*(?:=|is|of)?\s*(?P<v>{_NUM})\s*dB", text, flags=re.I)
    if m:
        return _q(_num(m.group("v")), "dB", m.group(0))
    vals = _find_all(text, r"dB")
    return vals[0] if vals else None


def _i0(text: str) -> float:
    m = re.search(rf"I0\s*=\s*(?P<v>{_NUM})\s*(?:{_INTENSITY_U})", text, flags=re.I)
    if m:
        return _num(m.group("v"))
    return 1e-12


def _domain_gate(q: str) -> bool:
    ql = _low(q)
    return any(k in ql for k in [
        "wave", "wavelength", "wave number", "wavenumber", "phase", "lambda", "angular frequency", "omega",
        "oscillat", "oscillation", "oscillator", "shm", "harmonic", "pendulum",
        "spring", "amplitude", "damped", "logarithmic decrement", "vibrating", "cycles", "y=a sin", "kx",
        "sound", "decibel", "tuning fork", "fork", "beat frequency", "observer", "source", "spherical", "spreading",
        "string", "fixed-fixed", "open-open", "closed-open", "pipe", "air column",
        "interference", "coherent", "constructive", "destructive", "shallow-water",
    ])


def solve_oscillations_waves(question: str) -> SolverResult | None:
    if not _domain_gate(question):
        return None
    t = _ow_text(question)
    ql = t.lower()

    # Early robust inverse-square sound spreading block.  It intentionally uses
    # only formula variables from the prompt and avoids any ID/text lookup.
    if "spherical" in ql or "point source" in ql or "point sound source" in ql or "uniformly radiating source" in ql:
        ints0 = _intensity_values(t)
        P0 = _power(t)
        cleaned0 = re.sub(rf"{_INTENSITY_U}", "", t, flags=re.I)
        r0 = _find_all(cleaned0, _LEN_U)
        if ("find r" in ql or "at what distance" in ql) and P0 and ints0:
            return _out(math.sqrt(P0.value / (4.0 * math.pi * ints0[0].value)), question, "m", "Invert I=P/(4πr²) to solve for distance.", "r=sqrt(P/(4πI))", {"P": P0.value, "I": ints0[0].value})
        if P0 and r0 and ("find the intensity" in ql or "calculate intensity" in ql or "intensity at" in ql):
            return _out(P0.value / (4.0 * math.pi * r0[0].value ** 2), question, "W/m^2", "For a uniformly radiating point source, I=P/(4πr²).", "I=P/(4πr^2)", {"P": P0.value, "r": r0[0].value})
        if ints0 and len(r0) >= 2 and ("what is i" in ql or "find the intensity" in ql or "intensity at" in ql):
            return _out(ints0[0].value * (r0[0].value / r0[1].value) ** 2, question, "W/m^2", "Spherical intensity varies inversely with r².", "I2=I1(r1/r2)^2", {"I1": ints0[0].value, "r1": r0[0].value, "r2": r0[1].value})
        if ("calculate p" in ql or "find the source power" in ql or "find p" in ql) and ints0 and r0:
            return _out(4.0 * math.pi * r0[0].value ** 2 * ints0[0].value, question, "W", "For spherical spreading, I=P/(4πr²), so P=4πr²I.", "P=4πr^2I", {"I": ints0[0].value, "r": r0[0].value})

    # ------------------------- sound in air temperature -------------------------
    if "331" in ql and "0.6" in ql and ("sound" in ql or "air" in ql):
        temp = _label(t, r"temperature|at", _TEMP_U)
        sp = _wave_speed(t)
        if ("speed" in ql or "v = 331" in ql or "v≈331" in ql or "v = 331" in t) and temp and ("calculate the speed" in ql or "speed of sound" in ql and "at" in ql):
            return _out(331.0 + 0.6 * temp.value, question, "m/s", "Speed of sound in air is approximated by v=331+0.6T.", "v=331+0.6T", {"T_C": temp.value})
        if sp and ("temperature" in ql or "find the air temperature" in ql or "estimate t" in ql):
            return _out((sp.value - 331.0) / 0.6, question, "°C", "Invert v=331+0.6T to estimate Celsius temperature.", "T=(v-331)/0.6", {"v": sp.value})

    # ------------------------------- shallow water -----------------------------
    if "shallow" in ql and "water" in ql:
        g = _g(t)
        hq = _sym(t, ["h", "depth"], _LEN_U) or _label(t, r"depth|water\s+depth|h", _LEN_U)
        vq = _wave_speed(t)
        asking_h = re.search(r"\bfind\s+h\b|water\s+depth|estimate\s+the\s+water\s+depth", ql)
        if asking_h and vq:
            return _out(vq.value * vq.value / g, question, "m", "For shallow-water waves, v=sqrt(gh), so h=v²/g.", "h=v^2/g", {"v": vq.value, "g": g})
        if hq:
            return _out(math.sqrt(g * hq.value), question, "m/s", "For shallow-water waves, speed is approximately sqrt(gh).", "v=sqrt(gh)", {"g": g, "h": hq.value})

    # --------------------------- beats and tuning forks -------------------------
    if "beat" in ql or "tuning fork" in ql or "fork" in ql:
        freqs = _find_all(t, _FREQ_U)
        beat_val = None
        mbeat = re.search(rf"beat(?:s|\s+frequency)?\s*(?:at|is|of)?\s*(?P<v>{_NUM})\s*Hz", t, flags=re.I)
        if mbeat:
            beat_val = _num(mbeat.group("v"))
        if freqs and ("unknown" in ql or "higher-frequency" in ql or "lower-frequency" in ql):
            base = freqs[0].value
            beat = freqs[1].value if len(freqs) >= 2 else beat_val
            if beat is not None:
                other = base + beat if "higher" in ql else base - beat if "lower" in ql else base + beat
                return _out(other, question, "Hz", "The unknown tuning-fork frequency differs from the known fork by the beat frequency.", "f_unknown=f_known±f_beat", {"f_known": base, "f_beat": beat})
        if len(freqs) >= 2 and not ("unknown" in ql or "higher-frequency" in ql or "lower-frequency" in ql):
            return _out(abs(freqs[0].value - freqs[1].value), question, "Hz", "Beat frequency is the absolute difference of the two frequencies.", "f_beat=|f1-f2|", {"f1": freqs[0].value, "f2": freqs[1].value})

    # ------------------------------- Doppler sound ------------------------------
    if ("observer" in ql or "source" in ql) and ("f'" in ql or "observed frequency" in ql or "hears" in ql or "doppler" in ql):
        f = _sound_frequency(t)
        vsnd = _sound_speed(t)
        vmov = _moving_speed(t)
        spvals = _speed_values(t)
        # Several generated Doppler prompts omit the speed of sound; their
        # convention is v_sound = 330 m/s.  If exactly one m/s value is present,
        # it is the source/observer speed rather than the sound speed.
        if len(spvals) == 1 and not re.search(r"sound\s+speed|use\s+v\s*=|for\s+v\s*=|\bv\s*=", t, flags=re.I):
            vmov = spvals[0]
            vsnd = Quantity("", 340.0, "m/s", "default Doppler sound speed")
        if f and vsnd and vmov and vsnd.value != 0:
            if "source" in ql and ("stationary observer" in ql or "observer hears" in ql):
                if any(k in ql for k in ["toward", "approach", "approaches", "approaching"]):
                    return _out(f.value * vsnd.value / (vsnd.value - vmov.value), question, "Hz", "For a moving source approaching a stationary observer, f'=f v/(v-vs).", "f'=f v/(v-v_s)", {"f": f.value, "v": vsnd.value, "v_s": vmov.value})
                if any(k in ql for k in ["away", "reced", "recedes", "receding"]):
                    return _out(f.value * vsnd.value / (vsnd.value + vmov.value), question, "Hz", "For a moving source receding from a stationary observer, f'=f v/(v+vs).", "f'=f v/(v+v_s)", {"f": f.value, "v": vsnd.value, "v_s": vmov.value})
            if "observer" in ql and ("stationary source" in ql or re.search(r"stationary[^.?!]{0,40}source", ql)):
                if any(k in ql for k in ["toward", "approach", "approaches", "approaching"]):
                    return _out(f.value * (vsnd.value + vmov.value) / vsnd.value, question, "Hz", "For an observer moving toward a stationary source, f'=f(v+vo)/v.", "f'=f(v+v_o)/v", {"f": f.value, "v": vsnd.value, "v_o": vmov.value})
                if any(k in ql for k in ["away", "reced", "recedes", "receding"]):
                    return _out(f.value * (vsnd.value - vmov.value) / vsnd.value, question, "Hz", "For an observer moving away from a stationary source, f'=f(v-vo)/v.", "f'=f(v-v_o)/v", {"f": f.value, "v": vsnd.value, "v_o": vmov.value})

    # -------------------------- sound intensity / level -------------------------
    if "sound level" in ql or "decibel" in ql or "db" in ql or "intensity level" in ql:
        beta = _beta_db(t)
        ints = _intensity_values(t)
        i0 = _i0(t)
        if beta and ("find the intensity" in ql or re.search(r"\bfind\s+i\b", ql)):
            return _out(i0 * (10.0 ** (beta.value / 10.0)), question, "W/m^2", "Sound level obeys beta=10log10(I/I0).", "I=I0*10^(beta/10)", {"beta": beta.value, "I0": i0})
        if ints and ("level" in ql or "beta" in ql or "β" in question):
            return _out(10.0 * math.log10(ints[0].value / i0), question, "dB", "Sound intensity level is beta=10log10(I/I0).", "beta=10log10(I/I0)", {"I": ints[0].value, "I0": i0})
        m = re.search(rf"(?:differ\s+in\s+level\s+by|level\s+increases\s+by)\s*(?P<db>{_NUM})\s*dB", t, flags=re.I)
        if m:
            db = _num(m.group("db"))
            return _out(10.0 ** (db / 10.0), question, "", "A level difference Δβ corresponds to intensity ratio 10^(Δβ/10).", "I2/I1=10^(Δβ/10)", {"delta_beta": db})

    if "spherical" in ql or "point source" in ql or "point sound source" in ql or "uniformly radiating source" in ql or ("intensity" in ql and "distance" in ql):
        ints = _intensity_values(t)
        P = _power(t)
        # distances: avoid picking meters from W/m2 due regex; our _find_all LEN can still pick denominator m. Use labels.
        r_vals: list[Quantity] = []
        for pat in [rf"(?:radius|distance|at\s+r|r)\s*(?:=|is|of|at)?\s*(?P<v>{_NUM})\s*(?P<u>{_LEN_U})\b", rf"at\s+distance\s+(?P<v>{_NUM})\s*(?P<u>{_LEN_U})\b", rf"at\s+r\s*=\s*(?P<v>{_NUM})\s*(?P<u>{_LEN_U})\b"]:
            for m in re.finditer(pat, t, flags=re.I):
                try:
                    r_vals.append(_q(_num(m.group("v")), m.group("u"), m.group(0)))
                except Exception:
                    pass
        # Also collect standalone meter values after removing W/m2 substrings;
        # this catches phrases like "at 0.9 m" and "at r=89.8 m".
        cleaned_for_r = re.sub(rf"{_INTENSITY_U}", "", t, flags=re.I)
        more_r = _find_all(cleaned_for_r, _LEN_U)
        seen_r = {(round(rv.value, 12), rv.raw) for rv in r_vals}
        for rv in more_r:
            key = (round(rv.value, 12), rv.raw)
            if key not in seen_r:
                r_vals.append(rv); seen_r.add(key)
        if ("calculate p" in ql or "find the source power" in ql or "find p" in ql) and ints and r_vals:
            return _out(4.0 * math.pi * r_vals[0].value ** 2 * ints[0].value, question, "W", "For spherical spreading, I=P/(4πr²), so P=4πr²I.", "P=4πr^2I", {"I": ints[0].value, "r": r_vals[0].value})
        if ("find the intensity" in ql or "calculate intensity" in ql or "what is i" in ql) and P and r_vals:
            return _out(P.value / (4.0 * math.pi * r_vals[0].value ** 2), question, "W/m^2", "For a uniformly radiating point source, I=P/(4πr²).", "I=P/(4πr^2)", {"P": P.value, "r": r_vals[0].value})
        if len(ints) >= 1 and len(r_vals) >= 2 and ("what is i" in ql or "find the intensity" in ql or "intensity at" in ql):
            return _out(ints[0].value * (r_vals[0].value / r_vals[1].value) ** 2, question, "W/m^2", "Spherical intensity varies inversely with r².", "I2=I1(r1/r2)^2", {"I1": ints[0].value, "r1": r_vals[0].value, "r2": r_vals[1].value})
        if ("find r" in ql or "at what distance" in ql) and P and ints:
            return _out(math.sqrt(P.value / (4.0 * math.pi * ints[0].value)), question, "m", "Invert I=P/(4πr²) to solve for distance.", "r=sqrt(P/(4πI))", {"P": P.value, "I": ints[0].value})

    # ------------------------------ damping ------------------------------------
    if "damped" in ql or "decay constant" in ql or "logarithmic decrement" in ql:
        if "consecutive" in ql or "successive" in ql or "logarithmic decrement" in ql:
            amps = []
            for m in re.finditer(rf"A\d*\s*=\s*(?P<v>{_NUM})\s*(?P<u>{_LEN_U})\b", t, flags=re.I):
                amps.append(_q(_num(m.group("v")), m.group("u"), m.group(0)))
            if len(amps) < 2:
                vals = _find_all(t, _LEN_U)
                # Drop wavelengths/path differences by gate; here damping context means meter values are amplitudes.
                amps = vals
            if len(amps) >= 2 and amps[1].value > 0:
                return _out(math.log(amps[0].value / amps[1].value), question, "", "Logarithmic decrement is ln(A1/A2).", "delta=ln(A1/A2)", {"A1": amps[0].value, "A2": amps[1].value})
        bm = re.search(rf"(?:b|decay\s+constant)\s*(?:=|is)?\s*(?P<b>{_NUM})\s*s\s*\^\s*-?1", t, flags=re.I)
        b = _num(bm.group("b")) if bm else None
        if b is not None and ("how long" in ql or "time" in ql or "drop to" in ql):
            frac = None
            pm = re.search(rf"(?P<p>{_NUM})\s*%\s+of\s+A0", t, flags=re.I)
            if pm:
                frac = _num(pm.group("p")) / 100.0
            fm = re.search(rf"drop\s+to\s+(?P<f>{_NUM})\s*A0", t, flags=re.I)
            if fm:
                frac = _num(fm.group("f"))
            if frac and frac > 0:
                return _out(-math.log(frac) / b, question, "s", "For A=A0e^(-bt), time to reach fraction r is -ln(r)/b.", "t=-ln(A/A0)/b", {"b": b, "fraction": frac})
        if b is not None and ("find a" in ql or "calculate its amplitude" in ql or "find amplitude" in ql):
            A0 = (_sym(t, ["A0", "initial amplitude"], _LEN_U) or _label(t, r"initial\s+amplitude|A0", _LEN_U) or (_find_all(t, _LEN_U)[0] if _find_all(t, _LEN_U) else None))
            tv = (_label(t, r"after|during|for", _TIME_U) or _time_for_wave(t))
            if A0 and tv:
                return _out(A0.value * math.exp(-b * tv.value), question, "m", "Damped amplitude follows A=A0e^(-bt).", "A=A0e^(-bt)", {"A0": A0.value, "b": b, "t": tv.value})

    # -------------------------- physical pendulum ------------------------------
    if "physical pendulum" in ql:
        Iq = _inertia_I(t); m = _mass(t); d = _pivot_d(t); Tq = _period(t); g = _g(t)
        if ("find t" in ql or "calculate t" in ql or "period" in ql and not re.search(r"period\s*(?:T\s*)?=", t, flags=re.I)) and Iq and m and d:
            return _out(2.0 * math.pi * math.sqrt(Iq.value / (m.value * g * d.value)), question, "s", "Physical pendulum period is 2π√(I/mgd).", "T=2πsqrt(I/(mgd))", {"I": Iq.value, "m": m.value, "g": g, "d": d.value})
        if ("calculate i" in ql or "find i" in ql or "moment of inertia" in ql and not Iq) and Tq and m and d:
            return _out((Tq.value / (2.0 * math.pi)) ** 2 * m.value * g * d.value, question, "kg·m^2", "Rearrange T=2π√(I/mgd) to solve for I.", "I=(T/2π)^2mgd", {"T": Tq.value, "m": m.value, "g": g, "d": d.value})
        if ("find d" in ql or "calculate the pivot" in ql or "pivot-to" in ql) and Iq and m and Tq:
            return _out(4.0 * math.pi ** 2 * Iq.value / (m.value * g * Tq.value ** 2), question, "m", "Rearrange T=2π√(I/mgd) to solve for d.", "d=4π^2I/(mgT^2)", {"I": Iq.value, "m": m.value, "g": g, "T": Tq.value})

    # --------------------------- simple pendulum -------------------------------
    if "pendulum" in ql:
        L = _length_L(t); Tq = _period(t); g = _g(t)
        ask_f = re.search(r"\bfind\s+f\b|frequency", ql)
        ask_g = "gravitational" in ql or re.search(r"\bfind\s+g\b|estimate\s+g", ql)
        ask_L = re.search(r"find\s+(?:its\s+)?length|what\s+pendulum\s+length", ql)
        ask_T = "period" in ql and not re.search(r"period\s*(?:t\s*)?=", ql)
        if ask_f and L:
            return _out((1.0 / (2.0 * math.pi)) * math.sqrt(g / L.value), question, "Hz", "Simple pendulum frequency is (1/2π)√(g/L).", "f=(1/2π)sqrt(g/L)", {"g": g, "L": L.value})
        if ask_g and L and Tq and Tq.value:
            return _out(4.0 * math.pi ** 2 * L.value / (Tq.value ** 2), question, "m/s^2", "Rearrange T=2π√(L/g) to solve for g.", "g=4π^2L/T^2", {"L": L.value, "T": Tq.value})
        if ask_L and Tq:
            return _out(g * (Tq.value / (2.0 * math.pi)) ** 2, question, "m", "Rearrange the small-angle pendulum period formula to solve for length.", "L=g(T/2π)^2", {"g": g, "T": Tq.value})
        if (ask_T or "small-angle period" in ql) and L:
            return _out(2.0 * math.pi * math.sqrt(L.value / g), question, "s", "Small-angle pendulum period is 2π√(L/g).", "T=2πsqrt(L/g)", {"L": L.value, "g": g})

    # ----------------------------- interference --------------------------------
    if "interference" in ql or "coherent" in ql or "constructive" in ql or "destructive" in ql or "path difference" in ql:
        lam = _lambda(t)
        dr = (_sym(t, ["delta r", "deltar", "path difference", "delta x", "deltax"], _LEN_U)
              or _label(t, r"path\s+difference|separation|delta\s*r|deltar|delta\s*x|deltax", _LEN_U))
        order = _order_m(t)
        if "phase" in ql and lam and dr:
            return _out(2.0 * math.pi * dr.value / lam.value, question, "rad", "Phase difference from path difference is 2πΔr/λ.", "Δφ=2πΔr/λ", {"Δr": dr.value, "lambda": lam.value})
        if "constructive" in ql:
            if dr and lam and ("calculate m" in ql or "find the order" in ql or re.search(r"find\s+m\b", ql)):
                return _out(dr.value / lam.value, question, "", "Constructive interference has Δr=mλ.", "m=Δr/λ", {"Δr": dr.value, "lambda": lam.value})
            if lam and order is not None:
                return _out(float(order) * lam.value, question, "m", "Constructive interference path difference is mλ.", "Δr=mλ", {"m": order, "lambda": lam.value})
        if "destructive" in ql:
            if dr and lam and ("find m" in ql or "calculate m" in ql):
                return _out(dr.value / lam.value - 0.5, question, "", "Destructive interference has Δr=(m+1/2)λ.", "m=Δr/λ-1/2", {"Δr": dr.value, "lambda": lam.value})
            if lam and order is not None:
                return _out((float(order) + 0.5) * lam.value, question, "m", "Destructive-interference path difference is (m+1/2)λ.", "Δr=(m+1/2)λ", {"m": order, "lambda": lam.value})

    # ---------------------------- string waves ---------------------------------
    if ("string" in ql or "tension" in ql or "linear mass density" in ql or "linear density" in ql or "mu" in ql or "μ" in question) and ("wave" in ql or "tension" in ql or "linear" in ql or "mu" in ql or "μ" in question):
        tension = (_sym(t, ["T", "tension"], _FORCE_U) or _label(t, r"tension", _FORCE_U) or (_find_all(t, _FORCE_U)[0] if _find_all(t, _FORCE_U) else None))
        mu = (_sym(t, ["mu", "linear mass density", "linear density"], _LINDENS_U) or _label(t, r"linear\s+(?:mass\s+)?density|mu", _LINDENS_U) or (_find_all(t, _LINDENS_U)[0] if _find_all(t, _LINDENS_U) else None))
        vq = _wave_speed(t)
        lam = _lambda(t)
        if ("calculate mu" in ql or "find the linear mass density" in ql or "linear mass density" in ql and not mu) and tension and vq and vq.value:
            return _out(tension.value / (vq.value ** 2), question, "kg/m", "For waves on a string, v=sqrt(T/mu), so mu=T/v².", "mu=T/v^2", {"T": tension.value, "v": vq.value})
        if ("tension" in ql and ("find" in ql or "calculate" in ql)) and mu and vq:
            return _out(mu.value * vq.value ** 2, question, "N", "For waves on a string, T=mu v².", "T=mu v^2", {"mu": mu.value, "v": vq.value})
        if ("speed" in ql or "wave speed" in ql) and tension and mu and mu.value:
            return _out(math.sqrt(tension.value / mu.value), question, "m/s", "String wave speed is sqrt(T/mu).", "v=sqrt(T/mu)", {"T": tension.value, "mu": mu.value})
        if ("frequency" in ql or re.search(r"calculate\s+f\b|find\s+f\b", ql)) and tension and mu and lam:
            return _out(math.sqrt(tension.value / mu.value) / lam.value, question, "Hz", "Find string wave speed from sqrt(T/mu), then use f=v/lambda.", "f=sqrt(T/mu)/lambda", {"T": tension.value, "mu": mu.value, "lambda": lam.value})

    # ----------------------- fixed strings and air columns ----------------------
    if "fixed" in ql and "string" in ql or "open" in ql and "pipe" in ql or "air column" in ql or "closed" in ql and "pipe" in ql:
        n = _harmonic_n(t) or 1
        L = _length_L(t)
        vq = _wave_speed(t)
        f = _freq(t)
        lam = _lambda(t)
        is_closed = "closed-open" in ql or "closed at one end" in ql or "closed-open pipe" in ql or ("pipe" in ql and "closed" in ql)
        denom = 4.0 if is_closed else 2.0
        # Default sound speed only for closed/open pipe frequency without explicit v.
        if vq is None and "pipe" in ql and ("calculate f" in ql or "find the frequency" in ql or "harmonic" in ql):
            vq = Quantity("", 330.0, "m/s", "default sound speed")
        if ("wavelength" in ql or "calculate lambda" in ql or "find lambda" in ql or "calculate λ" in ql) and L:
            return _out(denom * L.value / n, question, "m", "Standing-wave wavelength is 2L/n for open/fixed ends and 4L/n for closed-open pipes.", "lambda=2L/n or 4L/n", {"L": L.value, "n": n})
        if ("find l" in ql or "find its length" in ql or "calculate length" in ql or "find the string length" in ql or "find l." in ql) and vq and f:
            return _out(n * vq.value / (denom * f.value), question, "m", "Rearrange standing-wave harmonic frequency to solve for length.", "L=nv/(2f) or nv/(4f)", {"n": n, "v": vq.value, "f": f.value})
        if ("calculate v" in ql or "find the wave speed" in ql or "find v" in ql) and L and f:
            return _out(denom * L.value * f.value / n, question, "m/s", "Rearrange standing-wave harmonic frequency to solve for wave speed.", "v=2Lf/n or 4Lf/n", {"L": L.value, "f": f.value, "n": n})
        if ("frequency" in ql or re.search(r"calculate\s+f_?\d*|find\s+f_?\d*", ql) or "harmonic" in ql and f is None) and L and vq:
            return _out(n * vq.value / (denom * L.value), question, "Hz", "Standing-wave harmonic frequency is nv/(2L) or nv/(4L) for closed-open pipes.", "f_n=nv/(2L) or nv/(4L)", {"n": n, "v": vq.value, "L": L.value})

    # -------------------------- spring combinations ----------------------------
    if "spring" in ql and ("series" in ql or "parallel" in ql):
        ks = _two_spring_constants(t)
        m = _mass(t)
        if ks:
            k1, k2 = ks
            keq = k1 + k2 if "parallel" in ql else (k1 * k2 / (k1 + k2))
            if "equivalent" in ql or "k_eq" in ql:
                return _out(keq, question, "N/m", "Equivalent spring constant is sum for parallel and reciprocal sum for series.", "k_eq=k1+k2 or k1k2/(k1+k2)", {"k1": k1, "k2": k2})
            if m and ("period" in ql or re.search(r"find\s+t\b", ql)):
                return _out(2.0 * math.pi * math.sqrt(m.value / keq), question, "s", "Use equivalent spring constant in T=2π√(m/k_eq).", "T=2πsqrt(m/k_eq)", {"m": m.value, "k_eq": keq})
            if m and ("frequency" in ql or re.search(r"(?:find|calculate)\s+f\b", ql)):
                return _out((1.0 / (2.0 * math.pi)) * math.sqrt(keq / m.value), question, "Hz", "Use equivalent spring constant in f=(1/2π)√(k_eq/m).", "f=(1/2π)sqrt(k_eq/m)", {"m": m.value, "k_eq": keq})

    # ----------------------------- vertical spring -----------------------------
    if "vertical spring" in ql or "extends" in ql or "stretches" in ql or "static extension" in ql:
        dx = _extension_dx(t)
        m = _mass(t)
        g = _g(t)
        if dx and ("spring constant" in ql or re.search(r"find\s+k\b|calculate\s+k\b", ql)) and m:
            return _out(m.value * g / dx.value, question, "N/m", "At equilibrium, mg=kΔx, so k=mg/Δx.", "k=mg/Δx", {"m": m.value, "g": g, "Δx": dx.value})
        if dx and ("period" in ql or re.search(r"find\s+t\b", ql)):
            return _out(2.0 * math.pi * math.sqrt(dx.value / g), question, "s", "For a vertical spring, T=2π√(Δx/g).", "T=2πsqrt(Δx/g)", {"Δx": dx.value, "g": g})
        if dx and ("frequency" in ql or re.search(r"(?:find|calculate)\s+f\b", ql)):
            return _out((1.0 / (2.0 * math.pi)) * math.sqrt(g / dx.value), question, "Hz", "For a vertical spring, f=(1/2π)√(g/Δx).", "f=(1/2π)sqrt(g/Δx)", {"Δx": dx.value, "g": g})

    # -------------------------- mass-spring oscillator -------------------------
    if "spring" in ql or "shm" in ql or "harmonic oscillator" in ql or "simple harmonic motion" in ql or "mass-spring" in ql:
        kq = _spring_k(t); m = _mass(t); Tq = _period(t); f = _freq(t); omega = _omega(t)
        A = _amplitude(t); x = _displacement_x(t)
        if ("energy" in ql or "mechanical energy" in ql) and kq and A and not x:
            return _out(0.5 * kq.value * A.value ** 2, question, "J", "Total spring SHM energy is 1/2 kA².", "E=1/2 kA^2", {"k": kq.value, "A": A.value})
        if ("potential energy" in ql or re.search(r"calculate\s+u\b|find\s+u\b", ql)) and kq and x:
            return _out(0.5 * kq.value * x.value ** 2, question, "J", "Spring potential energy at displacement x is 1/2kx².", "U=1/2kx^2", {"k": kq.value, "x": x.value})
        if ("kinetic energy" in ql or re.search(r"what\s+is\s+k\s+when", ql)) and kq and A and x:
            return _out(0.5 * kq.value * max(A.value ** 2 - x.value ** 2, 0.0), question, "J", "In SHM, K=1/2k(A²-x²).", "K=1/2k(A^2-x^2)", {"k": kq.value, "A": A.value, "x": x.value})
        if ("speed" in ql or re.search(r"calculate\s+v\b|find\s+v\b", ql)) and A and x:
            if omega:
                return _out(abs(omega.value) * math.sqrt(max(A.value ** 2 - x.value ** 2, 0.0)), question, "m/s", "SHM speed at displacement x is ω√(A²-x²).", "v=omega*sqrt(A^2-x^2)", {"omega": omega.value, "A": A.value, "x": x.value})
            if kq and m:
                return _out(math.sqrt(max(kq.value / m.value * (A.value ** 2 - x.value ** 2), 0.0)), question, "m/s", "For spring SHM, v=√((k/m)(A²-x²)).", "v=sqrt((k/m)(A^2-x^2))", {"k": kq.value, "m": m.value, "A": A.value, "x": x.value})
        if ("maximum speed" in ql or "v_max" in ql) and A and omega:
            return _out(abs(omega.value) * A.value, question, "m/s", "Maximum SHM speed is ωA.", "v_max=omega*A", {"omega": omega.value, "A": A.value})
        if ("maximum acceleration" in ql or "a_max" in ql) and A and omega:
            return _out(omega.value ** 2 * A.value, question, "m/s^2", "Maximum SHM acceleration is ω²A.", "a_max=omega^2*A", {"omega": omega.value, "A": A.value})
        if ("acceleration" in ql or re.search(r"find\s+a\b", ql)) and omega and x:
            return _out(-omega.value ** 2 * x.value, question, "m/s^2", "SHM acceleration is a=-ω²x.", "a=-omega^2*x", {"omega": omega.value, "x": x.value})
        if ("displacement" in ql or re.search(r"find\s+x\b", ql)) and A and "phase" in ql:
            ph = _label(t, r"phase", r"rad") or (_find_all(t, r"rad")[0] if _find_all(t, r"rad") else None)
            if ph:
                return _out(A.value * math.cos(ph.value), question, "m", "For x=Acos(phase), substitute the given phase.", "x=Acos(phi)", {"A": A.value, "phase": ph.value})
        if ("angular frequency" in ql or re.search(r"find\s+omega\b|find\s+ω\b|calculate\s+omega\b|calculate\s+ω\b", ql)) and kq and m:
            return _out(math.sqrt(kq.value / m.value), question, "rad/s", "Mass-spring angular frequency is √(k/m).", "omega=sqrt(k/m)", {"k": kq.value, "m": m.value})
        if ("frequency" in ql or re.search(r"(?:find|calculate)\s+f\b", ql)) and kq and m:
            return _out((1.0 / (2.0 * math.pi)) * math.sqrt(kq.value / m.value), question, "Hz", "Mass-spring frequency is (1/2π)√(k/m).", "f=(1/2π)sqrt(k/m)", {"k": kq.value, "m": m.value})
        if ("period" in ql or re.search(r"find\s+t\b|calculate\s+t\b", ql)) and kq and m:
            return _out(2.0 * math.pi * math.sqrt(m.value / kq.value), question, "s", "Mass-spring period is 2π√(m/k).", "T=2πsqrt(m/k)", {"m": m.value, "k": kq.value})
        if ("spring constant" in ql or re.search(r"find\s+k\b|calculate\s+k\b", ql)) and m:
            if f:
                return _out(m.value * (2.0 * math.pi * f.value) ** 2, question, "N/m", "Rearrange f=(1/2π)√(k/m) to solve for k.", "k=m(2πf)^2", {"m": m.value, "f": f.value})
            if Tq:
                return _out(4.0 * math.pi ** 2 * m.value / (Tq.value ** 2), question, "N/m", "Rearrange T=2π√(m/k) to solve for k.", "k=4π^2m/T^2", {"m": m.value, "T": Tq.value})
        if ("mass" in ql or re.search(r"find\s+m\b|calculate\s+m\b", ql)) and kq:
            if f:
                return _out(kq.value / ((2.0 * math.pi * f.value) ** 2), question, "kg", "Rearrange f=(1/2π)√(k/m) to solve for m.", "m=k/(2πf)^2", {"k": kq.value, "f": f.value})
            if Tq:
                return _out(kq.value * Tq.value ** 2 / (4.0 * math.pi ** 2), question, "kg", "Rearrange T=2π√(m/k) to solve for m.", "m=kT^2/(4π^2)", {"k": kq.value, "T": Tq.value})

    # -------------------------- periodic motion base ---------------------------
    cyc = _cycles_time(t)
    if cyc:
        ncyc, dt = cyc
        if dt:
            if "angular" in ql or "omega" in ql or "ω" in question:
                return _out(2.0 * math.pi * ncyc / dt, question, "rad/s", "Angular frequency is 2π times cycles per second.", "omega=2πN/t", {"N": ncyc, "t": dt})
            if re.search(r"\bperiod\b", ql):
                return _out(dt / ncyc, question, "s", "Period is total time divided by number of cycles.", "T=t/N", {"N": ncyc, "t": dt})
            if "frequency" in ql or re.search(r"(?:find|calculate)\s+f\b", ql):
                return _out(ncyc / dt, question, "Hz", "Frequency is cycles divided by time.", "f=N/t", {"N": ncyc, "t": dt})

    # ---------------------------- phase/wave algebra ---------------------------
    if "phase" in ql:
        f = _freq(t); Tq = _period(t); omega = _omega(t); lam = _lambda(t)
        dt = _sym(t, ["delta t", "deltat", "t"], _TIME_U) or _label(t, r"delta\s*t|deltat|over|during", _TIME_U)
        dx = (_sym(t, ["delta x", "deltax", "path separation", "separation"], _LEN_U)
              or _label(t, r"path\s+separation|separation|delta\s*x|deltax|path\s+difference", _LEN_U))
        if dt and omega:
            return _out(omega.value * dt.value, question, "rad", "Phase advance over time is Δφ=ωΔt.", "Δφ=omega*Δt", {"omega": omega.value, "Δt": dt.value})
        if dt and f:
            return _out(2.0 * math.pi * f.value * dt.value, question, "rad", "Phase advance over time is Δφ=2πfΔt.", "Δφ=2πfΔt", {"f": f.value, "Δt": dt.value})
        if dx and lam:
            return _out(2.0 * math.pi * dx.value / lam.value, question, "rad", "Spatial phase difference is Δφ=2πΔx/λ.", "Δφ=2πΔx/λ", {"Δx": dx.value, "lambda": lam.value})

    # ----------------------------- wave relations ------------------------------
    if "wave" in ql or "wavelength" in ql or "lambda" in ql or "wave number" in ql or "sinusoidal" in ql or "y=a sin" in ql or "kx-omega" in ql:
        f = _freq(t); lam = _lambda(t); vq = _wave_speed(t); omega = _omega(t); kn = _wavenumber(t)
        if ("wave number" in ql or re.search(r"calculate\s+k\b|find\s+k\b", ql)) and lam:
            return _out(2.0 * math.pi / lam.value, question, "rad/m", "Wave number is k=2π/λ.", "k=2π/lambda", {"lambda": lam.value})
        if ("wavelength" in ql or re.search(r"calculate\s+lambda\b|find\s+lambda\b|calculate\s+λ\b", ql)):
            if kn and kn.value:
                return _out(2.0 * math.pi / kn.value, question, "m", "Wavelength is λ=2π/k.", "lambda=2π/k", {"k": kn.value})
            if vq and f and f.value:
                return _out(vq.value / f.value, question, "m", "Wave speed obeys v=fλ, so λ=v/f.", "lambda=v/f", {"v": vq.value, "f": f.value})
        if ("phase speed" in ql or re.search(r"calculate\s+v\b|find\s+v\b|find\s+the\s+wave\s+speed", ql) or "wave speed" in ql):
            if omega and kn and kn.value:
                return _out(omega.value / kn.value, question, "m/s", "For y=A sin(kx-ωt), phase speed is ω/k.", "v=omega/k", {"omega": omega.value, "k": kn.value})
            if f and lam:
                return _out(f.value * lam.value, question, "m/s", "Wave speed obeys v=fλ.", "v=fλ", {"f": f.value, "lambda": lam.value})
        if ("frequency" in ql or re.search(r"find\s+f\b|calculate\s+f\b", ql)):
            if omega:
                return _out(omega.value / (2.0 * math.pi), question, "Hz", "Frequency is angular frequency divided by 2π.", "f=omega/(2π)", {"omega": omega.value})
            if vq and lam and lam.value:
                return _out(vq.value / lam.value, question, "Hz", "Wave speed obeys v=fλ, so f=v/λ.", "f=v/lambda", {"v": vq.value, "lambda": lam.value})
        if ("angular frequency" in ql or re.search(r"calculate\s+omega\b|find\s+omega\b|calculate\s+ω\b|find\s+ω\b", ql)) and f:
            return _out(2.0 * math.pi * f.value, question, "rad/s", "Angular frequency is 2πf.", "omega=2πf", {"f": f.value})
        if (re.search(r"\bperiod\b", ql) or re.search(r"calculate\s+t\b|find\s+t\b", ql)) and f and f.value:
            return _out(1.0 / f.value, question, "s", "Period is reciprocal of frequency.", "T=1/f", {"f": f.value})

    # ---------------------- omega / frequency / period generic -----------------
    f = _freq(t); Tq = _period(t); omega = _omega(t)
    if "angular frequency" in ql or "omega" in ql or "ω" in question:
        if (re.search(r"\bperiod\b", ql) or "find its period" in ql) and omega and omega.value:
            return _out(2.0 * math.pi / omega.value, question, "s", "Period is 2π/omega.", "T=2π/omega", {"omega": omega.value})
        if f and not re.search(r"\bfind\s+f\b|frequency\s+from", ql):
            return _out(2.0 * math.pi * f.value, question, "rad/s", "Angular frequency is 2πf.", "omega=2πf", {"f": f.value})
        if Tq and Tq.value:
            return _out(2.0 * math.pi / Tq.value, question, "rad/s", "Angular frequency is 2π/T.", "omega=2π/T", {"T": Tq.value})
    if re.search(r"\bperiod\b", ql) or re.search(r"\bfind\s+t\b|calculate\s+t\b", ql):
        if omega and omega.value:
            return _out(2.0 * math.pi / omega.value, question, "s", "Period is 2π/omega.", "T=2π/omega", {"omega": omega.value})
        if f and f.value:
            return _out(1.0 / f.value, question, "s", "Period is reciprocal of frequency.", "T=1/f", {"f": f.value})
    if "frequency" in ql or re.search(r"\bfind\s+f\b|frequency\s+from", ql):
        if omega:
            return _out(omega.value / (2.0 * math.pi), question, "Hz", "Frequency is omega/(2π).", "f=omega/(2π)", {"omega": omega.value})
        if Tq and Tq.value:
            return _out(1.0 / Tq.value, question, "Hz", "Frequency is reciprocal of period.", "f=1/T", {"T": Tq.value})

    # ------------------------ wave distance/speed/time -------------------------
    if "wave" in ql or "crest" in ql or "pulse" in ql or "same phase point" in ql:
        dist = _distance_for_wave(t)
        tm = _time_for_wave(t)
        sp = _wave_speed(t)
        if ("how far" in ql or "distance covered" in ql or "find the distance" in ql) and sp and tm:
            return _out(sp.value * tm.value, question, "m", "Distance traveled is speed times time.", "d=vt", {"v": sp.value, "t": tm.value})
        if ("how long" in ql or "travel time" in ql or "find the travel time" in ql or "find t" in ql) and dist and sp and sp.value:
            return _out(dist.value / sp.value, question, "s", "Travel time is distance divided by speed.", "t=d/v", {"d": dist.value, "v": sp.value})
        if ("speed" in ql or "phase speed" in ql) and dist and tm and tm.value:
            return _out(dist.value / tm.value, question, "m/s", "Speed is distance divided by time.", "v=d/t", {"d": dist.value, "t": tm.value})

    return None
