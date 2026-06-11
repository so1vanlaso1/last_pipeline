from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Any, Callable
from dataclasses import dataclass
from typing import Any, Callable
try:
    from ...schemas import PredictRequest, SolverResult
except Exception:
    try:
        from ..schemas import PredictRequest, SolverResult
    except Exception:                                               
        class PredictRequest:                
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.model_extra = {k: v for k, v in kwargs.items() if k not in {"question", "type"}}
        class SolverResult:                
            def __init__(self, **kwargs):
                if "fol" not in kwargs:
                    kwargs["fol"] = None
                self.__dict__.update(kwargs)
COULOMB_K = 9.0e9
EPS0 = 8.85e-12
G = 9.8
_SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
})
_DECIMAL_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_EXP_PART = r"(?:\^\s*\{?\s*[-+]?\d+\s*\}?|[-+]\d+)"
VALUE_PATTERN = (
    r"(?:" + _DECIMAL_PATTERN + r"\s*(?:×|x|\*|·)\s*10\s*" + _EXP_PART + r")|"
    r"(?:[-+]?\d+(?:\.\d+)?\.10\s*" + _EXP_PART + r")|"
    r"(?:[-+]?10\s*" + _EXP_PART + r")|"
    r"(?:" + _DECIMAL_PATTERN + r"[eE][-+]?\d+)|"
    r"(?:" + _DECIMAL_PATTERN + r")"
)
UNIT_PATTERN = (
    r"cm\^2|cm²|mm\^2|mm²|m\^2|m²|"
    r"N/C|V/m|"
    r"microfarads?|microfarad|μF|µF|uF|mF|nF|pF|F|"
    r"mH|μH|µH|uH|H|"
    r"kΩ|kω|Ω|ω|kohm|ohms?|"
    r"kHz|Hz|"
    r"mA|A|"
    r"kV|mV|V|"
    r"S|siemens?|"
    r"mJ|μJ|µJ|uJ|J|"
    r"mC|μC|µC|uC|nC|pC|C|"
    r"km|cm|mm|m|"
    r"kg|g|"
    r"hours?|hrs?|h|minutes?|mins?|s|"
    r"N"
)
@dataclass(frozen=True)
class Quantity:
    symbol: str
    value: float
    unit: str
    raw: str
def _normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.translate(_SUPERSCRIPT_MAP)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("µ", "μ").replace("Ω", "Ω").replace("π", "pi")
    text = text.replace("⁄", "/")
    text = text.replace("^^", "^")
    text = re.sub(r"\((C|cm|mm|m|V|F|H|Hz|Ω|ohm|N|J|A)\)", r"\1", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def _lower(text: str) -> str:
    return _normalize_text(text).lower()
def _parse_number(value: str) -> float:
    s = _normalize_text(value)
    s = re.sub(r"\s+", "", s)
    s = s.replace("×", "x").replace("*", "x").replace("·", "x")
    s = re.sub(r"10\^?\{\s*([-+]?\d+)\s*\}", r"10^\1", s, flags=re.I)
    if "," in s:
        if "." in s:
            s = s.replace(",", "")
        elif re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:[eE][-+]?\d+)?", s) and not re.match(r"[-+]?0,", s):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    decimal_re = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    if re.fullmatch(decimal_re + r"(?:[eE][-+]?\d+)?", s):
        return float(s)
    m = re.fullmatch(
        rf"({decimal_re})x10(?:\^)?([-+]?\d+)",
        s,
        flags=re.I,
    )
    if m:
        return float(m.group(1)) * (10.0 ** int(m.group(2)))
    m = re.fullmatch(
        r"([-+]?\d+(?:\.\d+)?)\.10(?:\^)?([-+]?\d+)",
        s,
        flags=re.I,
    )
    if m and ("^" in s or re.search(r"\.10[-+]", s)):
        return float(m.group(1)) * (10.0 ** int(m.group(2)))
    m = re.fullmatch(r"([-+]?)10\^([-+]?\d+)", s, flags=re.I)
    if m:
        sign = -1.0 if m.group(1) == "-" else 1.0
        return sign * (10.0 ** int(m.group(2)))
    m = re.fullmatch(r"([-+]?)10([-+]\d+)", s, flags=re.I)
    if m:
        sign = -1.0 if m.group(1) == "-" else 1.0
        return sign * (10.0 ** int(m.group(2)))
    return float(s)
def _norm_unit(unit: str | None) -> str:
    u = _normalize_text(unit or "").strip().lower()
    u = u.replace("µ", "μ")
    u = u.replace("ω", "Ω".lower())
    u = u.replace("ohms", "ohm")
    u = u.replace("volts", "v")
    u = u.replace("microfarads", "μf").replace("microfarad", "μf")
    return u
def _to_si(value: float, unit: str | None) -> float:
    raw = _normalize_text(unit or "").strip()
    if raw in {"H"}:
        return value
    if raw in {"mH"}:
        return value * 1e-3
    if raw in {"μH", "µH", "uH"}:
        return value * 1e-6
    u = _norm_unit(unit)
    mult = {
        "v": 1.0, "mv": 1e-3, "kv": 1e3,
        "a": 1.0, "ma": 1e-3,
        "ohm": 1.0, "Ω".lower(): 1.0, "kohm": 1e3, "kω": 1e3, "kΩ".lower(): 1e3,
        "hz": 1.0, "khz": 1e3,
        "f": 1.0, "mf": 1e-3, "μf": 1e-6, "uf": 1e-6, "nf": 1e-9, "pf": 1e-12,
        "c": 1.0, "mc": 1e-3, "μc": 1e-6, "uc": 1e-6, "nc": 1e-9, "pc": 1e-12,
        "j": 1.0, "mj": 1e-3, "μj": 1e-6, "uj": 1e-6,
        "n": 1.0,
        "m": 1.0, "cm": 1e-2, "mm": 1e-3, "km": 1e3,
        "m^2": 1.0, "m²": 1.0, "m2": 1.0, "cm^2": 1e-4, "cm²": 1e-4, "cm2": 1e-4, "mm^2": 1e-6, "mm²": 1e-6, "mm2": 1e-6,
        "kg": 1.0, "g": 1e-3,
        "s": 1.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
        "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
        "v/m": 1.0, "n/c": 1.0,
    }.get(u, 1.0)
    return value * mult
def _format_number(x: float, places: int | None = None, sci_large: bool = False) -> str:
    if math.isnan(x) or math.isinf(x):
        return "Uncertain"
    if places is not None:
        return f"{x:.{places}f}"
    if sci_large and abs(x) >= 1e6:
        exp = int(math.floor(math.log10(abs(x))))
        mant = x / (10 ** exp)
        mant_s = f"{mant:.3g}".rstrip("0").rstrip(".")
        return f"{mant_s} × 10^{exp}"
    if abs(x - round(x)) < 1e-9 and abs(x) >= 1000:
        return str(int(round(x)))
    if 0 < abs(x) < 1e-3:
        return f"{x:.9g}"
    s = f"{x:.9f}".rstrip("0").rstrip(".")
    return s if s else "0"
def _rounding_places(question: str) -> int | None:
    q = _lower(question)
    if "nearest integer" in q or "nearest interger" in q:
        return 0
    m = re.search(r"rounded?\s+(?:to\s+)?(?:the\s+)?(\w+|\d+)\s+decimal", q)
    if not m:
        return None
    word = m.group(1)
    lookup = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    if word.isdigit():
        return int(word)
    return lookup.get(word)
def _normalize_single_numeric_answer(answer: str) -> str:
    s = str(answer).strip()
    if ";" in s:
        return s
    m = re.fullmatch(
        rf"[A-Za-z_][A-Za-z0-9_₀₁₂₃₄₅₆₇₈₉]*\s*=\s*(?P<num>{VALUE_PATTERN})(?:\s*[A-Za-z/%Ωωμµ]*)?",
        s,
        flags=re.I,
    )
    if m:
        return m.group("num").strip()
    return s
def _make_result(
    answer: str,
    unit: str | None,
    explanation: str,
    formula: str,
    quantities: dict[str, Any] | None = None,
    confidence: float = 0.86,
    warnings: list[str] | None = None,
) -> SolverResult:
    answer = _normalize_single_numeric_answer(str(answer))
    cot = [
        "Step 1: Extract the relevant physical quantities from the question.",
        f"Step 2: Select the formula {formula}.",
        "Step 3: Substitute the extracted quantities and compute the result.",
        f"Step 4: Final answer = {answer}{(' ' + unit) if unit else ''}.",
    ]
    return SolverResult(
        answer=answer,
        unit=unit,
        explanation=explanation,
        cot=cot,
        fol=None,
        premises=[formula],
        confidence=confidence,
        warnings=warnings or [],
        debug={"formula": formula, "quantities": quantities or {}},
    )
def _uncertain(question: str, warnings: list[str] | None = None, debug: dict[str, Any] | None = None) -> SolverResult:
    cot = [
        "Step 1: Extract quantities from the question.",
        "Step 2: Try supported deterministic physics formulas.",
        "Step 3: No formula matched safely.",
    ]
    return SolverResult(
        answer="Uncertain",
        unit=None,
        explanation="The system could not confidently identify a supported physics formula from the question.",
        cot=cot,
        fol=None,
        premises=[],
        confidence=0.25,
        warnings=warnings or ["PHYSICS_FORMULA_ERROR: no supported formula matched the question."],
        debug=debug or {},
    )
def _find_all_values(text: str, unit_regex: str | None = None) -> list[tuple[float, str, str]]:
    t = _normalize_text(text)
    unit_part = unit_regex or UNIT_PATTERN
    pattern = re.compile(rf"(?P<value>{VALUE_PATTERN})\s*(?P<unit>{unit_part})\b", flags=re.I)
    out: list[tuple[float, str, str]] = []
    for m in pattern.finditer(t):
        try:
            value = _parse_number(m.group("value"))
        except Exception:
            continue
        unit = m.group("unit")
        out.append((_to_si(value, unit), unit, m.group(0)))
    return out
def _find_symbol_values(text: str, symbols: list[str], unit_regex: str | None = None) -> list[Quantity]:
    t = _normalize_text(text)
    symbols_alt = "|".join(re.escape(s) for s in symbols)
    unit_part = unit_regex or UNIT_PATTERN
    pattern = re.compile(
        rf"\b(?P<sym>{symbols_alt})\s*=\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>{unit_part})?\b",
        flags=re.I,
    )
    out: list[Quantity] = []
    for m in pattern.finditer(t):
        try:
            value = _parse_number(m.group("value"))
        except Exception:
            continue
        unit = m.group("unit") or ""
        out.append(Quantity(m.group("sym"), _to_si(value, unit), unit, m.group(0)))
    return out
def _first(qs: list[Quantity]) -> Quantity | None:
    return qs[0] if qs else None
def _get_capacitance(text: str) -> Quantity | None:
    q = _first(_find_symbol_values(text, ["C"], r"microfarads?|μF|µF|uF|mF|nF|pF|F"))
    if q:
        return q
    m = re.search(rf"capacitance\s+(?:of\s+)?(?:is\s+)?(?P<value>{VALUE_PATTERN})\s*(?P<unit>microfarads?|μF|µF|uF|mF|nF|pF|F)", _normalize_text(text), flags=re.I)
    if m:
        unit = m.group("unit")
        return Quantity("C", _to_si(_parse_number(m.group("value")), unit), unit, m.group(0))
    return None
def _get_inductance(text: str) -> Quantity | None:
    q = _first(_find_symbol_values(text, ["L"], r"mH|μH|µH|uH|H"))
    if q:
        return q
    m = re.search(rf"inductance\s+(?:of\s+)?(?:is\s+)?(?P<value>{VALUE_PATTERN})\s*(?P<unit>mH|μH|µH|uH|H)", _normalize_text(text), flags=re.I)
    if m:
        unit = m.group("unit")
        return Quantity("L", _to_si(_parse_number(m.group("value")), unit), unit, m.group(0))
    return None
def _get_resistance(text: str) -> Quantity | None:
    q = _first(_find_symbol_values(text, ["R", "R1", "R2"], r"kΩ|kω|Ω|ω|kohm|ohms?"))
    if q:
        return q
    m = re.search(rf"resistance\s+(?:R\s*)?(?:=|is|of)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>kΩ|kω|Ω|ω|kohm|ohms?)", _normalize_text(text), flags=re.I)
    if m:
        unit = m.group("unit")
        return Quantity("R", _to_si(_parse_number(m.group("value")), unit), unit, m.group(0))
    return None
def _get_voltage(text: str) -> Quantity | None:
    q = _first(_find_symbol_values(text, ["U", "V", "u", "voltage"], r"kV|mV|V"))
    if q:
        return q
    t = _normalize_text(text)
    patterns = [
        rf"(?:voltage|potential difference|rms voltage)\s+across\s+(?:the\s+)?(?:capacitor|plates?|component|resistor|inductor|it|[A-Za-z0-9_]+)\s*(?:is|of|=|to)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>kV|mV|V)\b",
        rf"(?:voltage|potential difference|rms voltage)\s*(?:is|of|=|to)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>kV|mV|V)\b",
        rf"(?:charged\s+to|applied\s+across[^,.;]*?to|applied\s+voltage(?:\s+of)?)\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>kV|mV|V)\b",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.I)
        if m:
            unit = m.group("unit")
            return Quantity("V", _to_si(_parse_number(m.group("value")), unit), unit, m.group(0))
    vals = _find_all_values(t, r"kV|mV|V")
    if vals:
        value, unit, raw = vals[-1]
        return Quantity("V", value, unit, raw)
    return None
def _get_current(text: str) -> Quantity | None:
    q = _first(_find_symbol_values(text, ["I", "Imax", "I_max", "current"], r"mA|A"))
    if q:
        return q
    m = re.search(rf"(?:current|maximum current|rms current)\s+(?:is|of|=)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>mA|A)", _normalize_text(text), flags=re.I)
    if m:
        unit = m.group("unit")
        return Quantity("I", _to_si(_parse_number(m.group("value")), unit), unit, m.group(0))
    return None
def _get_frequency_values(text: str) -> list[Quantity]:
    out = _find_symbol_values(text, ["f", "f0", "f1", "f2"], r"kHz|Hz")
    t = _normalize_text(text)
    for m in re.finditer(rf"(?:resonates at|resonance\s*\(?|resonant frequency|frequency(?: increases to)?|when f\s*=?)\s*(?:f\w*\s*=\s*)?(?P<value>{VALUE_PATTERN})\s*(?P<unit>kHz|Hz)", t, flags=re.I):
        try:
            unit = m.group("unit")
            out.append(Quantity("f", _to_si(_parse_number(m.group("value")), unit), unit, m.group(0)))
        except Exception:
            pass
    seen = set()
    uniq = []
    for q in out:
        key = round(q.value, 12)
        if key not in seen:
            seen.add(key)
            uniq.append(q)
    return uniq
def _get_area(text: str) -> Quantity | None:
    m = re.search(rf"(?:plate area|area)\s+(?:A\s*)?(?:=|is|of)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>cm\^2|cm²|mm\^2|mm²|m\^2|m²)", _normalize_text(text), flags=re.I)
    if m:
        unit = m.group("unit")
        return Quantity("A", _to_si(_parse_number(m.group("value")), unit), unit, m.group(0))
    return None
def _get_distance_values(text: str) -> list[Quantity]:
    values: list[Quantity] = []
    t = _normalize_text(text)
    for sym in ["r", "d", "AB", "MO", "CA", "CB", "L"]:
        values.extend(_find_symbol_values(t, [sym], r"km|cm|mm|m"))
    for m in re.finditer(rf"(?:distance|separation|side length|legs?|away|apart|plate separation|distance between[^,]*|located[^,]*?)\s+(?:is|of|between the plates is|=)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>km|cm|mm|m)\b", t, flags=re.I):
        try:
            unit = m.group("unit")
            values.append(Quantity("D", _to_si(_parse_number(m.group("value")), unit), unit, m.group(0)))
        except Exception:
            pass
    for val, unit, raw in _find_all_values(t, r"km|cm|mm|m"):
        if unit.lower() == "m" and re.search(r"\b(?:A|V|F|H|J|N)\b", raw):
            continue
        values.append(Quantity("D", val, unit, raw))
    uniq: list[Quantity] = []
    seen = set()
    for q in values:
        key = (round(q.value, 12), q.raw)
        if key not in seen:
            seen.add(key)
            uniq.append(q)
    return uniq
def _get_energy_values(text: str) -> list[Quantity]:
    values: list[Quantity] = []
    for m in re.finditer(rf"(?:energy|stored energy|total energy|electric field energy|magnetic field energy|desired stored energy)\s+(?:is|of|=)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>mJ|μJ|µJ|uJ|J)", _normalize_text(text), flags=re.I):
        try:
            unit = m.group("unit")
            values.append(Quantity("E", _to_si(_parse_number(m.group("value")), unit), unit, m.group(0)))
        except Exception:
            pass
    for val, unit, raw in _find_all_values(text, r"mJ|μJ|µJ|uJ|J"):
        values.append(Quantity("E", val, unit, raw))
    uniq = []
    seen = set()
    for q in values:
        key = (round(q.value, 12), q.raw)
        if key not in seen:
            seen.add(key)
            uniq.append(q)
    return uniq
def _charge_quantities(text: str) -> list[Quantity]:
    t = _normalize_text(text)
    out: list[Quantity] = []
    for m in re.finditer(rf"(?P<prefix>(?:q[A-Za-z0-9]*\s*=\s*){ 2,} )(?P<value>{VALUE_PATTERN})\s*(?P<unit>mC|μC|µC|uC|nC|pC|C)\b", t, flags=re.I):
        syms = re.findall(r"q[A-Za-z0-9]*", m.group("prefix"), flags=re.I)
        unit = m.group("unit")
        value = _to_si(_parse_number(m.group("value")), unit)
        for s in syms:
            out.append(Quantity(s, value, unit, m.group(0)))
    out.extend(_find_symbol_values(t, ["q", "Q", "q1", "q2", "q3", "qA", "qB"], r"mC|μC|µC|uC|nC|pC|C"))
    m = re.search(rf"qA\s+and\s+qB[^.]*?both\s+equal\s+to\s+(?P<value>{VALUE_PATTERN})\s*(?P<unit>mC|μC|µC|uC|nC|pC|C)", t, flags=re.I)
    if m:
        unit = m.group("unit")
        value = _to_si(_parse_number(m.group("value")), unit)
        out.append(Quantity("qA", value, unit, m.group(0)))
        out.append(Quantity("qB", value, unit, m.group(0)))
    m = re.search(rf"(?:three\s+identical\s+charges|identical\s+charges),?\s*q\s*=\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>mC|μC|µC|uC|nC|pC|C)", t, flags=re.I)
    if m:
        unit = m.group("unit")
        value = _to_si(_parse_number(m.group("value")), unit)
        out.append(Quantity("q", value, unit, m.group(0)))
    return out
def _charge_values(text: str) -> dict[str, float]:
    d: dict[str, float] = {}
    for q in _charge_quantities(text):
        d[q.symbol.lower()] = q.value
    return d
def _extract_current_ratio(question: str) -> float | None:
    q = _lower(question)
    if (
        "current is halved" in q
        or "current becomes half" in q
        or "current is half" in q
        or "current decreases to 1/2" in q
        or "current decreases to one half" in q
        or "current drops to 1/2" in q
    ):
        return 2.0                       
    m = re.search(r"current(?:\s*\([^)]*\))?\s+(?:decreases|drops|is reduced)\s+to\s+(?P<num>\d+(?:\.\d+)?|one)\s*/\s*(?P<den>\d+(?:\.\d+)?)", q)
    if m:
        num = 1.0 if m.group("num") == "one" else float(m.group("num"))
        den = float(m.group("den"))
        if num > 0:
            return den / num
    currents = _find_symbol_values(question, ["I", "I1", "I2"], r"mA|A")
    if len(currents) >= 2 and currents[-1].value != 0:
        return currents[0].value / currents[-1].value
    return None
__all__ = [name for name in globals() if not name.startswith("__")]
_DEGREE_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:°|degrees?|deg)"
def _parse_number_expr(value: str) -> float:
    s = _normalize_text(value)
    s = s.replace("−", "-").replace("×", "*").replace("x", "*")
    s = re.sub(r"(\d)\s*√\s*(\d+)", r"\1*sqrt(\2)", s)
    s = re.sub(r"√\s*(\d+)", r"sqrt(\1)", s)
    s = re.sub(r"10\s*\^\s*([-+]?\d+)", r"10**\1", s)
    try:
        import sympy as sp
        return float(sp.N(sp.sympify(s, locals={"sqrt": sp.sqrt, "pi": sp.pi})))
    except Exception:
        return _parse_number(value)
def _find_all_values_expr(text: str, unit_regex: str | None = None) -> list[tuple[float, str, str]]:
    t = _normalize_text(text)
    unit_part = unit_regex or UNIT_PATTERN
    expr = rf"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*√\s*\d+)|{VALUE_PATTERN}"
    pattern = re.compile(rf"(?P<value>{expr})\s*(?P<unit>{unit_part})\b", flags=re.I)
    out: list[tuple[float, str, str]] = []
    for m in pattern.finditer(t):
        try:
            value = _parse_number_expr(m.group("value"))
        except Exception:
            continue
        unit = m.group("unit")
        out.append((_to_si(value, unit), unit, m.group(0)))
    return out
def _format_default(x: float, places: int | None = None) -> str:
    return _format_number(x, places)
def _all_resistances(question: str) -> list[float]:
    vals: list[float] = []
    t = _normalize_text(question)
    for q in _find_symbol_values(t, ["R1", "R2", "R3", "R4", "R_1", "R_2", "R_3", "R_4", "R"], r"kΩ|kω|Ω|ω|kohm|ohms?"):
        vals.append(q.value)
    for m in re.finditer(rf"resistance\s+(?:of\s+)?(?P<value>{VALUE_PATTERN})\s*(?P<unit>kΩ|kω|Ω|ω|kohm|ohms?)", t, flags=re.I):
        try:
            vals.append(_to_si(_parse_number(m.group("value")), m.group("unit")))
        except Exception:
            pass
    out: list[float] = []
    for v in vals:
        if not any(abs(v - x) <= max(1e-12, abs(v) * 1e-12) for x in out):
            out.append(v)
    return out
def _all_capacitances(question: str) -> list[Quantity]:
    t = _normalize_text(question)
    vals: list[Quantity] = []
    vals.extend(_find_symbol_values(t, ["C1", "C2", "C3", "C4", "C_1", "C_2", "C_3", "C_4", "C"], r"microfarads?|μF|µF|uF|mF|nF|pF|F"))
    for m in re.finditer(rf"capacitance(?:s)?\s+(?:of\s+)?(?P<value>{VALUE_PATTERN})\s*(?P<unit>microfarads?|μF|µF|uF|mF|nF|pF|F)", t, flags=re.I):
        try:
            vals.append(Quantity("C", _to_si(_parse_number(m.group("value")), m.group("unit")), m.group("unit"), m.group(0)))
        except Exception:
            pass
    out: list[Quantity] = []
    for q in vals:
        if not any(abs(q.value - x.value) <= max(1e-18, abs(q.value) * 1e-12) for x in out):
            out.append(q)
    return out
def _all_voltages(question: str) -> list[Quantity]:
    t = _normalize_text(question)
    vals: list[Quantity] = []
    vals.extend(_find_symbol_values(t, ["U1", "U2", "U3", "U_1", "U_2", "V1", "V2", "U", "V"], r"kV|mV|V"))
    for m in re.finditer(rf"(?:voltage|potential difference|charged to)\s+(?:of\s+|to\s+|=\s*)?(?P<value>{VALUE_PATTERN})\s*(?P<unit>kV|mV|V)", t, flags=re.I):
        try:
            vals.append(Quantity("V", _to_si(_parse_number(m.group("value")), m.group("unit")), m.group("unit"), m.group(0)))
        except Exception:
            pass
    out: list[Quantity] = []
    for q in vals:
        if not any(abs(q.value - x.value) <= max(1e-12, abs(q.value) * 1e-12) for x in out):
            out.append(q)
    return out
def _cap_energy_unit(question: str, C: Quantity | None, value_j: float) -> tuple[float, str, int | None]:
    q = _lower(question)
    if re.search(r"\(\s*j\s*\)|\bin\s+j\b|\bunit\s+j\b|energy\s+j\b", q):
        return value_j, "J", _rounding_places(question)
    if "mj" in q or "millijoule" in q:
        return value_j / 1e-3, "mJ", _rounding_places(question) or 2
    if "μj" in q or "uj" in q or "microjoule" in q:
        return value_j / 1e-6, "μJ", _rounding_places(question)
    if "nj" in q or "nanojoule" in q:
        return value_j / 1e-9, "nJ", _rounding_places(question) or 2
    cu = _norm_unit(C.unit if C else "")
    if cu in {"pf"}:
        nj = value_j / 1e-9
        if abs(nj) >= 1000:
            return value_j / 1e-6, "μJ", _rounding_places(question) or 2
        return nj, "nJ", _rounding_places(question) or 2
    if cu in {"μf", "uf", "microfarad"}:
        if abs(value_j) >= 0.1:
            return value_j, "J", _rounding_places(question)
        uj = value_j / 1e-6
        if abs(uj) >= 1000:
            return value_j / 1e-3, "mJ", _rounding_places(question) or 2
        return uj, "μJ", _rounding_places(question)
    return value_j, "J", _rounding_places(question)
def _generic_voltage(question: str) -> Quantity | None:
    vals = _find_all_values_expr(question, r"kV|mV|V")
    if vals:
        v, unit, raw = vals[-1]
        return Quantity("V", v, unit, raw)
    return None
def _generic_capacitance(question: str) -> Quantity | None:
    vals = _find_all_values_expr(question, r"microfarads?|μF|µF|uF|mF|nF|pF|F")
    if vals:
        v, unit, raw = vals[0]
        return Quantity("C", v, unit, raw)
    return None
def _generic_inductance(question: str) -> Quantity | None:
    vals = _find_all_values_expr(question, r"mH|μH|µH|uH|H")
    if vals:
        v, unit, raw = vals[0]
        return Quantity("L", v, unit, raw)
    return None
def _generic_frequency(question: str) -> Quantity | None:
    vals = _find_all_values_expr(question, r"kHz|Hz")
    if vals:
        v, unit, raw = vals[-1]
        return Quantity("f", v, unit, raw)
    return None
def _generic_impedance(question: str) -> Quantity | None:
    t = _normalize_text(question)
    patterns = [
        rf"\bZ\s*=\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>kΩ|kω|Ω|ω|kohm|ohms?)",
        rf"impedance[^.?,;]*?(?:=|is|be|to be|measured\s+to\s+be)\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>kΩ|kω|Ω|ω|kohm|ohms?)",
        rf"total\s+impedance[^.?,;]*?(?:=|is|at\s+this\s+point\s+is)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>kΩ|kω|Ω|ω|kohm|ohms?)",
        rf"impedance[^.?,;]*?\s+(?P<value>{VALUE_PATTERN})\s*(?P<unit>kΩ|kω|Ω|ω|kohm|ohms?)",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.I)
        if m:
            return Quantity("Z", _to_si(_parse_number(m.group("value")), m.group("unit")), m.group("unit"), m.group(0))
    return None
def _generic_charge_quantity(question: str) -> Quantity | None:
    t = _normalize_text(question)
    patterns = [
        rf"\bQ\s*=\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>mC|μC|µC|uC|nC|pC|C)\b",
        rf"(?:maximum\s+)?charge(?:\s+on\s+[^.?,;]*?|\s+of\s+[^.?,;]*?|\s+varies\s+from\s+0\s+to)?\s*(?:is|=)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>mC|μC|µC|uC|nC|pC|C)\b",
        rf"charged\s+capacitor\s+has\s+a\s+charge\s+of\s+(?P<value>{VALUE_PATTERN})\s*(?P<unit>mC|μC|µC|uC|nC|pC|C)\b",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.I)
        if m:
            return Quantity("Q", _to_si(_parse_number(m.group("value")), m.group("unit")), m.group("unit"), m.group(0))
    return None
def _extract_turn_count(question: str) -> float | None:
    t = _normalize_text(question)
    m = re.search(r"(?P<value>\d+(?:\.\d+)?)\s*turns", t, flags=re.I)
    if m:
        return _parse_number(m.group("value"))
    return None
def _extract_magnetic_field_T(question: str) -> float | None:
    t = _normalize_text(question)
    m = re.search(rf"(?:magnetic\s+field(?:\s+is)?|B\s*=)\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>T|tesla)", t, flags=re.I)
    if m:
        return _parse_number(m.group("value"))
    return None
def _extract_radius(question: str) -> Quantity | None:
    t = _normalize_text(question)
    m = re.search(rf"radius\s*(?:R\s*)?(?:=|is|of)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>km|cm|mm|m)\b", t, flags=re.I)
    if m:
        return Quantity("R", _to_si(_parse_number(m.group("value")), m.group("unit")), m.group("unit"), m.group(0))
    return None
def _display_in_original_unit(value_si: float, unit: str) -> float:
    scale = _to_si(1.0, unit)
    return value_si / scale if scale else value_si
__all__ = [name for name in globals() if not name.startswith("__")]
def _foundational_float_expr(s: str) -> float:
    s = _normalize_text(str(s or ""))
    s = s.replace("√", "sqrt")
    s = s.replace("pi", "math.pi")
    s = s.replace("^", "**")
    s = re.sub(r"(?<=\d)\s*sqrt", "*math.sqrt", s)
    s = re.sub(r"(?<=\))\s*(?=\d)", "*", s)
    s = re.sub(r"(?<=\d)\s*(?=math\.pi)", "*", s)
    s = re.sub(r"(?<=math\.pi)\s*(?=\d)", "*", s)
    return float(eval(s, {"__builtins__": {}}, {"math": math}))
def _foundational_all_symbol_values(text: str, sym: str, unit_regex: str) -> list[Quantity]:
    t = _normalize_text(text)
    out = _find_symbol_values(t, [sym], unit_regex)
    pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(sym)}\s*=\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>{unit_regex})", flags=re.I)
    for m in pat.finditer(t):
        try:
            out.append(Quantity(sym, _to_si(_parse_number(m.group('value')), m.group('unit')), m.group('unit'), m.group(0)))
        except Exception:
            pass
    seen = set(); uniq=[]
    for qv in out:
        k=(qv.symbol.lower(), round(qv.value, 15), qv.unit)
        if k not in seen:
            seen.add(k); uniq.append(qv)
    return uniq
def _foundational_cap_values(text: str) -> list[Quantity]:
    vals=[]
    for sym in ["C", "C1", "C2", "C_1", "C_2"]:
        vals.extend(_foundational_all_symbol_values(text, sym, r"microfarads?|μF|µF|uF|mF|nF|pF|F"))
    t=_normalize_text(text)
    for m in re.finditer(rf"capacitance(?:\s+of)?\s+(?:is\s+|=\s*)?(?P<value>{VALUE_PATTERN})\s*(?P<unit>microfarads?|μF|µF|uF|mF|nF|pF|F)", t, flags=re.I):
        try: vals.append(Quantity("C", _to_si(_parse_number(m.group('value')), m.group('unit')), m.group('unit'), m.group(0)))
        except Exception: pass
    seen=set(); out=[]
    for v in vals:
        k=(round(v.value, 15), v.raw)
        if k not in seen:
            seen.add(k); out.append(v)
    return out
def _foundational_voltage_values(text: str) -> list[Quantity]:
    vals=[]
    for sym in ["U", "U1", "U2", "V", "V1", "V2"]:
        vals.extend(_foundational_all_symbol_values(text, sym, r"kV|mV|V"))
    t=_normalize_text(text)
    for m in re.finditer(rf"(?:voltage|potential difference|source voltage|rms voltage|applied voltage)\s*(?:of|is|=)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>kV|mV|V)", t, flags=re.I):
        try: vals.append(Quantity("V", _to_si(_parse_number(m.group('value')), m.group('unit')), m.group('unit'), m.group(0)))
        except Exception: pass
    return vals
def _foundational_current_values(text: str) -> list[Quantity]:
    vals=[]
    for sym in ["I", "I0", "I_0", "Imax", "current"]:
        vals.extend(_foundational_all_symbol_values(text, sym, r"mA|A"))
    t=_normalize_text(text)
    for m in re.finditer(rf"(?:current|rms current|maximum current)\s*(?:is|of|=)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>mA|A)", t, flags=re.I):
        try: vals.append(Quantity("I", _to_si(_parse_number(m.group('value')), m.group('unit')), m.group('unit'), m.group(0)))
        except Exception: pass
    return vals
def _foundational_l_values(text: str) -> list[Quantity]:
    vals=[]
    for sym in ["L"]:
        vals.extend(_foundational_all_symbol_values(text, sym, r"mH|μH|µH|uH|H"))
    t=_normalize_text(text)
    for m in re.finditer(rf"(?:inductor|inductance)\s*(?:with|of|is|=)?\s*L?\s*=?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>mH|μH|µH|uH|H)", t, flags=re.I):
        try: vals.append(Quantity("L", _to_si(_parse_number(m.group('value')), m.group('unit')), m.group('unit'), m.group(0)))
        except Exception: pass
    return vals
def _foundational_resistance_values(text: str) -> list[Quantity]:
    vals=[]
    for sym in ["R", "R1", "R2", "Z", "XL", "XC", "ZL"]:
        vals.extend(_foundational_all_symbol_values(text, sym, r"kΩ|kω|Ω|ω|kohm|ohms?"))
    return vals
def _foundational_time_values(text: str) -> list[Quantity]:
    vals=[]
    t=_normalize_text(text)
    for m in re.finditer(rf"(?:time|period|in|over|t\s*=)\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>hours?|hrs?|h|minutes?|mins?|s|ms)", t, flags=re.I):
        try:
            u=m.group('unit')
            mult=1e-3 if u.lower()=="ms" else 1.0
            vals.append(Quantity("t", _to_si(_parse_number(m.group('value')), u)*mult, u, m.group(0)))
        except Exception: pass
    return vals
def _foundational_output_num(x: float, question: str, default_places: int | None = None, sci_large: bool = False) -> str:
    places = _rounding_places(question)
    if places is None:
        places = default_places
    return _format_number(x, places, sci_large=sci_large)
__all__ = [name for name in globals() if not name.startswith("__")]
def _electromagnetic_freqs(text: str) -> list[float]:
    t = _normalize_text(text)
    vals=[]
    for m in re.finditer(rf"(?P<v>{VALUE_PATTERN})\s*(?P<u>kHz|Hz)\b", t, flags=re.I):
        try: vals.append(_to_si(_parse_number(m.group('v')), m.group('u')))
        except Exception: pass
    return vals
def _electromagnetic_lengths(text: str) -> list[float]:
    vals=[]
    for m in re.finditer(rf"(?P<v>{VALUE_PATTERN})\s*(?P<u>km|cm|mm|m)\b", _normalize_text(text), flags=re.I):
        try: vals.append(_to_si(_parse_number(m.group('v')), m.group('u')))
        except Exception: pass
    return vals
def _electromagnetic_area(text: str) -> Quantity | None:
    t=_normalize_text(text)
    for pat in [
        rf"(?:area|plate area|cross-sectional area)\s*(?:S|A)?\s*(?:=|is|of)?\s*(?P<v>{VALUE_PATTERN})\s*(?P<u>cm\^2|cm²|mm\^2|mm²|m\^2|m²)",
        rf"\bS\s*=\s*(?P<v>{VALUE_PATTERN})\s*(?P<u>cm\^2|cm²|mm\^2|mm²|m\^2|m²)",
    ]:
        m=re.search(pat,t,flags=re.I)
        if m:
            return Quantity('S', _to_si(_parse_number(m.group('v')), m.group('u')), m.group('u'), m.group(0))
    m=re.search(rf"(?P<a>{VALUE_PATTERN})\s*m\s*(?:x|×)\s*(?P<b>{VALUE_PATTERN})\s*m", t, flags=re.I)
    if m:
        return Quantity('S', _parse_number(m.group('a'))*_parse_number(m.group('b')), 'm^2', m.group(0))
    return None
def _electromagnetic_currents_all(text: str) -> list[float]:
    t=_normalize_text(text)
    vals=[]
    m=re.search(rf"from\s+(?P<a>{VALUE_PATTERN})\s*(?:A)?\s+to\s+(?P<b>{VALUE_PATTERN})\s*A", t, flags=re.I)
    if m:
        vals.extend([_parse_number(m.group('a')), _parse_number(m.group('b'))])
    for x,_,_ in _find_all_values(text, r"mA|A"):
        vals.append(x)
    out=[]
    for v in vals:
        if not any(abs(v-u)<1e-12 for u in out): out.append(v)
    return out
def _electromagnetic_fluxes(text: str) -> list[float]:
    vals=[]
    for m in re.finditer(rf"(?P<v>{VALUE_PATTERN})\s*Wb\b", _normalize_text(text), flags=re.I):
        try: vals.append(_parse_number(m.group('v')))
        except Exception: pass
    return vals
def _electromagnetic_round_for_gold(x: float, question: str, places: int | None = None, sci: bool = False) -> str:
    p = _rounding_places(question)
    if p is None: p=places
    return _format_number(x, p, sci_large=sci)
__all__ = [name for name in globals() if not name.startswith("__")]
def _ac_match_value_unit(text: str, pattern: str, units: str = UNIT_PATTERN) -> Quantity | None:
    m = re.search(pattern, _normalize_text(text), flags=re.I)
    if not m:
        return None
    try:
        value = _parse_number(m.group('value'))
        unit = m.groupdict().get('unit') or ''
        return Quantity('', _to_si(value, unit), unit, m.group(0))
    except Exception:
        return None
def _ac_symbol(text: str, name: str, units: str = UNIT_PATTERN) -> Quantity | None:
    t = _normalize_text(text)
    name_re = re.escape(name).replace('\\ ', r'\\s*')
    m = re.search(rf"\b{name_re}\s*=\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>{units})?\b", t, flags=re.I)
    if not m:
        return None
    try:
        unit = m.group('unit') or ''
        return Quantity(name, _to_si(_parse_number(m.group('value')), unit), unit, m.group(0))
    except Exception:
        return None
def _ac_plain_number_after(text: str, key_re: str) -> float | None:
    m = re.search(rf"{key_re}\s*(?:=|is|of|as|to)?\s*(?P<value>{VALUE_PATTERN})", _normalize_text(text), flags=re.I)
    if not m:
        return None
    try:
        return _parse_number(m.group('value'))
    except Exception:
        return None
def _ac_r1_r2(text: str) -> tuple[float | None, float | None]:
    t = _normalize_text(text)
    def one(label: str) -> float | None:
        m = re.search(rf"\b{label}\s*=\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>kΩ|kω|Ω|ω|kohm|ohms?)", t, flags=re.I)
        if m:
            return _to_si(_parse_number(m.group('value')), m.group('unit'))
        return None
    return one('R1'), one('R2')
def _ac_voltage_from_ac_source(text: str) -> tuple[float | None, float | None]:
    t = _normalize_text(text)
    m = re.search(rf"u\s*=\s*(?P<A>{VALUE_PATTERN})\s*(?:√\s*2|sqrt\s*\(?\s*2\s*\)?)\s*cos\s*(?P<w>{VALUE_PATTERN})\s*pi\s*t", t, flags=re.I)
    if m:
        return _parse_number(m.group('A')), _parse_number(m.group('w')) * math.pi
    m = re.search(rf"u\s*=\s*(?P<A>{VALUE_PATTERN})\s*cos\s*(?P<w>{VALUE_PATTERN})\s*pi\s*t", t, flags=re.I)
    if m:
        return _parse_number(m.group('A')) / math.sqrt(2), _parse_number(m.group('w')) * math.pi
    return None, None
def _ac_expr_inductance(text: str) -> float | None:
    t = _normalize_text(text)
    m = re.search(r"L\s*=\s*(?:(?P<num>[-+]?\d+(?:\.\d+)?)\s*)?/\s*pi\s*H", t, flags=re.I)
    if m:
        num = float(m.group('num') or 1.0)
        return num / math.pi
    q = _get_inductance(t)
    if q:
        return q.value
    m = re.search(rf"(?P<value>{VALUE_PATTERN})\s*(?P<unit>mH|μH|µH|uH|H)\s+inductor", t, flags=re.I)
    if m:
        return _to_si(_parse_number(m.group('value')), m.group('unit'))
    return None
def _ac_expr_capacitance(text: str) -> float | None:
    t = _normalize_text(text)
    m = re.search(r"C\s*=\s*(?P<num>10[-+]\d+|(?:[-+]?\d+(?:\.\d+)?)\s*(?:x|×|\*)\s*10\^?[-+]?\d+|10\^?[-+]?\d+|[-+]?\d+(?:\.\d+)?)\s*/\s*\(?\s*(?P<den>[-+]?\d+(?:\.\d+)?)?\s*pi\s*\)?\s*F", t, flags=re.I)
    if m:
        num_s = re.sub(r"\s+", "", m.group('num'))
        mm = re.fullmatch(r"10([-+]\d+)", num_s)
        num = 10.0 ** int(mm.group(1)) if mm else _parse_number(num_s)
        den = float(m.group('den') or 1.0)
        return num / (den * math.pi)
    q = _get_capacitance(t)
    if q:
        return q.value
    return None
def _ac_area_from_radius(text: str) -> float | None:
    t = _normalize_text(text)
    area = _get_area(t)
    if area:
        return area.value
    sm = re.search(rf"(?:area\s*)?S\s*=\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>cm\^2|cm2|cm²|mm\^2|mm2|mm²|m\^2|m2|m²)", t, flags=re.I)
    if sm:
        unit = sm.group('unit').replace('2','^2') if sm.group('unit').endswith('2') else sm.group('unit')
        return _to_si(_parse_number(sm.group('value')), unit)
    am = re.search(rf"area(?: of)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>cm\^2|cm2|cm²|mm\^2|mm2|mm²|m\^2|m2|m²)", t, flags=re.I)
    if am:
        unit = am.group('unit').replace('2','^2') if am.group('unit').endswith('2') else am.group('unit')
        return _to_si(_parse_number(am.group('value')), unit)
    m = re.search(rf"radius\s*(?:R\s*)?(?:=|is|of)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>km|cm|mm|m)", t, flags=re.I)
    if m:
        r = _to_si(_parse_number(m.group('value')), m.group('unit'))
        return math.pi * r * r
    return None
def _ac_dominant_distance(text: str) -> float | None:
    vals = _get_distance_values(text)
    if not vals:
        return None
    for v in vals:
        if re.search(r"plate separation|distance between|separation|plates", v.raw, flags=re.I):
            return v.value
    return vals[0].value
def _ac_capacitor_values(text: str) -> tuple[Quantity | None, Quantity | None, Quantity | None, Quantity | None]:
    t = _normalize_text(text)
    charges = _charge_quantities(t)
    q = charges[0] if charges else None
    if q is None:
        m = re.search(rf"charge(?: of)?\s*(?:is|=)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>μC|µC|uC|mC|nC|pC|C)", t, flags=re.I)
        if m:
            q = Quantity('Q', _to_si(_parse_number(m.group('value')), m.group('unit')), m.group('unit'), m.group(0))
    v = _get_voltage(t)
    if v is None:
        m = re.search(rf"(?P<value>{VALUE_PATTERN})\s*V\b", t, flags=re.I)
        if m:
            v = Quantity('V', _parse_number(m.group('value')), 'V', m.group(0))
    c = _get_capacitance(t)
    if c is None:
        m = re.search(rf"(?P<value>{VALUE_PATTERN})\s*(?P<unit>μF|µF|uF|mF|nF|pF|F)\s+capacitor", t, flags=re.I)
        if m:
            c = Quantity('C', _to_si(_parse_number(m.group('value')), m.group('unit')), m.group('unit'), m.group(0))
    energies = _get_energy_values(t)
    return c, v, q, (energies[0] if energies else None)
__all__ = [name for name in globals() if not name.startswith("__")]
MU0 = 4.0 * math.pi * 1e-7
E_CHARGE = 1.602176634e-19
E_MASS = 9.1093837015e-31
def _geometry_fmt(x: float, places: int | None = None, sci: bool = False) -> str:
    if places is not None and not (math.isnan(x) or math.isinf(x)):
        eps = 1e-12 if x >= 0 else -1e-12
        return f"{x + eps:.{places}f}"
    return _format_number(x, places, sci_large=sci)
def _geometry_numbers(text: str) -> list[float]:
    return [_parse_number(m.group(0)) for m in re.finditer(VALUE_PATTERN, _normalize_text(text))]
def _geometry_unit_values(text: str, unit_regex: str) -> list[Quantity]:
    vals: list[Quantity] = []
    t = _normalize_text(text)
    for m in re.finditer(rf"(?P<value>{VALUE_PATTERN})\s*(?P<unit>{unit_regex})\b", t, flags=re.I):
        try:
            vals.append(Quantity("", _to_si(_parse_number(m.group("value")), m.group("unit")), m.group("unit"), m.group(0)))
        except Exception:
            pass
    return vals
def _geometry_symbol_value(text: str, symbols: list[str], unit_regex: str | None = None) -> Quantity | None:
    vals = _find_symbol_values(_normalize_text(text), symbols, unit_regex)
    return vals[0] if vals else None
def _geometry_symbol_values(text: str, symbols: list[str], unit_regex: str | None = None) -> list[Quantity]:
    return _find_symbol_values(_normalize_text(text), symbols, unit_regex)
def _geometry_first_unit_value(text: str, unit_regex: str) -> Quantity | None:
    vals = _geometry_unit_values(text, unit_regex)
    return vals[0] if vals else None
def _geometry_get_area(text: str) -> Quantity | None:
    t = _normalize_text(text)
    patterns = [
        rf"(?:cross-sectional\s+area|plate\s+area|area|S)\s*(?:A\s*)?(?:=|is|of)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>cm\^2|cm2|cm²|mm\^2|mm2|mm²|m\^2|m2|m²)",
        rf"(?P<value>{VALUE_PATTERN})\s*(?P<unit>cm\^2|cm2|cm²|mm\^2|mm2|mm²|m\^2|m2|m²)",
    ]
    mult = {"m^2":1.0,"m2":1.0,"m²":1.0,"cm^2":1e-4,"cm2":1e-4,"cm²":1e-4,"mm^2":1e-6,"mm2":1e-6,"mm²":1e-6}
    for pat in patterns:
        m = re.search(pat, t, flags=re.I)
        if m:
            unit = m.group("unit")
            val = _parse_number(m.group("value")) * mult.get(unit.lower(), 1.0)
            return Quantity("A", val, unit, m.group(0))
    return None
def _geometry_extract_B(text: str) -> float | None:
    t = _normalize_text(text)
    m = re.search(rf"(?:magnetic\s+(?:field(?:\s+strength|\s+density)?|flux\s+density)(?:\s+of|\s+is)?|B\s*=)\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>T|tesla)", t, flags=re.I)
    if m:
        try:
            return _parse_number(m.group("value"))
        except Exception:
            return None
    return _extract_magnetic_field_T(text)
def _geometry_all_charges(text: str) -> list[Quantity]:
    vals = _charge_quantities(text)
    t = _normalize_text(text)
    for sym in ["q0", "q_0", "q", "q1", "q2", "q3", "Q"]:
        vals.extend(_find_symbol_values(t, [sym], r"mC|μC|µC|uC|nC|pC|C"))
    out: list[Quantity] = []
    seen = set()
    for v in vals:
        key = (v.symbol.lower().replace("_", ""), round(v.value, 18), v.raw)
        if key not in seen:
            seen.add(key); out.append(v)
    return out
def _geometry_charge_map(text: str) -> dict[str, float]:
    d: dict[str, float] = {}
    for qv in _geometry_all_charges(text):
        d[qv.symbol.lower().replace("_", "")] = qv.value
    return d
def _geometry_two_source_and_test(text: str) -> tuple[float, float, float] | None:
    m = _geometry_charge_map(text)
    q1 = m.get("q1") or m.get("qa")
    q2 = m.get("q2") or m.get("qb")
    qt = m.get("q3") or m.get("q0") or m.get("q")
    if q1 is not None and q2 is not None and qt is not None:
        return q1, q2, qt
    vals = [v.value for v in _geometry_all_charges(text)]
    if len(vals) >= 3:
        return vals[0], vals[1], vals[2]
    return None
def _geometry_get_g(question: str) -> float:
    t = _normalize_text(question)
    m = re.search(rf"\bg\s*=\s*(?P<value>{VALUE_PATTERN})\s*m\s*/\s*s\^?2", t, flags=re.I)
    if m:
        try: return _parse_number(m.group("value"))
        except Exception: pass
    return G
def _geometry_triangle_point_from_distances(r1: float, r2: float, ab: float) -> tuple[float, float]:
    x = (r1*r1 - r2*r2 + ab*ab) / (2.0 * ab) if ab else 0.0
    y2 = max(0.0, r1*r1 - x*x)
    return x, math.sqrt(y2)
def _geometry_force_vec(q_source: float, q_test: float, src: tuple[float, float], pt: tuple[float, float]) -> tuple[float, float]:
    dx = pt[0] - src[0]; dy = pt[1] - src[1]
    r2 = dx*dx + dy*dy
    if r2 <= 0:
        return (0.0, 0.0)
    r = math.sqrt(r2)
    coef = COULOMB_K * q_source * q_test / (r2 * r)
    return (coef * dx, coef * dy)
def _geometry_field_vec(q_source: float, src: tuple[float, float], pt: tuple[float, float]) -> tuple[float, float]:
    dx = pt[0] - src[0]; dy = pt[1] - src[1]
    r2 = dx*dx + dy*dy
    if r2 <= 0:
        return (0.0, 0.0)
    r = math.sqrt(r2)
    coef = COULOMB_K * q_source / (r2 * r)
    return (coef * dx, coef * dy)
def _geometry_mag(v: tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])
def _geometry_cap_energy_output(question: str, Cq: Quantity | None, value_j: float) -> tuple[float, str, int | None]:
    q = _lower(question)
    if "mj" in q or "(mj" in q or "millijoule" in q:
        return value_j / 1e-3, "mJ", _rounding_places(question) or 2
    if "μj" in q or "uj" in q or "microjoule" in q:
        return value_j / 1e-6, "μJ", _rounding_places(question)
    if "nj" in q or "nanojoule" in q:
        return value_j / 1e-9, "nJ", _rounding_places(question) or 2
    cu = _norm_unit(Cq.unit if Cq else "")
    if cu == "pf":
        nJ = value_j / 1e-9
        if abs(nJ) >= 1000:
            return value_j / 1e-6, "μJ", _rounding_places(question) or 2
        return nJ, "nJ", _rounding_places(question) or 2
    if cu in {"μf", "uf"}:
        if abs(value_j) < 1e-3:
            return value_j / 1e-6, "μJ", _rounding_places(question)
        if abs(value_j) < 1:
            return value_j / 1e-3, "mJ", _rounding_places(question) or 2
    return value_j, "J", _rounding_places(question)
__all__ = [name for name in globals() if not name.startswith("__")]
def _priority_sci(x: float, sig: int = 3, spaced: bool = True) -> str:
    if x == 0 or not math.isfinite(x):
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / (10 ** exp)
    ms = f"{mant:.{sig}g}".rstrip("0").rstrip(".")
    return f"{ms} × 10^{exp}" if spaced else f"{ms}×10^{exp}"
def _priority_places(question: str, default: int | None = None) -> int | None:
    return _rounding_places(question) if _rounding_places(question) is not None else default
def _exact_result_by_question(question: str) -> SolverResult | None:
    return None
def _priority_eps(question: str) -> float:
    t = _normalize_text(question)
    q = t.lower()
    m = re.search(rf"(?:dielectric\s+(?:constant|medium)|relative\s+permittivity|ε(?:_r)?|epsilon)\s*(?:=|of|is)?\s*(?P<e>{VALUE_PATTERN})", t, flags=re.I)
    if m:
        try:
            return _parse_number(m.group('e'))
        except Exception:
            pass
    return 1.0
def _priority_len_after(label_regex: str, text: str) -> float | None:
    t = _normalize_text(text)
    m = re.search(rf"(?:{label_regex})\s*(?:=|is|of|being|being\s+)?\s*(?P<v>{VALUE_PATTERN})\s*(?P<u>km|cm|mm|m)\b", t, flags=re.I)
    if m:
        return _to_si(_parse_number(m.group('v')), m.group('u'))
    return None
def _priority_all_lengths(text: str) -> list[float]:
    vals=[]
    for qv in _get_distance_values(text):
        if qv.value > 0 and qv.value < 1e5:
            vals.append(qv.value)
    out=[]
    for v in vals:
        if not any(abs(v-x) <= max(1e-12, abs(v)*1e-9) for x in out):
            out.append(v)
    return out
def _priority_charge_map2(text: str) -> dict[str, float]:
    t = _normalize_text(text)
    d = _geometry_charge_map(text).copy()
    m = re.search(rf"q\s*1\s*=\s*-\s*q\s*2\s*=\s*(?P<v>{VALUE_PATTERN})\s*(?P<u>mC|μC|µC|uC|nC|pC|C)", t, flags=re.I)
    if m:
        val = _to_si(_parse_number(m.group('v')), m.group('u'))
        d['q1'] = val; d['q2'] = -val
    m = re.search(rf"q\s*1\s*=\s*q\s*2\s*=\s*(?P<v>{VALUE_PATTERN})\s*(?P<u>mC|μC|µC|uC|nC|pC|C)", t, flags=re.I)
    if m:
        val = _to_si(_parse_number(m.group('v')), m.group('u'))
        d['q1'] = val; d['q2'] = val
    m = re.search(rf"q(?:′|')\s*=\s*(?P<v>{VALUE_PATTERN})\s*(?P<u>mC|μC|µC|uC|nC|pC|C)", t, flags=re.I)
    if m:
        d['qprime'] = _to_si(_parse_number(m.group('v')), m.group('u'))
    return d
__all__ = [name for name in globals() if not name.startswith("__")]
__all__ = [name for name in globals() if not name.startswith("__")]
