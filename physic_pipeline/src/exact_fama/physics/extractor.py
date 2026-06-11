from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any
UNIT_MULTIPLIERS = {
    "v": 1.0,
    "volt": 1.0,
    "volts": 1.0,
    "a": 1.0,
    "ampere": 1.0,
    "amperes": 1.0,
    "ma": 1e-3,
    "ka": 1e3,
    "ohm": 1.0,
    "ohms": 1.0,
    "ω": 1.0,
    "kohm": 1e3,
    "kω": 1e3,
    "mohm": 1e6,
    "w": 1.0,
    "kw": 1e3,
    "mw": 1e-3,
    "j": 1.0,
    "f": 1.0,
    "mf": 1e-3,
    "uf": 1e-6,
    "μf": 1e-6,
    "microfarad": 1e-6,
    "microfarads": 1e-6,
    "nf": 1e-9,
    "pf": 1e-12,
    "c": 1.0,
    "mc": 1e-3,
    "μc": 1e-6,
    "uc": 1e-6,
    "nc": 1e-9,
    "pc": 1e-12,
    "n/c": 1.0,
    "v/m": 1.0,
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "km": 1e3,
    "ev": 1.602176634e-19,
    "t": 1.0,
    "tesla": 1.0,
}
SYMBOL_ALIASES = {
    "u": "V",
    "v": "V",
    "voltage": "V",
    "potential": "V",
    "potential_difference": "V",
    "i": "I",
    "current": "I",
    "r": "R",
    "resistance": "R",
    "p": "P",
    "power": "P",
    "c": "C",
    "capacitance": "C",
    "q": "Q",
    "charge": "Q",
    "e": "E",
    "electric_field": "Efield",
    "field": "Efield",
    "f": "Fforce",
    "force": "Fforce",
    "d": "D",
    "distance": "D",
}
UNIT_PATTERN = r"microfarads?|μF|uF|mF|nF|pF|F|kΩ|kω|Ω|ω|kohm|ohm|ohms|mA|kA|A|mW|kW|W|J|mC|μC|uC|nC|pC|C|N/C|V/m|km|cm|mm|m|eV|tesla|T|V|volts?"
VALUE_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*[eE]\s*[-+]?\d+)?"
@dataclass
class Quantity:
    symbol: str
    value: float
    unit: str
    raw: str
def normalize_unit(unit: str) -> str:
    return unit.strip().lower().replace("Ω", "ω").replace("µ", "μ")
def parse_number(value: str) -> float:
    return float(re.sub(r"\s+", "", value))
def to_si(value: float, unit: str) -> float:
    unit_norm = normalize_unit(unit)
    return value * UNIT_MULTIPLIERS.get(unit_norm, 1.0)
def infer_symbol(raw_sym: str, unit: str) -> str:
    raw = raw_sym.strip()
    if re.fullmatch(r"[Rr]\d+", raw):
        return raw.upper()
    if re.fullmatch(r"[Cc]\d+", raw):
        return raw.upper()
    unit_norm = normalize_unit(unit)
    if raw.lower() == "c" and unit_norm in {"c", "mc", "μc", "uc", "nc", "pc"}:
        return "Q"
    if raw.lower() in {"e", "field"} and unit_norm in {"n/c", "v/m"}:
        return "Efield"
    return SYMBOL_ALIASES.get(raw.lower(), raw.upper())
def extract_quantities(question: str) -> dict[str, Quantity]:
    text = question.replace("µ", "μ").replace("Ω", "Ω")
    quantities: dict[str, Quantity] = {}
    explicit_pattern = re.compile(
        rf"\b(?P<sym>[A-Za-z]\d*)\s*=\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>{UNIT_PATTERN})\b",
        flags=re.I,
    )
    for m in explicit_pattern.finditer(text):
        unit = m.group("unit")
        sym = infer_symbol(m.group("sym"), unit)
        value = to_si(parse_number(m.group("value")), unit)
        quantities[sym] = Quantity(sym, value, unit, m.group(0))
    named_pattern = re.compile(
        rf"\b(?P<name>voltage|current|resistance|power|capacitance|charge|force|distance|electric field|field|potential)\s+(?:is|=|of)?\s*(?P<value>{VALUE_PATTERN})\s*(?P<unit>{UNIT_PATTERN})\b",
        flags=re.I,
    )
    for m in named_pattern.finditer(text):
        name = m.group("name").lower().replace(" ", "_")
        unit = m.group("unit")
        sym = infer_symbol(name, unit)
        value = to_si(parse_number(m.group("value")), unit)
        quantities[sym] = Quantity(sym, value, unit, m.group(0))
    return quantities
def quantities_from_structured_parse(structured_parse: dict[str, Any] | None) -> dict[str, Quantity]:
    if not structured_parse:
        return {}
    out: dict[str, Quantity] = {}
    for item in structured_parse.get("quantities") or []:
        if not isinstance(item, dict):
            continue
        raw_symbol = str(item.get("symbol") or item.get("name") or "").strip()
        unit = str(item.get("unit") or item.get("unit_raw") or "").strip()
        if not raw_symbol:
            continue
        raw_text = str(item.get("raw_text") or item.get("raw") or f"{raw_symbol}={item.get('value')} {unit}")
        try:
            if item.get("value_si") not in (None, ""):
                value_si = float(item["value_si"])
            else:
                value_si = to_si(float(item.get("value")), unit)
        except Exception:
            continue
        sym = infer_symbol(raw_symbol, unit)
        out[sym] = Quantity(sym, value_si, unit, raw_text)
    return out
