from __future__ import annotations
from pathlib import Path
import re

path = Path("src/exact_fama/physics/physics_solvers/domains/physics_engine.py")
text = path.read_text(encoding="utf-8")
backup = path.with_suffix(path.suffix + ".bak_lc_robust")
backup.write_text(text, encoding="utf-8")

new_func = r'''def _solve_clean_lc_resonance_design_hotfix(question: str) -> SolverResult | None:
    """Robust LC resonance design fallback: C=1/((2πf)^2L), L=1/((2πf)^2C)."""
    q = _lower(question)
    t = _normalize_text(question)
    if not any(k in q for k in ["lc", "resonate", "resonance", "resonant", "f0", "oscillator", "oscillating circuit"]):
        return None

    freq_unit = r"kHz|Hz"
    ind_unit = r"mH|μH|µH|uH|H"
    cap_unit = r"microfarads?|μF|µF|uF|mF|nF|pF|F"

    def _first_frequency() -> float | None:
        for pat in [
            rf"\bf\s*_?\s*0\s*=\s*(?P<v>{VALUE_PATTERN})\s*(?P<u>{freq_unit})\b",
            rf"(?:resonant\s+frequency|resonance\s+frequency|frequency|resonate\s+at|resonates\s+at|must\s+resonate\s+at|at)\s*(?:f\s*_?\s*0\s*)?(?:=|is|of)?\s*(?P<v>{VALUE_PATTERN})\s*(?P<u>{freq_unit})\b",
            rf"(?P<v>{VALUE_PATTERN})\s*(?P<u>{freq_unit})\b",
        ]:
            m = re.search(pat, t, flags=re.I)
            if m:
                return _to_si(_parse_number(m.group("v")), m.group("u"))
        return None

    def _first_inductance() -> float | None:
        for pat in [
            rf"\bL\s*=\s*(?P<v>{VALUE_PATTERN})\s*(?P<u>{ind_unit})\b",
            rf"(?:inductance|inductor)\s*(?:L\s*)?(?:of|is|=|with|has|uses|using)?\s*(?P<v>{VALUE_PATTERN})\s*(?P<u>{ind_unit})\b",
            rf"uses\s+an\s+inductor\s+of\s*(?P<v>{VALUE_PATTERN})\s*(?P<u>{ind_unit})\b",
        ]:
            m = re.search(pat, t, flags=re.I)
            if m:
                return _to_si(_parse_number(m.group("v")), m.group("u"))
        vals = _eng_inductance_values(question)
        return vals[0].value if vals else None

    def _first_capacitance() -> float | None:
        for pat in [
            rf"\bC\s*=\s*(?P<v>{VALUE_PATTERN})\s*(?P<u>{cap_unit})\b",
            rf"(?:capacitance|capacitor)\s*(?:C\s*)?(?:of|is|=|with|has|uses|using)?\s*(?P<v>{VALUE_PATTERN})\s*(?P<u>{cap_unit})\b",
        ]:
            m = re.search(pat, t, flags=re.I)
            if m:
                return _to_si(_parse_number(m.group("v")), m.group("u"))
        vals = _eng_cap_values(question)
        return vals[0].value if vals else None

    f = _first_frequency()
    L = _first_inductance()
    C_given = _first_capacitance()

    asks_C = bool(f and L and (
        "required capacitance" in q or "capacitance required" in q or "what capacitance" in q or
        "capacitor value" in q or re.search(r"(?:calculate|find|determine)\s+(?:the\s+)?(?:required\s+)?(?:capacitance|capacitor\s+value|c\b)", q)
    ))
    if asks_C:
        C = 1.0 / (((2.0 * math.pi * f) ** 2) * L)
        unit = (_expected_unit(question) or ("μF" if re.search(r"μf|µf|uf|microfarad", q, flags=re.I) else "F")).replace("µ", "μ")
        val = _scale_to_unit(C, unit)
        places = _rounding_places(question)
        ans = _eng_fmt(val, places if places is not None else 4 if abs(val) >= 1 else 6)
        return _result(ans, unit, "Use the LC resonance relation and solve for capacitance.", "C=1/((2πf)^2L)", {"C": C, "L": L, "f": f}, conf=0.95)

    asks_L = bool(f and C_given and (
        "required inductance" in q or "inductance needed" in q or "what inductance" in q or
        re.search(r"(?:calculate|find|determine)\s+(?:the\s+)?(?:required\s+)?(?:inductance|l\b)", q)
    ))
    if asks_L:
        Lcalc = 1.0 / (((2.0 * math.pi * f) ** 2) * C_given)
        unit = (_expected_unit(question) or ("mH" if Lcalc < 1 else "H")).replace("µ", "μ")
        val = _scale_to_unit(Lcalc, unit)
        places = _rounding_places(question)
        ans = _eng_fmt(val, places if places is not None else 4 if abs(val) >= 1 else 6)
        return _result(ans, unit, "Use the LC resonance relation and solve for inductance.", "L=1/((2πf)^2C)", {"L": Lcalc, "C": C_given, "f": f}, conf=0.95)

    return None
'''

pattern = r"def _solve_clean_lc_resonance_design_hotfix\(question: str\) -> SolverResult \| None:\n.*?\n(?=def solve_clean_physics_engine\()"
new_text, n = re.subn(pattern, new_func + "\n", text, count=1, flags=re.S)
if n != 1:
    raise SystemExit("Could not replace _solve_clean_lc_resonance_design_hotfix; pattern not found.")
path.write_text(new_text, encoding="utf-8")
print(f"Patched {path}")
print(f"Backup saved to {backup}")
