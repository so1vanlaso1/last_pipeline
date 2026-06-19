from __future__ import annotations

import math
import re
from typing import Any

from ..common import SolverResult, VALUE_PATTERN, _make_result, _normalize_text, _parse_number

# Deterministic, formula-only solver for astrophysics / cosmology.
# No sample-id lookup, no question-text lookup, no answer-table lookup.

G_NEWTON = 6.67430e-11                 # m^3 kg^-1 s^-2
C_MS = 299_792_458.0                   # m/s
C_KMS = 299_792.458                    # km/s
AU_M = 1.495978707e11                  # m
PC_M = 3.085677581491367e16            # m
MPC_M = 1.0e6 * PC_M
M_SUN_KG = 1.98847e30                  # kg
R_SUN_M = 6.957e8                      # m
L_SUN_W = 3.828e26                     # W
T_SUN_K = 5772.0                       # K
M_EARTH_KG = 5.9722e24                 # kg
R_EARTH_M = 6.371e6                    # m
M_JUP_KG = 1.89813e27                  # kg
SIGMA_SB = 5.670374419e-8              # W m^-2 K^-4
WIEN_B = 2.897771955e-3                # m K
DAY_S = 86400.0
YR_S = 365.25 * DAY_S
GYR_S = 1.0e9 * YR_S
ARCSEC_PER_RAD = 206264.80624709636

_V = VALUE_PATTERN


def _clean(question: str) -> str:
    text = _normalize_text(question)
    text = re.sub(r"\[expected_unit:\s*[^\]]+\]", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _lower(question: str) -> str:
    return _clean(question).lower()


def _num(raw: str) -> float:
    return float(_parse_number(raw))


def _m(pattern: str, text: str, flags: int = re.I) -> re.Match[str] | None:
    return re.search(pattern, text, flags=flags)


def _expected_unit(question: str) -> str | None:
    m = re.search(r"\[expected_unit:\s*([^\]]+)\]", str(question or ""), flags=re.I)
    if not m:
        return None
    unit = m.group(1).strip()
    return None if unit.lower() in {"", "none", "null"} else unit


def _canonical_unit(u: str | None) -> str | None:
    if not u:
        return u
    raw = _normalize_text(u).strip()
    compact = raw.lower().replace(" ", "")
    aliases = {
        "meters": "m", "meter": "m", "m": "m", "km": "km", "kilometer": "km", "kilometers": "km",
        "s": "s", "sec": "s", "second": "s", "seconds": "s", "yr": "yr", "year": "yr", "years": "yr",
        "gyr": "Gyr", "au": "AU", "pc": "pc", "mpc": "Mpc",
        "m/s": "m/s", "ms^-1": "m/s", "km/s": "km/s",
        "m/s^2": "m/s^2", "m/s2": "m/s^2", "n": "N", "j": "J", "j/kg": "J/kg",
        "kg/m^3": "kg/m^3", "kg/m3": "kg/m^3", "m^3/s^2": "m^3/s^2", "m3/s2": "m^3/s^2",
        "w": "W", "w/m^2": "W/m^2", "w/m2": "W/m^2", "k": "K", "nm": "nm",
        "mag": "mag", "arcsec": "arcsec", "%": "%", "percent": "%",
        "m_sun": "M_sun", "msun": "M_sun", "solar masses": "M_sun",
        "l_sun": "L_sun", "lsun": "L_sun", "r_sun": "R_sun", "rsun": "R_sun",
        "r_earth": "R_earth", "rearth": "R_earth", "km/s/mpc": "km/s/Mpc",
    }
    return aliases.get(compact, raw)


_SCALE_TO_SI = {
    "m": 1.0, "km": 1e3, "AU": AU_M, "pc": PC_M, "Mpc": MPC_M,
    "s": 1.0, "yr": YR_S, "Gyr": GYR_S, "days": DAY_S, "day": DAY_S,
    "m/s": 1.0, "km/s": 1e3, "m/s^2": 1.0,
    "N": 1.0, "J": 1.0, "J/kg": 1.0, "kg/m^3": 1.0, "m^3/s^2": 1.0,
    "W": 1.0, "W/m^2": 1.0, "K": 1.0, "nm": 1e-9,
    "M_sun": M_SUN_KG, "R_sun": R_SUN_M, "L_sun": L_SUN_W,
    "M_earth": M_EARTH_KG, "R_earth": R_EARTH_M, "M_jup": M_JUP_KG,
    "arcsec": 1.0 / ARCSEC_PER_RAD, "%": 0.01, "mag": 1.0,
    "km/s/Mpc": 1000.0 / MPC_M,
}


def _convert(value: float, from_unit: str | None, to_unit: str | None) -> tuple[float, str | None]:
    fu = _canonical_unit(from_unit)
    tu = _canonical_unit(to_unit)
    if not tu or tu == fu:
        return value, fu
    if fu in _SCALE_TO_SI and tu in _SCALE_TO_SI:
        return value * _SCALE_TO_SI[fu] / _SCALE_TO_SI[tu], tu
    # Dimensionless aliases.
    if fu in {None, "", "1"} and tu in {"", "1"}:
        return value, tu
    return value, fu


def _fmt(x: float) -> str:
    if not math.isfinite(x):
        return "Uncertain"
    if abs(x) == 0:
        return "0"
    ax = abs(x)
    if ax >= 1.0e6 or ax < 1.0e-4:
        s = f"{x:.6e}"
        s = re.sub(r"e([+-])0+(\d+)$", r"e\1\2", s)
        return s
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s and s != "-0" else "0"


def _out(value: float, question: str, unit: str | None, formula: str, q: dict[str, Any] | None = None, conf: float = 0.985) -> SolverResult:
    eu = _expected_unit(question)
    value2, unit2 = _convert(value, unit, eu)
    answer = _fmt(value2)
    explanation = f"Apply the astrophysics/cosmology relation {formula}; computed answer = {answer}{(' ' + unit2) if unit2 else ''}."
    return _make_result(answer, unit2, explanation, formula, q or {}, confidence=conf)


def _guard(question: str) -> bool:
    t = _lower(question)
    keys = (
        "astronomical", "gravitational", "spherical body", "central mass", "planet", "star", "stellar", "solar",
        "asteroid", "moon", "dwarf planet", "surface gravity", "escape velocity", "blackbody",
        "m_sun", "r_sun", "l_sun", "m_earth", "r_earth", "m_jup", "black hole", "schwarzschild",
        "orbit", "orbital", "kepler", "vis-viva", "semi-major", "perihelion", "aphelion",
        "hill sphere", "roche", "parallax", "proper motion", "apparent magnitude", "absolute magnitude",
        "bolometric", "luminosity", "observed flux", "transit", "exoplanet", "angular diameter",
        "small-angle", "spectral line", "lambda", "redshift", "hubble", "lambda-cdm", "omega_m",
        "cmb", "light-years", "cosmological", "critical density", "universe", "galaxy",
    )
    return any(k in t for k in keys)


def _first_two_numbers(text: str) -> tuple[float, float] | None:
    vals = [_num(m.group(0)) for m in re.finditer(_V, text, flags=re.I)]
    if len(vals) >= 2:
        return vals[0], vals[1]
    return None


def solve_astrophysics_cosmology(question: str) -> SolverResult | None:
    if not _guard(question):
        return None
    s = _clean(question)
    t = s.lower()

    # 1) Newtonian gravitation: F = G m1 m2 / r^2
    # Order-independent Newtonian gravitation for generated prompts that state
    # M, m and r in varying order.
    if ("gravitational force" in t or "newtonian gravitational force" in t) and re.search(r"\bM\s*=", s) and re.search(r"\bm\s*=", s) and re.search(r"\br\s*=", s):
        mm = re.search(rf"\bM\s*=\s*(?P<M>{_V})\s*kg", s)
        sm = re.search(rf"\bm\s*=\s*(?P<m>{_V})\s*kg", s)
        rr = re.search(rf"\br\s*=\s*(?P<r>{_V})\s*(?:m|km|AU|pc|Mpc)", s)
        if mm and sm and rr:
            unit_match = re.search(rf"\br\s*=\s*{_V}\s*(?P<u>m|km|AU|pc|Mpc)", s, flags=re.I)
            ru = unit_match.group("u") if unit_match else "m"
            r_si, _ = _convert(_num(rr["r"]), ru, "m")
            if r_si != 0:
                return _out(G_NEWTON * _num(mm["M"]) * _num(sm["m"]) / r_si**2, question, "N", "F = G M m / r^2", {"M": _num(mm["M"]), "m": _num(sm["m"]), "r_m": r_si})

    m = _m(rf"m1\s*=\s*(?P<m1>{_V})\s*kg.*?m2\s*=\s*(?P<m2>{_V})\s*kg.*?r\s*=\s*(?P<r>{_V})\s*m.*?gravitational force", s)
    if m:
        m1, m2, r = _num(m["m1"]), _num(m["m2"]), _num(m["r"])
        return _out(G_NEWTON * m1 * m2 / r**2, question, "N", "F = G m1 m2 / r^2", {"m1": m1, "m2": m2, "r": r})

    # 2) Gravitational field / surface gravity: g = GM/r^2 or GM/R^2
    m = _m(rf"mass\s+M\s*=\s*(?P<M>{_V})\s*kg.*?gravitational field strength.*?distance\s+r\s*=\s*(?P<r>{_V})\s*m", s)
    if m:
        M, r = _num(m["M"]), _num(m["r"])
        return _out(G_NEWTON * M / r**2, question, "m/s^2", "g = GM/r^2", {"M": M, "r": r})

    m = _m(rf"mass\s+M\s*=\s*(?P<M>{_V})\s*kg\s+and\s+radius\s+R\s*=\s*(?P<R>{_V})\s*m.*?surface gravity", s)
    if m:
        M, R = _num(m["M"]), _num(m["R"])
        return _out(G_NEWTON * M / R**2, question, "m/s^2", "g = GM/R^2", {"M": M, "R": R})

    # 3) Weight on a planet: W = m GM/R^2
    m = _m(rf"object of mass\s+(?P<mass>{_V})\s*kg.*?planet with mass\s+(?P<M>{_V})\s*kg\s+and\s+radius\s+(?P<R>{_V})\s*m.*?weight", s)
    if m:
        mass, M, R = _num(m["mass"]), _num(m["M"]), _num(m["R"])
        return _out(mass * G_NEWTON * M / R**2, question, "N", "W = mGM/R^2", {"m": mass, "M": M, "R": R})

    # 4) Gravitational potential energy: U = -GMm/r
    # Order-independent celestial potential energy.  This prevents generic
    # mechanics mgh from handling orbital/central-mass prompts.
    if ("potential energy" in t or "gravitational potential" in t) and re.search(r"\bM\s*=", s) and re.search(r"\bm\s*=", s) and re.search(r"\br\s*=", s):
        mm = re.search(rf"\bM\s*=\s*(?P<M>{_V})\s*kg", s)
        sm = re.search(rf"\bm\s*=\s*(?P<m>{_V})\s*kg", s)
        rr = re.search(rf"\br\s*=\s*(?P<r>{_V})\s*(?P<u>m|km|AU|pc|Mpc)", s)
        if mm and sm and rr:
            r_si, _ = _convert(_num(rr["r"]), rr["u"], "m")
            if r_si != 0:
                return _out(-G_NEWTON * _num(mm["M"]) * _num(sm["m"]) / r_si, question, "J", "U = -GMm/r", {"M": _num(mm["M"]), "m": _num(sm["m"]), "r_m": r_si})

    # Gravitational potential per unit mass: phi = -GM/r.
    if ("gravitational potential" in t or "potential per unit mass" in t) and re.search(r"\bM\s*=", s) and re.search(r"\br\s*=", s) and not re.search(r"\bm\s*=", s):
        mm = re.search(rf"\bM\s*=\s*(?P<M>{_V})\s*kg", s)
        rr = re.search(rf"\br\s*=\s*(?P<r>{_V})\s*(?P<u>m|km|AU|pc|Mpc)", s)
        if mm and rr:
            r_si, _ = _convert(_num(rr["r"]), rr["u"], "m")
            if r_si != 0:
                return _out(-G_NEWTON * _num(mm["M"]) / r_si, question, "J/kg", "phi = -GM/r", {"M": _num(mm["M"]), "r_m": r_si})

    m = _m(rf"mass\s+m\s*=\s*(?P<mass>{_V})\s*kg.*?distance\s+r\s*=\s*(?P<r>{_V})\s*m.*?central mass\s+M\s*=\s*(?P<M>{_V})\s*kg.*?potential energy", s)
    if m:
        mass, r, M = _num(m["mass"]), _num(m["r"]), _num(m["M"])
        return _out(-G_NEWTON * M * mass / r, question, "J", "U = -GMm/r", {"m": mass, "M": M, "r": r})

    # 5) Escape velocity: v_esc = sqrt(2GM/r)
    m = _m(rf"distance\s+r\s*=\s*(?P<r>{_V})\s*m.*?mass\s+M\s*=\s*(?P<M>{_V})\s*kg.*?escape velocity", s)
    if not m:
        m = _m(rf"escape velocity.*?mass\s+M\s*=\s*(?P<M>{_V})\s*kg\s+and\s+radius\s+R\s*=\s*(?P<r>{_V})\s*m", s)
    if m:
        M, r = _num(m["M"]), _num(m["r"])
        return _out(math.sqrt(2.0 * G_NEWTON * M / r), question, "m/s", "v_esc = sqrt(2GM/r)", {"M": M, "r": r})

    # 6) Circular orbital speed and period.
    m = _m(rf"circular orbit of radius\s+r\s*=\s*(?P<r>{_V})\s*m.*?(?:central mass|body of mass|mass)\s+M\s*=\s*(?P<M>{_V})\s*kg.*?orbital speed", s)
    if m:
        r, M = _num(m["r"]), _num(m["M"])
        return _out(math.sqrt(G_NEWTON * M / r), question, "m/s", "v = sqrt(GM/r)", {"M": M, "r": r})

    m = _m(rf"circular orbit of radius\s+r\s*=\s*(?P<r>{_V})\s*m.*?mass\s+M\s*=\s*(?P<M>{_V})\s*kg.*?orbital period", s)
    if not m:
        m = _m(rf"orbital period.*?circular orbit of radius\s+(?P<r>{_V})\s*m\s+around a mass\s+(?P<M>{_V})\s*kg", s)
    if m:
        r, M = _num(m["r"]), _num(m["M"])
        return _out(2.0 * math.pi * math.sqrt(r**3 / (G_NEWTON * M)), question, "s", "T = 2*pi*sqrt(r^3/GM)", {"M": M, "r": r})

    # 7) Kepler's third law in solar units: M = a^3/T^2 and a = (M T^2)^(1/3).
    m = _m(rf"semi-major axis\s+a\s*=\s*(?P<a>{_V})\s*AU\s+and\s+period\s+T\s*=\s*(?P<T>{_V})\s*yr.*?star'?s mass", s)
    if m:
        a, T = _num(m["a"]), _num(m["T"])
        return _out(a**3 / T**2, question, "M_sun", "M/M_sun = a_AU^3/T_yr^2", {"a_AU": a, "T_yr": T})

    m = _m(rf"star of mass\s+(?P<M>{_V})\s*M_sun\s+with period\s+(?P<T>{_V})\s*yr.*?semi-major axis", s)
    if m:
        M, T = _num(m["M"]), _num(m["T"])
        return _out((M * T**2) ** (1.0 / 3.0), question, "AU", "a_AU = (M_sun_units*T_yr^2)^(1/3)", {"M_Msun": M, "T_yr": T})

    # 8) Vis-viva equation in AU/M_sun -> km/s.
    m = _m(rf"star of mass\s+(?P<M>{_V})\s*M_sun.*?semi-major axis\s+a\s*=\s*(?P<a>{_V})\s*AU.*?distance\s+r\s*=\s*(?P<r>{_V})\s*AU.*?vis-viva", s)
    if m:
        M, a, r = _num(m["M"]), _num(m["a"]), _num(m["r"])
        mu = G_NEWTON * M * M_SUN_KG
        v = math.sqrt(mu * (2.0 / (r * AU_M) - 1.0 / (a * AU_M))) / 1000.0
        return _out(v, question, "km/s", "v = sqrt(GM(2/r - 1/a))", {"M_Msun": M, "a_AU": a, "r_AU": r})

    # 9) Ellipse perihelion/aphelion.
    m = _m(rf"semi-major axis\s+a\s*=\s*(?P<a>{_V})\s*AU\s+and\s+eccentricity\s+e\s*=\s*(?P<e>{_V}).*?perihelion", s)
    if m:
        a, e = _num(m["a"]), _num(m["e"])
        return _out(a * (1.0 - e), question, "AU", "q = a(1-e)", {"a_AU": a, "e": e})
    m = _m(rf"semi-major axis\s+a\s*=\s*(?P<a>{_V})\s*AU\s+and\s+eccentricity\s+e\s*=\s*(?P<e>{_V}).*?aphelion", s)
    if m:
        a, e = _num(m["a"]), _num(m["e"])
        return _out(a * (1.0 + e), question, "AU", "Q = a(1+e)", {"a_AU": a, "e": e})

    # 10) Specific orbital energy and gravitational parameter.
    m = _m(rf"around mass\s+M\s*=\s*(?P<M>{_V})\s*kg\s+with semi-major axis\s+a\s*=\s*(?P<a>{_V})\s*m.*?specific orbital energy", s)
    if m:
        M, a = _num(m["M"]), _num(m["a"])
        return _out(-G_NEWTON * M / (2.0 * a), question, "J/kg", "epsilon = -GM/(2a)", {"M": M, "a": a})
    m = _m(rf"radius\s+r\s*=\s*(?P<r>{_V})\s*m\s+and\s+orbital speed\s+v\s*=\s*(?P<v>{_V})\s*m/s.*?(?:gravitational parameter|\bmu\b)", s)
    if m:
        r, v = _num(m["r"]), _num(m["v"])
        return _out(v**2 * r, question, "m^3/s^2", "mu = v^2 r", {"r": r, "v": v})

    # 11) Hill sphere and Roche limit.
    m = _m(rf"planet of mass\s+(?P<mp>{_V})\s*kg\s+orbits a star of mass\s+(?P<ms>{_V})\s*M_sun\s+at\s+a\s*=\s*(?P<a>{_V})\s*AU.*?Hill sphere", s)
    if m:
        mp, ms, a = _num(m["mp"]), _num(m["ms"]), _num(m["a"])
        return _out(a * (mp / (3.0 * ms * M_SUN_KG)) ** (1.0 / 3.0), question, "AU", "r_H = a(m/3M)^(1/3)", {"m": mp, "M_Msun": ms, "a_AU": a})
    m = _m(rf"Roche distance.*?primary radius\s+R\s*=\s*(?P<R>{_V})\s*m,?\s+primary density\s+(?P<rho1>{_V})\s*kg/m\^?3,?\s+and\s+satellite density\s+(?P<rho2>{_V})\s*kg/m\^?3", s)
    if m:
        R, rho1, rho2 = _num(m["R"]), _num(m["rho1"]), _num(m["rho2"])
        return _out(2.44 * R * (rho1 / rho2) ** (1.0 / 3.0), question, "m", "d_Roche = 2.44 R_primary (rho_primary/rho_satellite)^(1/3)", {"R": R, "rho_primary": rho1, "rho_sat": rho2})

    # 12) Mean density of spherical bodies and exoplanets.
    m = _m(rf"astronomical object has mass\s+M\s*=\s*(?P<M>{_V})\s*kg\s+and\s+radius\s+R\s*=\s*(?P<R>{_V})\s*m.*?average density", s)
    if m:
        M, R = _num(m["M"]), _num(m["R"])
        return _out(M / ((4.0 / 3.0) * math.pi * R**3), question, "kg/m^3", "rho = M/(4*pi*R^3/3)", {"M": M, "R": R})
    m = _m(rf"exoplanet has mass\s+M\s*=\s*(?P<M>{_V})\s*M_earth\s+and\s+radius\s+R\s*=\s*(?P<R>{_V})\s*R_earth.*?mean density", s)
    if m:
        M, R = _num(m["M"]), _num(m["R"])
        rho = M * M_EARTH_KG / ((4.0 / 3.0) * math.pi * (R * R_EARTH_M) ** 3)
        return _out(rho, question, "kg/m^3", "rho = M/(4*pi*R^3/3)", {"M_Mearth": M, "R_Rearth": R})

    # 13) Stellar luminosity, radius, flux, magnitudes.
    m = _m(rf"star has radius\s+R\s*=\s*(?P<R>{_V})\s*R_sun\s+and\s+effective temperature\s+T\s*=\s*(?P<T>{_V})\s*K.*?luminosity in solar", s)
    if m:
        R, T = _num(m["R"]), _num(m["T"])
        return _out(R**2 * (T / T_SUN_K) ** 4, question, "L_sun", "L/L_sun = (R/R_sun)^2 (T/T_sun)^4", {"R_Rsun": R, "T": T})
    m = _m(rf"star has luminosity\s+L\s*=\s*(?P<L>{_V})\s*L_sun\s+and\s+effective temperature\s+T\s*=\s*(?P<T>{_V})\s*K.*?radius in solar", s)
    if m:
        L, T = _num(m["L"]), _num(m["T"])
        return _out(math.sqrt(L) / (T / T_SUN_K) ** 2, question, "R_sun", "R/R_sun = sqrt(L/L_sun)/(T/T_sun)^2", {"L_Lsun": L, "T": T})
    m = _m(rf"star has radius\s+R\s*=\s*(?P<R>{_V})\s*m\s+and\s+effective temperature\s+T\s*=\s*(?P<T>{_V})\s*K.*?luminosity as a blackbody", s)
    if m:
        R, T = _num(m["R"]), _num(m["T"])
        return _out(4.0 * math.pi * R**2 * SIGMA_SB * T**4, question, "W", "L = 4*pi*R^2*sigma*T^4", {"R": R, "T": T})
    m = _m(rf"luminosity\s+L\s*=\s*(?P<L>{_V})\s*L_sun\s+and\s+is at distance\s+d\s*=\s*(?P<d>{_V})\s*pc.*?observed flux", s)
    if m:
        L, d = _num(m["L"]), _num(m["d"])
        return _out(L * L_SUN_W / (4.0 * math.pi * (d * PC_M) ** 2), question, "W/m^2", "F = L/(4*pi*d^2)", {"L_Lsun": L, "d_pc": d})
    m = _m(rf"source has luminosity\s+L\s*=\s*(?P<L>{_V})\s*L_sun\s+and\s+observed flux\s+F\s*=\s*(?P<F>{_V})\s*W/m\^?2.*?distance", s)
    if m:
        L, F = _num(m["L"]), _num(m["F"])
        return _out(math.sqrt(L * L_SUN_W / (4.0 * math.pi * F)) / PC_M, question, "pc", "d = sqrt(L/(4*pi*F))", {"L_Lsun": L, "F": F})
    m = _m(rf"identical stars.*?d1\s*=\s*(?P<d1>{_V})\s*pc\s+and\s+d2\s*=\s*(?P<d2>{_V})\s*pc.*?flux ratio\s+F1/F2", s)
    if m:
        d1, d2 = _num(m["d1"]), _num(m["d2"])
        return _out((d2 / d1) ** 2, question, "", "F1/F2 = (d2/d1)^2", {"d1_pc": d1, "d2_pc": d2})
    m = _m(rf"absolute magnitude\s+M\s*=\s*(?P<M>{_V})\s+and\s+distance\s+d\s*=\s*(?P<d>{_V})\s*pc.*?apparent magnitude", s)
    if m:
        M, d = _num(m["M"]), _num(m["d"])
        return _out(M + 5.0 * math.log10(d / 10.0), question, "mag", "m = M + 5 log10(d/10 pc)", {"M": M, "d_pc": d})
    m = _m(rf"apparent magnitude\s+m\s*=\s*(?P<m>{_V})\s+and\s+absolute magnitude\s+M\s*=\s*(?P<M>{_V}).*?distance", s)
    if m:
        app, abs_m = _num(m["m"]), _num(m["M"])
        return _out(10.0 ** ((app - abs_m + 5.0) / 5.0), question, "pc", "d = 10^((m-M+5)/5)", {"m": app, "M": abs_m})
    m = _m(rf"observed fluxes\s+F1\s*=\s*(?P<F1>{_V})\s*W/m\^?2\s+and\s+F2\s*=\s*(?P<F2>{_V})\s*W/m\^?2.*?m2\s*-\s*m1", s)
    if m:
        F1, F2 = _num(m["F1"]), _num(m["F2"])
        return _out(-2.5 * math.log10(F2 / F1), question, "mag", "m2-m1 = -2.5 log10(F2/F1)", {"F1": F1, "F2": F2})
    m = _m(rf"luminosity\s+L\s*=\s*(?P<L>{_V})\s*L_sun.*?bolometric absolute magnitude.*?Mbol_sun\s*=\s*(?P<Mbol>{_V})", s)
    if m:
        L, Mbol = _num(m["L"]), _num(m["Mbol"])
        return _out(Mbol - 2.5 * math.log10(L), question, "mag", "Mbol = Mbol_sun - 2.5 log10(L/L_sun)", {"L_Lsun": L, "Mbol_sun": Mbol})

    # 14) Proper motion, transit and exoplanet equilibrium/RV.
    m = _m(rf"proper motion\s+mu\s*=\s*(?P<mu>{_V})\s*arcsec/yr\s+and\s+distance\s+d\s*=\s*(?P<d>{_V})\s*pc.*?transverse velocity", s)
    if m:
        mu, d = _num(m["mu"]), _num(m["d"])
        return _out(4.74047 * mu * d, question, "km/s", "v_t = 4.74047 mu d", {"mu_arcsec_per_yr": mu, "d_pc": d})
    m = _m(rf"radius\s+Rp\s*=\s*(?P<Rp>{_V})\s*R_earth\s+transits a star of radius\s+Rs\s*=\s*(?P<Rs>{_V})\s*R_sun.*?transit depth", s)
    if m:
        Rp, Rs = _num(m["Rp"]), _num(m["Rs"])
        depth_percent = ((Rp * R_EARTH_M) / (Rs * R_SUN_M)) ** 2 * 100.0
        return _out(depth_percent, question, "%", "transit depth = (Rp/Rs)^2", {"Rp_Rearth": Rp, "Rs_Rsun": Rs})
    m = _m(rf"fractional depth\s+delta\s*=\s*(?P<delta>{_V})\s+around a star of radius\s+Rs\s*=\s*(?P<Rs>{_V})\s*R_sun.*?planet radius", s)
    if m:
        delta, Rs = _num(m["delta"]), _num(m["Rs"])
        Rp_rearth = math.sqrt(delta) * Rs * R_SUN_M / R_EARTH_M
        return _out(Rp_rearth, question, "R_earth", "Rp = Rs sqrt(delta)", {"delta": delta, "Rs_Rsun": Rs})
    m = _m(rf"orbits at\s+a\s*=\s*(?P<a>{_V})\s*AU\s+from a star of luminosity\s+L\s*=\s*(?P<L>{_V})\s*L_sun\s+and\s+has Bond albedo\s+A\s*=\s*(?P<A>{_V}).*?equilibrium temperature", s)
    if m:
        a, L, A = _num(m["a"]), _num(m["L"]), _num(m["A"])
        T_eq = (L * L_SUN_W * (1.0 - A) / (16.0 * math.pi * SIGMA_SB * (a * AU_M) ** 2)) ** 0.25
        return _out(T_eq, question, "K", "T_eq = [L(1-A)/(16*pi*sigma*a^2)]^(1/4)", {"a_AU": a, "L_Lsun": L, "A": A})
    m = _m(rf"Mp\s*=\s*(?P<Mp>{_V})\s*M_jup,?\s+period\s+P\s*=\s*(?P<P>{_V})\s*days,?\s+and\s+star mass\s+M\s*=\s*(?P<M>{_V})\s*M_sun.*?radial-velocity", s)
    if m:
        Mp, P_days, M = _num(m["Mp"]), _num(m["P"]), _num(m["M"])
        K = (2.0 * math.pi * G_NEWTON / (P_days * DAY_S)) ** (1.0 / 3.0) * (Mp * M_JUP_KG) / (M * M_SUN_KG) ** (2.0 / 3.0)
        return _out(K, question, "m/s", "K = (2*pi*G/P)^(1/3) Mp/Mstar^(2/3)", {"Mp_Mjup": Mp, "P_days": P_days, "M_Msun": M})

    # 15) Small-angle approximation.
    m = _m(rf"physical diameter\s+D\s*=\s*(?P<D>{_V})\s*m\s+is at distance\s+d\s*=\s*(?P<d>{_V})\s*m.*?angular diameter", s)
    if m:
        D, d = _num(m["D"]), _num(m["d"])
        return _out((D / d) * ARCSEC_PER_RAD, question, "arcsec", "theta_arcsec = (D/d)*206265", {"D": D, "d": d})
    m = _m(rf"object at distance\s+d\s*=\s*(?P<d>{_V})\s*pc\s+has angular size\s+theta\s*=\s*(?P<th>{_V})\s*arcsec.*?physical size", s)
    if m:
        d, th = _num(m["d"]), _num(m["th"])
        return _out(d * th, question, "AU", "D_AU = theta_arcsec * d_pc", {"d_pc": d, "theta_arcsec": th})

    # 16) Wavelength redshift and Hubble law.
    m = _m(rf"rest wavelength\s+lambda0\s*=\s*(?P<l0>{_V})\s*nm\s+and\s+observed wavelength\s+lambda_obs\s*=\s*(?P<lo>{_V})\s*nm.*?redshift", s)
    if m:
        l0, lo = _num(m["l0"]), _num(m["lo"])
        return _out((lo - l0) / l0, question, "", "z = (lambda_obs-lambda0)/lambda0", {"lambda0_nm": l0, "lambda_obs_nm": lo})
    m = _m(rf"rest wavelength\s+lambda0\s*=\s*(?P<l0>{_V})\s*nm.*?redshift\s+z\s*=\s*(?P<z>{_V}).*?observed wavelength", s)
    if m:
        l0, z = _num(m["l0"]), _num(m["z"])
        return _out(l0 * (1.0 + z), question, "nm", "lambda_obs = lambda0(1+z)", {"lambda0_nm": l0, "z": z})
    m = _m(rf"Hubble\'?s law with\s+H0\s*=\s*(?P<H>{_V})\s*km/s/Mpc.*?distance\s+d\s*=\s*(?P<d>{_V})\s*Mpc", s)
    if m:
        H, d = _num(m["H"]), _num(m["d"])
        return _out(H * d, question, "km/s", "v = H0 d", {"H0": H, "d_Mpc": d})
    m = _m(rf"recession velocity\s+v\s*=\s*(?P<v>{_V})\s*km/s.*?H0\s*=\s*(?P<H>{_V})\s*km/s/Mpc.*?distance", s)
    if m:
        v, H = _num(m["v"]), _num(m["H"])
        return _out(v / H, question, "Mpc", "d = v/H0", {"v_km_s": v, "H0": H})
    m = _m(rf"recession speed\s+v\s*=\s*(?P<v>{_V})\s*km/s.*?redshift\s+z\s*=\s*v/c", s)
    if m:
        v = _num(m["v"])
        return _out(v / C_KMS, question, "", "z = v/c", {"v_km_s": v})

    # 17) Lambda-CDM and critical/matter density.
    m = _m(rf"Lambda-CDM model with\s+H0\s*=\s*(?P<H>{_V})\s*km/s/Mpc,?\s+Omega_m\s*=\s*(?P<Om>{_V}),?\s+and\s+Omega_Lambda\s*=\s*(?P<Ol>{_V}),?\s+calculate H\(z\) at z\s*=\s*(?P<z>{_V})", s)
    if m:
        H, Om, Ol, z = _num(m["H"]), _num(m["Om"]), _num(m["Ol"]), _num(m["z"])
        return _out(H * math.sqrt(Om * (1.0 + z) ** 3 + Ol), question, "km/s/Mpc", "H(z)=H0 sqrt(Omega_m(1+z)^3+Omega_Lambda)", {"H0": H, "Omega_m": Om, "Omega_Lambda": Ol, "z": z})
    m = _m(rf"H0\s*=\s*(?P<H>{_V})\s*km/s/Mpc\s+and\s+matter density parameter\s+Omega_m\s*=\s*(?P<Om>{_V}).*?present matter density", s)
    if m:
        H, Om = _num(m["H"]), _num(m["Om"])
        H_si = H * 1000.0 / MPC_M
        return _out(Om * 3.0 * H_si**2 / (8.0 * math.pi * G_NEWTON), question, "kg/m^3", "rho_m0 = Omega_m*3H0^2/(8*pi*G)", {"H0": H, "Omega_m": Om})
    m = _m(rf"H0\s*=\s*(?P<H>{_V})\s*km/s/Mpc.*?critical density", s)
    if m:
        H = _num(m["H"])
        H_si = H * 1000.0 / MPC_M
        return _out(3.0 * H_si**2 / (8.0 * math.pi * G_NEWTON), question, "kg/m^3", "rho_c = 3H0^2/(8*pi*G)", {"H0": H})
    m = _m(rf"H0\s*=\s*(?P<H>{_V})\s*km/s/Mpc.*?Hubble time\s+1/H0", s)
    if m:
        H = _num(m["H"])
        H_si = H * 1000.0 / MPC_M
        return _out((1.0 / H_si) / GYR_S, question, "Gyr", "t_H = 1/H0", {"H0": H})

    # 18) Distances/parallax/light-travel time/cosmological scale/CMB.
    m = _m(rf"distance of\s+(?P<d>{_V})\s*light-years.*?light-travel time", s)
    if m:
        d = _num(m["d"])
        return _out(d, question, "yr", "t_years = distance_lightyears", {"distance_ly": d})
    m = _m(rf"redshift\s+z\s*=\s*(?P<z>{_V}).*?scale factor\s+a.*?a0\s*=\s*(?P<a0>{_V})", s)
    if m:
        z, a0 = _num(m["z"]), _num(m["a0"])
        return _out(a0 / (1.0 + z), question, "", "a = a0/(1+z)", {"z": z, "a0": a0})
    m = _m(rf"present CMB temperature is\s+(?P<T0>{_V})\s*K.*?redshift\s+z\s*=\s*(?P<z>{_V})", s)
    if m:
        T0, z = _num(m["T0"]), _num(m["z"])
        return _out(T0 * (1.0 + z), question, "K", "T_CMB(z) = T0(1+z)", {"T0": T0, "z": z})
    m = _m(rf"star is at distance\s+d\s*=\s*(?P<d>{_V})\s*pc.*?parallax angle", s)
    if m:
        d = _num(m["d"])
        return _out(1.0 / d, question, "arcsec", "p_arcsec = 1/d_pc", {"d_pc": d})
    m = _m(rf"parallax\s+p\s*=\s*(?P<p>{_V})\s*arcsec.*?distance", s)
    if m:
        p = _num(m["p"])
        return _out(1.0 / p, question, "pc", "d_pc = 1/p_arcsec", {"p_arcsec": p})

    # 19) Black holes and Wien's law.
    m = _m(rf"black hole'?s Schwarzschild radius as a sphere radius.*?mass\s+M\s*=\s*(?P<M>{_V})\s*M_sun.*?average density", s)
    if m:
        M_solar = _num(m["M"])
        M_kg = M_solar * M_SUN_KG
        Rs = 2.0 * G_NEWTON * M_kg / C_MS**2
        return _out(M_kg / ((4.0 / 3.0) * math.pi * Rs**3), question, "kg/m^3", "rho = M/(4*pi*R_s^3/3)", {"M_Msun": M_solar, "R_s": Rs})
    m = _m(rf"Schwarzschild radius.*?mass\s+M\s*=\s*(?P<M>{_V})\s*M_sun", s)
    if m:
        M = _num(m["M"])
        return _out(2.0 * G_NEWTON * M * M_SUN_KG / C_MS**2 / 1000.0, question, "km", "R_s = 2GM/c^2", {"M_Msun": M})
    m = _m(rf"Schwarzschild radius.*?mass\s+M\s*=\s*(?P<M>{_V})\s*kg", s)
    if m:
        M = _num(m["M"])
        return _out(2.0 * G_NEWTON * M / C_MS**2, question, "m", "R_s = 2GM/c^2", {"M": M})
    m = _m(rf"blackbody at temperature\s+T\s*=\s*(?P<T>{_V})\s*K.*?Wien", s)
    if m:
        T = _num(m["T"])
        return _out(WIEN_B / T * 1.0e9, question, "nm", "lambda_max = b/T", {"T": T})

    # 20) Earth circular orbit altitude from period.
    m = _m(rf"circular orbit around Earth.*?period\s+T\s*=\s*(?P<T>{_V})\s*min.*?Earth'?s mass and radius", s)
    if m:
        T_min = _num(m["T"])
        r = (G_NEWTON * M_EARTH_KG * (T_min * 60.0) ** 2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)
        return _out((r - R_EARTH_M) / 1000.0, question, "km", "h = (GM_E T^2/4pi^2)^(1/3)-R_E", {"T_min": T_min})

    return None
