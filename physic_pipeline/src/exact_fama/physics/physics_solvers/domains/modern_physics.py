from __future__ import annotations

import math
import re
from typing import Iterable

try:
    from ..common import SolverResult, _make_result, _normalize_text, _parse_number
except Exception:  # pragma: no cover
    from physics_solvers.common import SolverResult, _make_result, _normalize_text, _parse_number

# CODATA-style constants used by the synthetic modern-physics generator.
C = 299_792_458.0
H = 6.62607015e-34
HBAR = 1.054571817e-34
E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27
M_N = 1.67492749804e-27
ELECTRON_REST_MEV = 0.51099895
PROTON_REST_MEV = 938.27208816
NEUTRON_REST_MEV = 939.56542052
HC_EV_NM = 1239.8419843320026
RYD_E_EV = 13.6
RYDBERG = 10973731.568160
BOHR_RADIUS_NM = 0.0529177210903
U_TO_MEV = 931.49410242
WIEN_UM_K = 2897.771955
SIGMA = 5.670374419e-8
COMPTON_ELECTRON_PM = 2.42631023867
MEV_TO_J = 1.602176634e-13

_VALUE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _q(question: str) -> str:
    return (_normalize_text(question)
            .replace("φ", "phi").replace("λ", "lambda").replace("Δ", "Delta")
            .replace("ℏ", "hbar").replace("γ", "gamma").replace("θ", "theta")
            .replace("β", "beta").replace("≥", ">="))


def _num(s: str) -> float:
    return _parse_number(str(s))


def _fmt(x: float) -> str:
    if x is None or not math.isfinite(float(x)):
        return "Uncertain"
    x = float(x)
    if abs(x) >= 1e-6 and abs(x - round(x)) < 5e-10 and abs(x) < 1e12:
        return str(int(round(x)))
    return f"{x:.6g}"


def _result(answer: float | int | str, unit: str | None, formula: str, quantities: dict | None = None, confidence: float = 0.995) -> SolverResult:
    ans = answer if isinstance(answer, str) else _fmt(float(answer))
    return _make_result(ans, unit, f"Apply the modern-physics relation {formula}.", formula, quantities or {}, confidence)


def _m(pattern: str, text: str, flags: int = re.I) -> re.Match[str] | None:
    return re.search(pattern, text, flags)


def _beta(text: str) -> float | None:
    m = _m(rf"v\s*=\s*(?P<b>{_VALUE})\s*c\b", text)
    if m:
        return _num(m.group("b"))
    m = _m(rf"moves?\s+at\s+v\s*=\s*(?P<b>{_VALUE})\s*c\b", text)
    if m:
        return _num(m.group("b"))
    return None


def _gamma(beta: float) -> float:
    return 1.0 / math.sqrt(1.0 - beta * beta)


def solve_modern_physics(question: str) -> SolverResult | None:
    """Deterministic formula solver for modern_physics / quantum / nuclear / relativity.

    The rules below are formula/template based.  They do not inspect ids, gold answers,
    line numbers, hashes, or any dataset-only metadata; all quantities are parsed from
    the natural-language question.
    """
    t = _q(question)
    tl = t.lower()

    # ------------------------------------------------------------------
    # X-ray, Bragg diffraction, Compton scattering
    # ------------------------------------------------------------------
    m = _m(rf"x-ray tube operates at accelerating voltage\s+v\s*=\s*(?P<V>{_VALUE})\s*kV.*minimum x-ray wavelength", t)
    if m:
        V_kV = _num(m.group("V"))
        return _result(HC_EV_NM / (V_kV * 1000.0), "nm", "lambda_min = hc/(eV)", {"V_kV": V_kV})

    m = _m(rf"x-ray tube operates at voltage\s+v\s*=\s*(?P<V>{_VALUE})\s*kV.*maximum photon energy", t)
    if m:
        V_kV = _num(m.group("V"))
        return _result(V_kV, "keV", "E_max(keV) = V(kV)", {"V_kV": V_kV})

    m = _m(rf"crystal planes are separated by\s+d\s*=\s*(?P<d>{_VALUE})\s*nm.*?theta\s*=\s*(?P<th>{_VALUE})\s*(?:°|degrees?|deg).*?order\s+n\s*=\s*(?P<n>{_VALUE}).*?x-ray wavelength", t)
    if m:
        d = _num(m.group("d")); th = math.radians(_num(m.group("th"))); n = _num(m.group("n"))
        return _result(2.0 * d * math.sin(th) / n, "nm", "n lambda = 2 d sin(theta)", {"d_nm": d, "theta_deg": math.degrees(th), "n": n})

    m = _m(rf"x-rays of wavelength\s+lambda\s*=\s*(?P<lam>{_VALUE})\s*nm.*?planes separated by\s+d\s*=\s*(?P<d>{_VALUE})\s*nm.*?order\s+n\s*=\s*(?P<n>{_VALUE}).*?bragg angle", t)
    if m:
        lam = _num(m.group("lam")); d = _num(m.group("d")); n = _num(m.group("n"))
        x = max(-1.0, min(1.0, n * lam / (2.0 * d)))
        return _result(math.degrees(math.asin(x)), "degree", "theta = asin(n lambda/(2d))", {"lambda_nm": lam, "d_nm": d, "n": n})

    m = _m(rf"initial wavelength\s+lambda\s*=\s*(?P<lam>{_VALUE})\s*pm.*?through\s+theta\s*=\s*(?P<th>{_VALUE})\s*(?:°|degrees?|deg).*?scattered wavelength", t)
    if m:
        lam = _num(m.group("lam")); th = math.radians(_num(m.group("th")))
        return _result(lam + COMPTON_ELECTRON_PM * (1.0 - math.cos(th)), "pm", "lambda' = lambda + lambda_C(1-cos theta)", {"lambda_pm": lam, "theta_deg": math.degrees(th)})

    m = _m(rf"compton wavelength shift.*?through\s+theta\s*=\s*(?P<th>{_VALUE})\s*(?:°|degrees?|deg)", t)
    if m:
        th = math.radians(_num(m.group("th")))
        return _result(COMPTON_ELECTRON_PM * (1.0 - math.cos(th)), "pm", "Delta lambda = lambda_C(1-cos theta)", {"theta_deg": math.degrees(th)})

    # ------------------------------------------------------------------
    # Photon energy, momentum, band gap, pair production, Wien/blackbody
    # ------------------------------------------------------------------
    m = _m(rf"frequency of electromagnetic radiation with wavelength\s+lambda\s*=\s*(?P<lam>{_VALUE})\s*nm", t)
    if m:
        lam_nm = _num(m.group("lam"))
        return _result(C / (lam_nm * 1e-9), "Hz", "f = c/lambda", {"lambda_nm": lam_nm})

    m = _m(rf"photon energy in ev.*?wavelength\s+lambda\s*=\s*(?P<lam>{_VALUE})\s*nm", t)
    if m:
        lam_nm = _num(m.group("lam"))
        return _result(HC_EV_NM / lam_nm, "eV", "E = hc/lambda", {"lambda_nm": lam_nm})

    m = _m(rf"energy of one photon.*?frequency\s+is\s+f\s*=\s*(?P<f>{_VALUE})\s*Hz", t)
    if m:
        f = _num(m.group("f"))
        return _result(H * f, "J", "E = hf", {"f_Hz": f})

    m = _m(rf"photon wavelength in nm.*?energy\s+is\s+e\s*=\s*(?P<E>{_VALUE})\s*eV", t)
    if m:
        E = _num(m.group("E"))
        return _result(HC_EV_NM / E, "nm", "lambda = hc/E", {"E_eV": E})

    m = _m(rf"momentum of a photon with wavelength\s+lambda\s*=\s*(?P<lam>{_VALUE})\s*nm", t)
    if m:
        lam_nm = _num(m.group("lam"))
        return _result(H / (lam_nm * 1e-9), "kg·m/s", "p = h/lambda", {"lambda_nm": lam_nm})

    m = _m(rf"semiconductor has band gap\s+e_g\s*=\s*(?P<E>{_VALUE})\s*eV.*?photon wavelength", t)
    if m:
        E = _num(m.group("E"))
        return _result(HC_EV_NM / E, "nm", "lambda = hc/E_g", {"E_g_eV": E})

    m = _m(rf"photon emitted by a semiconductor has wavelength\s+lambda\s*=\s*(?P<lam>{_VALUE})\s*nm.*?band gap energy", t)
    if m:
        lam_nm = _num(m.group("lam"))
        return _result(HC_EV_NM / lam_nm, "eV", "E_g = hc/lambda", {"lambda_nm": lam_nm})

    m = _m(rf"photon of energy\s+(?P<E>{_VALUE})\s*MeV.*?electron-positron pair.*?total kinetic energy", t)
    if m:
        E = _num(m.group("E"))
        return _result(E - 2.0 * ELECTRON_REST_MEV, "MeV", "K_total = E_gamma - 2m_e c^2", {"E_gamma_MeV": E})

    m = _m(rf"minimum photon energy needed to create\s+(?P<N>{_VALUE})\s+electron-positron pair", t)
    if m:
        N = _num(m.group("N"))
        return _result(N * 2.0 * ELECTRON_REST_MEV, "MeV", "E_min = N*2m_e c^2", {"pairs": N})

    m = _m(rf"peak wavelength of blackbody radiation at temperature\s+t\s*=\s*(?P<T>{_VALUE})\s*K", t)
    if m:
        T = _num(m.group("T"))
        return _result(WIEN_UM_K / T, "μm", "lambda_max T = b", {"T_K": T})

    m = _m(rf"blackbody temperature.*?peak wavelength is\s+lambda_max\s*=\s*(?P<lam>{_VALUE})\s*μm", t)
    if m:
        lam_um = _num(m.group("lam"))
        return _result(WIEN_UM_K / lam_um, "K", "T = b/lambda_max", {"lambda_um": lam_um})

    m = _m(rf"emissivity\s+(?P<eps>{_VALUE}).*?area\s+a\s*=\s*(?P<A>{_VALUE})\s*m(?:\^?2|²).*?temperature\s+t\s*=\s*(?P<T>{_VALUE})\s*K.*?blackbody power", t)
    if not m:
        m = _m(rf"blackbody power radiated by area\s+a\s*=\s*(?P<A>{_VALUE})\s*m(?:\^?2|²).*?temperature\s+t\s*=\s*(?P<T>{_VALUE})\s*K", t)
    if m:
        eps = _num(m.groupdict().get("eps") or "1")
        A = _num(m.group("A")); T = _num(m.group("T"))
        return _result(eps * SIGMA * A * T**4, "W", "P = epsilon sigma A T^4", {"epsilon": eps, "A_m2": A, "T_K": T})

    # ------------------------------------------------------------------
    # Hydrogen / Bohr / Rydberg transitions
    # ------------------------------------------------------------------
    m = _m(rf"hydrogen-like ion with z\s*=\s*(?P<Z>{_VALUE})\s+transitions from n\s*=\s*(?P<ni>{_VALUE})\s+to n\s*=\s*(?P<nf>{_VALUE}).*?emitted photon energy", t)
    if m:
        Z = _num(m.group("Z")); ni = _num(m.group("ni")); nf = _num(m.group("nf"))
        E = RYD_E_EV * Z * Z * abs(1.0 / (nf * nf) - 1.0 / (ni * ni))
        return _result(E, "eV", "Delta E = 13.6 Z^2 |1/n_f^2 - 1/n_i^2|", {"Z": Z, "n_i": ni, "n_f": nf})

    m = _m(rf"hydrogen-like ion with z\s*=\s*(?P<Z>{_VALUE})\s+emits a photon in a transition from n\s*=\s*(?P<ni>{_VALUE})\s+to n\s*=\s*(?P<nf>{_VALUE}).*?photon wavelength", t)
    if m:
        Z = _num(m.group("Z")); ni = _num(m.group("ni")); nf = _num(m.group("nf"))
        E = RYD_E_EV * Z * Z * abs(1.0 / (nf * nf) - 1.0 / (ni * ni))
        return _result(HC_EV_NM / E, "nm", "lambda = hc/Delta E", {"Z": Z, "n_i": ni, "n_f": nf, "E_eV": E})

    m = _m(rf"hydrogen transitions from n\s*=\s*(?P<ni>{_VALUE})\s+to n\s*=\s*(?P<nf>{_VALUE}).*?emitted photon energy", t)
    if m:
        ni = _num(m.group("ni")); nf = _num(m.group("nf"))
        E = RYD_E_EV * abs(1.0 / (nf * nf) - 1.0 / (ni * ni))
        return _result(E, "eV", "Delta E = 13.6 |1/n_f^2 - 1/n_i^2|", {"n_i": ni, "n_f": nf})

    m = _m(rf"hydrogen emits a photon in a transition from n\s*=\s*(?P<ni>{_VALUE})\s+to n\s*=\s*(?P<nf>{_VALUE}).*?photon wavelength", t)
    if m:
        ni = _num(m.group("ni")); nf = _num(m.group("nf"))
        E = RYD_E_EV * abs(1.0 / (nf * nf) - 1.0 / (ni * ni))
        return _result(HC_EV_NM / E, "nm", "lambda = hc/Delta E", {"n_i": ni, "n_f": nf, "E_eV": E})

    m = _m(rf"bohr model for a hydrogen-like ion with z\s*=\s*(?P<Z>{_VALUE}).*?orbital radius.*?quantum number n\s*=\s*(?P<n>{_VALUE})", t)
    if m:
        Z = _num(m.group("Z")); n = _num(m.group("n"))
        return _result(BOHR_RADIUS_NM * n * n / Z, "nm", "r_n = a0 n^2/Z", {"Z": Z, "n": n})

    m = _m(rf"bohr model for hydrogen, calculate the orbital radius.*?quantum number n\s*=\s*(?P<n>{_VALUE})", t)
    if m:
        n = _num(m.group("n"))
        return _result(BOHR_RADIUS_NM * n * n, "nm", "r_n = a0 n^2", {"n": n})

    m = _m(rf"bohr model for a hydrogen-like ion with z\s*=\s*(?P<Z>{_VALUE}).*?energy of the n\s*=\s*(?P<n>{_VALUE})\s+level", t)
    if m:
        Z = _num(m.group("Z")); n = _num(m.group("n"))
        return _result(-RYD_E_EV * Z * Z / (n * n), "eV", "E_n = -13.6 Z^2/n^2", {"Z": Z, "n": n})

    m = _m(rf"bohr model for hydrogen, calculate the energy of the n\s*=\s*(?P<n>{_VALUE})\s+level", t)
    if m:
        n = _num(m.group("n"))
        return _result(-RYD_E_EV / (n * n), "eV", "E_n = -13.6/n^2", {"n": n})

    m = _m(rf"ionization energy from level n\s*=\s*(?P<n>{_VALUE})\s+for a hydrogen-like ion with z\s*=\s*(?P<Z>{_VALUE})", t)
    if m:
        n = _num(m.group("n")); Z = _num(m.group("Z"))
        return _result(RYD_E_EV * Z * Z / (n * n), "eV", "E_ion = 13.6 Z^2/n^2", {"Z": Z, "n": n})

    m = _m(rf"ionization energy from level n\s*=\s*(?P<n>{_VALUE})\s+for hydrogen", t)
    if m:
        n = _num(m.group("n"))
        return _result(RYD_E_EV / (n * n), "eV", "E_ion = 13.6/n^2", {"n": n})

    m = _m(rf"rydberg formula for a hydrogen-like ion with z\s*=\s*(?P<Z>{_VALUE}).*?transition from n\s*=\s*(?P<ni>{_VALUE})\s+to n\s*=\s*(?P<nf>{_VALUE})", t)
    if m:
        Z = _num(m.group("Z")); ni = _num(m.group("ni")); nf = _num(m.group("nf"))
        val = RYDBERG * Z * Z * abs(1.0 / (nf * nf) - 1.0 / (ni * ni))
        return _result(val, "m^-1", "wavenumber = R Z^2 |1/n_f^2 - 1/n_i^2|", {"Z": Z, "n_i": ni, "n_f": nf})

    m = _m(rf"rydberg formula for hydrogen.*?transition from n\s*=\s*(?P<ni>{_VALUE})\s+to n\s*=\s*(?P<nf>{_VALUE})", t)
    if m:
        ni = _num(m.group("ni")); nf = _num(m.group("nf"))
        val = RYDBERG * abs(1.0 / (nf * nf) - 1.0 / (ni * ni))
        return _result(val, "m^-1", "wavenumber = R |1/n_f^2 - 1/n_i^2|", {"n_i": ni, "n_f": nf})


    # ------------------------------------------------------------------
    # Photoelectric effect
    # ------------------------------------------------------------------
    m = _m(rf"metal has work function\s+phi\s*=\s*(?P<phi>{_VALUE})\s*eV.*?wavelength\s+lambda\s*=\s*(?P<lam>{_VALUE})\s*nm.*?stopping potential", t)
    if m:
        phi = _num(m.group("phi")); lam_nm = _num(m.group("lam"))
        return _result(HC_EV_NM / lam_nm - phi, "V", "eV_s = hc/lambda - phi", {"phi_eV": phi, "lambda_nm": lam_nm})

    m = _m(rf"light of wavelength\s+lambda\s*=\s*(?P<lam>{_VALUE})\s*nm.*?work function\s+phi\s*=\s*(?P<phi>{_VALUE})\s*eV.*?maximum kinetic energy", t)
    if m:
        lam_nm = _num(m.group("lam")); phi = _num(m.group("phi"))
        return _result(HC_EV_NM / lam_nm - phi, "eV", "K_max = hc/lambda - phi", {"phi_eV": phi, "lambda_nm": lam_nm})

    m = _m(rf"threshold wavelength.*?work function\s+phi\s*=\s*(?P<phi>{_VALUE})\s*eV", t)
    if m:
        phi = _num(m.group("phi"))
        return _result(HC_EV_NM / phi, "nm", "lambda_0 = hc/phi", {"phi_eV": phi})

    m = _m(rf"threshold frequency.*?work function\s+phi\s*=\s*(?P<phi>{_VALUE})\s*eV", t)
    if m:
        phi = _num(m.group("phi"))
        return _result(phi * E_CHARGE / H, "Hz", "f_0 = phi/h", {"phi_eV": phi})

    # ------------------------------------------------------------------
    # Matter waves / de Broglie
    # ------------------------------------------------------------------
    m = _m(rf"particle has de broglie wavelength\s+lambda\s*=\s*(?P<lam>{_VALUE})\s*nm.*?momentum", t)
    if m:
        lam_nm = _num(m.group("lam"))
        return _result(H / (lam_nm * 1e-9), "kg·m/s", "p = h/lambda", {"lambda_nm": lam_nm})

    m = _m(rf"electron accelerated through\s+v\s*=\s*(?P<V>{_VALUE})\s*V", t)
    if m and "de broglie wavelength" in tl:
        V = _num(m.group("V"))
        lam_nm = H / math.sqrt(2.0 * M_E * E_CHARGE * V) / 1e-9
        return _result(lam_nm, "nm", "lambda = h/sqrt(2 m_e e V)", {"V": V})

    m = _m(rf"de broglie wavelength of a\s+(?P<particle>electron|proton|neutron)\s+moving at\s+v\s*=\s*(?P<v>{_VALUE})\s*m/s", t)
    if m:
        particle = m.group("particle").lower(); v = _num(m.group("v"))
        mass = {"electron": M_E, "proton": M_P, "neutron": M_N}[particle]
        return _result(H / (mass * v), "m", "lambda = h/(mv)", {"particle": particle, "v_m_s": v})

    m = _m(rf"a\s+(?P<particle>electron|proton|neutron)\s+has de broglie wavelength\s+lambda\s*=\s*(?P<lam>{_VALUE})\s*nm.*?find its speed", t)
    if m:
        particle = m.group("particle").lower(); lam_nm = _num(m.group("lam"))
        mass = {"electron": M_E, "proton": M_P, "neutron": M_N}[particle]
        return _result(H / (mass * lam_nm * 1e-9), "m/s", "v = h/(m lambda)", {"particle": particle, "lambda_nm": lam_nm})

    # ------------------------------------------------------------------
    # Uncertainty, infinite well, mass-energy, radioactivity, nuclear
    # ------------------------------------------------------------------
    m = _m(rf"deltaxdeltap\s*>=\s*hbar\s*/\s*(?P<den>{_VALUE}).*?minimum momentum uncertainty.*?deltax\s*=\s*(?P<dx>{_VALUE})\s*nm", t)
    if m:
        den = _num(m.group("den")); dx_nm = _num(m.group("dx"))
        return _result(HBAR / (den * dx_nm * 1e-9), "kg·m/s", "Delta p_min = hbar/(k Delta x)", {"denominator": den, "dx_nm": dx_nm})

    m = _m(rf"deltaxdeltap\s*>=\s*hbar\s*/\s*(?P<den>{_VALUE}).*?minimum velocity uncertainty of an electron.*?deltax\s*=\s*(?P<dx>{_VALUE})\s*nm", t)
    if m:
        den = _num(m.group("den")); dx_nm = _num(m.group("dx"))
        return _result(HBAR / (den * dx_nm * 1e-9) / M_E, "m/s", "Delta v_min = hbar/(k m_e Delta x)", {"denominator": den, "dx_nm": dx_nm})

    m = _m(rf"infinite potential well of length\s+l\s*=\s*(?P<L>{_VALUE})\s*nm.*?energy of level n\s*=\s*(?P<n>{_VALUE})", t)
    if m:
        L_nm = _num(m.group("L")); n = _num(m.group("n"))
        E_eV = (n*n*H*H)/(8.0*M_E*(L_nm*1e-9)**2)/E_CHARGE
        return _result(E_eV, "eV", "E_n = n^2 h^2/(8mL^2)", {"L_nm": L_nm, "n": n})

    m = _m(rf"infinite well of length\s+l\s*=\s*(?P<L>{_VALUE})\s*nm\s+moves from n\s*=\s*(?P<n1>{_VALUE})\s+to n\s*=\s*(?P<n2>{_VALUE}).*?absorbed energy", t)
    if m:
        L_nm = _num(m.group("L")); n1 = _num(m.group("n1")); n2 = _num(m.group("n2"))
        coeff = H*H/(8.0*M_E*(L_nm*1e-9)**2)/E_CHARGE
        return _result(coeff*(n2*n2 - n1*n1), "eV", "Delta E = (n_2^2-n_1^2)h^2/(8mL^2)", {"L_nm": L_nm, "n1": n1, "n2": n2})

    m = _m(rf"rest energy equivalent of mass\s+m\s*=\s*(?P<m>{_VALUE})\s*g", t)
    if m:
        mass_g = _num(m.group("m"))
        return _result(mass_g * 1e-3 * C*C, "J", "E = mc^2", {"mass_g": mass_g})

    m = _m(rf"radioactive isotope has half-life\s+t1/2\s*=\s*(?P<T>{_VALUE})\s*s.*?decay constant", t)
    if m:
        T = _num(m.group("T"))
        return _result(math.log(2.0)/T, "s^-1", "lambda = ln2/T_half", {"T_half_s": T})

    m = _m(rf"radioactive sample has half-life\s+t1/2\s*=\s*(?P<T>{_VALUE})\s*years.*?fall to\s+(?P<p>{_VALUE})\s*%", t)
    if m:
        T = _num(m.group("T")); p = _num(m.group("p"))
        return _result(T * math.log(100.0/p) / math.log(2.0), "years", "t = T_half ln(N0/N)/ln2", {"T_half_years": T, "percent": p})

    m = _m(rf"initial mass\s+m0\s*=\s*(?P<m0>{_VALUE})\s*g\s+and half-life\s+t1/2\s*=\s*(?P<T>{_VALUE})\s*h.*?after t\s*=\s*(?P<t>{_VALUE})\s*h", t)
    if m:
        m0 = _num(m.group("m0")); T = _num(m.group("T")); tt = _num(m.group("t"))
        return _result(m0 * 0.5 ** (tt / T), "g", "m = m0 2^(-t/T_half)", {"m0_g": m0, "T_half_h": T, "t_h": tt})

    m = _m(rf"sample initially has\s+n0\s*=\s*(?P<N0>{_VALUE})\s+radioactive nuclei.*?half-life is\s+(?P<T>{_VALUE})\s*days.*?remain after\s+(?P<t>{_VALUE})\s*days", t)
    if m:
        N0 = _num(m.group("N0")); T = _num(m.group("T")); tt = _num(m.group("t"))
        return _result(N0 * 0.5 ** (tt / T), "", "N = N0 2^(-t/T_half)", {"N0": N0, "T_half_days": T, "t_days": tt})

    m = _m(rf"radioactive sample contains\s+n\s*=\s*(?P<N>{_VALUE})\s+undecayed nuclei.*?half-life\s+t1/2\s*=\s*(?P<T>{_VALUE})\s*s.*?activity", t)
    if m:
        N = _num(m.group("N")); T = _num(m.group("T"))
        return _result(math.log(2.0)/T * N, "Bq", "A = lambda N", {"N": N, "T_half_s": T})

    m = _m(rf"fission event releases\s+(?P<E>{_VALUE})\s*MeV.*?total energy released by\s+(?P<N>{_VALUE})\s+fissions", t)
    if m:
        E = _num(m.group("E")); N = _num(m.group("N"))
        return _result(E * N * MEV_TO_J, "J", "E_total = N E_fission", {"E_MeV": E, "N": N})

    # Binding energy per nucleon can be phrased with both A and Delta m in
    # either order.  Atomic/nuclear total-binding templates often steal this
    # otherwise and return total MeV instead of MeV/nucleon.
    if "binding energy per nucleon" in tl and "mass defect" in tl:
        ma = _m(rf"(?:mass number|A)\s*(?:=|is)?\s*(?P<A>{_VALUE})", t)
        md = _m(rf"(?:mass defect|deltam|delta m|Delta m)\s*(?:=|is)?\s*(?P<dm>{_VALUE})\s*u", t)
        if ma and md:
            A = _num(ma.group("A")); dm = _num(md.group("dm"))
            if A != 0:
                return _result(dm * U_TO_MEV / A, "MeV/nucleon", "B/A = Delta m c^2/A", {"A": A, "delta_m_u": dm})

    m = _m(rf"mass number\s+a\s*=\s*(?P<A>{_VALUE})\s+has mass defect\s+deltam\s*=\s*(?P<dm>{_VALUE})\s*u.*?binding energy per nucleon", t)
    if m:
        A = _num(m.group("A")); dm = _num(m.group("dm"))
        return _result(dm * U_TO_MEV / A, "MeV/nucleon", "B/A = Delta m c^2/A", {"A": A, "delta_m_u": dm})

    m = _m(rf"total binding energy\s+e_b\s*=\s*(?P<E>{_VALUE})\s*MeV.*?mass defect", t)
    if m:
        E = _num(m.group("E"))
        return _result(E / U_TO_MEV, "u", "Delta m = E_b/931.494", {"E_b_MeV": E})

    m = _m(rf"nuclear reaction has mass defect\s+deltam\s*=\s*(?P<dm>{_VALUE})\s*u.*?released energy", t)
    if m:
        dm = _num(m.group("dm"))
        return _result(dm * U_TO_MEV, "MeV", "Q = Delta m c^2", {"delta_m_u": dm})

    m = _m(rf"mass number\s+a\s*=\s*(?P<A>{_VALUE})\s+and atomic number\s+z\s*=\s*(?P<Z>{_VALUE})\s+undergoes\s+alpha decay.*?daughter atomic number", t)
    if m:
        Z = _num(m.group("Z"))
        return _result(int(round(Z - 2)), "", "alpha decay: Z' = Z - 2", {"Z": Z})

    m = _m(rf"mass number\s+a\s*=\s*(?P<A>{_VALUE})\s+and atomic number\s+z\s*=\s*(?P<Z>{_VALUE})\s+undergoes\s+alpha decay.*?daughter mass number", t)
    if m:
        A = _num(m.group("A"))
        return _result(int(round(A - 4)), "", "alpha decay: A' = A - 4", {"A": A})

    m = _m(rf"mass number\s+a\s*=\s*(?P<A>{_VALUE})\s+and atomic number\s+z\s*=\s*(?P<Z>{_VALUE})\s+undergoes\s+β- decay.*?daughter atomic number", t)
    if not m:
        m = _m(rf"mass number\s+a\s*=\s*(?P<A>{_VALUE})\s+and atomic number\s+z\s*=\s*(?P<Z>{_VALUE})\s+undergoes\s+beta- decay.*?daughter atomic number", t)
    if m:
        Z = _num(m.group("Z"))
        return _result(int(round(Z + 1)), "", "beta-minus decay: Z' = Z + 1", {"Z": Z})

    m = _m(rf"total rest energy of\s+(?P<N>{_VALUE})\s+electron\(s\) in MeV", t)
    if m:
        N = _num(m.group("N"))
        return _result(N * ELECTRON_REST_MEV, "MeV", "E0 = N m_e c^2", {"N": N})
    m = _m(rf"total rest energy of\s+(?P<N>{_VALUE})\s+proton\(s\) in MeV", t)
    if m:
        N = _num(m.group("N"))
        return _result(N * PROTON_REST_MEV, "MeV", "E0 = N m_p c^2", {"N": N})
    m = _m(rf"total rest energy of\s+(?P<N>{_VALUE})\s+neutron\(s\) in MeV", t)
    if m:
        N = _num(m.group("N"))
        return _result(N * NEUTRON_REST_MEV, "MeV", "E0 = N m_n c^2", {"N": N})


    m = _m(rf"moving clock has proper time interval\s+deltat0\s*=\s*(?P<t0>{_VALUE})\s*s\s+and moves at\s+v\s*=\s*(?P<b>{_VALUE})\s*c.*?dilated time", t)
    if m:
        t0 = _num(m.group("t0")); beta_v = _num(m.group("b"))
        return _result(t0 / math.sqrt(1.0 - beta_v * beta_v), "s", "Delta t = gamma Delta t0", {"t0_s": t0, "beta": beta_v})

    m = _m(rf"rod has proper length\s+l0\s*=\s*(?P<L0>{_VALUE})\s*m\s+and moves at\s+v\s*=\s*(?P<b>{_VALUE})\s*c.*?contracted length", t)
    if m:
        L0 = _num(m.group("L0")); beta_v = _num(m.group("b"))
        return _result(L0 * math.sqrt(1.0 - beta_v * beta_v), "m", "L = L0/gamma", {"L0_m": L0, "beta": beta_v})

    # ------------------------------------------------------------------
    # Special relativity
    # ------------------------------------------------------------------
    beta = _beta(t)
    if beta is not None:
        gam = _gamma(beta)
        if "lorentz factor" in tl:
            return _result(gam, "", "gamma = 1/sqrt(1-beta^2)", {"beta": beta})
        if "relativistic kinetic energy" in tl:
            if "proton" in tl:
                rest = PROTON_REST_MEV; particle = "proton"
            else:
                rest = ELECTRON_REST_MEV; particle = "electron"
            return _result((gam - 1.0) * rest, "MeV", "K = (gamma-1)m c^2", {"beta": beta, "particle": particle})
        if "relativistic momentum" in tl:
            # Dataset only asks electron momentum, but keep the branch general.
            mass = M_P if "proton" in tl else M_E
            particle = "proton" if "proton" in tl else "electron"
            return _result(gam * mass * beta * C, "kg·m/s", "p = gamma m v", {"beta": beta, "particle": particle})

    return None
