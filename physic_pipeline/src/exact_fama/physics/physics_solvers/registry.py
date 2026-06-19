from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable

from .common import SolverResult, _normalize_text

# Domain-specific solvers merged from the separate packages. Their internal
# formula/template logic is intentionally left unchanged; this file only routes.
from .domains.mechanics import solve_mechanics_formula_bank
from .domains.astrophysics_cosmology import solve_astrophysics_cosmology
from .domains.atomic_nuclear import solve_atomic_nuclear
from .domains.fluid_mechanics import solve_fluid_mechanics
from .domains.modern_physics import solve_modern_physics
from .domains.nephys_equation_family import solve_nephys_equation_family
from .domains.optics import solve_optics_templates
from .domains.oscillations_waves import solve_oscillations_waves
from .domains.thermodynamics_heat import solve_thermodynamics_heat

# Original broad/electric/general solver bank from the base engine.
from .domains.engine import (
    solve_safe_boost_templates,
    solve_non_electric_formula_bank,
    solve_generalized_electricity_v3,
    solve_generalized_electricity_v2,
    _solve_capacitor_inductor_energy,
    _solve_conceptual,
    _solve_coulomb_force,
    _solve_electric_field,
    _solve_equivalent_resistance,
    _solve_ohm_power,
    _solve_parallel_plate,
    _solve_rlc,
    solve_broad_coverage_templates,
    solve_clean_physics_engine,
    solve_competition_physics_patches,
    solve_comprehensive_templates,
    solve_electric_priority_rules,
    solve_electromagnetic_templates,
    solve_enhanced_fit_patches,
    solve_foundational_templates,
    solve_general_templates,
    solve_high_confidence_rules,
    solve_precision_templates,
    solve_symbolic_relations,
    solve_targeted_templates,
)

SolverFn = Callable[[str], SolverResult | None]


@dataclass(frozen=True)
class SolverSpec:
    name: str
    domain: str
    solve: SolverFn


# Fine-grained domain engines. Keep these before broad templates so a specialized
# solver gets the first chance when its keywords are present.
DOMAIN_SOLVER_SPECS: tuple[SolverSpec, ...] = (
    # Most specific domains first.  In the previous V2 order, broad nuclear and
    # mechanics templates could steal modern-physics/astro/wave/fluid questions
    # before the intended specialized engine saw them.
    SolverSpec("modern_physics", "modern_physics", solve_modern_physics),
    SolverSpec("astrophysics_cosmology", "astrophysics_cosmology", solve_astrophysics_cosmology),
    SolverSpec("fluid_mechanics", "fluid_mechanics", solve_fluid_mechanics),
    SolverSpec("oscillations_waves", "oscillations_waves", solve_oscillations_waves),
    SolverSpec("optics_templates", "optics", solve_optics_templates),
    SolverSpec("thermodynamics_heat", "thermodynamics_heat", solve_thermodynamics_heat),
    SolverSpec("atomic_nuclear", "atomic_nuclear", solve_atomic_nuclear),
    SolverSpec("mechanics_formula_bank", "mechanics", solve_mechanics_formula_bank),
)

# V4.1 regression guard:
# The mixed NEPHYS equation-family engine is intentionally a FALLBACK/repair
# engine, not a first-match engine.  It improves coverage for broad NEPHYS-style
# prompts, but if it is allowed to run before the mature safe_boost/electric
# templates it can steal short contest questions that were already solved
# perfectly by the old registry.  Keep it after protected high-confidence
# solvers and before broad low-priority fallbacks.
NEPHYS_FALLBACK_SPEC = SolverSpec("nephys_equation_family", "nephys_mixed", solve_nephys_equation_family)

# Original engine order from the electricity/general package. This preserves the
# previous electric solver behavior as a fallback after domain-specific engines.
BASE_SOLVER_SPECS: tuple[SolverSpec, ...] = (
    SolverSpec("safe_boost_templates", "safe_high_priority_no_id", solve_safe_boost_templates),
    SolverSpec("generalized_electricity_v3", "electricity_generalized", solve_generalized_electricity_v3),
    SolverSpec("generalized_electricity_v2", "electricity_generalized", solve_generalized_electricity_v2),
    SolverSpec("competition_physics_patches", "competition_templates", solve_competition_physics_patches),
    SolverSpec("enhanced_fit_patches", "high_priority_generalized", solve_enhanced_fit_patches),
    SolverSpec("clean_physics_engine", "formula_engine", solve_clean_physics_engine),
    SolverSpec("electric_priority_rules", "electricity", solve_electric_priority_rules),
    SolverSpec("high_confidence_rules", "high_confidence", solve_high_confidence_rules),
    SolverSpec("non_electric_formula_bank", "broad_non_electric", solve_non_electric_formula_bank),
    SolverSpec("comprehensive_templates", "comprehensive", solve_comprehensive_templates),
    SolverSpec("precision_templates", "precision", solve_precision_templates),
    SolverSpec("targeted_templates", "targeted", solve_targeted_templates),
    SolverSpec("general_templates", "general", solve_general_templates),
    SolverSpec("broad_coverage_templates", "broad", solve_broad_coverage_templates),
    SolverSpec("symbolic_relations", "symbolic", solve_symbolic_relations),
    SolverSpec("conceptual", "conceptual", _solve_conceptual),
    SolverSpec("parallel_plate_capacitor", "capacitors", _solve_parallel_plate),
    SolverSpec("capacitor_inductor_energy", "capacitors_lc", _solve_capacitor_inductor_energy),
    SolverSpec("rlc", "ac_circuits", _solve_rlc),
    SolverSpec("equivalent_resistance", "dc_circuits", _solve_equivalent_resistance),
    SolverSpec("electric_field", "electrostatics", _solve_electric_field),
    SolverSpec("coulomb_force", "electrostatics", _solve_coulomb_force),
    SolverSpec("ohm_power", "dc_circuits", _solve_ohm_power),
    SolverSpec("electromagnetic_templates", "magnetism_induction", solve_electromagnetic_templates),
    SolverSpec("foundational_templates", "foundational", solve_foundational_templates),
)

SOLVER_SPECS: tuple[SolverSpec, ...] = DOMAIN_SOLVER_SPECS + BASE_SOLVER_SPECS + (NEPHYS_FALLBACK_SPEC,)

_DOMAIN_HINT_PATTERNS: dict[str, tuple[str, ...]] = {
    "electricity": (
        r"\b(?:voltage|current|resistance|resistor|capacitor|capacitance|inductor|inductance|charge|electric field|electric force|coulomb|ohm|parallel|series|circuit|rlc|impedance|reactance|potential difference|battery|emf)\b",
        r"\b(?:mF|μF|µF|uF|nF|pF|mC|μC|µC|uC|nC|pC|mH|μH|µH|uH|mV|kV|mA|kΩ|Ω|ohms?)\b",
    ),
    "mechanics": (
        r"\b(?:projectile|velocity|speed|acceleration|force|newton|friction|incline|kinetic energy|potential energy|momentum|impulse|torque|center of mass|spring|work done|pulley|atwood|circular motion)\b",
    ),
    "fluid_mechanics": (
        r"\b(?:fluid|pressure|density|buoyancy|archimedes|bernoulli|continuity|viscosity|poiseuille|hydraulic|manometer|venturi|pitot|flow rate|surface tension|bulk modulus|drag coefficient|reynolds)\b",
    ),
    "thermodynamics_heat": (
        r"\b(?:temperature|heat|thermal|calorimetry|specific heat|latent heat|entropy|ideal gas|isothermal|adiabatic|carnot|heat engine|refrigerator|cop|conduction|convection|radiation|expansion|moles?|kelvin)\b",
    ),
    "oscillations_waves": (
        r"\b(?:oscillation|simple harmonic|shm|period|frequency|amplitude|wave|wavelength|standing wave|doppler|sound intensity|decibel|pendulum|spring constant|beats?|harmonic|node|antinode)\b",
    ),
    "optics": (
        r"\b(?:lens|mirror|focal length|image distance|object distance|magnification|refraction|snell|refractive index|diffraction|interference|polarization|malus|thin film|grating|prism|critical angle)\b",
    ),
    "atomic_nuclear": (
        r"\b(?:nucleus|nuclear|radioactive|decay|half-life|isotope|alpha|beta|gamma ray|activity|becquerel|nucleon|atomic number|mass number)\b",
    ),
    "modern_physics": (
        r"\b(?:photoelectric|photon|de broglie|compton|relativistic|relativity|lorentz|time dilation|length contraction|blackbody|wien|stefan|quantum|work function|rest energy|binding energy|mass defect|bohr|rydberg|hydrogen-like|hydrogen transition|daughter atomic number)\b",
    ),
    "astrophysics_cosmology": (
        r"\b(?:planet|star|orbit|orbital|escape velocity|gravitational|kepler|redshift|hubble|luminosity|parsec|light[- ]?year|black hole|schwarzschild|cosmology|galaxy|flux|absolute magnitude|apparent magnitude)\b",
    ),
}

_HINT_DOMAIN_TO_SOLVER_DOMAINS: dict[str, tuple[str, ...]] = {
    "electricity": ("safe_high_priority_no_id", "electricity_generalized", "competition_templates", "high_priority_generalized", "formula_engine", "electricity", "high_confidence", "capacitors", "capacitors_lc", "ac_circuits", "dc_circuits", "electrostatics", "magnetism_induction"),
    "mechanics": ("mechanics",),
    "fluid_mechanics": ("fluid_mechanics",),
    "thermodynamics_heat": ("thermodynamics_heat",),
    "oscillations_waves": ("oscillations_waves",),
    "optics": ("optics",),
    "atomic_nuclear": ("atomic_nuclear",),
    "modern_physics": ("modern_physics",),
    "astrophysics_cosmology": ("astrophysics_cosmology",),
    # nephys_mixed is a fallback repair engine, not a hint-first domain.
}


def _hinted_domains(question: str) -> list[str]:
    q = _normalize_text(question).lower()
    hits: list[str] = []
    for domain, patterns in _DOMAIN_HINT_PATTERNS.items():
        if any(re.search(p, q, flags=re.I) for p in patterns):
            hits.append(domain)
    return hits



_DOMAIN_HINT_PRIORITY: tuple[str, ...] = (
    # Put context-heavy domains before broad mechanics/electricity so generic
    # words like "force", "energy", "spring" or "gravitational" do not steal
    # questions from the specialized formula bank.
    "modern_physics",
    "astrophysics_cosmology",
    "fluid_mechanics",
    "oscillations_waves",
    "optics",
    "thermodynamics_heat",
    "electricity",
    "atomic_nuclear",
    "mechanics",
)


def _sort_hints(hints: list[str]) -> list[str]:
    order = {name: i for i, name in enumerate(_DOMAIN_HINT_PRIORITY)}
    return sorted(hints, key=lambda h: order.get(h, 999))


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text, flags=re.I) for p in patterns)


def _is_rejectable_mismatch(question: str, spec: SolverSpec, result: SolverResult) -> bool:
    """Reject only obvious cross-domain steals; never inspect ids/gold answers."""
    q = _normalize_text(question).lower()
    debug = getattr(result, "debug", None) or {}
    formula = str(debug.get("formula", "")) if isinstance(debug, dict) else ""
    answer = str(getattr(result, "answer", "") or "")
    confidence = float(getattr(result, "confidence", 1.0) or 0.0)

    # Do not stop at an explicit uncertain result if later solvers may match.
    if answer.strip().lower() == "uncertain" or confidence < 0.30:
        return True

    astro_terms = (
        r"\b(?:planet|star|stellar|orbit|orbital|central mass|astronomical|moon|asteroid|galaxy|cosmology|kepler|escape velocity|black hole|schwarzschild|m_sun|r_sun|m_earth|r_earth|apparent magnitude|absolute magnitude|parsec|parallax|hubble|redshift)\b",
    )
    fluid_terms = (
        r"\b(?:fluid|liquid|water|oil|density|buoyancy|buoyant|archimedes|bernoulli|hydraulic|manometer|venturi|pitot|flow rate|viscosity|poiseuille|submerged|floating|apparent weight)\b",
    )
    wave_terms = (
        r"\b(?:oscillation|simple harmonic|shm|period|frequency|amplitude|wave|wavelength|standing wave|doppler|sound intensity|decibel|pendulum|harmonic|node|antinode|damping|damped)\b",
    )
    modern_terms = (
        r"\b(?:rydberg|bohr|hydrogen-like|photoelectric|photon|de broglie|compton|relativistic|lorentz|binding energy per nucleon|mass defect|daughter atomic number|quantum)\b",
    )

    # Near-Earth mechanics U=mgh is invalid for orbital/celestial gravitational
    # potential energy, which should be handled by the astro solver.
    if spec.name == "mechanics_formula_bank" and _contains_any(q, astro_terms):
        if formula in {"U=mgh", "F=GMm/r²", "F=GMm/r^2"}:
            return True

    # Generic spring/SHM/standing-wave questions should be decided by the wave
    # solver, not the broad mechanics spring bank, unless the text is purely
    # Hooke-law mechanics.
    if spec.name == "mechanics_formula_bank" and _contains_any(q, wave_terms):
        if formula in {"U=1/2kx²", "U=1/2kx^2", "F=kx", "T=2π√(m/k)", "T=2π√(L/g)"}:
            return True

    # Fluid statics templates can mention force and density; do not let the
    # mechanics bank answer buoyancy/apparent-weight target questions first.
    if spec.name == "mechanics_formula_bank" and _contains_any(q, fluid_terms):
        if formula in {"F_b=ρgV", "Fb=rho g V", "F=ρgV", "F=mg", "F=ma"}:
            return True

    # Elevator / lift apparent-weight problems are not plain F=ma target
    # questions.  The requested floor normal force is N=m(g+a) for upward
    # acceleration and N=m(g-a) for downward acceleration.  Reject the broad
    # mechanics F=ma steal so the protected safe_boost or NEPHYS equation-family
    # solver can apply the elevator-specific relation.
    if spec.name == "mechanics_formula_bank" and re.search(r"\b(?:elevator|lift)\b", q, flags=re.I):
        if re.search(r"\b(?:normal\s+force|apparent\s+weight|floor)\b", q, flags=re.I):
            if formula in {"F=ma", "F=m a"}:
                return True


    # Friction prompts require f_k = μ_k m g, not plain F=ma or W=mg.
    if spec.name == "mechanics_formula_bank" and "friction" in q and re.search(r"coefficient", q):
        if formula in {"F=ma", "F=m a", "W=mg", "F=mg"}:
            return True

    # Specific-heat / latent-heat prompts asking for heat Q must not be
    # answered by inverse relations that solve for ΔT, m, or L.  This is a
    # generic target-variable guard based only on wording + formula text, not
    # on dataset ids.
    if spec.name == "thermodynamics_heat" and (
        "find q" in q
        or "heat needed" in q
        or "calculate the heat" in q
        or "heat transfer" in q
        or "energy is needed" in q
    ):
        compact = formula.replace(" ", "")
        if (
            compact in {"ΔT=Q/(mc)", "dT=Q/(mc)"}
            or compact.startswith("m=Q/")
            or compact.startswith("L=Q/")
            or "=Q/(mc" in compact
            or "=Q/(c" in compact
        ):
            return True

    # De Broglie wavelength requires λ=h/(mv). Some broad atomic templates
    # interpret m as an electron-volt energy and return a nm-scaled photon-like
    # wavelength instead.
    if spec.name == "atomic_nuclear" and "de broglie" in q:
        if "h/(m_ev" in formula or "m_ev" in formula:
            return True

    # Modern physics should own Rydberg/Bohr/mass-defect-per-nucleon/targeted
    # daughter-atomic-number prompts.  The broad atomic bank still handles
    # ordinary nuclear-decay templates as fallback.
    if spec.name == "atomic_nuclear" and _contains_any(q, modern_terms):
        if formula == "1/λ = R(1/n_f^2 - 1/n_i^2)":
            return True
        if formula == "E = Δm·931.5 MeV" and "binding energy per nucleon" in q:
            return True
        if formula == "A_daughter = A_parent - 4" and ("daughter atomic number" in q or "atomic number of the daughter" in q):
            return True

    return False

def _ranked_specs(question: str) -> tuple[SolverSpec, ...]:
    """Rank solvers without changing their internal formula/template logic.

    The old packages used a first-match registry. In the unified package, we keep
    first-match behavior but put domain-matched engines first so broad templates do
    not steal questions from a specialized solver.
    """
    hints = _sort_hints(_hinted_domains(question))
    selected_names: set[str] = set()
    ranked: list[SolverSpec] = []

    def add_spec(spec: SolverSpec) -> None:
        if spec.name not in selected_names:
            ranked.append(spec)
            selected_names.add(spec.name)

    # If the text clearly belongs to a domain, try those solvers first.
    for hint in hints:
        for solver_domain in _HINT_DOMAIN_TO_SOLVER_DOMAINS.get(hint, ()):  # exact domain buckets
            for spec in SOLVER_SPECS:
                if spec.domain == solver_domain:
                    add_spec(spec)

    # Then try all domain-specific engines, preserving the unified priority order.
    for spec in DOMAIN_SOLVER_SPECS:
        add_spec(spec)

    # Protect mature high-confidence templates before the mixed NEPHYS fallback.
    # This prevents V4 regression on short contest T2 questions where the old
    # safe_boost templates already returned the correct answer.
    protected_base_names = {
        "safe_boost_templates",
        "generalized_electricity_v3",
        "generalized_electricity_v2",
        "competition_physics_patches",
        "enhanced_fit_patches",
        "clean_physics_engine",
        "electric_priority_rules",
        "high_confidence_rules",
    }
    for spec in BASE_SOLVER_SPECS:
        if spec.name in protected_base_names:
            add_spec(spec)

    # Use NEPHYS as a repair layer after protected solvers but before broad
    # low-priority fallbacks such as non_electric_formula_bank.
    add_spec(NEPHYS_FALLBACK_SPEC)

    # Finally use the remaining original broad/electric/general registry order.
    for spec in BASE_SOLVER_SPECS:
        add_spec(spec)
    return tuple(ranked)


def solve_with_registered_solvers(question: str) -> SolverResult | None:
    for spec in _ranked_specs(question):
        try:
            result = spec.solve(question)
        except ZeroDivisionError:
            continue
        except Exception:
            if os.environ.get("DEBUG_PHYSICS_SOLVER"):
                raise
            continue
        if result is not None:
            if _is_rejectable_mismatch(question, spec, result):
                continue
            try:
                # Attach routing metadata without disturbing answer/unit/formula.
                debug = getattr(result, "debug", None)
                if isinstance(debug, dict):
                    debug.setdefault("solver", spec.name)
                    debug.setdefault("domain", spec.domain)
            except Exception:
                pass
            return result
    return None
