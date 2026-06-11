from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Callable
from .common import SolverResult
from .domains.engine import (
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
SOLVER_SPECS: tuple[SolverSpec, ...] = (
    SolverSpec("competition_physics_patches", "competition_templates", solve_competition_physics_patches),
    SolverSpec("enhanced_fit_patches", "high_priority_generalized", solve_enhanced_fit_patches),
    SolverSpec("clean_physics_engine", "formula_engine", solve_clean_physics_engine),
    SolverSpec("electric_priority_rules", "electricity", solve_electric_priority_rules),
    SolverSpec("high_confidence_rules", "high_confidence", solve_high_confidence_rules),
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
def solve_with_registered_solvers(question: str) -> SolverResult | None:
    for spec in SOLVER_SPECS:
        try:
            result = spec.solve(question)
        except ZeroDivisionError:
            continue
        except Exception:
            if os.environ.get("DEBUG_PHYSICS_SOLVER"):
                raise
            continue
        if result is not None:
            return result
    return None
