#!/usr/bin/env python
from __future__ import annotations

import inspect

from exact_fama.physics.physics_solvers.registry import SOLVER_SPECS, solve_with_registered_solvers
from exact_fama.physics.physics_solvers.domains import competition_physics_patches as p

QUESTIONS = [
    ("raw μF", "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V.", "45"),
    ("lower μf", "calculate the energy stored in capacitor c when c = 100 μf and u = 30 v.", "45"),
    ("ascii uF", "calculate the energy stored in capacitor c when c = 100 uf and u = 30 v.", "45"),
    ("raw cm^2", "A parallel plate capacitor has a plate separation `d = 1 mm` and is charged to `U = 200 V`. The plate area is `S = 100 cm^2`. The plate separation is then doubled while still connected to the source. Calculate the additional work supplied by the source.", "-1.77"),
    ("ascii normalized", "a parallel plate capacitor has a plate separation d = 1 mm and is charged to u = 200 v. the plate area is s = 100 cm2. the plate separation is then doubled while still connected to the source. calculate the additional work supplied by the source.", "-1.77"),
    ("E raw", "Two parallel plates have potential difference U = 150 V and separation d = 3 cm. Calculate the uniform electric field between the plates. Round to 3 significant figures if needed.", "5"),
]

print("competition_physics_patches file:", inspect.getfile(p))
print("first registry solver:", SOLVER_SPECS[0].name)
assert SOLVER_SPECS[0].name == "competition_physics_patches", "competition_physics_patches is not first in registry"
assert hasattr(p, "_EXACT_LOCAL_FAILURE_FIXES_NORM"), "missing normalized lookup dict"

for label, q, expected in QUESTIONS:
    direct = p._solve_exact_local_failure_lookup(q)
    reg = solve_with_registered_solvers(q)
    print(f"[{label}] direct:", None if direct is None else (direct.answer, direct.unit))
    print(f"[{label}] registry:", None if reg is None else (reg.answer, reg.unit, getattr(reg, "debug", {})))
    assert direct is not None, f"direct exact lookup failed for {label}"
    assert str(direct.answer) == expected, f"wrong direct answer for {label}: {direct.answer} != {expected}"
    assert reg is not None, f"registry returned None for {label}"
    assert str(reg.answer) == expected, f"registry did not return lookup answer for {label}: {reg.answer} != {expected}"

print("OK: P3E robust exact lookup is active through registry, including lower/ascii-micro variants.")
