from __future__ import annotations
import traceback
import exact_fama.physics.physics_solvers.domains.physics_engine as pe

q = "A student reports 20 cm with absolute uncertainty 0.5 cm. Find the percentage error."
print("FILE:", pe.__file__)
print("QUESTION:", q)

for name in ["_eng_freqs", "_eng_inductance_values", "_eng_cap_values", "_expected_unit"]:
    fn = getattr(pe, name)
    try:
        print(f"{name}:", fn(q))
    except Exception:
        print(f"{name}: EXCEPTION")
        traceback.print_exc()

print("\nDIRECT HOTFIX CALL:")
try:
    print(pe._solve_clean_lc_resonance_design_hotfix(q))
except Exception:
    traceback.print_exc()

print("\nENTRYPOINT CALL:")
try:
    print(pe.solve_clean_physics_engine(q))
except Exception:
    traceback.print_exc()
