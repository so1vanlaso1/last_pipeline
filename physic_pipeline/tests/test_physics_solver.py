from exact_fama.physics.solver import solve_physics
from exact_fama.schemas import PredictRequest


def test_capacitor_energy():
    req = PredictRequest(type="physics", question="Calculate the energy stored in capacitor C when C = 100 microfarads and U = 30 V.")
    res = solve_physics(req)
    assert float(res.answer) == 0.045
    assert res.unit == "J"


def test_ohm_current():
    req = PredictRequest(type="physics", question="Calculate the current when V = 12 V and R = 4 ohms.")
    res = solve_physics(req)
    assert float(res.answer) == 3.0
    assert res.unit == "A"


def test_equivalent_resistance_series():
    req = PredictRequest(type="physics", question="Calculate equivalent resistance for resistors in series when R1 = 2 ohms and R2 = 3 ohms.")
    res = solve_physics(req)
    assert float(res.answer) == 5.0
    assert res.unit == "ohm"


def test_equivalent_resistance_parallel():
    req = PredictRequest(type="physics", question="Calculate equivalent resistance for resistors in parallel when R1 = 6 ohms and R2 = 3 ohms.")
    res = solve_physics(req)
    assert float(res.answer) == 2.0
    assert res.unit == "ohm"
