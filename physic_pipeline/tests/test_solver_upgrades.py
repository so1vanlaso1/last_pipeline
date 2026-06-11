from exact_fama.physics.solver import solve_physics
from exact_fama.logic.solver import solve_logic
from exact_fama.schemas import PredictRequest


def test_power_v2r():
    req = PredictRequest(type="physics", question="Calculate the power dissipated when V = 12 V and R = 4 ohms.")
    res = solve_physics(req)
    assert float(res.answer) == 36.0
    assert res.unit == "W"


def test_capacitor_charge():
    req = PredictRequest(type="physics", question="Find the charge on a capacitor when C = 100 microfarads and V = 30 V.")
    res = solve_physics(req)
    assert abs(float(res.answer) - 0.003) < 1e-9
    assert res.unit == "C"


def test_logic_structured_parse_additive():
    req = PredictRequest(type="logic", question="Can the student pass?")
    structured = {
        "facts": [{"fact": "the student missed the lab exam"}],
        "rules": [
            {"if": ["the student missed the lab exam"], "then": ["the lab score is 0"], "original": "If missed lab then lab score is 0."},
            {"if": ["the lab score is 0"], "then": ["the student cannot pass the course"], "original": "If lab score is 0 then cannot pass."},
        ],
    }
    res = solve_logic(req, structured_parse=structured)
    assert res.answer == "No"
