from exact_fama.logic.solver import solve_logic
from exact_fama.schemas import PredictRequest


def test_simple_no_pass_rule():
    req = PredictRequest(
        type="logic",
        question="Can the student pass the course if they missed the lab exam?",
        **{"premises-NL": [
            "If a student misses the lab exam, the lab score is 0.",
            "If the lab score is 0, the student cannot pass the course.",
        ]},
    )
    res = solve_logic(req)
    assert res.answer == "No"
    assert res.cot
