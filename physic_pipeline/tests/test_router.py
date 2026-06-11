from exact_fama.router import route_task
from exact_fama.schemas import PredictRequest


def test_route_physics_by_keywords():
    req = PredictRequest(question="Calculate current when V = 12 V and R = 4 ohms.")
    assert route_task(req) == "physics"


def test_route_logic_by_premises():
    req = PredictRequest(question="Can the student pass?", **{"premises-NL": ["If A then B."]})
    assert route_task(req) == "logic"
