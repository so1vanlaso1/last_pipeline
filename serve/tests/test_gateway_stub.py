"""No-GPU wiring test for the competition schema.

Run with the stub LLM backend so no model/GPU is needed:

    GATEWAY_LLM=stub PYTHONPATH=serve:physic_pipeline/src:logic_pipeline/src \
        python -m pytest serve/tests -q

It checks that /predict accepts the competition input, routes by type, and
returns a JSON list of well-formed result objects.
"""

from __future__ import annotations

import os

os.environ.setdefault("GATEWAY_LLM", "stub")
os.environ.setdefault("PHYSICS_LLM_FALLBACK", "0")

import pytest
from fastapi.testclient import TestClient

from gateway.app import app

client = TestClient(app)


def _check_common(obj: dict, qid: str) -> None:
    assert obj["query_id"] == qid
    assert isinstance(obj["answer"], str) and obj["answer"] != ""
    assert isinstance(obj["explanation"], str) and obj["explanation"] != ""
    assert "unit" in obj and isinstance(obj["unit"], str)
    assert isinstance(obj["premises_used"], list)


def test_health_and_models():
    assert client.get("/health").status_code == 200
    data = client.get("/v1/models").json()
    assert data["object"] == "list"
    assert data["data"] and "id" in data["data"][0]


def test_type1_choice_returns_listed_option():
    q = {
        "query_id": "T1_0001",
        "type": "type1",
        "query": "Is Student A eligible for graduation?",
        "premises": [
            "A student with >= 120 credits is eligible.",
            "Student A has 118 credits.",
        ],
        "options": ["Yes", "No", "Uncertain"],
    }
    out = client.post("/predict", json=q).json()
    assert isinstance(out, list) and len(out) == 1
    r = out[0]
    _check_common(r, "T1_0001")
    assert r["answer"] in q["options"]          # must be exactly one of the options
    assert r["unit"] == ""                       # Type 1 -> empty unit
    assert all(0 <= i < len(q["premises"]) for i in r["premises_used"])


def test_type1_free_form_number():
    q = {
        "query_id": "T1_0002",
        "type": "type1",
        "query": "How many more credits does Student A need to graduate?",
        "premises": [
            "A student with >= 120 credits is eligible.",
            "Student A has 118 credits.",
        ],
        "options": [],
    }
    r = client.post("/predict", json=q).json()[0]
    _check_common(r, "T1_0002")
    assert r["unit"] == ""


def test_type2_physics_shape():
    q = {
        "query_id": "T2_0001",
        "type": "type2",
        "query": "Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a 12V battery. Find the total current.",
        "premises": [],
        "options": [],
    }
    r = client.post("/predict", json=q).json()[0]
    _check_common(r, "T2_0001")
    assert r["premises_used"] == []              # Type 2 -> empty premises_used


def test_two_model_vote():
    """The Type 1 path votes across the resident line-up (cascade.finalize_by_vote).
    Drive it directly with two stub judges to exercise the multi-model path."""
    from gateway.logic_adapter import answer_type1
    from gateway.schema import PredictQuery
    from gateway.vllm_client import LLMClient

    judges = [
        (LLMClient(mode="stub", base_url="http://localhost:8001/v1", model="judge-a-4b"), 1.0, "4b"),
        (LLMClient(mode="stub", base_url="http://localhost:8002/v1", model="judge-b-4b"), 1.0, "4b"),
    ]
    q = PredictQuery(
        query_id="V1", type="type1",
        query="Does it follow that the statement holds?",
        premises=["All A are B.", "x is an A."],
        options=["Yes", "No", "Uncertain"],
    )
    r = answer_type1(judges, q)
    assert r.answer in q.options
    assert r.query_id == "V1"
    assert r.reasoning.type == "fol"


def test_vote_mode_still_works(monkeypatch):
    """The legacy weighted-vote flow remains available via LOGIC_MODE=vote."""
    monkeypatch.setenv("LOGIC_MODE", "vote")
    from gateway.logic_adapter import answer_type1
    from gateway.schema import PredictQuery
    from gateway.vllm_client import LLMClient

    judges = [
        (LLMClient(mode="stub", base_url="http://localhost:8001/v1", model="a"), 1.0, "4b"),
        (LLMClient(mode="stub", base_url="http://localhost:8002/v1", model="b"), 1.0, "4b"),
    ]
    q = PredictQuery(query_id="VOTE", type="type1", query="Entailed?",
                     premises=["All A are B.", "x is A."], options=["Yes", "No", "Uncertain"])
    r = answer_type1(judges, q)
    assert r.answer in q.options
    assert r.query_id == "VOTE"


def test_arbiter_single_8b_two_passes(monkeypatch):
    """Single-model line-up: the model makes two generator passes + arbitrates."""
    monkeypatch.setenv("LOGIC_MODE", "arbiter")
    from gateway.logic_adapter import answer_type1
    from gateway.schema import PredictQuery
    from gateway.vllm_client import LLMClient

    judges = [(LLMClient(mode="stub", base_url="http://localhost:8001/v1", model="solo-8b"), 1.5, "8b")]
    q = PredictQuery(query_id="SOLO", type="type1", query="Entailed?",
                     premises=["All A are B.", "x is A."], options=["Yes", "No", "Uncertain"])
    r = answer_type1(judges, q)
    assert r.answer in q.options
    assert r.reasoning.type == "fol"


def _stub(model: str, weight: float, cls: str, role: str = ""):
    from gateway.vllm_client import LLMClient
    return (LLMClient(mode="stub", base_url="http://localhost:8001/v1", model=model),
            weight, cls, role)


def test_split_lineup_uses_role_tags():
    """The role-tagged line-up: the two generators generate, the Gemma-4 8B judges."""
    from gateway.logic_adapter import split_lineup

    judges = [
        _stub("qwen-4b", 1.0, "4b", "generator"),
        _stub("gemma-e2b", 1.0, "4b", "generator"),
        _stub("gemma-8b-judge", 1.5, "8b", "judge"),
    ]
    gens, arbiter = split_lineup(judges)
    assert [g.model for g in gens] == ["qwen-4b", "gemma-e2b"]
    assert arbiter.model == "gemma-8b-judge"


def test_split_lineup_without_roles_keeps_old_rule():
    """No role tags → everything generates, the highest-weight model arbitrates."""
    from gateway.logic_adapter import split_lineup

    judges = [_stub("a-4b", 1.0, "4b")[:3], _stub("b-8b", 1.5, "8b")[:3]]
    gens, arbiter = split_lineup(judges)
    assert [g.model for g in gens] == ["a-4b", "b-8b"]
    assert arbiter.model == "b-8b"


def test_arbiter_generators_plus_gemma_judge(monkeypatch):
    """The full generate→judge flow over the 3-model role line-up."""
    monkeypatch.setenv("LOGIC_MODE", "arbiter")
    from gateway.logic_adapter import answer_type1
    from gateway.schema import PredictQuery

    judges = [
        _stub("qwen-4b", 1.0, "4b", "generator"),
        _stub("gemma-e2b", 1.0, "4b", "generator"),
        _stub("gemma-8b-judge", 1.5, "8b", "judge"),
    ]
    q = PredictQuery(query_id="J1", type="type1", query="Entailed?",
                     premises=["All A are B.", "x is A."], options=["Yes", "No", "Uncertain"])
    r = answer_type1(judges, q)
    assert r.answer in q.options
    assert r.query_id == "J1"
    assert r.reasoning.type == "fol"
    assert all(0 <= i < len(q.premises) for i in r.premises_used)


def test_config_lineup_roles_and_budget():
    from gateway import config
    models = config.load_models()
    assert models, "expected at least one resident model"
    # The line-up must fit the configured residency budget (the launch guard).
    assert config.total_params_b(models) <= config.max_resident_b() + 1e-9
    # Generate→judge design: at least one generator + exactly one 8B judge.
    roles = [m["role"] for m in models]
    assert roles.count("generator") >= 1
    assert roles.count("judge") == 1
    judge = next(m for m in models if m["role"] == "judge")
    assert judge["id"] == "google/gemma-4-E4B-it"
    # Compliance: with the swap only ONE group is on the GPU at a time, so the peak
    # momentary load = max(sum of generators, judge) must stay within 8B (§6.3 / Q3).
    gens_b = sum(m["params_b"] for m in models if m["role"] != "judge")
    peak = max(gens_b, judge["params_b"]) if config.swap_active(models) else config.total_params_b(models)
    assert peak <= 8.0 + 1e-9, f"peak resident {peak}B exceeds the 8B limit"


def test_batch_list_input():
    batch = [
        {"query_id": "A", "type": "type1", "query": "Q?", "premises": ["p1"], "options": ["Yes", "No", "Uncertain"]},
        {"query_id": "B", "type": "type2", "query": "Find I with R=2 ohm, V=4 V.", "premises": [], "options": []},
    ]
    out = client.post("/predict", json=batch).json()
    assert [o["query_id"] for o in out] == ["A", "B"]


def test_arbiter_does_not_borrow_mismatched_junior_premises(monkeypatch):
    """#3 regression: when the judge's answer differs from a junior's, the submitted
    premises_used/explanation must NOT be copied from that mismatched junior — they
    would justify a different option than the answer we send. Only a junior whose
    answer maps to the SAME option may supply them."""
    monkeypatch.setenv("LOGIC_MODE", "arbiter")
    import gateway.logic_adapter as la
    from gateway.schema import PredictQuery

    cands = [
        {"label": "j1", "canon": "No", "display": "No", "answer_raw": "No",
         "premises_used": [0], "explanation": "Premise 1 contradicts the claim."},
        {"label": "j2", "canon": "Yes", "display": "Yes", "answer_raw": "Yes",
         "premises_used": [1], "explanation": "Premise 2 entails the claim."},
    ]
    monkeypatch.setattr(la, "_run_generators", lambda gen_specs, record: cands)
    # Judge answers "Yes" but omits chosen/premises_used/explanation (a realistic
    # terse thinking-model verdict). chosen is absent -> _chosen_candidate -> None.
    monkeypatch.setattr(la, "_chat", lambda *a, **k: '{"answer": "Yes"}')

    judges = [
        _stub("qwen-4b", 1.0, "4b", "generator"),
        _stub("gemma-e2b", 1.0, "4b", "generator"),
        _stub("gemma-8b-judge", 1.5, "8b", "judge"),
    ]
    q = PredictQuery(query_id="MM1", type="type1", query="Entailed?",
                     premises=["All A are B.", "x is A."], options=["Yes", "No", "Uncertain"])
    r = la.answer_type1(judges, q)
    assert r.answer == "Yes"
    # Premises/explanation come from the junior that ALSO answered Yes, never the "No" one.
    assert r.premises_used == [1]
    assert "contradicts" not in r.explanation


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
