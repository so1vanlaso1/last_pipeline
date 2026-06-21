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


def test_type1_free_form_numbers_are_plain():
    import gateway.logic_adapter as la

    assert la._plain_free_form_answer("1.2300e-4") == "0.000123"
    assert la._plain_free_form_answer("17,28") == "17.28"
    assert la._plain_free_form_answer("$42$") == "42"
    assert la._plain_free_form_answer("Unknown") == "Unknown"


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
    assert r["reasoning"]["type"] == "cot"


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


def _physics_adapter_with_stub_lineup(monkeypatch, deterministic):
    import gateway.physics_adapter as pa

    judges = [
        _stub("qwen-4b", 1.0, "4b", "generator"),
        _stub("qwen-4b-instruct", 1.0, "4b", "generator"),
        _stub("gemma-8b-judge", 1.5, "8b", "judge"),
    ]
    adapter = pa.PhysicsAdapter(judges)
    monkeypatch.setattr(
        pa.PhysicsAdapter,
        "_deterministic_candidate",
        lambda self, q, question: deterministic,
    )
    return adapter


def test_type2_cascade_uses_judge_answer(monkeypatch):
    """Type 2 now runs deterministic -> two generators -> physics judge."""
    import gateway.physics_adapter as pa
    from gateway.schema import PredictQuery

    deterministic = pa._candidate(
        label="deterministic", source="deterministic", answer="4", unit="A",
        explanation="Deterministic solver answer.", steps=["solver step"], confidence=0.9,
    )
    adapter = _physics_adapter_with_stub_lineup(monkeypatch, deterministic)
    q = PredictQuery(query_id="P-JUDGE", type="type2", query="Find the current.")

    r = adapter.answer(q)
    assert r.answer == "6"
    assert r.unit == "A"
    assert r.premises_used == []
    assert r.reasoning.type == "cot"
    assert "senior physics arbiter" in r.explanation.lower()


def test_type2_judge_prompt_includes_deterministic_candidate(monkeypatch):
    import gateway.physics_adapter as pa
    from gateway.schema import PredictQuery
    from gateway.vllm_client import LLMClient

    deterministic = pa._candidate(
        label="deterministic", source="deterministic", answer="4", unit="A",
        explanation="Deterministic solver answer.", steps=["solver step"], confidence=0.9,
    )
    adapter = _physics_adapter_with_stub_lineup(monkeypatch, deterministic)
    judge_users = []

    def fake_chat(self, system, user, **kwargs):
        stage = kwargs.get("log_context", "")
        if "physics.judge" in stage:
            judge_users.append(user)
            return '{"chosen": "self", "answer": "6", "unit": "A", "explanation": "Judge.", "steps": ["judge step"]}'
        return '{"answer": "5", "unit": "A", "explanation": "Generator.", "steps": ["gen step"], "confidence": 0.6}'

    monkeypatch.setattr(LLMClient, "chat", fake_chat)
    adapter.answer(PredictQuery(query_id="P-PROMPT", type="type2", query="Find the current."))

    assert judge_users
    assert "Deterministic" in judge_users[0]
    assert "answer = 4 A" in judge_users[0]
    assert "reference only" in judge_users[0]


def test_type2_judge_reconciles_against_consensus(monkeypatch):
    """When the judge produces a usable answer that DISAGREES with a unanimous
    reference consensus, a reconciliation pass runs and its corrected answer (still
    8B-authored) becomes the final answer — no deterministic value is substituted."""
    import gateway.physics_adapter as pa
    from gateway.schema import PredictQuery
    from gateway.vllm_client import LLMClient

    deterministic = pa._candidate(
        label="deterministic", source="deterministic", answer="7200", unit="V/m",
        explanation="Fields add at the dipole midpoint.", steps=["E=|E1+E2|"], confidence=0.93,
    )
    adapter = _physics_adapter_with_stub_lineup(monkeypatch, deterministic)
    calls = []

    def fake_chat(self, system, user, **kwargs):
        ctx = kwargs.get("log_context", "")
        if "physics.judge.reconcile" in ctx:
            calls.append("reconcile")
            return ('{"chosen": "deterministic", "answer": "7200", "unit": "V/m", '
                    '"explanation": "On re-check both fields point the same way and add.", "steps": ["x"]}')
        if "physics.judge" in ctx:                  # phase-1 judge: a wrong, dissenting answer
            calls.append("judge")
            return '{"chosen": "self", "answer": "0", "unit": "V/m", "explanation": "Cancel.", "steps": ["x"]}'
        return '{"answer": "7200", "unit": "V/m", "explanation": "g", "steps": ["x"], "confidence": 0.9}'

    monkeypatch.setattr(LLMClient, "chat", fake_chat)
    r = adapter.answer(PredictQuery(query_id="P-RECON", type="type2",
                                    query="Field at the midpoint of a +/-4nC dipole 20cm apart."))
    assert "reconcile" in calls          # the reconciliation pass fired
    assert r.answer == "7200"            # the judge corrected itself to the consensus
    assert r.unit == "V/m"


def test_type2_judge_does_not_reconcile_without_consensus(monkeypatch):
    """No reconciliation when the references do NOT unanimously agree: the judge's own
    answer stands and no extra pass runs."""
    import gateway.physics_adapter as pa
    from gateway.schema import PredictQuery
    from gateway.vllm_client import LLMClient

    deterministic = pa._candidate(
        label="deterministic", source="deterministic", answer="4", unit="A",
        explanation="det", steps=["s"], confidence=0.9,
    )
    adapter = _physics_adapter_with_stub_lineup(monkeypatch, deterministic)
    calls = []

    def fake_chat(self, system, user, **kwargs):
        ctx = kwargs.get("log_context", "")
        if "physics.judge.reconcile" in ctx:
            calls.append("reconcile")
            return '{"chosen": "self", "answer": "99", "unit": "A", "explanation": "x", "steps": ["x"]}'
        if "physics.judge" in ctx:
            return '{"chosen": "self", "answer": "6", "unit": "A", "explanation": "Judge.", "steps": ["x"]}'
        # generators disagree with each other and with deterministic -> no consensus
        return '{"answer": "5", "unit": "A", "explanation": "g", "steps": ["x"], "confidence": 0.6}'

    monkeypatch.setattr(LLMClient, "chat", fake_chat)
    r = adapter.answer(PredictQuery(query_id="P-NOCONS", type="type2", query="Find the current."))
    assert "reconcile" not in calls      # references did not agree -> no reconciliation
    assert r.answer == "6"               # the judge's own answer stands
    assert r.unit == "A"


def test_type2_judge_prompt_has_consensus_rule(monkeypatch):
    """The judge prompt must push extra reasoning when it dissents from a 3-way
    consensus (the dipole-midpoint failure: deterministic + both juniors say 7200
    V/m, a wrong judge says 0). It must treat the agreement as evidence its own work
    is wrong, recheck vector directions, and adopt the consensus unless it can name a
    concrete error."""
    import gateway.physics_adapter as pa
    from gateway.schema import PredictQuery
    from gateway.vllm_client import LLMClient

    deterministic = pa._candidate(
        label="deterministic", source="deterministic", answer="7200", unit="V/m",
        explanation="Fields add at the midpoint of a dipole.", steps=["E=|E1+E2|"],
        confidence=0.93,
    )
    adapter = _physics_adapter_with_stub_lineup(monkeypatch, deterministic)
    judge_systems = []

    def fake_chat(self, system, user, **kwargs):
        if "physics.judge" in kwargs.get("log_context", ""):
            judge_systems.append(system)
            return ('{"chosen": "deterministic", "answer": "7200", "unit": "V/m", '
                    '"explanation": "Both fields point the same way and add.", "steps": ["x"]}')
        return '{"answer": "7200", "unit": "V/m", "explanation": "g", "steps": ["x"], "confidence": 0.9}'

    monkeypatch.setattr(LLMClient, "chat", fake_chat)
    r = adapter.answer(PredictQuery(query_id="P-CONS", type="type2",
                                    query="Field at the midpoint of a +/-4nC dipole 20cm apart."))
    assert judge_systems
    sys = judge_systems[0].lower()
    assert "consensus" in sys                 # the 3-way agreement rule is present
    assert "not always right" in sys          # the judge is told it can be wrong
    assert "direction" in sys                 # recheck the error-prone vector step
    # When the judge endorses the consensus value, that is the final answer + unit.
    assert r.answer == "7200"
    assert r.unit == "V/m"


def test_type2_invalid_judge_returns_uncertain_not_deterministic(monkeypatch):
    """No-fallback policy: when the 8B judge yields nothing usable even after the
    recovery ladder (here phase-1 prose AND the phase-2 re-extract both fail), the
    result is Uncertain — never the deterministic reference value."""
    import gateway.physics_adapter as pa
    from gateway.schema import PredictQuery
    from gateway.vllm_client import LLMClient

    deterministic = pa._candidate(
        label="deterministic", source="deterministic", answer="4", unit="A",
        explanation="Deterministic solver answer.", steps=["solver step"], confidence=0.9,
    )
    adapter = _physics_adapter_with_stub_lineup(monkeypatch, deterministic)

    def fake_chat(self, system, user, **kwargs):
        # "physics.judge" matches BOTH the phase-1 judge and the phase-2 reextract,
        # so the entire judge ladder yields unparseable text here.
        if "physics.judge" in kwargs.get("log_context", ""):
            return "not json"
        return '{"answer": "5", "unit": "A", "explanation": "Generator.", "steps": ["gen step"], "confidence": 0.6}'

    monkeypatch.setattr(LLMClient, "chat", fake_chat)
    r = adapter.answer(PredictQuery(query_id="P-DET", type="type2", query="Find the current."))
    assert r.answer == "Uncertain"        # NOT the deterministic "4"
    assert r.unit == ""


def test_type2_judge_json_allows_latex_explanation(monkeypatch):
    import gateway.physics_adapter as pa
    from gateway.schema import PredictQuery
    from gateway.vllm_client import LLMClient

    deterministic = pa._candidate(
        label="deterministic", source="deterministic", answer="17.28", unit="N",
        explanation="Deterministic solver answer.", steps=["solver step"], confidence=0.98,
    )
    adapter = _physics_adapter_with_stub_lineup(monkeypatch, deterministic)

    def fake_chat(self, system, user, **kwargs):
        if "physics.judge" in kwargs.get("log_context", ""):
            return (
                r'Final Answer: $0.02304 \text{ N}$.'
                r'{"chosen": "self", "answer": 0.02304, "unit": "N", '
                r'"explanation": "Use $r=\sqrt{(0.03)^2+(0.04)^2}$ and $F=q\cdot E$.", '
                r'"steps": ["$r=\sqrt{(0.03)^2+(0.04)^2}=0.05$ m", '
                r'"$F=(2\times 10^{-6})(11520)=0.02304$ N"]}'
            )
        return '{"answer": "17.28", "unit": "N", "explanation": "Generator.", "steps": ["gen step"], "confidence": 0.6}'

    monkeypatch.setattr(LLMClient, "chat", fake_chat)
    r = adapter.answer(PredictQuery(query_id="P-LATEX", type="type2", query="Find the force."))
    assert r.answer == "0.02304"
    assert r.unit == "N"
    assert r.explanation.startswith("Use $r=\\sqrt")
    assert r.reasoning.steps[0].startswith("$r=\\sqrt")


def test_type2_answers_are_plain_numbers():
    import gateway.physics_adapter as pa

    assert pa._candidate(label="x", source="test", answer=0.02304)["answer"] == "0.02304"
    assert pa._candidate(label="x", source="test", answer="1.2300e-4")["answer"] == "0.000123"
    assert pa._candidate(label="x", source="test", answer="17,28")["answer"] == "17.28"
    assert pa._candidate(label="x", source="test", answer=r"9\sqrt{3} x 10^-27")["answer"].startswith(
        "0.000000000000000000000000015588"
    )
    assert pa._candidate(label="x", source="test", answer=r"\frac{1}{2}")["answer"] == "0.5"
    assert pa._candidate(label="x", source="test", answer=r"$0.02304 \text{ N}$")["answer"] == "0.02304"


def test_type2_uncertain_deterministic_no_fallback_to_generator(monkeypatch):
    """Even when the deterministic reference abstained, a failed judge does NOT fall
    back to a generator's answer — the generators are references only."""
    import gateway.physics_adapter as pa
    from gateway.schema import PredictQuery
    from gateway.vllm_client import LLMClient

    deterministic = pa._candidate(
        label="deterministic", source="deterministic", answer="Uncertain", unit="",
        explanation="Solver abstained.", steps=["solver abstained"], confidence=0.0,
    )
    adapter = _physics_adapter_with_stub_lineup(monkeypatch, deterministic)

    def fake_chat(self, system, user, **kwargs):
        if "physics.judge" in kwargs.get("log_context", ""):
            return "not json"
        return '{"answer": "5", "unit": "A", "explanation": "Generator.", "steps": ["gen step"], "confidence": 0.6}'

    monkeypatch.setattr(LLMClient, "chat", fake_chat)
    r = adapter.answer(PredictQuery(query_id="P-GEN", type="type2", query="Find the current."))
    assert r.answer == "Uncertain"        # NOT the generator's "5"
    assert r.unit == ""


def test_type2_judge_regex_recovery(monkeypatch):
    """When the judge reasons in prose without a JSON object, the regex step recovers
    its answer/unit and the (network) phase-2 re-extract is NOT invoked."""
    import gateway.physics_adapter as pa
    from gateway.schema import PredictQuery
    from gateway.vllm_client import LLMClient

    deterministic = pa._candidate(
        label="deterministic", source="deterministic", answer="4", unit="A",
        explanation="Deterministic solver answer.", steps=["solver step"], confidence=0.9,
    )
    adapter = _physics_adapter_with_stub_lineup(monkeypatch, deterministic)

    def fake_chat(self, system, user, **kwargs):
        ctx = kwargs.get("log_context", "")
        if "physics.judge.reextract" in ctx:
            raise AssertionError("phase-2 re-extract must not run when regex recovers")
        if "physics.judge" in ctx:
            return "Worked it through. answer: 6 A"          # prose, no JSON object
        return '{"answer": "5", "unit": "A", "explanation": "Generator.", "steps": ["gen step"], "confidence": 0.6}'

    monkeypatch.setattr(LLMClient, "chat", fake_chat)
    r = adapter.answer(PredictQuery(query_id="P-REGEX", type="type2", query="Find the current."))
    assert r.answer == "6"
    assert r.unit == "A"


def test_type2_judge_phase2_reextract(monkeypatch):
    """When phase-1 has no parseable JSON and no regex-recoverable answer, the no-think
    guided-JSON phase-2 re-extract of the judge's own reasoning supplies the answer."""
    import gateway.physics_adapter as pa
    from gateway.schema import PredictQuery
    from gateway.vllm_client import LLMClient

    deterministic = pa._candidate(
        label="deterministic", source="deterministic", answer="4", unit="A",
        explanation="Deterministic solver answer.", steps=["solver step"], confidence=0.9,
    )
    adapter = _physics_adapter_with_stub_lineup(monkeypatch, deterministic)

    def fake_chat(self, system, user, **kwargs):
        ctx = kwargs.get("log_context", "")
        if "physics.judge.reextract" in ctx:
            return '{"chosen": "self", "answer": "6", "unit": "A", "explanation": "Re-extracted.", "steps": ["x"]}'
        if "physics.judge" in ctx:
            return "The reasoning here is inconclusive prose with no verdict object."
        return '{"answer": "5", "unit": "A", "explanation": "Generator.", "steps": ["gen step"], "confidence": 0.6}'

    monkeypatch.setattr(LLMClient, "chat", fake_chat)
    r = adapter.answer(PredictQuery(query_id="P-PHASE2", type="type2", query="Find the current."))
    assert r.answer == "6"
    assert r.unit == "A"


def test_type2_candidate_captures_trailing_unit():
    """A unit crammed into the answer field is moved to the unit field, not discarded."""
    import gateway.physics_adapter as pa

    c = pa._candidate(label="x", source="test", answer="5 A")
    assert c["answer"] == "5" and c["unit"] == "A"
    c2 = pa._candidate(label="x", source="test", answer="3 V/m")
    assert c2["answer"] == "3" and c2["unit"] == "V/m"
    c3 = pa._candidate(label="x", source="test", answer="9.1e-31 kg")
    assert c3["unit"] == "kg" and float(c3["answer"]) == 9.1e-31
    # An explicit unit field wins and the answer is not double-stripped.
    c4 = pa._candidate(label="x", source="test", answer="5", unit="A")
    assert c4["answer"] == "5" and c4["unit"] == "A"


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


def test_config_lineup_roles_and_budget(monkeypatch):
    monkeypatch.delenv("MODEL_TEMPERATURE", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    from gateway import config
    models = config.load_models()
    assert models, "expected at least one resident model"
    assert all("temperature" in m for m in models)
    assert all(m["temperature"] == 0.0 for m in models)
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


def test_llm_client_uses_configured_temperature(monkeypatch):
    from gateway.vllm_client import LLMClient

    seen = {}

    def fake_chat_vllm(self, system, user, **kwargs):
        seen["temperature"] = kwargs["temperature"]
        kwargs["raw_out"]("{}")
        return "{}"

    monkeypatch.setattr(LLMClient, "_chat_vllm", fake_chat_vllm)
    client_under_test = LLMClient(mode="vllm", temperature=0.7)

    client_under_test.chat("system", "user")

    assert seen["temperature"] == 0.7


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


def test_type1_arbiter_reconciles_when_juniors_agree(monkeypatch):
    """Both generators agree (Yes) but the judge dissents (No): a reconciliation pass
    runs and the judge's corrected answer (Yes) is used."""
    monkeypatch.setenv("LOGIC_MODE", "arbiter")
    import gateway.logic_adapter as la
    from gateway.schema import PredictQuery

    cands = [
        {"label": "j1", "canon": "Yes", "display": "Yes", "answer_raw": "Yes",
         "premises_used": [0], "explanation": "Premise 1 entails it."},
        {"label": "j2", "canon": "Yes", "display": "Yes", "answer_raw": "Yes",
         "premises_used": [0], "explanation": "Premise 1 entails it."},
    ]
    monkeypatch.setattr(la, "_run_generators", lambda gen_specs, record: cands)
    calls = []

    def fake_chat(client, system, user, **kwargs):
        if "reconcile" in kwargs.get("stage", ""):
            calls.append("reconcile")
            return '{"chosen": 1, "answer": "Yes", "premises_used": [1], "explanation": "On re-check the juniors are right."}'
        calls.append("judge")
        return '{"answer": "No", "premises_used": [1], "explanation": "Judge dissent."}'

    monkeypatch.setattr(la, "_chat", fake_chat)
    judges = [
        _stub("qwen-4b", 1.0, "4b", "generator"),
        _stub("gemma-e2b", 1.0, "4b", "generator"),
        _stub("gemma-8b-judge", 1.5, "8b", "judge"),
    ]
    q = PredictQuery(query_id="R1", type="type1", query="Entailed?",
                     premises=["All A are B.", "x is A."], options=["Yes", "No", "Uncertain"])
    r = la.answer_type1(judges, q)
    assert "reconcile" in calls          # the reconciliation pass fired
    assert r.answer == "Yes"             # judge corrected itself to the juniors' consensus


def test_type1_arbiter_no_reconcile_when_juniors_disagree(monkeypatch):
    """When the two generators disagree, there is no consensus, so no reconciliation
    pass runs and the judge's own answer stands."""
    monkeypatch.setenv("LOGIC_MODE", "arbiter")
    import gateway.logic_adapter as la
    from gateway.schema import PredictQuery

    cands = [
        {"label": "j1", "canon": "No", "display": "No", "answer_raw": "No",
         "premises_used": [0], "explanation": "a"},
        {"label": "j2", "canon": "Yes", "display": "Yes", "answer_raw": "Yes",
         "premises_used": [1], "explanation": "b"},
    ]
    monkeypatch.setattr(la, "_run_generators", lambda gen_specs, record: cands)
    calls = []

    def fake_chat(client, system, user, **kwargs):
        if "reconcile" in kwargs.get("stage", ""):
            calls.append("reconcile")
            return '{"answer": "Uncertain"}'
        return '{"answer": "Yes", "premises_used": [2], "explanation": "Judge."}'

    monkeypatch.setattr(la, "_chat", fake_chat)
    judges = [
        _stub("qwen-4b", 1.0, "4b", "generator"),
        _stub("gemma-e2b", 1.0, "4b", "generator"),
        _stub("gemma-8b-judge", 1.5, "8b", "judge"),
    ]
    q = PredictQuery(query_id="R2", type="type1", query="Entailed?",
                     premises=["All A are B.", "x is A."], options=["Yes", "No", "Uncertain"])
    r = la.answer_type1(judges, q)
    assert "reconcile" not in calls      # generators disagreed -> no reconciliation
    assert r.answer == "Yes"             # the judge's own answer stands


def test_type1_free_form_arbiter_reconciles(monkeypatch):
    """Free-form arbiter: both juniors agree (42), the judge dissents (7), and the
    reconciliation pass corrects it back to 42."""
    monkeypatch.setenv("LOGIC_MODE", "arbiter")
    import gateway.logic_adapter as la
    from gateway.schema import PredictQuery

    calls = []

    def fake_chat(client, system, user, **kwargs):
        stage = kwargs.get("stage", "")
        if "free_form_generator" in stage:
            return '{"answer": "42", "premises_used": [1], "explanation": "gen"}'
        if "reconcile" in stage:
            calls.append("reconcile")
            return '{"answer": "42", "premises_used": [1], "explanation": "On re-check the juniors are right."}'
        if "free_form_judge" in stage:
            calls.append("judge")
            return '{"answer": "7", "premises_used": [1], "explanation": "Judge dissent."}'
        return "{}"

    monkeypatch.setattr(la, "_chat", fake_chat)
    judges = [
        _stub("qwen-4b", 1.0, "4b", "generator"),
        _stub("gemma-e2b", 1.0, "4b", "generator"),
        _stub("gemma-8b-judge", 1.5, "8b", "judge"),
    ]
    q = PredictQuery(query_id="R3", type="type1", query="How many?",
                     premises=["There are 42."], options=[])
    r = la.answer_type1(judges, q)
    assert "reconcile" in calls
    assert r.answer == "42"


# ── Provability answer-label fix (hybrid detection) ──────────────────────────

def _ynn_lineup():
    return [
        _stub("qwen-4b", 1.0, "4b", "generator"),
        _stub("gemma-e2b", 1.0, "4b", "generator"),
        _stub("gemma-8b-judge", 1.5, "8b", "judge"),
    ]


_UNKNOWN_CANDS = [
    {"label": "j1", "canon": "Unknown", "display": "Not Given", "answer_raw": "Not Given",
     "premises_used": [0], "explanation": "a"},
    {"label": "j2", "canon": "Unknown", "display": "Not Given", "answer_raw": "Not Given",
     "premises_used": [0], "explanation": "b"},
]


def _choice_query(query, options=("Yes", "No", "Uncertain")):
    from gateway.schema import PredictQuery
    return PredictQuery(
        query_id="PV", type="type1", query=query,
        premises=["If audit-ready and a manager signs off, then the case is closed.",
                  "The case is audit-ready."],
        options=list(options),
    )


def _run_choice(monkeypatch, query, judge_json, cands=None, options=("Yes", "No", "Uncertain")):
    monkeypatch.setenv("LOGIC_MODE", "arbiter")
    import gateway.logic_adapter as la
    monkeypatch.setattr(la, "_run_generators", lambda gs, rec: list(cands or _UNKNOWN_CANDS))
    monkeypatch.setattr(la, "_chat", lambda *a, **k: judge_json)
    return la.answer_type1(_ynn_lineup(), _choice_query(query, options))


def test_provability_choice_missing_step_is_no(monkeypatch):
    """Regex-classified provability + Not Given (no contradiction) -> No."""
    r = _run_choice(
        monkeypatch,
        "Do the premises prove that the Atlas case can be formally closed?",
        '{"answer": "Not Given", "conflict": false, "premises_used": [1,2], "explanation": "A manager sign-off is not stated."}',
    )
    assert r.answer == "No"


def test_provability_choice_model_label_widens_backstop(monkeypatch):
    """Regex misses (no trigger, no veto) but the model self-labels provability and
    there is no veto -> the backstop still collapses Not Given to No."""
    r = _run_choice(
        monkeypatch,
        "Is the Atlas case ready to wrap up now?",
        '{"answer": "Not Given", "question_mode": "provability", "conflict": false, "explanation": "A required step is missing."}',
    )
    assert r.answer == "No"


def test_entailment_veto_blocks_model_widening(monkeypatch):
    """A clear entailment phrasing vetoes the model's provability self-label, so a
    genuine Not Given is preserved."""
    r = _run_choice(
        monkeypatch,
        "Does it follow that the Atlas case is closed?",
        '{"answer": "Not Given", "question_mode": "provability", "explanation": "Neither proved."}',
    )
    assert r.answer == "Uncertain"


def test_provability_contradiction_stays_uncertain_flag(monkeypatch):
    """Provability but the model flagged a genuine contradiction -> Uncertain stands."""
    r = _run_choice(
        monkeypatch,
        "Do the premises establish that the River Codex is safe for public release?",
        '{"answer": "Not Given", "conflict": true, "explanation": "P2 and P5 are mutually exclusive."}',
    )
    assert r.answer == "Uncertain"


def test_provability_contradiction_stays_uncertain_explanation(monkeypatch):
    """No conflict flag, but the explanation names a premise-vs-premise contradiction."""
    r = _run_choice(
        monkeypatch,
        "Do the premises establish that the River Codex is safe for public release?",
        '{"answer": "Not Given", "explanation": "Premise 2 directly contradicts premise 5."}',
    )
    assert r.answer == "Uncertain"


def test_provability_definite_no_survives(monkeypatch):
    r = _run_choice(
        monkeypatch,
        "Do the premises prove that the Atlas case can be formally closed?",
        '{"answer": "No", "explanation": "A required step is missing."}',
    )
    assert r.answer == "No"


def test_entailment_choice_not_given_stays_uncertain(monkeypatch):
    r = _run_choice(
        monkeypatch,
        "Does it follow that the statement holds?",
        '{"answer": "Not Given", "explanation": "Neither the claim nor its negation is proved."}',
    )
    assert r.answer == "Uncertain"


def _run_free_form(monkeypatch, query, judge_answer, gen_answer="Uncertain"):
    monkeypatch.setenv("LOGIC_MODE", "arbiter")
    import gateway.logic_adapter as la
    from gateway.schema import PredictQuery

    def fake_chat(client, system, user, **kw):
        stage = kw.get("stage", "")
        if "free_form_generator" in stage:
            return '{"answer": "%s", "premises_used": [1], "explanation": "g"}' % gen_answer
        if "free_form_judge" in stage:
            return '{"answer": "%s", "premises_used": [1], "explanation": "A required step is missing."}' % judge_answer
        return "{}"

    monkeypatch.setattr(la, "_chat", fake_chat)
    q = PredictQuery(query_id="FF", type="type1", query=query,
                     premises=["If pick and weight-check then prepared.", "Kappa can pick."],
                     options=[])
    return la.answer_type1(_ynn_lineup(), q)


def test_provability_free_form_is_no(monkeypatch):
    r = _run_free_form(
        monkeypatch,
        "Does Robot Kappa satisfy every requirement for preparing outbound orders?",
        judge_answer="Uncertain",
    )
    assert r.answer == "No"


def test_entailment_free_form_stays_uncertain(monkeypatch):
    r = _run_free_form(monkeypatch, "Is it true that the order is prepared?", judge_answer="Uncertain")
    assert r.answer == "Uncertain"


# ── Pure-function unit tests ──────────────────────────────────────────────────

def test_classify_type1_mode_and_veto():
    from prompts import classify_type1_mode, entailment_veto

    provability = [
        "Do the premises prove that X?",
        "Do the premises establish that X?",
        "Does having sensors guarantee Y?",
        "Does Robot Kappa satisfy every requirement for Z?",
        "Do the premises support the conclusion that X holds?",
    ]
    for q in provability:
        assert classify_type1_mode(q, None) == "provability", q

    entailment = [
        "Does it follow that X?",
        "Is it true that X?",
        "Can we conclude that X?",
        "Do the premises entail that X?",
        "Do the premises imply that X?",
        "Does Sophia like cats?",
        "What is the total current?",
    ]
    for q in entailment:
        assert classify_type1_mode(q, None) == "entailment", q

    assert entailment_veto("Does it follow that X?")
    # The entailment veto wins over an incidental provability verb.
    assert classify_type1_mode("Does it follow that the premises establish X?", None) == "entailment"
    # Content options => MCQ; Yes/No(/Uncertain) options stay non-MCQ.
    assert classify_type1_mode("Which option holds?", ["a cat is here", "no animal", "birds fly"]) == "mcq"
    assert classify_type1_mode("Do the premises prove X?", ["Yes", "No", "Uncertain"]) == "provability"


def test_has_real_contradiction():
    from gateway.logic_adapter import has_real_contradiction

    assert has_real_contradiction("P2 directly contradicts P5.")
    assert has_real_contradiction("Premise 2 contradicts premise 5.")
    assert has_real_contradiction("Premise 2 and premise 5 are mutually exclusive.")
    assert not has_real_contradiction("There is no contradiction here.")
    assert not has_real_contradiction("Premise 3 requires a review before proceeding.")


def test_build_user_prompt_shapes():
    from schema import Record, AnswerType
    from prompts import build_user, rules_for

    prov = Record(id="a", premises_nl=["p"], answer_type=AnswerType.YES_NO_UNKNOWN,
                  question_nl="Do the premises prove that the case is closed?")
    ent = Record(id="b", premises_nl=["p"], answer_type=AnswerType.YES_NO_UNKNOWN,
                 question_nl="Does it follow that X?")
    assert "ESTABLISH" in build_user(prov)
    assert "Does the statement logically follow" in build_user(ent)
    assert "Yes, No, or Not Given" in build_user(ent)
    rf = rules_for(prov)
    assert "MODE B" in rf and "Reply in EXACTLY" not in rf


def test_add_question_named_premises_includes_named_feature():
    from gateway.logic_adapter import add_question_named_premises
    from schema import Record, AnswerType

    premises = [
        "If a satellite has calibrated thermal sensors, then it can monitor surface temperature.",
        "If a satellite can monitor surface temperature and has cloud-penetrating radar, then it can support disaster mapping.",
        "If a satellite supports disaster mapping, then it can provide emergency response data.",
        "Satellite Vega has calibrated thermal sensors.",
        "Satellite Vega does not have cloud-penetrating radar.",
        "Satellite Vega has a high-resolution optical camera.",
        "All satellites with high-resolution optical cameras can capture daytime images.",
    ]
    rec = Record(id="t", premises_nl=premises, answer_type=AnswerType.YES_NO_UNKNOWN,
                 question_nl="Does having thermal sensors and an optical camera guarantee emergency response data in this case?")
    out = add_question_named_premises(rec, [0, 1, 2, 3, 4, 5])
    assert 6 in out                                   # the optical-camera consequence rule
    assert set([0, 1, 2, 3, 4, 5]).issubset(set(out))  # never drops existing citations


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
