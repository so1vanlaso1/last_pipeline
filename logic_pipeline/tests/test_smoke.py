"""No-GPU smoke tests: data loading, prompt/answer normalization, the
generate→judge flow, and the weighted soft-vote wiring via StubModel.

Run:  python -m pytest NEWpipeline/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cascade import (  # noqa: E402
    DECIDER_GEN_AGREE, DECIDER_GEN_FALLBACK, DECIDER_JUDGE, DECIDER_JUDGE_CHOSEN,
    WEIGHT_4B, WEIGHT_8B, finalize_by_vote, finalize_judged, generate_candidates,
    judge_decide, query, tally,
)
from chat_model import StubModel  # noqa: E402
from data_load import load_records, parse_mcq_question  # noqa: E402
from prompts import (  # noqa: E402
    build_user, canonicalize_mcq, canonicalize_ynn, extract_last_json_object,
    generator_system, judge_system, judge_user, parse_generator_reply,
    parse_judge_reply, parse_reply, strip_thinking, to_zero_based,
)
from run_cascade import STAGE_ORDER, parse_stages  # noqa: E402
from schema import AnswerType, ModelReply, Record  # noqa: E402

DATA = ROOT / "Logic_Based_Educational_Queries.json"


# ── helpers ───────────────────────────────────────────────────────────────────
def _ynn(rid: str = "r") -> Record:
    return Record(id=rid, premises_nl=["All cats are animals."],
                  question_nl="Are cats animals?", answer_type=AnswerType.YES_NO_UNKNOWN)


def _reply(answer, weight, cls="4b", why="because premise 1") -> ModelReply:
    return ModelReply(model_label=f"m@{weight}", model_id="stub", prompt="", raw="",
                      answer=answer, answer_display=(answer or ""), explanation=why,
                      weight=weight, model_class=cls)


# ── data / normalization (unchanged behavior) ─────────────────────────────────
def test_dataset_loads_and_classifies():
    recs = load_records(DATA)
    assert len(recs) > 500
    kinds = {r.answer_type for r in recs}
    assert AnswerType.MCQ in kinds and AnswerType.YES_NO_UNKNOWN in kinds
    for r in recs:
        if r.answer is None:
            continue
        if r.answer_type == AnswerType.MCQ:
            assert r.answer == "Unknown" or (len(r.answer) == 1 and r.answer.isalpha())
        else:
            assert r.answer in {"Yes", "No", "Unknown"}


def test_mcq_question_split():
    stem, opts = parse_mcq_question("Pick one\nA. first\nB. second\nC. third")
    assert stem == "Pick one"
    assert opts == ["first", "second", "third"]


def test_ynn_normalization():
    assert canonicalize_ynn("Yes") == "Yes"
    assert canonicalize_ynn("no.") == "No"
    assert canonicalize_ynn("Not Given") == "Unknown"
    assert canonicalize_ynn("unknown") == "Unknown"
    assert canonicalize_ynn("no information in the premises") == "Unknown"
    assert canonicalize_ynn("banana") is None


def test_mcq_normalization():
    opts = ["alpha", "beta", "gamma"]
    assert canonicalize_mcq("B", opts)[0] == "B"
    assert canonicalize_mcq("(C) gamma", opts)[0] == "C"
    assert canonicalize_mcq("beta", opts)[0] == "B"
    assert canonicalize_mcq("Unknown", opts)[0] == "Unknown"


def test_mcq_no_substring_false_match():
    opts = ["Sophia is eligible for the program",
            "Sophia needs to pass to get an honors diploma",
            "John's GPA is insufficient"]
    assert canonicalize_mcq("No", opts)[0] is None
    assert canonicalize_mcq("no", opts)[0] is None
    assert canonicalize_mcq("John's GPA is insufficient", opts)[0] == "C"


def test_canon_gold_handles_not_given():
    from data_load import _canon_gold
    from schema import AnswerType as AT
    assert _canon_gold("Not Given", AT.YES_NO_UNKNOWN, None) == "Unknown"
    assert _canon_gold("no information available", AT.YES_NO_UNKNOWN, None) == "Unknown"
    assert _canon_gold("No", AT.YES_NO_UNKNOWN, None) == "No"
    assert _canon_gold("Unknown", AT.MCQ, ["a", "b"]) == "Unknown"


# ── new prompt template ───────────────────────────────────────────────────────
def test_build_user_ynn_has_statement():
    u = build_user(_ynn())
    assert "Premises:" in u and "Definitions: None" in u
    assert "Question:" in u and "Statement: Are cats animals?" in u
    assert "Options:" not in u


def test_build_user_mcq_has_options_not_statement():
    rec = Record(id="m", premises_nl=["p1", "p2"], question_nl="Which follows?",
                 answer_type=AnswerType.MCQ, options=["first", "second"])
    u = build_user(rec)
    assert "Question: Which follows?" in u
    assert "Options:" in u and "A. first" in u and "B. second" in u
    assert "Statement:" not in u
    # premises are numbered from 1 so the rules can cite "premise 2"
    assert "1. p1" in u and "2. p2" in u


# ── parsing: inline WHY + thinking strip ──────────────────────────────────────
def test_parse_reply_answer_first():
    rec = _ynn()
    canon, display, why = parse_reply("ANSWER: Not Given\nWHY: nothing entails it.", rec)
    assert canon == "Unknown"
    assert "nothing entails it" in why.lower()


def test_parse_reply_inline_answer_and_why():
    rec = _ynn()
    canon, display, why = parse_reply("ANSWER: Yes WHY: premise 1 entails it.", rec)
    assert canon == "Yes"
    assert display == "Yes"  # the inline WHY is not glued onto the answer token
    assert "premise 1" in why.lower()


def test_parse_reply_strips_thinking_block():
    rec = _ynn()
    raw = "<think>It might be No, but actually...</think>\nANSWER: Yes WHY: by premise 1."
    canon, _disp, why = parse_reply(raw, rec)
    assert canon == "Yes"  # the 'No' inside <think> must not win
    assert "<think>" not in why and "It might be No" not in why


def test_strip_thinking_handles_unclosed_close_tag():
    assert "reasoning" not in strip_thinking("reasoning here</think> ANSWER: Yes").lower()


def test_why_ignores_reasoning_before_the_answer():
    # Repro of the live bug: thinking-mode reasoning preceded the ANSWER and there
    # was no WHY line, so the explanation must NOT pick up the reasoning prose.
    rec = Record(id="m", premises_nl=["p"], question_nl="Which?",
                 answer_type=AnswerType.MCQ, options=["x", "y"])
    raw = "thought The user wants me to act as a strict logic examiner and pick.\nANSWER: A"
    canon, _disp, why = parse_reply(raw, rec)
    assert canon == "A"
    assert "thought" not in why.lower() and "wants me to act" not in why.lower()


def test_why_comes_from_after_answer_when_reasoning_precedes_it():
    rec = _ynn()
    raw = "Let me think. The premises say cats are animals.\nANSWER: Yes\nWHY: premise 1 states it."
    canon, _disp, why = parse_reply(raw, rec)
    assert canon == "Yes"
    assert "premise 1" in why and "Let me think" not in why


def test_strip_thinking_handles_gemma_thought_tags():
    raw = "<start_of_thought>maybe No<end_of_thought>\nANSWER: Yes WHY: by premise 1."
    rec = _ynn()
    canon, _disp, why = parse_reply(raw, rec)
    assert canon == "Yes" and "maybe No" not in why


# ── weighted soft vote ────────────────────────────────────────────────────────
def test_weights_defaults():
    assert WEIGHT_4B == 1.0 and WEIGHT_8B == 1.5


def test_tally_sums_weights():
    s = tally([_reply("Yes", 1.0), _reply("Yes", 1.0), _reply("No", 1.5, "8b")])
    assert s == {"Yes": 2.0, "No": 1.5}


def test_one_8b_outvotes_single_4b():
    rec = _ynn()
    final = finalize_by_vote(rec, [_reply("Yes", 1.0), _reply("No", 1.5, "8b", why="8B says no")])
    assert final.answer == "No" and not final.agreed
    assert abs(final.confidence - 1.5 / 2.5) < 1e-9
    assert final.explanation == "8B says no"  # winner's explanation is the 8B's


def test_two_4b_outvote_one_8b():
    rec = _ynn()
    final = finalize_by_vote(
        rec, [_reply("Yes", 1.0), _reply("Yes", 1.0), _reply("No", 1.5, "8b")])
    assert final.answer == "Yes"
    assert abs(final.confidence - 2.0 / 3.5) < 1e-9


def test_unanimous_is_flagged_and_full_confidence():
    rec = _ynn()
    final = finalize_by_vote(rec, [_reply("Yes", 1.0), _reply("Yes", 1.5, "8b")])
    assert final.answer == "Yes" and final.agreed
    assert abs(final.confidence - 1.0) < 1e-9


def test_pure_4b_tie_is_broken_deterministically():
    rec = _ynn()
    final = finalize_by_vote(rec, [_reply("Yes", 1.0), _reply("No", 1.0)])
    # tie on weight → defer to the later (more authoritative) vote
    assert final.answer == "No" and not final.agreed
    assert abs(final.confidence - 0.5) < 1e-9


def test_explanation_prefers_strongest_model_with_a_reason():
    rec = _ynn()
    final = finalize_by_vote(rec, [
        _reply("Yes", 1.0, why="4B reason"),
        _reply("Yes", 1.5, "8b", why=""),       # strongest, but no WHY
        _reply("Yes", 1.0, why="later 4B reason"),
    ])
    # Skips the empty-WHY 8B; among equal-weight explainers, prefers the later one
    # (same rule as the vote tie-break).
    assert final.explanation == "later 4B reason"


def test_explanation_uses_8b_when_it_explains():
    rec = _ynn()
    final = finalize_by_vote(rec, [
        _reply("Yes", 1.0, why="4B reason"),
        _reply("Yes", 1.5, "8b", why="8B reason"),
    ])
    assert final.explanation == "8B reason"  # strongest model that explained wins


def test_no_parseable_answer_yields_none():
    rec = _ynn()
    final = finalize_by_vote(rec, [_reply(None, 1.0), _reply(None, 1.5, "8b")])
    assert final.answer is None and final.confidence == 0.0
    assert not final.agreed


# ── end-to-end via StubModel.query ────────────────────────────────────────────
def test_query_copies_weight_and_class_onto_reply():
    rec = _ynn()
    m = StubModel("Big", lambda u: "No", vote_weight=1.5, model_class="8b")
    rep = query(m, rec)
    assert rep.answer == "No" and rep.weight == 1.5 and rep.model_class == "8b"


def test_full_three_stage_vote_via_stubs():
    rec = Record(id="dis", premises_nl=["X."], question_nl="Does she get a scholarship?",
                 answer_type=AnswerType.YES_NO_UNKNOWN)
    qwen = StubModel("Qwen-4B", lambda u: "Yes", vote_weight=1.0, model_class="4b")
    gemma_s = StubModel("Gemma-E2B", lambda u: "No", vote_weight=1.0, model_class="4b")
    gemma_b = StubModel("Gemma-E4B", lambda u: "Not Given", vote_weight=1.5, model_class="8b")
    liquid = StubModel("Liquid", lambda u: "Yes", vote_weight=1.5, model_class="8b")
    reps = [query(m, rec) for m in (qwen, gemma_s, gemma_b, liquid)]
    final = finalize_by_vote(rec, reps)
    # Yes = 1.0(qwen) + 1.5(liquid) = 2.5 ; No = 1.0 ; Unknown = 1.5
    assert final.scores == {"Yes": 2.5, "No": 1.0, "Unknown": 1.5}
    assert final.answer == "Yes"
    assert len(final.replies) == 4  # all four models recorded for the log


# ── positional alignment is robust to duplicate ids ───────────────────────────
def test_predictions_dict_keeps_records_with_a_shared_id():
    from logio import predictions_dict
    rec = _ynn("dup")
    f1 = finalize_by_vote(rec, [_reply("Yes", 1.0)])
    f2 = finalize_by_vote(rec, [_reply("No", 1.0)])
    out = predictions_dict([rec, rec], [f1, f2])
    assert len(out) == 2  # both survive even though the records share an id
    assert sorted(v["answer"] for v in out.values()) == ["No", "Yes"]


def test_score_aligns_by_position_not_id():
    from score import score
    r1 = Record(id="x", premises_nl=["p"], question_nl="q1?",
                answer_type=AnswerType.YES_NO_UNKNOWN, answer="Yes")
    r2 = Record(id="x", premises_nl=["p"], question_nl="q2?",
                answer_type=AnswerType.YES_NO_UNKNOWN, answer="No")
    rep = score([r1, r2], [finalize_by_vote(r1, [_reply("Yes", 1.0)]),
                           finalize_by_vote(r2, [_reply("No", 1.0)])])
    assert rep.overall.correct == 2  # both scored independently, not collapsed


# ── stage selection ───────────────────────────────────────────────────────────
def test_parse_stages_orders_and_dedupes():
    assert parse_stages("liquid8b,4b,4b") == ["4b", "liquid8b"]
    assert parse_stages("4b,gemma8b,liquid8b") == STAGE_ORDER


def test_parse_stages_rejects_unknown():
    import pytest
    with pytest.raises(Exception):
        parse_stages("4b,bogus")


# ══ Generate→judge flow ═══════════════════════════════════════════════════════
def _gen_reply(label, answer, pu=(0,), why="cites premise 1", weight=1.0) -> ModelReply:
    return ModelReply(model_label=label, model_id="stub", prompt="", raw="",
                      answer=answer, answer_display=(answer or ""), explanation=why,
                      weight=weight, model_class="4b", role="generator",
                      premises_used=list(pu))


def _judge_reply(answer, chosen=None, pu=(1,), why="judge cites premise 2") -> ModelReply:
    r = ModelReply(model_label="LFM", model_id="stub", prompt="", raw="",
                   answer=answer, answer_display=(answer or ""), explanation=why,
                   weight=1.5, model_class="8b", role="judge", premises_used=list(pu))
    r.chosen = chosen  # type: ignore[attr-defined]
    return r


def test_extract_last_json_object_picks_the_final_one():
    raw = ('I think {"answer": "No"} at first... but actually\n'
           '{"answer": "Yes", "premises_used": [1], "explanation": "p1."}')
    assert extract_last_json_object(raw)["answer"] == "Yes"
    # broken braces in the reasoning prose must not kill the real object
    raw2 = 'reasoning {step by step ... \n{"answer": "No", "premises_used": [2]}'
    assert extract_last_json_object(raw2)["answer"] == "No"
    assert extract_last_json_object("no json here at all") is None


def test_to_zero_based_clamps_and_dedupes():
    assert to_zero_based([1, 2, 2, "3", 99, -1, "x"], 3) == [0, 1, 2]
    assert to_zero_based(None, 3) == []


def test_json_prompts_drop_the_answer_why_format_line():
    # The examiner rules end with "Reply in EXACTLY this format: ANSWER:/WHY:".
    # Embedded in a JSON-format prompt that line would compete with the JSON
    # instruction — generator and judge prompts must strip it.
    for rec in (_ynn(), Record(id="m", premises_nl=["p"], question_nl="Which?",
                               answer_type=AnswerType.MCQ, options=["x", "y"])):
        for sys_p in (generator_system(rec), judge_system(rec, 2)):
            assert "Reply in EXACTLY this format" not in sys_p
            assert "JSON object" in sys_p


def test_generator_prompt_and_parse_roundtrip():
    rec = _ynn()
    sys_p = generator_system(rec)
    assert "JSON object" in sys_p and "premises_used" in sys_p
    raw = ('<think>maybe No?</think>\nLet me check premise 1.\n'
           '{"answer": "Yes", "premises_used": [1], "explanation": "Premise 1 states it."}')
    canon, display, why, pu = parse_generator_reply(raw, rec)
    assert canon == "Yes" and pu == [0] and "Premise 1" in why


def test_generator_parse_falls_back_to_answer_why():
    rec = _ynn()
    canon, _d, why, pu = parse_generator_reply("ANSWER: No\nWHY: premise 1 negates it.", rec)
    assert canon == "No" and pu == [] and "premise 1" in why


def test_judge_prompt_shows_both_juniors():
    rec = _ynn()
    cands = [_gen_reply("Qwen-4B", "Yes"), _gen_reply("Gemma-E2B", "No")]
    user = judge_user(rec, cands)
    assert "Junior 1 (Qwen-4B)" in user and "Junior 2 (Gemma-E2B)" in user
    assert "premises_used = [1]" in user            # 1-based for the model
    sys_p = judge_system(rec, len(cands))
    assert "SENIOR examiner" in sys_p and '"chosen"' in sys_p


def test_parse_judge_reply_reads_chosen():
    rec = _ynn()
    raw = ('Junior 2 used the converse — invalid.\n'
           '{"chosen": 1, "answer": "Yes", "premises_used": [1], '
           '"explanation": "Premise 1 entails it directly."}')
    v = parse_judge_reply(raw, rec)
    assert v["canon"] == "Yes" and v["chosen"] == 1 and v["premises_used"] == [0]


def test_finalize_judged_judge_wins_over_both_juniors():
    rec = _ynn()
    cands = [_gen_reply("A", "Yes"), _gen_reply("B", "Yes")]
    final = finalize_judged(rec, cands, _judge_reply("No", chosen=None))
    assert final.answer == "No" and final.decider == DECIDER_JUDGE
    assert final.premises_used == [1]               # the judge's own derivation
    assert final.explanation == "judge cites premise 2"
    assert not final.agreed


def test_finalize_judged_unparseable_judge_endorsement():
    rec = _ynn()
    cands = [_gen_reply("A", "Yes"), _gen_reply("B", "No", pu=(0, 1))]
    final = finalize_judged(rec, cands, _judge_reply(None, chosen=2, pu=(), why=""))
    assert final.answer == "No" and final.decider == DECIDER_JUDGE_CHOSEN
    assert final.premises_used == [0, 1]            # falls back to the junior's
    assert final.explanation == "cites premise 1"


def test_finalize_judged_unparseable_judge_generator_agreement():
    rec = _ynn()
    cands = [_gen_reply("A", "Yes"), _gen_reply("B", "Yes")]
    final = finalize_judged(rec, cands, _judge_reply(None, chosen=None, pu=(), why=""))
    assert final.answer == "Yes" and final.decider == DECIDER_GEN_AGREE


def test_finalize_judged_split_fallback_is_first_generator():
    rec = _ynn()
    cands = [_gen_reply("A", "Yes"), _gen_reply("B", "No")]
    final = finalize_judged(rec, cands, None)       # judge stage never ran
    assert final.answer == "Yes" and final.decider == DECIDER_GEN_FALLBACK


def test_finalize_judged_agreed_flag_when_everyone_concurs():
    rec = _ynn()
    cands = [_gen_reply("A", "Yes"), _gen_reply("B", "Yes")]
    final = finalize_judged(rec, cands, _judge_reply("Yes", chosen=1))
    assert final.answer == "Yes" and final.agreed
    assert abs(final.confidence - 1.0) < 1e-9


def test_judge_flow_end_to_end_via_stubs():
    rec = _ynn("e2e")
    gens = [StubModel("Qwen-4B", lambda u: "Yes"),
            StubModel("Gemma-E2B", lambda u: "No")]
    judge = StubModel("LFM", lambda u: "Yes", vote_weight=1.5, model_class="8b",
                      always_cot=True)
    cands = generate_candidates(gens, rec, max_new_tokens=64)
    assert [c.answer for c in cands] == ["Yes", "No"]
    assert all(c.role == "generator" and c.premises_used == [0] for c in cands)
    jr = judge_decide(judge, rec, cands, max_new_tokens=64)
    assert jr.role == "judge" and jr.answer == "Yes" and jr.chosen == 1
    final = finalize_judged(rec, cands, jr)
    assert final.answer == "Yes" and final.decider == DECIDER_JUDGE
    assert len(final.replies) == 3


def test_submission_list_maps_answers_to_competition_format():
    from logio import submission_list
    mcq = Record(id="m1", premises_nl=["p1", "p2"], question_nl="Which?",
                 answer_type=AnswerType.MCQ, options=["first", "second"])
    ynn = _ynn("y1")
    f_mcq = finalize_judged(mcq, [_gen_reply("A", "B", pu=(1,))], _judge_reply("B", pu=(1,)))
    f_ynn = finalize_judged(ynn, [_gen_reply("A", "Unknown")], _judge_reply("Unknown"))
    out = submission_list([mcq, ynn], [f_mcq, f_ynn])
    assert out[0]["query_id"] == "m1"
    assert out[0]["answer"] == "second"              # exact option text, not "B"
    assert out[0]["premises_used"] == [1] and out[0]["unit"] == ""
    assert out[1]["answer"] == "Uncertain"           # Unknown → competition label
    assert out[1]["reasoning"]["type"] == "fol" and out[1]["reasoning"]["steps"]
