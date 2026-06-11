"""Type 1 (logic) adapter.

Reuses the logic-cascade's *exact* prompt construction, answer parsing, and
WEIGHTED VOTE (`logic_pipeline/src/{prompts,cascade,schema}.py`) so the decision
logic is preserved — only the model backend changes from local HuggingFace to one
or more shared vLLM servers.

The Type 1 answer is `cascade.finalize_by_vote` over every resident model (1 or 2,
chosen in serve/logic_config.yaml). For a single-model line-up the "vote" is just
that model; for the two-4B line-up it is the cascade's real weighted vote.

Sub-paths by question shape:
  * Yes/No(/Uncertain) choice -> cascade Yes/No/Not-Given examiner, vote, map to option.
  * content multiple-choice    -> cascade MCQ examiner, vote, map to option.
  * no options (free-form)      -> primary model, JSON CoT (number/text) answer.

`premises_used` (0-based) comes from the winning model's premise citations, with a
cheap JSON fallback call (primary model) that never changes the voted answer.
"""

from __future__ import annotations

import concurrent.futures
import os
from typing import Any, Dict, List, Optional, Tuple

from . import _paths  # noqa: F401  (side-effect: put logic_pipeline/src on sys.path)
from . import config as cfg
from .io_log import model_labels
from .residency import get_manager
from .schema import PredictQuery, PredictResult, Reasoning
from .units import (
    clamp_indices, find_no_option, find_uncertain_option, find_yes_option,
    looks_like_ynn, map_letter_to_option, match_text_to_option, premises_from_text,
)
from .vllm_client import LLMClient, extract_json_object, extract_last_json_object

# Cascade IP (pure-python modules: re + dataclasses, no torch).
from cascade import finalize_by_vote  # type: ignore
from prompts import (  # type: ignore
    build_user, canonicalize, parse_reply, rules_for, system_for,
)
from schema import AnswerType, ModelReply, Record  # type: ignore

# A resident model: client + cascade vote weight + size class (+ optional role,
# "generator"/"judge", from serve/logic_config.yaml). Old 3-tuples still work.
Judge = Tuple[LLMClient, float, str, str]


def _role(j) -> str:
    return (j[3] if len(j) > 3 else "") or ""


def _chat(
    client: LLMClient,
    system: str,
    user: str,
    *,
    query_id: str,
    stage: str,
    loaded_clients: List[LLMClient],
    max_tokens: int,
    temperature: float = 0.0,
    enable_thinking: Optional[bool] = None,
) -> str:
    return client.chat(
        system,
        user,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        log_context=f"type1 query_id={query_id or 'q'} stage={stage}",
        loaded_models=model_labels(loaded_clients),
    )


def _chat_json(
    client: LLMClient,
    system: str,
    user: str,
    *,
    query_id: str,
    stage: str,
    loaded_clients: List[LLMClient],
    max_tokens: int,
    temperature: float = 0.0,
) -> Optional[dict]:
    return client.chat_json(
        system,
        user,
        max_tokens=max_tokens,
        temperature=temperature,
        log_context=f"type1 query_id={query_id or 'q'} stage={stage}",
        loaded_models=model_labels(loaded_clients),
    )


def _make_record(q: PredictQuery, as_mcq: bool) -> Record:
    return Record(
        id=q.query_id or "q",
        premises_nl=list(q.premises or []),
        question_nl=q.query or "",
        answer_type=AnswerType.MCQ if as_mcq else AnswerType.YES_NO_UNKNOWN,
        answer=None,
        options=list(q.options or []) or None,
        raw={},
    )


def _collect_votes(judges: List[Judge], record: Record) -> List[ModelReply]:
    """Ask every resident model the same record; build one cascade ModelReply each."""
    system = system_for(record)
    user = build_user(record)
    replies: List[ModelReply] = []
    loaded_clients = [j[0] for j in judges]
    for client, weight, cls, *_ in judges:
        try:
            raw = _chat(
                client, system, user, query_id=record.id, stage="vote",
                loaded_clients=loaded_clients, max_tokens=512,
            )
        except Exception as exc:  # a dead judge just abstains; the vote continues
            raw = f"<generation error: {exc}>"
        canon, display, why = parse_reply(raw, record)
        replies.append(ModelReply(
            model_label=client.model, model_id=client.model,
            prompt=user, raw=raw, answer=canon,
            answer_display=display or (canon or ""), explanation=why,
            elapsed_s=0.0, weight=weight, model_class=cls,
        ))
    return replies


def _premises_used(
    primary: LLMClient, why: str, premises: List[str], question: str, answer: str,
    query_id: str = "", loaded_clients: Optional[List[LLMClient]] = None,
) -> List[int]:
    """0-based premise indices. Prefer citations in the winning WHY; otherwise ask
    the primary model in a small JSON call that does not touch the answer."""
    if not premises:
        return []
    cited = premises_from_text(why, len(premises))
    if cited:
        return cited
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(premises))
    system = (
        "You identify which premises are needed to justify a given answer. "
        "Return JSON only, exactly: {\"premises_used\": [<1-based premise numbers>]}. "
        "Include only premises that are actually required; no prose."
    )
    user = (
        f"Premises:\n{numbered}\n\nQuestion: {question}\nGiven answer: {answer}\n\n"
        "Which premise numbers are needed to derive that answer?"
    )
    try:
        data = _chat_json(
            primary, system, user, query_id=query_id, stage="premises_used",
            loaded_clients=loaded_clients or [primary], max_tokens=128,
        )
    except Exception:
        data = None
    if isinstance(data, dict) and isinstance(data.get("premises_used"), list):
        zero = [int(x) - 1 for x in data["premises_used"] if str(x).strip().lstrip("-").isdigit()]
        return clamp_indices(zero, len(premises))
    return []


def _fol_reasoning(why: str, premises_used: List[int]) -> Reasoning:
    steps: List[str] = []
    if why:
        steps.append(why)
    if premises_used:
        steps.append("Premises used: " + ", ".join(str(i) for i in premises_used) + " (0-based).")
    if not steps:
        steps.append("Derived from the given premises.")
    return Reasoning(type="fol", steps=steps)


def _choice(judges: List[Judge], q: PredictQuery) -> PredictResult:
    options = list(q.options or [])
    as_mcq = not looks_like_ynn(options)        # YNN examiner for Yes/No/Uncertain
    record = _make_record(q, as_mcq=as_mcq)
    replies = _collect_votes(judges, record)
    final = finalize_by_vote(record, replies)
    canon = final.answer                         # letter | "Yes"/"No"/"Unknown" | None
    display = final.answer_display
    why = final.explanation

    answer = _map_to_option(canon, display, options, as_mcq)
    if answer is None:
        loaded_clients = [j[0] for j in judges]
        answer = _choose_option_fallback(judges[0][0], q, options, loaded_clients) or (
            options[find_uncertain_option(options) or 0] if options else (display or "Uncertain")
        )

    loaded_clients = [j[0] for j in judges]
    pu = _premises_used(
        judges[0][0], why, list(q.premises or []), q.query, answer,
        q.query_id, loaded_clients,
    )
    explanation = (why or f"Based on the premises, the answer is {answer}.").strip()
    return PredictResult(
        query_id=q.query_id, answer=answer, unit="", explanation=explanation,
        premises_used=pu, reasoning=_fol_reasoning(why, pu),
    )


def _map_to_option(
    canon: Optional[str], display: str, options: List[str], as_mcq: bool
) -> Optional[str]:
    if not options:
        return None
    if not as_mcq:                               # Yes / No / Uncertain
        idx = None
        if canon == "Yes":
            idx = find_yes_option(options)
        elif canon == "No":
            idx = find_no_option(options)
        elif canon == "Unknown":
            idx = find_uncertain_option(options)
        if idx is not None:
            return options[idx]
        return match_text_to_option(display or "", options)
    # content MCQ
    if canon and canon != "Unknown":
        return map_letter_to_option(canon, options) or match_text_to_option(display or canon, options)
    if canon == "Unknown":
        idx = find_uncertain_option(options)
        if idx is not None:
            return options[idx]
    return None


def _choose_option_fallback(
    client: LLMClient, q: PredictQuery, options: List[str],
    loaded_clients: Optional[List[LLMClient]] = None,
) -> Optional[str]:
    """If the vote did not land on a listed option, ask the primary model to pick
    the index of the best option (constrained), then map it back exactly."""
    numbered = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
    prem = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(q.premises or [])) or "(none)"
    system = (
        "You pick the single best answer option using only the premises. "
        "Return JSON only: {\"option_index\": <0-based index of the chosen option>}."
    )
    user = f"Premises:\n{prem}\n\nQuestion: {q.query}\n\nOptions:\n{numbered}"
    try:
        data = _chat_json(
            client, system, user, query_id=q.query_id, stage="option_fallback",
            loaded_clients=loaded_clients or [client], max_tokens=64,
        )
    except Exception:
        data = None
    if isinstance(data, dict):
        try:
            idx = int(data.get("option_index"))
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < len(options):
            return options[idx]
    return None


def _free_form(client: LLMClient, q: PredictQuery) -> PredictResult:
    premises = list(q.premises or [])
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(premises)) or "(none)"
    system = (
        "You are a careful logic and arithmetic assistant for an educational QA task. "
        "Use ONLY the given premises and the question; do not use outside knowledge. "
        "Return JSON only, exactly these fields: "
        "{\"answer\": <the final answer as a number or short text>, "
        "\"premises_used\": [<1-based premise numbers actually used>], "
        "\"explanation\": <one or two sentences>}."
    )
    user = f"Premises:\n{numbered}\n\nQuestion: {q.query}"
    try:
        data = _chat_json(
            client, system, user, query_id=q.query_id, stage="free_form",
            loaded_clients=[client], max_tokens=512,
        )
    except Exception:
        data = None

    if not isinstance(data, dict):
        text = _chat(
            client,
            "Answer the question using only the premises. Reply with just the final answer.",
            user,
            query_id=q.query_id,
            stage="free_form_fallback",
            loaded_clients=[client],
            max_tokens=256,
        ).strip()
        answer = text.splitlines()[0].strip() if text else "Uncertain"
        return PredictResult(
            query_id=q.query_id, answer=answer or "Uncertain", unit="",
            explanation=f"The answer is {answer}.", premises_used=[],
            reasoning=Reasoning(type="fol", steps=[f"Answer: {answer}"]),
        )

    answer = str(data.get("answer", "")).strip() or "Uncertain"
    zero = [int(x) - 1 for x in (data.get("premises_used") or []) if str(x).strip().lstrip("-").isdigit()]
    pu = clamp_indices(zero, len(premises))
    explanation = str(data.get("explanation", "")).strip() or f"The answer is {answer}."
    return PredictResult(
        query_id=q.query_id, answer=answer, unit="", explanation=explanation,
        premises_used=pu, reasoning=_fol_reasoning(explanation, pu),
    )


# ── Arbiter (judge) flow ─────────────────────────────────────────────────────
# Stage 1: the GENERATOR models (role "generator"; the two thinking 4B juniors —
# Qwen3.5-4B + Gemma-4-E2B) answer CONCURRENTLY, each emitting {answer,
# premises_used, explanation}. Stage 2: the JUDGE model (role "judge"; the
# Gemma-4-E4B 8B) inspects the original premises + question plus both
# candidates (reference only) and decides the truly correct answer, RE-DERIVING
# premises_used + explanation itself. The submission object is then assembled by
# deterministic code (canonicalize → exact option mapping → PredictResult).
# Line-ups without roles keep the old behaviour: the strongest model arbitrates;
# a single-model line-up makes two passes (strict + skeptical) and self-judges.

_SKEPTIC = (
    "\n\nFor THIS pass be especially skeptical: actively check whether the opposite "
    "conclusion — or 'Not Given' / no-entailment — is in fact the correct answer."
)


def _think_tokens() -> int:
    """Max tokens for the reasoning calls (generators' THINK pass + the judge).

    Raised from 1024 → 4096: thinking-mode models (Qwen3-4B-Thinking-2507 always
    thinks; gemma-4-E4B-it thinks when asked) spend the budget on a <think> chain
    BEFORE the final JSON. At 1024 the chain ate the whole budget → finish_reason
    =length with EMPTY content after the think block is stripped (the "empty
    content / corrupt generation" symptom). 4096 leaves room to finish reasoning
    AND emit the JSON within max_model_len=8192. Env LOGIC_THINK_TOKENS overrides.
    """
    return int(os.environ.get("LOGIC_THINK_TOKENS", "4096"))


def _answer_space(record: Record) -> str:
    if record.answer_type == AnswerType.MCQ and record.options:
        letters = ", ".join(chr(65 + i) for i in range(len(record.options)))
        return f"<exactly one option letter: {letters}>"
    return "<exactly one of: Yes, No, Not Given>"


def _gen_format(record: Record) -> str:
    return (
        "\n\nWork through the premises step by step (you may think first). Then Finish your "
        "reply with ONE JSON object on its own line and nothing after it:\n"
        '{"answer": ' + _answer_space(record)
        + ', "premises_used": [<1-based numbers of the premises you actually used>], '
        '"explanation": <2-3 sentences citing those premise numbers>}'
    )


def _to_zero_based(values: Any, n: int) -> List[int]:
    zero = [int(x) - 1 for x in (values or []) if str(x).strip().lstrip("-").isdigit()]
    return clamp_indices(zero, n)


def _generate_candidate(
    client: LLMClient, record: Record, persona: str, loaded_clients: List[LLMClient],
) -> Dict[str, Any]:
    # rules_for strips the examiner rules' trailing "Reply in EXACTLY this
    # format: ANSWER:/WHY:" line — it would compete with the JSON instruction.
    system = rules_for(record) + persona + _gen_format(record)
    user = build_user(record)
    try:
        # enable_thinking defaults to the model's configured thinking mode.
        text = _chat(
            client, system, user, query_id=record.id, stage="arbiter.generator",
            loaded_clients=loaded_clients, max_tokens=_think_tokens(),
        )
    except Exception:
        text = ""
    data = extract_last_json_object(text) or {}
    ans_raw = str(data.get("answer", "")).strip()
    canon, display = canonicalize(ans_raw, record) if ans_raw else (None, "")
    return {
        "label": client.model,
        "canon": canon,
        "display": display or ans_raw,
        "answer_raw": ans_raw,
        "premises_used": _to_zero_based(data.get("premises_used"), len(record.premises_nl)),
        "explanation": str(data.get("explanation", "")).strip(),
    }


def _render_candidates(cands: List[Dict[str, Any]]) -> str:
    out = []
    for i, c in enumerate(cands, 1):
        pu1 = [j + 1 for j in c["premises_used"]]
        out.append(
            f"Junior {i} ({c['label']}): answer = {c['display'] or c['answer_raw'] or 'N/A'}; "
            f"premises_used = {pu1}; explanation = {c['explanation'] or '(none)'}"
        )
    return "\n".join(out)


def _chosen_candidate(data: Dict[str, Any], cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    try:
        idx = int(data.get("chosen"))
    except (TypeError, ValueError):
        return cands[0] if cands else None
    if 1 <= idx <= len(cands):
        return cands[idx - 1]
    return cands[0] if cands else None


def _run_generators(gen_specs, record: Record) -> List[Dict[str, Any]]:
    loaded_clients = [spec[0] for spec in gen_specs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gen_specs)) as ex:
        return list(ex.map(
            lambda s: _generate_candidate(s[0], record, s[1], loaded_clients),
            gen_specs,
        ))


def _arbiter_choice(gen_specs, arbiter: LLMClient, q: PredictQuery) -> PredictResult:
    options = list(q.options or [])
    as_mcq = not looks_like_ynn(options)
    record = _make_record(q, as_mcq=as_mcq)

    mgr = get_manager()
    mgr.ensure_generators()                      # generators awake for stage 1
    cands = _run_generators(gen_specs, record)

    system = (
        "You are the SENIOR examiner and arbiter. Two junior examiners each answered the "
        "same question. Using ONLY the premises, decide which junior's ANSWER is correct; "
        "if both are wrong, give the correct answer yourself. Then INDEPENDENTLY work out "
        "which premises are actually required and write the explanation in your own words "
        "— use the juniors' work only as a reference, do NOT copy it.\n\n"
        "Apply these examiner rules:\n" + rules_for(record)
        + "\n\nFinish your reply with ONE JSON object on its own line:\n"
        '{"chosen": <1 or 2 — the junior you judged correct>, "answer": ' + _answer_space(record)
        + ', "premises_used": [<1-based premise numbers you determine are needed>], '
        '"explanation": <2-4 sentences in your own words citing those premises>}'
    )
    user = build_user(record) + "\n\nJunior answers (reference only):\n" + _render_candidates(cands)

    # Stage 2: swap the judge in (generators sleep) for the arbitration AND any
    # arbiter-backed fallbacks below, then the generators wake back up on exit.
    with mgr.judge():
        try:
            text = _chat(
                arbiter, system, user, query_id=q.query_id, stage="arbiter.judge",
                loaded_clients=[arbiter], max_tokens=_think_tokens(),
            )
        except Exception:
            text = ""
        data = extract_last_json_object(text) or {}

        ans_raw = str(data.get("answer", "")).strip()
        canon, display = canonicalize(ans_raw, record) if ans_raw else (None, "")
        answer = _map_to_option(canon, display, options, as_mcq)
        chosen = _chosen_candidate(data, cands)
        if answer is None and chosen and chosen["canon"]:
            answer = _map_to_option(chosen["canon"], chosen["display"], options, as_mcq)
        if answer is None:
            answer = _choose_option_fallback(arbiter, q, options, [arbiter]) or (
                options[find_uncertain_option(options) or 0] if options else "Uncertain")

        pu = _to_zero_based(data.get("premises_used"), len(q.premises or []))
        explanation = str(data.get("explanation", "")).strip()
        if not pu and chosen:
            pu = chosen["premises_used"]
        if not explanation and chosen:
            explanation = chosen["explanation"]
        if not pu:
            pu = _premises_used(
                arbiter, explanation, list(q.premises or []), q.query, answer,
                q.query_id, [arbiter],
            )
    if not explanation:
        explanation = f"Based on the premises, the answer is {answer}."
    return PredictResult(
        query_id=q.query_id, answer=answer, unit="", explanation=explanation,
        premises_used=pu, reasoning=_fol_reasoning(explanation, pu),
    )


def _arbiter_free_form(gen_specs, arbiter: LLMClient, q: PredictQuery) -> PredictResult:
    premises = list(q.premises or [])
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(premises)) or "(none)"
    gsys = (
        "You are a careful logic and arithmetic assistant. Use ONLY the premises and the "
        "question; no outside knowledge. You may think first, then Finish your reply with "
        'ONE JSON object: {"answer": <a number or short text>, "premises_used": '
        '[<1-based premise numbers used>], "explanation": <1-2 sentences>}.'
    )
    guser = f"Premises:\n{numbered}\n\nQuestion: {q.query}"

    def gen(spec):
        c, persona = spec
        try:
            text = _chat(
                c, gsys + persona, guser, query_id=q.query_id,
                stage="arbiter.free_form_generator",
                loaded_clients=[s[0] for s in gen_specs], max_tokens=_think_tokens(),
            )
        except Exception:
            text = ""
        d = extract_last_json_object(text) or {}
        return {"label": c.model, "answer": str(d.get("answer", "")).strip(),
                "premises_used": _to_zero_based(d.get("premises_used"), len(premises)),
                "explanation": str(d.get("explanation", "")).strip()}

    mgr = get_manager()
    mgr.ensure_generators()                      # generators awake for stage 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gen_specs)) as ex:
        cands = list(ex.map(gen, gen_specs))

    asys = (
        "You are the senior arbiter. Two juniors answered the same question. Using ONLY the "
        "premises, decide which answer is correct (or give the correct one yourself). Then "
        "INDEPENDENTLY determine premises_used and write the explanation in your own words "
        "(reference the juniors, do not copy). Finish with ONE JSON object: "
        '{"chosen": <1 or 2>, "answer": <a number or short text>, "premises_used": '
        '[<1-based>], "explanation": <1-2 sentences>}.'
    )
    auser = guser + "\n\nJunior answers (reference only):\n" + "\n".join(
        f"Junior {i} ({c['label']}): answer={c['answer'] or 'N/A'}; "
        f"premises_used={[j + 1 for j in c['premises_used']]}; explanation={c['explanation'] or '(none)'}"
        for i, c in enumerate(cands, 1))
    with mgr.judge():                            # stage 2: swap the judge in
        try:
            text = _chat(
                arbiter, asys, auser, query_id=q.query_id,
                stage="arbiter.free_form_judge", loaded_clients=[arbiter],
                max_tokens=_think_tokens(),
            )
        except Exception:
            text = ""
    data = extract_last_json_object(text) or {}

    answer = str(data.get("answer", "")).strip()
    chosen = _chosen_candidate(data, cands)
    if not answer:
        answer = (chosen["answer"] if chosen else "") or (cands[0]["answer"] if cands else "Uncertain")
    pu = _to_zero_based(data.get("premises_used"), len(premises))
    explanation = str(data.get("explanation", "")).strip()
    if not pu and chosen:
        pu = chosen["premises_used"]
    if not explanation and chosen:
        explanation = chosen["explanation"]
    if not explanation:
        explanation = f"The answer is {answer}."
    return PredictResult(
        query_id=q.query_id, answer=answer or "Uncertain", unit="", explanation=explanation,
        premises_used=pu, reasoning=_fol_reasoning(explanation, pu),
    )


def split_lineup(judges: List[Judge]) -> Tuple[List[LLMClient], LLMClient]:
    """(generator clients, the judge client) from the resident line-up.

    Role-tagged line-ups are explicit: role "generator" models generate, the
    role "judge" model (the Gemma-4 8B) arbitrates. Without roles, the old rule
    applies — everything generates and the highest-weight model arbitrates."""
    gens = [j[0] for j in judges if _role(j) == "generator"]
    tagged = [j[0] for j in judges if _role(j) == "judge"]
    arbiter = tagged[0] if tagged else max(judges, key=lambda j: j[1])[0]
    if not gens:
        # No "generator" tags: every resident model generates (the arbiter
        # included) — exactly the pre-role behaviour.
        gens = [j[0] for j in judges]
    return gens, arbiter


def _arbiter_flow(judges: List[Judge], q: PredictQuery) -> PredictResult:
    gens, arbiter = split_lineup(judges)
    if len(gens) >= 2:
        gen_specs = [(gens[0], ""), (gens[1], "")]                # the two 4B juniors
    else:
        gen_specs = [(gens[0], ""), (gens[0], _SKEPTIC)]           # one generator, two passes
    if q.options:
        return _arbiter_choice(gen_specs, arbiter, q)
    return _arbiter_free_form(gen_specs, arbiter, q)


def answer_type1(judges: List[Judge], q: PredictQuery) -> PredictResult:
    """`judges` is the resident model line-up: [(client, weight, model_class), ...]."""
    if cfg.load_mode() == "arbiter":
        return _arbiter_flow(judges, q)
    if q.options:
        return _choice(judges, q)
    return _free_form(judges[0][0], q)
