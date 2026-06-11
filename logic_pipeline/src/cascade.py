"""The cascade decision logic — two flows over the same prompts/parsing.

1. Generate→judge (default, `--mode judge`):
   Stage 1 — the two 4B generators (Qwen3.5-4B, Gemma-4-E2B) answer CONCURRENTLY
   in thinking mode, each emitting {answer, premises_used, explanation}.
   Stage 2 — the 8B judge (LiquidAI/LFM2.5-8B-A1B) sees the original premises +
   question plus both candidates (reference only) and decides the final answer,
   re-deriving premises_used + explanation itself. `finalize_judged` then builds
   the FinalAnswer deterministically (the submission JSON comes from code, not
   from the model).

2. Weighted soft vote (legacy, `--mode vote`): every selected model votes; each
   contributes its weight (4B → 1.0, 8B → 1.5) to the label it picked and the
   heaviest label wins.

The VRAM invariant (≤ two 4B models OR one 8B model resident) is enforced by the
caller (`run_cascade.py`), which loads/runs/unloads one stage at a time. This
module is pure logic.
"""

from __future__ import annotations

import concurrent.futures

from schema import FinalAnswer, ModelReply, Record
from prompts import (
    build_user, generator_system, judge_system, judge_user, parse_generator_reply,
    parse_judge_reply, parse_reply, system_for,
)

# Default vote weights by model size class.
WEIGHT_4B = 1.0
WEIGHT_8B = 1.5

DECIDER_UNANIMOUS = "unanimous vote"
DECIDER_VOTE = "weighted vote"
DECIDER_NONE = "no parseable answer"
DECIDER_JUDGE = "8B judge"
DECIDER_JUDGE_CHOSEN = "8B judge (endorsed a junior)"
DECIDER_GEN_AGREE = "generators agree (judge unparseable)"
DECIDER_GEN_FALLBACK = "generator fallback (judge unparseable)"

_EPS = 1e-9


def query(model, record: Record, max_new_tokens: int = 256) -> ModelReply:
    """Ask one model for its answer + compact explanation on one record.

    Never raises: a generation failure becomes a ModelReply with answer=None and
    the error captured in `raw` (so it still shows up in the model-I/O log, and
    the record simply gets one fewer vote instead of aborting the whole batch)."""
    system = system_for(record)
    user = build_user(record)
    # Render the prompt up front so it is logged even if generation throws.
    try:
        prompt = (model.render(system, user) if hasattr(model, "render")
                  else f"[SYSTEM]\n{system}\n\n[USER]\n{user}")
    except Exception:  # noqa: BLE001
        prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
    raw, elapsed = "", 0.0
    try:
        raw, prompt2, elapsed = model.generate(system, user, max_new_tokens=max_new_tokens)
        prompt = prompt2 or prompt
    except Exception as e:  # noqa: BLE001 — keep the batch alive, log the failure
        raw = f"<generation error: {type(e).__name__}: {e}>"
    canon, display, why = parse_reply(raw, record)
    return ModelReply(
        model_label=model.label,
        model_id=model.model_id,
        prompt=prompt,
        raw=raw,
        answer=canon,
        answer_display=display or (canon or ""),
        explanation=why,
        elapsed_s=elapsed,
        weight=float(getattr(model, "vote_weight", WEIGHT_4B)),
        model_class=getattr(model, "model_class", "4b"),
    )


# ── Weighted soft vote ────────────────────────────────────────────────────────
def tally(replies: list[ModelReply]) -> dict[str, float]:
    """Sum each model's weight onto the label it voted for (None votes abstain)."""
    scores: dict[str, float] = {}
    for r in replies:
        if r.answer is None:
            continue
        scores[r.answer] = scores.get(r.answer, 0.0) + r.weight
    return scores


def _break_tie(winners: list[str], replies: list[ModelReply]) -> str:
    """Tied total weight → defer to the single strongest individual vote; if
    still tied, to the latest stage that voted for it (more authoritative). For a
    pure 2×4B split (1.0 vs 1.0) this resolves to the later judge."""
    best_label, best_key = winners[0], None
    for idx, r in enumerate(replies):
        if r.answer in winners:
            key = (r.weight, idx)  # higher weight first, then later in the run
            if best_key is None or key > best_key:
                best_key, best_label = key, r.answer
    return best_label


def _pick_explainer(label: str, replies: list[ModelReply]) -> ModelReply | None:
    """The explanation for the final answer comes from the strongest model that
    voted for `label` and actually wrote a WHY; falls back to the strongest such
    model even if its WHY is empty."""
    voters = [r for r in replies if r.answer == label]
    if not voters:
        return None
    # Highest weight first, then later stage (an 8B's reasoning beats a 4B's).
    voters_ranked = sorted(
        range(len(voters)), key=lambda i: (voters[i].weight, i), reverse=True
    )
    for i in voters_ranked:
        if voters[i].explanation.strip():
            return voters[i]
    return voters[voters_ranked[0]]


def finalize_by_vote(record: Record, replies: list[ModelReply]) -> FinalAnswer:
    """Combine every model's vote on one record into the final answer."""
    scores = tally(replies)
    elapsed = sum(r.elapsed_s for r in replies)
    if not scores:
        # Nobody produced a parseable answer.
        return FinalAnswer(
            id=record.id, answer_type=record.answer_type, answer=None,
            answer_display="", explanation="", decider=DECIDER_NONE,
            agreed=False, confidence=0.0, replies=replies, scores=scores,
            elapsed_s=elapsed,
        )

    total = sum(scores.values())
    best = max(scores.values())
    winners = [lbl for lbl, w in scores.items() if abs(w - best) < _EPS]
    win = winners[0] if len(winners) == 1 else _break_tie(winners, replies)

    voters = [r for r in replies if r.answer is not None]
    unanimous = len(scores) == 1 and len(voters) == len(replies)
    explainer = _pick_explainer(win, replies)

    return FinalAnswer(
        id=record.id,
        answer_type=record.answer_type,
        answer=win,
        answer_display=(explainer.answer_display if explainer else win),
        explanation=(explainer.explanation if explainer else ""),
        decider=DECIDER_UNANIMOUS if unanimous else DECIDER_VOTE,
        agreed=unanimous,
        confidence=best / total if total else 0.0,
        replies=replies,
        scores=scores,
        elapsed_s=elapsed,
    )


# ══ Generate→judge flow ═══════════════════════════════════════════════════════
def _safe_generate(model, system: str, user: str, max_new_tokens: int) -> tuple[str, str, float]:
    """generate() that never raises — a failure becomes an error-marker raw text
    (logged verbatim) and the record just loses that candidate/verdict."""
    try:
        prompt = (model.render(system, user) if hasattr(model, "render")
                  else f"[SYSTEM]\n{system}\n\n[USER]\n{user}")
    except Exception:  # noqa: BLE001
        prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
    try:
        raw, prompt2, elapsed = model.generate(system, user, max_new_tokens=max_new_tokens)
        return raw, (prompt2 or prompt), elapsed
    except Exception as e:  # noqa: BLE001
        return f"<generation error: {type(e).__name__}: {e}>", prompt, 0.0


def generate_candidate(model, record: Record, max_new_tokens: int = 1024) -> ModelReply:
    """Stage 1: one generator thinks and emits {answer, premises_used, explanation}."""
    system = generator_system(record)
    user = build_user(record)
    raw, prompt, elapsed = _safe_generate(model, system, user, max_new_tokens)
    canon, display, why, pu = parse_generator_reply(raw, record)
    return ModelReply(
        model_label=model.label, model_id=model.model_id, prompt=prompt, raw=raw,
        answer=canon, answer_display=display or (canon or ""), explanation=why,
        elapsed_s=elapsed, weight=float(getattr(model, "vote_weight", WEIGHT_4B)),
        model_class=getattr(model, "model_class", "4b"),
        role="generator", premises_used=pu,
    )


def generate_candidates(models, record: Record, max_new_tokens: int = 1024) -> list[ModelReply]:
    """Both resident generators answer the SAME record concurrently (the models
    coexist in VRAM; generate() releases the GIL during CUDA work). Reply order
    follows `models` order, so logs stay deterministic."""
    if len(models) == 1:
        return [generate_candidate(models[0], record, max_new_tokens)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as ex:
        return list(ex.map(lambda m: generate_candidate(m, record, max_new_tokens), models))


def judge_decide(model, record: Record, candidates: list[ModelReply],
                 max_new_tokens: int = 1024) -> ModelReply:
    """Stage 2: the 8B judge inspects the candidates and rules. The parsed
    verdict dict rides along for finalize_judged (stored on the reply)."""
    system = judge_system(record, len(candidates))
    user = judge_user(record, candidates)
    raw, prompt, elapsed = _safe_generate(model, system, user, max_new_tokens)
    verdict = parse_judge_reply(raw, record)
    reply = ModelReply(
        model_label=model.label, model_id=model.model_id, prompt=prompt, raw=raw,
        answer=verdict["canon"], answer_display=verdict["display"] or (verdict["canon"] or ""),
        explanation=verdict["explanation"], elapsed_s=elapsed,
        weight=float(getattr(model, "vote_weight", WEIGHT_8B)),
        model_class=getattr(model, "model_class", "8b"),
        role="judge", premises_used=verdict["premises_used"],
    )
    reply.chosen = verdict["chosen"]  # type: ignore[attr-defined] — judge-only extra
    return reply


def _chosen_candidate(judge: ModelReply | None, candidates: list[ModelReply]) -> ModelReply | None:
    idx = getattr(judge, "chosen", None) if judge is not None else None
    if isinstance(idx, int) and 1 <= idx <= len(candidates):
        return candidates[idx - 1]
    return None


def finalize_judged(record: Record, candidates: list[ModelReply],
                    judge: ModelReply | None) -> FinalAnswer:
    """Deterministically combine the candidates + the judge's verdict.

    Precedence: the judge's own answer; else the junior the judge endorsed via
    "chosen"; else generator agreement; else the first generator with an answer.
    The explanation/premises_used come from whoever decided (judge first, with
    the endorsed/agreeing candidate as fallback when the judge's are empty)."""
    replies = list(candidates) + ([judge] if judge is not None else [])
    elapsed = sum(r.elapsed_s for r in replies)
    scores = tally(replies)
    answered = [r for r in replies if r.answer is not None]
    agreed = bool(answered) and len({r.answer for r in answered}) == 1 \
        and len(answered) == len(replies)

    chosen = _chosen_candidate(judge, candidates)
    win: str | None = None
    source: ModelReply | None = None
    if judge is not None and judge.answer is not None:
        win, source = judge.answer, judge
        decider = DECIDER_UNANIMOUS if agreed else DECIDER_JUDGE
    elif chosen is not None and chosen.answer is not None:
        win, source = chosen.answer, chosen
        decider = DECIDER_JUDGE_CHOSEN
    else:
        cand_answers = {c.answer for c in candidates if c.answer is not None}
        if len(cand_answers) == 1:
            win = next(iter(cand_answers))
            source = next(c for c in candidates if c.answer == win)
            decider = DECIDER_GEN_AGREE
        else:
            source = next((c for c in candidates if c.answer is not None), None)
            win = source.answer if source else None
            decider = DECIDER_GEN_FALLBACK if source else DECIDER_NONE

    # Explanation/premises: the decider's own, falling back to the endorsed (or
    # any agreeing) candidate so the final record is never silently empty.
    explanation = (source.explanation if source else "").strip()
    premises_used = list(source.premises_used) if source else []
    backups = [c for c in (chosen, *candidates) if c is not None and c.answer == win]
    for b in backups:
        if not explanation and b.explanation.strip():
            explanation = b.explanation.strip()
        if not premises_used and b.premises_used:
            premises_used = list(b.premises_used)

    total = sum(scores.values())
    confidence = (scores.get(win, 0.0) / total) if (win is not None and total) else 0.0
    return FinalAnswer(
        id=record.id, answer_type=record.answer_type, answer=win,
        answer_display=(source.answer_display if source else (win or "")),
        explanation=explanation, decider=decider, agreed=agreed,
        confidence=confidence, replies=replies, scores=scores,
        elapsed_s=elapsed, premises_used=premises_used,
    )
