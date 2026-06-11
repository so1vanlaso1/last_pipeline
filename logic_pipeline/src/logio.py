"""Run logging — timestamped .txt files under Result/, in the spirit of the old
pipeline's run_*.txt / *_model_io.txt.

Four artifacts share one timestamp:
  run_cascade_<stamp>.txt              human-readable per-record summary + accuracy
  run_cascade_<stamp>_model_io.txt     the FULL prompt + raw output of EVERY model
                                       invocation (generators and judge / voters)
  run_cascade_<stamp>.json             machine-readable predictions (full detail)
  run_cascade_<stamp>_submission.json  competition-format result list (Section 4):
                                       deterministic code, not model output
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from schema import AnswerType, FinalAnswer, Record


def result_dir(root: Path) -> Path:
    d = root / "Result"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _wrap(s: str, n: int = 120) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _kind(rec: Record) -> str:
    return "MCQ" if rec.answer_type == AnswerType.MCQ else "YNN"


def _fmt_scores(scores: dict[str, float]) -> str:
    """Render a weighted tally, strongest label first: 'Yes=2.0, No=1.5'."""
    if not scores:
        return "(no votes)"
    return ", ".join(f"{k}={v:g}" for k, v in
                     sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def write_run_summary(
    path: Path, header: dict, records: list[Record], finals: list[FinalAnswer],
    n_correct: int, n_scored: int, elapsed_s: float,
) -> Path:
    """`finals` is aligned to `records` by position (finals[i] is records[i])."""
    lines: list[str] = ["=" * 78, f"Run summary — {datetime.now():%Y-%m-%d %H:%M:%S}"]
    for k, v in header.items():
        lines.append(f"{k}={v}")
    acc = f"{n_correct}/{n_scored} = {n_correct / n_scored:.1%}" if n_scored else "n/a (no --show-gold)"
    lines.append(f"records={len(records)}   accuracy={acc}   elapsed={elapsed_s:.1f}s")
    lines.append("=" * 78)

    for i, (rec, f) in enumerate(zip(records, finals), 1):
        lines.append("")
        lines.append(f"### [{i}/{len(records)}] {rec.id}   [{_kind(rec)}]")
        lines.append(f"Q: {_wrap(rec.question_nl)}")
        if rec.answer_type == AnswerType.MCQ and rec.options:
            for j, o in enumerate(rec.options):
                lines.append(f"   {chr(ord('A') + j)}. {_wrap(o, 90)}")
        if f is None:
            lines.append("  (no verdict)")
            continue
        # Every model's reply, in the order the stages ran (role marks the flow).
        for rep in f.replies:
            tag = rep.role if rep.role != "voter" else f"w={rep.weight:g}"
            pu = f"  premises={[p + 1 for p in rep.premises_used]}" if rep.premises_used else ""
            lines.append(f"  {rep.role:<9} {rep.model_label:<18} ({tag}, {rep.model_class}) "
                         f"-> {rep.answer!r}  ({_wrap(rep.answer_display, 60)}){pu}")
        lines.append(f"  tally: {_fmt_scores(f.scores)}   "
                     f"[{'UNANIMOUS' if f.agreed else 'split'}]")
        lines.append(f"  >> ANSWER: {f.answer_display!r}   (confidence {f.confidence:.2f}, via {f.decider})")
        if f.premises_used:
            lines.append(f"     premises_used (0-based): {f.premises_used}")
        if f.explanation:
            lines.append(f"     WHY: {_wrap(f.explanation, 200)}")
        if f.gold is not None:
            ok = (f.answer or "").strip().lower() == f.gold.strip().lower()
            lines.append(f"     gold: {f.gold!r}   [{'CORRECT' if ok else 'WRONG'}]")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_model_io(path: Path, records: list[Record], finals: list[FinalAnswer]) -> Path:
    """Verbatim record of every model call: the exact prompt sent and the exact
    raw text returned, in invocation/stage order. `finals` is aligned to
    `records` by position."""
    lines = [f"MODEL I/O — {datetime.now():%Y-%m-%d %H:%M:%S}",
             "Every model invocation, verbatim (prompt in, raw text out).",
             "=" * 78]
    for i, (rec, f) in enumerate(zip(records, finals), 1):
        lines.append("")
        lines.append(f"### [{i}] {rec.id}   [{_kind(rec)}]")
        if not f.replies:
            lines.append("  (no model output)")
            continue
        for n, rep in enumerate(f.replies, 1):
            lines.append("")
            lines.append(f"-- MODEL {n}: {rep.model_label}  ({rep.model_id})  "
                         f"[w={rep.weight:g} {rep.model_class}]  [{rep.elapsed_s:.2f}s] --")
            lines.append("  IN  (prompt) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            lines.append(_indent(rep.prompt))
            lines.append("  OUT (raw)    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            lines.append(_indent(rep.raw))
            lines.append(f"  PARSED: answer={rep.answer!r}  display={rep.answer_display!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + ln for ln in (text or "").splitlines()) or (prefix + "(empty)")


def predictions_dict(records: list[Record], finals: list[FinalAnswer]) -> dict:
    """`finals` is aligned to `records` by position. Keyed by record id (ids are
    globally unique, see data_load._record_id); if a duplicate id ever slipped
    through, a `#N` suffix keeps both entries instead of silently dropping one."""
    out: dict[str, dict] = {}
    for rec, f in zip(records, finals):
        key = rec.id if rec.id not in out else f"{rec.id}#{len(out)}"
        out[key] = {
            "answer_type": rec.answer_type.value,
            "answer": f.answer,
            "answer_display": f.answer_display,
            "explanation": f.explanation,
            "premises_used": f.premises_used,
            "agreed": f.agreed,
            "decider": f.decider,
            "confidence": f.confidence,
            "scores": {k: round(v, 3) for k, v in f.scores.items()},
            "gold": f.gold,
            "elapsed_s": round(f.elapsed_s, 3),
            "votes": [
                {"model": r.model_label, "role": r.role, "answer": r.answer,
                 "display": r.answer_display, "premises_used": r.premises_used,
                 "weight": r.weight, "class": r.model_class}
                for r in f.replies
            ],
        }
    return out


def write_predictions_json(path: Path, records: list[Record], finals: dict[str, FinalAnswer]) -> Path:
    path.write_text(
        json.dumps(predictions_dict(records, finals), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


# ── Competition submission format (Submission Guide, Section 4) ───────────────
def _submission_answer(rec: Record, f: FinalAnswer) -> str:
    """The exact answer string the competition expects: for MCQ the chosen
    option's text (the endpoint must return exactly one of the given options);
    for Yes/No/Not-Given the Yes/No/Uncertain label. Built from the canonical
    answer by code — never from raw model prose."""
    canon = f.answer
    if rec.answer_type == AnswerType.MCQ:
        if canon and len(canon) == 1 and rec.options:
            idx = ord(canon.upper()) - 65
            if 0 <= idx < len(rec.options):
                return rec.options[idx]
        return "Uncertain" if canon in (None, "Unknown") else str(canon)
    return {"Yes": "Yes", "No": "No", "Unknown": "Uncertain"}.get(canon or "", "Uncertain")


def submission_list(records: list[Record], finals: list[FinalAnswer]) -> list[dict]:
    """`finals` aligned to `records` by position → the Section 4 result list:
    [{query_id, answer, unit, explanation, premises_used, reasoning}, …]."""
    out: list[dict] = []
    for rec, f in zip(records, finals):
        answer = _submission_answer(rec, f)
        explanation = (f.explanation or "").strip() or f"Based on the premises, the answer is {answer}."
        steps = [explanation]
        if f.premises_used:
            steps.append("Premises used: " + ", ".join(str(i) for i in f.premises_used) + " (0-based).")
        out.append({
            "query_id": rec.id,
            "answer": answer,
            "unit": "",
            "explanation": explanation,
            "premises_used": list(f.premises_used),
            "reasoning": {"type": "fol", "steps": steps},
        })
    return out


def write_submission_json(path: Path, records: list[Record], finals: list[FinalAnswer]) -> Path:
    path.write_text(
        json.dumps(submission_list(records, finals), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
