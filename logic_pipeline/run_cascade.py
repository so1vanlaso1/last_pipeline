#!/usr/bin/env python
"""Generate→judge runner for the EXACT-style logic dataset.

Default flow (--mode judge)
---------------------------
    Question
       │
       ├─ Qwen/Qwen3.5-4B      ┐ run CONCURRENTLY (both resident), thinking mode,
       ├─ google/gemma-4-E2B-it┘ each emits {answer, premises_used, explanation}
       │                          … then both are UNLOADED …
       ├─ LiquidAI/LFM2.5-8B-A1B (the 8B judge): sees the ORIGINAL premises +
       │   question plus both juniors' answers (reference only) and decides the
       │   truly correct answer, re-deriving premises_used + explanation itself.
       ▼
    deterministic code → competition-format submission JSON (Section 4)

The two stages still run ONE AT A TIME (two 4B resident, then one 8B resident),
so the 12 GB VRAM invariant holds. If the judge's reply is unparseable the final
answer falls back deterministically: the junior the judge endorsed ("chosen"),
then generator agreement, then the first generator with an answer.

Legacy flow (--mode vote)
-------------------------
The weighted soft vote: every selected stage votes on every record (4B → w1.0,
8B → w1.5) and the heaviest label wins. Line-up via --stages (any combination of
{4b, gemma8b, liquid8b}); --stages is only honoured in vote mode.

Question types: MCQ (pick a letter) and Yes / No / Not Given (the dataset writes
"Not Given" as "Unknown"). The type is decided from the question's structure.

Precision is a single switch applied to every model: 4bit | 8bit | bf16.
NOTE: two 4B models in bf16 will not fit 12 GB — use 4bit (or 8bit) there.

Examples
--------
    # The generate→judge flow (default), 4-bit, scored against gold:
    python run_cascade.py --precision 4bit --show-gold --limit 20

    # Same, but thinking disabled for a faster (weaker) pass:
    python run_cascade.py --no-think --precision 4bit --show-gold

    # Legacy weighted vote over all three stages:
    python run_cascade.py --mode vote --stages 4b,gemma8b,liquid8b --precision 4bit --show-gold

    # Run a specific index window — questions 100..199 of the (filtered) set:
    python run_cascade.py --precision 4bit --start 100 --end 200 --show-gold

    # No-GPU wiring smoke test (fake models, exercises both stages + logging):
    python run_cascade.py --backend stub --show-gold --limit 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Reduce CUDA allocator fragmentation across the cascade's repeated load/unload
# cycles. Freeing one stage can leave gaps that strand a few hundred MiB and
# OOM-kill a right-at-the-edge next load (e.g. the 8B MoE on a ~16 GB card, which
# died only ~224 MiB short during weight conversion). Must be set BEFORE torch /
# CUDA initialises — hence up here, ahead of the local imports that pull torch.
# Respect a value the user already exported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cascade import (  # noqa: E402
    finalize_by_vote, finalize_judged, generate_candidates, judge_decide, query,
)
from data_load import load_records  # noqa: E402
from logio import (  # noqa: E402
    result_dir, write_model_io, write_predictions_json, write_run_summary,
    write_submission_json,
)
from schema import AnswerType, FinalAnswer, ModelReply, Record  # noqa: E402
from score import score  # noqa: E402

# ── Default model ids ─────────────────────────────────────────────────────────
# These are the repo ids the user specified. If a Gemma download 404s, the
# equivalent current repos are google/gemma-3n-E2B-it / google/gemma-3n-E4B-it —
# override with --gemma-small-model / --gemma-big-model.
DEFAULT_QWEN = "Qwen/Qwen3.5-4B"
DEFAULT_GEMMA_SMALL = "google/gemma-4-E2B-it"
DEFAULT_GEMMA_BIG = "google/gemma-4-E4B-it"
DEFAULT_LIQUID = "LiquidAI/LFM2.5-8B-A1B"
DEFAULT_DATA = ROOT / "Logic_Based_Educational_Queries.json"

# Canonical stage order (4B judges first, then the 8B models). A stage is a unit
# that is loaded together; the "4b" stage holds the two judges, each 8B stage one
# model — which is exactly the VRAM invariant. In judge mode the line-up is fixed
# ("4b" generators → "liquid8b" judge); --stages only applies to vote mode.
STAGE_ORDER = ["4b", "gemma8b", "liquid8b"]
DEFAULT_STAGES = "4b,gemma8b,liquid8b"
JUDGE_GEN_STAGE = "4b"
JUDGE_STAGE = "liquid8b"


def gate(rec: Record) -> tuple[Record, str | None]:
    """Strip gold before inference; re-derive the answer type from structure.
    Returns (model-facing record, gold). `raw` is emptied except for a
    `definitions` list (never gold) so it can't leak the gold answer/explanation
    into the prompt."""
    atype = AnswerType.MCQ if rec.options else AnswerType.YES_NO_UNKNOWN
    safe_raw = {}
    if isinstance(rec.raw, dict) and rec.raw.get("definitions"):
        safe_raw["definitions"] = rec.raw["definitions"]
    gated = Record(
        id=rec.id, premises_nl=rec.premises_nl, question_nl=rec.question_nl,
        answer_type=atype, answer=None, options=rec.options, raw=safe_raw,
    )
    return gated, rec.answer


def _exc_chain(e: BaseException, limit: int = 6) -> str:
    """Flatten an exception's cause/context chain so a wrapped import failure
    shows its ROOT (e.g. the actual 'cannot import name …') instead of just the
    transformers wrapper 'Could not import module …'."""
    msgs, cur, seen = [], e, 0
    while cur is not None and seen < limit:
        msgs.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return "  <= caused by: ".join(msgs)


def parse_stages(spec: str) -> list[str]:
    """'4b , gemma8b' → ['4b', 'gemma8b'] in canonical order, validated/deduped."""
    picked = {tok.strip().lower() for tok in spec.split(",") if tok.strip()}
    bad = picked - set(STAGE_ORDER)
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown stage(s) {sorted(bad)}; choose from {STAGE_ORDER} "
            "(comma-separated, e.g. 4b,gemma8b)"
        )
    if not picked:
        raise argparse.ArgumentTypeError("at least one stage is required")
    return [s for s in STAGE_ORDER if s in picked]


def stage_registry(args) -> dict[str, list[dict]]:
    """Per-stage model specs. `stub` is the deterministic answer fn used by the
    no-GPU backend (it exercises agreement, splits, and every weight)."""
    w4, w8 = args.weight_4b, args.weight_8b
    def _mcq(u: str) -> bool:
        return "Options:" in u

    return {
        "4b": [
            dict(id=args.qwen_model, label="Qwen-4B", cls="4b", weight=w4, cot=False,
                 stub=lambda u: "A" if _mcq(u) else "Yes"),
            dict(id=args.gemma_small_model, label="Gemma-E2B", cls="4b", weight=w4, cot=False,
                 stub=lambda u: ("B" if _mcq(u)
                                 else ("No" if "scholarship" in u.lower() else "Yes"))),
        ],
        "gemma8b": [
            dict(id=args.gemma_big_model, label="Gemma-E4B(8B)", cls="8b", weight=w8, cot=False,
                 stub=lambda u: "A" if _mcq(u) else "Not Given"),
        ],
        "liquid8b": [
            dict(id=args.liquid_model, label="LFM2.5-8B-A1B", cls="8b", weight=w8, cot=True,
                 stub=lambda u: "A" if _mcq(u) else "Yes"),
        ],
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows cp1252 → force UTF-8
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--mode", choices=["judge", "vote"], default="judge",
                    help="judge (default): 2 concurrent thinking 4B generators -> the "
                         "Liquid 8B judges; vote: legacy weighted soft vote")
    ap.add_argument("--stages", type=parse_stages, default=None,
                    help=f"[vote mode only] which model groups vote, comma-separated from "
                         f"{STAGE_ORDER} (default: {DEFAULT_STAGES}); ignored in judge mode")
    ap.add_argument("--qwen-model", default=DEFAULT_QWEN, help="4B generator/judge A")
    ap.add_argument("--gemma-small-model", default=DEFAULT_GEMMA_SMALL, help="4B generator/judge B")
    ap.add_argument("--gemma-big-model", default=DEFAULT_GEMMA_BIG, help="the Gemma 8B (vote mode)")
    ap.add_argument("--liquid-model", default=DEFAULT_LIQUID,
                    help="the Liquid 8B (LFM2.5-8B-A1B) — the judge in judge mode")
    ap.add_argument("--weight-4b", type=float, default=1.0, help="vote weight of each 4B model")
    ap.add_argument("--weight-8b", type=float, default=1.5, help="vote weight of each 8B model")
    ap.add_argument("--think", action=argparse.BooleanOptionalAction, default=None,
                    help="reasoning/thinking before the answer (Qwen <think>; Liquid always "
                         "reasons; Gemma ignores it). Default: ON in judge mode, OFF in vote "
                         "mode; --no-think disables. Needs more tokens.")
    ap.add_argument("--backend", choices=["hf", "stub"], default="hf")
    ap.add_argument("--precision", choices=["4bit", "8bit", "bf16", "fp16", "fp32"], default="4bit",
                    help="applied to every model. Two 4B models in bf16 will not fit 12 GB.")
    ap.add_argument("--compute-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                    help="compute dtype for the 4bit/8bit paths (bfloat16 on Blackwell/Ampere)")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="answer budget; auto-raised for thinking / always-reasoning models")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N questions (0 = all)")
    ap.add_argument("--start", type=int, default=0,
                    help="first index to run (inclusive), over the post---only set")
    ap.add_argument("--end", type=int, default=0,
                    help="stop BEFORE this index (exclusive); 0 = run to the end. "
                         "Pair with --start to run a window: --start 100 --end 200")
    ap.add_argument("--only", choices=["ynn", "mcq", "all"], default="all")
    ap.add_argument("--show-gold", action="store_true", help="score against the gold answers")
    ap.add_argument("--out", type=Path, default=None, help="extra path to also write predictions JSON")
    args = ap.parse_args()

    # Thinking defaults ON for the generate→judge flow (the user-facing design is
    # "two 4B thinking generators"); vote mode keeps the old opt-in behaviour.
    think: bool = args.think if args.think is not None else (args.mode == "judge")
    if args.mode == "judge":
        if args.stages:
            print(f"[warn] --stages is ignored in judge mode (fixed line-up: "
                  f"'{JUDGE_GEN_STAGE}' generators -> '{JUDGE_STAGE}' judge); use --mode vote.")
        stages: list[str] = [JUDGE_GEN_STAGE, JUDGE_STAGE]
    else:
        stages = args.stages or parse_stages(DEFAULT_STAGES)

    # ── Load + gate ───────────────────────────────────────────────────────────
    records_all = load_records(args.data)
    gated = [gate(r) for r in records_all]
    if args.only == "ynn":
        gated = [g for g in gated if g[0].answer_type == AnswerType.YES_NO_UNKNOWN]
    elif args.only == "mcq":
        gated = [g for g in gated if g[0].answer_type == AnswerType.MCQ]
    # Index window over the filtered set: process [start, end). --limit further
    # caps the window (process at most N of it). All indices are positions in the
    # post-(--only)-filter list, matching --start's existing meaning.
    n_filtered = len(gated)
    lo = max(0, args.start)
    hi = args.end if args.end > 0 else n_filtered
    if args.end and args.end <= lo:
        ap.error(f"--end ({args.end}) must be greater than --start ({lo})")
    gated = gated[lo:hi]
    if args.limit:
        gated = gated[: args.limit]
    records = [g[0] for g in gated]
    golds = [g[1] for g in gated]  # aligned to records by position (ids may recur)

    registry = stage_registry(args)
    print(f"[info] loaded {len(records_all)} questions; processing {len(records)} "
          f"({sum(1 for r in records if r.answer_type == AnswerType.MCQ)} MCQ / "
          f"{sum(1 for r in records if r.answer_type == AnswerType.YES_NO_UNKNOWN)} YNN)")
    print(f"[info] index window: [{lo}, {min(hi, n_filtered)}) of {n_filtered} after --only"
          + (f"; capped to first {args.limit}" if args.limit else ""))
    print(f"[info] backend={args.backend}  precision={args.precision}  "
          f"mode={args.mode}  think={think}")
    if args.mode == "judge":
        gens = ", ".join(s["label"] for s in registry[JUDGE_GEN_STAGE])
        print(f"[info] generators (concurrent, thinking={think}): {gens}")
        print(f"[info] judge: {registry[JUDGE_STAGE][0]['label']}")
    else:
        print(f"[info] stages={stages}  weights: 4B={args.weight_4b}  8B={args.weight_8b}")
        for st in stages:
            labels = ", ".join(f"{s['label']} (w={s['weight']:g})" for s in registry[st])
            print(f"[info]   stage '{st}': {labels}")

    def max_tokens_for(model) -> int:
        n = args.max_new_tokens
        if think or getattr(model, "always_cot", False):
            # Thinking + the trailing JSON object needs more room in judge mode.
            n = max(n, 1024 if args.mode == "judge" else 768)
        return n

    def build_stage(specs: list[dict]) -> list:
        """Instantiate (load) a stage's models. The 4B stage builds two models
        that coexist; the 8B stages build one. If the SECOND model of a stage
        fails to load, the first is unloaded before the error propagates — so a
        partial load never leaves a model resident and break the VRAM invariant."""
        def _make(s: dict):
            if args.backend == "stub":
                from chat_model import StubModel
                return StubModel(f"{s['label']}(stub)", s["stub"], vote_weight=s["weight"],
                                 model_class=s["cls"], always_cot=s["cot"])
            from chat_model import ChatModel
            return ChatModel(
                s["id"], precision=args.precision, device_map=args.device_map,
                compute_dtype=args.compute_dtype, enable_thinking=think,
                label=s["label"], vote_weight=s["weight"], model_class=s["cls"],
                always_cot=s["cot"])

        models: list = []
        try:
            for s in specs:
                models.append(_make(s))
        except Exception:
            for m in models:  # roll back any already-loaded models in this stage
                try:
                    m.unload()
                except Exception:  # noqa: BLE001
                    pass
            raise
        return models

    # ── Run the models, one stage resident at a time ──────────────────────────
    # The try/finally GUARANTEES a stage's models are unloaded before the next
    # stage loads, so the VRAM invariant (≤ two 4B OR one 8B resident) always
    # holds. Generation never raises (failures become error-marker replies).
    # Results are accumulated BY POSITION (not by id) so records are never
    # conflated even if two share an id.
    stage_errors: list[tuple[str, str]] = []
    t_start = time.perf_counter()

    def load_stage(name: str, specs: list[dict]) -> list:
        """build_stage + uniform error capture; returns [] if it won't load."""
        print(f"\n[load] stage '{name}': {', '.join(s['label'] for s in specs)}")
        try:
            return build_stage(specs)
        except Exception as e:  # noqa: BLE001 — a stage that won't load is skipped
            msg = _exc_chain(e)
            stage_errors.append((name, msg))
            print(f"[error] stage '{name}' failed to load ({msg}); skipping it.")
            if "out of memory" in msg.lower():
                print(f"        [hint] CUDA OOM on '{name}'. Cross-stage VRAM fragmentation is "
                      f"the usual cause (expandable_segments is already enabled). On a tight "
                      f"GPU, give this stage the whole card by running it alone in a fresh "
                      f"process.")
            return []

    def kind_of(rec: Record) -> str:
        return "MCQ" if rec.answer_type == AnswerType.MCQ else "YNN"

    if args.mode == "judge":
        # Stage 1 — both 4B generators resident, answering CONCURRENTLY per record.
        candidates_by_pos: list[list[ModelReply]] = [[] for _ in records]
        models = load_stage(JUDGE_GEN_STAGE, registry[JUDGE_GEN_STAGE])
        if models:
            try:
                budget = max(max_tokens_for(m) for m in models)
                for i, rec in enumerate(records):
                    cands = generate_candidates(models, rec, max_new_tokens=budget)
                    candidates_by_pos[i] = cands
                    answers = "  ".join(f"{r.model_label}={r.answer!r}" for r in cands)
                    print(f"[gen  ][{i + 1:>4}/{len(records)}] {rec.id:<22} {kind_of(rec)}  {answers}")
            finally:
                for m in models:
                    m.unload()
                print(f"[unload] stage '{JUDGE_GEN_STAGE}' freed")

        # Stage 2 — the Liquid 8B judge (alone resident) rules on every record.
        judge_by_pos: list[ModelReply | None] = [None] * len(records)
        jmodels = load_stage(JUDGE_STAGE, registry[JUDGE_STAGE])
        if jmodels:
            try:
                budget = max_tokens_for(jmodels[0])
                for i, rec in enumerate(records):
                    jr = judge_decide(jmodels[0], rec, candidates_by_pos[i], max_new_tokens=budget)
                    judge_by_pos[i] = jr
                    print(f"[judge][{i + 1:>4}/{len(records)}] {rec.id:<22} {kind_of(rec)}  "
                          f"{jr.model_label}={jr.answer!r} (chosen={getattr(jr, 'chosen', None)})")
            finally:
                for m in jmodels:
                    m.unload()
                print(f"[unload] stage '{JUDGE_STAGE}' freed")

        # ── Decide — deterministic code, judge first, generator fallbacks ─────
        finals: list[FinalAnswer] = [
            finalize_judged(rec, candidates_by_pos[i], judge_by_pos[i])
            for i, rec in enumerate(records)
        ]
    else:
        replies_by_pos: list[list] = [[] for _ in records]
        for st in stages:
            models = load_stage(st, registry[st])
            if not models:
                continue
            try:
                for i, rec in enumerate(records):
                    reps = [query(m, rec, max_new_tokens=max_tokens_for(m)) for m in models]
                    replies_by_pos[i].extend(reps)
                    votes = "  ".join(f"{r.model_label}={r.answer!r}" for r in reps)
                    print(f"[{st}][{i + 1:>4}/{len(records)}] {rec.id:<22} {kind_of(rec)}  {votes}")
            finally:
                for m in models:
                    m.unload()
                print(f"[unload] stage '{st}' freed")

        # ── Combine votes ──────────────────────────────────────────────────────
        # A list aligned to `records` by position — never an id-keyed dict, so two
        # records can never collide and silently drop one.
        finals = [
            finalize_by_vote(rec, replies_by_pos[i]) for i, rec in enumerate(records)
        ]

    # ── Scoring + logs ────────────────────────────────────────────────────────
    if args.show_gold:
        for f, gold in zip(finals, golds):
            f.gold = gold
    elapsed = time.perf_counter() - t_start
    n_correct = n_scored = 0
    if args.show_gold:
        rep = score(records, finals)
        n_correct, n_scored = rep.overall.correct, rep.overall.total
        print(f"\n[accuracy] {n_correct}/{n_scored} = "
              f"{(n_correct / n_scored if n_scored else 0):.1%}")
        for k, v in rep.by_type.items():
            print(f"           {k:<16} {v.correct}/{v.total} = {v.accuracy:.1%}")
        print(f"[vote]     unanimous: {rep.unanimous}   split: {rep.split}")
    else:
        unan = sum(1 for f in finals if f.agreed)
        print(f"\n[vote]     unanimous: {unan}   split: {len(finals) - unan}")
    if stage_errors:
        for st, msg in stage_errors:
            print(f"[warn] stage '{st}' did not run: {msg}")
    print(f"[done] {len(records)} questions in {elapsed:.1f}s "
          f"({elapsed / max(len(records), 1):.1f}s/q)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = result_dir(ROOT)
    header = {
        "backend": args.backend, "precision": args.precision, "mode": args.mode,
        "think": think,
        "qwen": args.qwen_model, "gemma_small": args.gemma_small_model,
        "liquid": args.liquid_model,
    }
    if args.mode == "judge":
        header["flow"] = (f"generators({JUDGE_GEN_STAGE}: concurrent, thinking) "
                          f"-> judge({JUDGE_STAGE})")
    else:
        header["stages"] = ",".join(stages)
        header["weights"] = f"4B={args.weight_4b} 8B={args.weight_8b}"
        header["gemma_big"] = args.gemma_big_model
    if stage_errors:
        header["stage_errors"] = "; ".join(f"{st}: {msg}" for st, msg in stage_errors)
    summary = write_run_summary(rd / f"run_cascade_{stamp}.txt", header, records, finals,
                                n_correct, n_scored, elapsed)
    model_io = write_model_io(rd / f"run_cascade_{stamp}_model_io.txt", records, finals)
    preds = write_predictions_json(rd / f"run_cascade_{stamp}.json", records, finals)
    submission = write_submission_json(rd / f"run_cascade_{stamp}_submission.json",
                                       records, finals)
    print(f"[wrote] {summary}")
    print(f"[wrote] {model_io}")
    print(f"[wrote] {preds}")
    print(f"[wrote] {submission}  (competition Section 4 format, built by code)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_predictions_json(args.out, records, finals)
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
