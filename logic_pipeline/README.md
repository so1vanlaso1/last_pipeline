# Generate→Judge NL-QA Pipeline

A **two-stage generate→judge** ensemble for the EXACT-style logic dataset
(`Logic_Based_Educational_Queries.json`). It answers each question using **only**
the natural-language premises and the question — never the gold answer, never the
FOL. The legacy **weighted soft-vote** flow is still available via `--mode vote`.

## The flow (`--mode judge`, default)

```
  Question
     │
     ├─ Qwen/Qwen3.5-4B       ┐  stage 1: both 4B generators resident, answering
     ├─ google/gemma-4-E2B-it ┘  CONCURRENTLY in thinking mode; each emits
     │                           {answer, premises_used, explanation}
     │        … both unloaded, then …
     ├─ LiquidAI/LFM2.5-8B-A1B   stage 2: the 8B judge sees the ORIGINAL premises
     │                           + question plus both juniors' answers (reference
     │                           only) and decides the truly correct answer,
     │                           re-deriving premises_used + explanation itself
     ▼
  deterministic code → competition-format submission JSON (Section 4 schema)
```

* **Stage 1 — generators.** The two 4B models are loaded together and queried
  **concurrently** (one thread per model; `generate()` releases the GIL during
  CUDA work). Thinking is ON by default in judge mode (`--no-think` disables).
  Each generator must end its reply with one JSON object:
  `{"answer": …, "premises_used": [1-based], "explanation": …}`.
* **Stage 2 — the judge.** `LiquidAI/LFM2.5-8B-A1B` (MoE, 8.3B total / 1.5B
  active — always reasons) gets the original problem **plus** both candidates,
  rules `{"chosen": 1|2, "answer": …, "premises_used": […], "explanation": …}`,
  and is instructed to use the juniors only as reference, never to copy.
* **Deterministic finalize.** Code — not a model — builds the final record. If
  the judge's reply is unparseable the answer falls back in order: the junior
  the judge endorsed (`chosen`) → generator agreement → the first generator
  with an answer. The submission JSON (`*_submission.json`) is always assembled
  by code from canonical fields.

The two stages still run **one at a time** (two 4B resident, then one 8B
resident — never more), so the 12 GB VRAM invariant holds. A stage that fails to
load is skipped and the fallbacks above kick in.

## Legacy flow (`--mode vote`)

Every selected stage votes on every record; each model adds its weight (4B →
1.0, 8B → 1.5; tune with `--weight-4b/--weight-8b`) to the label it picked and
the heaviest label wins. Line-up via `--stages` — any comma-combination of
`{4b, gemma8b, liquid8b}` (`gemma8b` = `google/gemma-4-E4B-it`, vote mode only).

## Question types

* **MCQ** — pick an option letter `A`–`H` (or `Unknown` if none follows).
* **Yes / No / Not Given** — the dataset writes "Not Given" as **`Unknown`**.

The type is decided from the question's *structure* (lettered options → MCQ, else
Yes/No/Not-Given), so it never peeks at the gold answer.

## Prompts

The system prompts are **strict formal-logic examiner** rules that target the
exact mistakes 4B models keep making — reading "not enough information" as
**No** (lack of proof ⇒ *Not Given*, not *No*), the **converse**/**inverse**
fallacies, and confusing **some** with **all**. The user turn follows a fixed
template:

```
Premises:        (numbered from 1, so the model can cite "premise 7")
Definitions:     None        (the dataset folds definitions into the premises)
Question:        the decision task (YNN) / the stem (MCQ)
Statement:       the claim to test          (YNN only)
Options:         A. … B. …                  (MCQ only)
```

Judge mode appends the JSON reply format (and, for the judge, the juniors'
answers). The **last** balanced JSON object in the completion is parsed, so
thinking prose containing braces can't shadow the verdict; `<think>`-style
blocks are stripped first. Vote mode keeps the `ANSWER:`/`WHY:` format.

## Precision

One switch, applied to every model: `--precision {4bit,8bit,bf16}`.

> ⚠️ Two 4B models in **bf16** will not fit a 12 GB card. Use **4bit** (≈5–6 GB
> for both generators) or **8bit** there. bf16 is fine if you have the headroom.

On a **single GPU**, the loader pins each model fully to GPU 0 (it does **not**
let `device_map="auto"` offload to CPU/disk, which would otherwise make
bitsandbytes int8 refuse to load an 8B model that actually fits). If a model is
genuinely too big it OOMs cleanly and that stage is skipped. The Liquid 8B MoE
fits at **8bit** (~8–9 GB) or **4bit** (~4.5 GB) on a 12 GB card — drop to 4bit
if 8bit is tight because other processes hold VRAM.

## Setup

```bash
cd logic_pipeline
# Gemma repos are gated — accept the license on huggingface.co first:
export HF_TOKEN=hf_xxx
chmod +x setup.sh run.sh
./setup.sh          # GPU check → venv → torch(cu128) → deps → download models
```

`setup.sh` env overrides: `QWEN_ID`, `GEMMA_SMALL_ID`, `GEMMA_BIG_ID`,
`LIQUID_ID`, `CUDA_WHL`, `HF_TOKEN`. **If a Gemma repo 404s**, the current
equivalents are `google/gemma-3n-E2B-it` / `google/gemma-3n-E4B-it`. **Liquid**
(`LiquidAI/LFM2.5-8B-A1B`) is **not gated** but needs **`transformers>=5.0`**
(pinned in `requirements.txt`). You only need the models for the stages you run.

## Run

```bash
# the generate→judge flow (default), 4-bit, scored against gold:
python run_cascade.py --precision 4bit --show-gold --limit 20
# same, thinking disabled (faster, weaker):
python run_cascade.py --no-think --precision 4bit --show-gold
# legacy weighted vote over all three stages:
python run_cascade.py --mode vote --stages 4b,gemma8b,liquid8b --precision 4bit --show-gold
# no-GPU wiring test (fake models, exercises both stages + logging):
python run_cascade.py --backend stub --show-gold --limit 8
```

Useful flags: `--mode {judge,vote}`, `--think/--no-think`, `--limit N`,
`--start/--end N`, `--only {ynn,mcq,all}`, `--max-new-tokens`, `--out path.json`,
`--qwen-model/--gemma-small-model/--gemma-big-model/--liquid-model` (override
repo ids), and (vote mode) `--stages`, `--weight-4b/--weight-8b`.

## Output / logs

Every run drops four timestamped files in `Result/`:

| File | Contents |
|------|----------|
| `run_cascade_<stamp>.txt` | per-record summary: each generator's answer + premises, the judge's ruling, the final answer + confidence + WHY, gold |
| `run_cascade_<stamp>_model_io.txt` | **every model invocation, verbatim** — the exact prompt in and raw text out |
| `run_cascade_<stamp>.json` | machine-readable predictions (answer, explanation, premises_used, per-reply detail, decider) |
| `run_cascade_<stamp>_submission.json` | **competition-format result list** (Section 4): `{query_id, answer, unit, explanation, premises_used, reasoning}` — built by deterministic code |

## Layout

```
run_cascade.py        entry point — stage load/run/unload, judge + vote modes, logging
src/
  schema.py           Record / ModelReply (+role/premises_used) / FinalAnswer
  data_load.py        dataset → Record (MCQ split, gold canonicalization, unique ids)
  prompts.py          examiner prompts, generator/judge JSON prompts + parsing
  chat_model.py       ChatModel (4bit/8bit/bf16, thinking, load/unload) + StubModel
  cascade.py          generate_candidates / judge_decide / finalize_judged + the vote
  logio.py            the four Result/ writers (incl. the submission format)
  score.py            accuracy + per-type + unanimous/split counts
tests/test_smoke.py   no-GPU tests (data, normalization, judge flow, vote, stages)
configs/default.yaml  documented defaults
setup.sh / run.sh     quickstart + convenience wrapper
```

## Notes

* Decoding is greedy (`temperature=0`) so every reply is reproducible.
* Only `premises-NL` + `questions` are read from the dataset; `premises-FOL` and
  the gold `answers`/`explanation` are never shown to a model.
* Record ids are made globally unique (`r<pos>_<idx>_q<n>`) because the dataset's
  `idx` is a list of premise indices, not a unique key, and recurs across records.
