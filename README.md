# EXACT 2026 — Logic + Physics QA Submission

A complete submission for the **EXACT 2026** educational-QA challenge. It answers
two kinds of questions behind **one public HTTP endpoint** (`POST /predict`),
served by **vLLM**, deployable to a fresh GPU box (e.g. vast.ai) with **a single
command**.

```
                          POST /predict  { query_id, type, query, premises, options }
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │  Gateway (FastAPI, :8000)   │
                    ├────────────────────────────┤
   type1 (logic) ──▶│ generate → judge flow over │   Qwen3.5-4B  ┐ concurrent
                    │ the resident vLLM line-up   │   Gemma-4-E2B ┘ thinking generators
                    │                             │   LFM2.5-8B / Gemma-4-E4B  ← the JUDGE
   type2 (physics)─▶│ deterministic solver        │   (rules on the candidates)
                    │ (+ optional LLM fallback)   │
                    └────────────┬───────────────┘
                                 ▼
        [ { query_id, answer, unit, explanation, premises_used, reasoning } ]
```

- **Type 1 (logic)** — a *generate→judge* ensemble: small generators answer
  concurrently, then an 8B-class **judge** decides the correct answer and
  re-derives which premises were used. **The judge is swappable** between
  `LiquidAI/LFM2.5-8B-A1B` and `google/gemma-4-E4B-it` via one env var.
- **Type 2 (physics)** — a deterministic formula/template **solver** decides the
  numeric answer + unit; the LLM only rewrites the explanation (never the answer).

---

## Table of contents

1. [Repository layout](#1-repository-layout)
2. [Prerequisites](#2-prerequisites)
3. [Quick start (the one command)](#3-quick-start-the-one-command)
4. [Switching the judge model (Liquid ↔ Gemma)](#4-switching-the-judge-model-liquid--gemma)
5. [Choosing the resident line-up](#5-choosing-the-resident-line-up)
6. [GPU sizing, swap & quantization](#6-gpu-sizing-swap--quantization)
7. [Environment variables](#7-environment-variables)
8. [The `/predict` API](#8-the-predict-api)
9. [Operating the server (stop / relaunch / logs)](#9-operating-the-server-stop--relaunch--logs)
10. [No-GPU wiring test](#10-no-gpu-wiring-test)
11. [Running the pipelines standalone (development)](#11-running-the-pipelines-standalone-development)
12. [Compliance — the ≤ 8B rule](#12-compliance--the-8b-rule)
13. [Submission artifacts](#13-submission-artifacts)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Repository layout

```
last/
├── setup.sh                  ← ONE-SHOT setup for the GPU box (install → download → launch)
├── README.md                 ← you are here
├── EXACT 2026 - Submission Guide.pdf
│
├── serve/                    ← the competition gateway (what actually gets graded)
│   ├── run_server.sh         launch vLLM server(s) + gateway + Cloudflare tunnel
│   ├── stop.sh               stop everything
│   ├── logic_config.yaml     the resident model line-up (generators + judge)
│   ├── requirements.txt
│   ├── gateway/              FastAPI app + adapters + config loader
│   │   ├── app.py            /predict, /health, /v1/models
│   │   ├── config.py         reads logic_config.yaml (env-overridable)
│   │   ├── logic_adapter.py  Type 1 generate→judge flow
│   │   ├── physics_adapter.py Type 2 bridge into physic_pipeline
│   │   ├── residency.py      sleep/wake VRAM swap for the judge
│   │   └── vllm_client.py    OpenAI-compatible vLLM client (+ stub backend)
│   ├── submission/           SOLUTION.md, urls.txt (generated), notation_mapping.csv
│   └── tests/                no-GPU stub tests
│
├── logic_pipeline/           ← standalone Type 1 research pipeline (generate→judge)
│   ├── run_cascade.py        CLI entry point
│   ├── setup.sh / run.sh
│   └── src/{prompts,cascade,chat_model,schema,...}.py
│
└── physic_pipeline/          ← standalone Type 2 research pipeline (solver-first)
    ├── scripts/              run_inference, evaluate, data prep, ...
    ├── configs/              eval_*.yaml
    └── src/exact_fama/       solvers, parser, pipeline, api
```

The **gateway** (`serve/`) imports the two pipelines' pure-Python logic and
serves them behind one endpoint. The pipeline folders are also usable on their
own for development and evaluation (see [§11](#11-running-the-pipelines-standalone-development)).

---

## 2. Prerequisites

| Need | Why |
|---|---|
| **Linux + NVIDIA GPU** with recent driver | vLLM requires CUDA. 24 GB (4090/5070-class) runs the default line-up in swap mode; 48 GB runs it all-resident. |
| **Python 3.10+** (3.11 preferred) | `setup.sh` builds a `.venv`. |
| **A recent vLLM** | Auto-installed by `setup.sh`. Needed for LFM2.5 and Gemma-4 model support. |
| **`HF_TOKEN`** (Hugging Face token) | **Only for gated Gemma repos.** Accept the model license on huggingface.co, then `export HF_TOKEN=hf_...`. Liquid (`LiquidAI/LFM2.5-8B-A1B`) is **ungated**. |
| Internet egress | To download model weights and (optionally) the Cloudflare tunnel binary. |

> The default line-up includes `google/gemma-4-E2B-it` (a generator), so you
> already need `HF_TOKEN` for the default run — even before switching the judge
> to Gemma.

---

## 3. Quick start (the one command)

On the GPU box, from the repo root:

```bash
export HF_TOKEN=hf_xxx        # required for the gated Gemma model(s)
bash setup.sh
```

`setup.sh` will:

1. install system + Python deps into a local `.venv`,
2. install **vLLM** (pulls a CUDA-matched torch) + gateway/physics deps,
3. **pre-cache both judge candidates** (Liquid + Gemma) so you can A/B them,
4. download the models in `serve/logic_config.yaml`,
5. launch **one vLLM server per model** (each exposes `/v1/models`),
6. launch the **gateway** (the single `/predict` endpoint),
7. open a **public URL** (Cloudflare quick tunnel) and write it to
   `serve/submission/urls.txt`.

Servers run in the background and survive SSH disconnect.

**Relaunch later without reinstalling:**

```bash
SKIP_INSTALL=1 bash setup.sh
```

**Verify it's up:**

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models      # lists every resident model (proves ≤ 8B line-up)
```

---

## 4. Switching the judge model (Liquid ↔ Gemma)

The Type 1 **judge** is selected at launch with the **`JUDGE_MODEL`** env var —
no code or config edit needed.

| `JUDGE_MODEL=` | Judge used |
|---|---|
| *(unset)* or `liquid` | `LiquidAI/LFM2.5-8B-A1B` (default; ungated, MoE 8.3B total / 1.5B active) |
| `gemma` | `google/gemma-4-E4B-it` (gated → needs `HF_TOKEN`; ~8B with embeddings) |
| *any full HF repo id* | that repo verbatim |

```bash
# Default judge (Liquid). setup.sh also pre-pulls Gemma if HF_TOKEN is set:
export HF_TOKEN=hf_xxx
bash setup.sh

# Switch to the Gemma judge and relaunch fast (weights already cached):
JUDGE_MODEL=gemma  SKIP_INSTALL=1 bash setup.sh

# ...and back to Liquid:
JUDGE_MODEL=liquid SKIP_INSTALL=1 bash setup.sh
```

Related knobs:

- **`JUDGE_PARAMS_B`** (default `8.3`) — the size the residency budget counts for
  the judge. A safe over-estimate that covers both candidates; lower it if your
  counting convention treats Gemma-4-E4B as smaller.
- **`PREFETCH_JUDGES`** (default `1`) — `setup.sh` pre-downloads **both** judge
  candidates so switching needs no re-download. Set `0` to fetch only the active
  one. Customize the list with `JUDGE_CANDIDATES="repoA repoB"`.

> **Notes on the Gemma judge.** It is **gated** (set `HF_TOKEN`) and is a newer
> multimodal model — ensure your vLLM is recent enough to serve it. In `swap:
> true` mode the judge **must** load or the launch aborts; if Gemma won't load
> under vLLM, test it first with `SWAP=0` (see [§6](#6-gpu-sizing-swap--quantization)).

Under the hood: `serve/logic_config.yaml`'s judge entry is
`id: ${JUDGE_MODEL:-LiquidAI/LFM2.5-8B-A1B}`, and `serve/gateway/config.py`
resolves the `liquid`/`gemma` shortcuts to full repo ids.

---

## 5. Choosing the resident line-up

The resident models live in **`serve/logic_config.yaml`**. `setup.sh` launches
**one vLLM server per model** listed there, downloads only those models, and
**refuses to start if the total exceeds `max_resident_b`**.

Each model entry takes:

```yaml
models:
  - id: Qwen/Qwen3.5-4B          # HF repo id (supports ${ENV:-default} substitution)
    params_b: 4                  # params the residency budget counts
    weight: 1.0                  # vote weight (used by mode: vote)
    role: generator              # "generator" | "judge" | "" (untagged)
    thinking: true               # per-model override of the global thinking: default
    # quantization: 8bit         # per-model override of the global quantization: default
```

Top-level keys: `mode` (`arbiter` = generate→judge, default; or `vote` =
weighted soft vote), `swap`, `quantization`, `thinking`, `max_resident_b`.

The default line-up: **`Qwen/Qwen3.5-4B` + `google/gemma-4-E2B-it`** (generators)
**+ the swappable judge** (Liquid or Gemma-4-E4B). Strictly-compliant alternatives
(2×4B, or a single ≤8B self-judge) are kept commented at the bottom of the file —
switching is a 30-second edit. See [§12](#12-compliance--the-8b-rule).

---

## 6. GPU sizing, swap & quantization

**Swap (`swap: true`, default).** The two 4B generators stay co-resident and the
8B judge is **slept/woken per query** (vLLM sleep mode), so peak VRAM is
`max(generators, judge) ≈ 18 GB` — the whole line-up **fits a 24 GB card**. The
judge boots first (alone, full card), is slept immediately, and the generators
load into the freed memory. Needs ~18 GB CPU RAM to hold the slept weights, and
`VLLM_SERVER_DEV_MODE=1` (set automatically) for the `/sleep` `/wake_up` endpoints.

**No swap (`swap: false` or `SWAP=0`).** All models resident at once (~40+ GB for
the default line-up → a 48 GB card). `GPU_MEM_UTIL` is split across servers in
proportion to each model's `params_b`. Use this to debug a judge that won't load.

**Quantization** (`quantization:` yaml key or `QUANTIZATION` env; per-model
override allowed) shrinks every model:

| Value | Method | Footprint | Requires |
|---|---|---|---|
| `none` (default) | bf16 | ~2 B/param | — |
| `8bit` | online FP8 | ~1 B/param | Ada/Hopper/Blackwell GPU (4090/5070); **not** Ampere |
| `4bit` | bitsandbytes NF4 | ~0.5 B/param | `bitsandbytes` (installed by requirements) |

At `4bit`, even the all-resident line-up fits ~12 GB.

---

## 7. Environment variables

Set any of these before `bash setup.sh` (or `SKIP_INSTALL=1 bash setup.sh`).

### Judge & line-up
| Var | Default | Meaning |
|---|---|---|
| `JUDGE_MODEL` | `liquid` | the swappable judge: `liquid` / `gemma` / any HF repo id |
| `JUDGE_PARAMS_B` | `8.3` | size the residency budget counts for the judge |
| `PREFETCH_JUDGES` | `1` | pre-download both judge candidates in `setup.sh` |
| `JUDGE_CANDIDATES` | `LiquidAI/LFM2.5-8B-A1B google/gemma-4-E4B-it` | which candidates to pre-cache |
| `LOGIC_CONFIG` | `serve/logic_config.yaml` | path to the line-up config |
| `LOGIC_MODE` | yaml `mode:` (default `arbiter`) | `arbiter` (generate→judge) or `vote` |
| `MAX_RESIDENT_B` | yaml `max_resident_b:` (default 8) | residency budget the launch guard enforces |

### Runtime / GPU
| Var | Default | Meaning |
|---|---|---|
| `SWAP` | yaml `swap:` (default `true`) | sleep/wake-swap the judge (24 GB-friendly) |
| `QUANTIZATION` | yaml `quantization:` (default `none`) | `none` / `8bit` / `4bit` for every model |
| `THINKING` | yaml `thinking:` (default `true`) | reasoning-call think mode (per-model override wins) |
| `LOGIC_THINK_TOKENS` | `1024` | max tokens per thinking generate/judge call |
| `RESIDENCY_SLEEP_LEVEL` | `1` | vLLM sleep level (1 = offload weights to CPU RAM) |
| `MAX_MODEL_LEN` | `8192` | vLLM context length |
| `GPU_MEM_UTIL` | `0.90` | total GPU fraction (split ∝ params_b when not swapping) |

### Server / deployment
| Var | Default | Meaning |
|---|---|---|
| `HF_TOKEN` | — | required for gated Gemma repos |
| `GATEWAY_PORT` | `8000` | the `/predict` port |
| `VLLM_BASE_PORT` | `8001` | first vLLM port (servers use base, base+1, …) |
| `CF_TUNNEL` | `1` | auto Cloudflare quick tunnel for a public URL |
| `PHYSICS_LLM_FALLBACK` | `1` | LLM fills Type 2 answers only when the solver abstains |
| `GATEWAY_LLM` | `vllm` | set `stub` for the no-GPU wiring test |
| `SKIP_INSTALL` | `0` | `1` = skip pip install, just (re)launch |
| `MODEL_ID` | `LiquidAI/LFM2.5-8B-A1B` | fallback single model if the yaml is absent |

---

## 8. The `/predict` API

### Endpoints
| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict` | answer one query or a list of queries |
| `GET` | `/health` | liveness + the resident model list |
| `GET` | `/v1/models` | aggregated `/v1/models` across every resident server |

### Request (Section 3 schema)
One object, or a JSON list of them:

```json
{
  "query_id": "q-001",
  "type": "type1",
  "query": "Is the statement entailed by the premises?",
  "premises": ["All birds can fly.", "A penguin is a bird."],
  "options": ["Yes", "No", "Not Given"]
}
```

- `type`: `"type1"` (logic) or `"type2"` (physics).
- `options`: present for MCQ / Yes-No-NotGiven; empty for free-form/physics.

### Response (Section 4 schema)
Always a JSON **list** (one result per query, even for a single query):

```json
[
  {
    "query_id": "q-001",
    "answer": "No",
    "unit": "",
    "explanation": "Premise 1 only licenses flight for typical birds...",
    "premises_used": [0, 1],
    "reasoning": { "type": "fol", "steps": ["..."] }
  }
]
```

- `premises_used` are **0-based** indices into `premises` (Type 1; 50% of the score).
- `unit` is an ASCII unit string for Type 2; `""` for Type 1.

### Example
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"query_id":"q1","type":"type1","query":"Does it follow?","premises":["A→B","A"],"options":["Yes","No","Not Given"]}'
```

---

## 9. Operating the server (stop / relaunch / logs)

```bash
bash serve/stop.sh             # stop vLLM servers + gateway + tunnel
SKIP_INSTALL=1 bash setup.sh   # relaunch without reinstalling
```

Logs live in `serve/logs/`:

| File | Contents |
|---|---|
| `vllm_<port>.log` | each vLLM server's startup + serving log |
| `gateway.log` | the FastAPI gateway |
| `cloudflared.log` | the public tunnel (contains the public URL) |
| `config.err` | the resolved line-up summary / budget errors |

The public URLs are (re)written to `serve/submission/urls.txt` on every launch.

---

## 10. No-GPU wiring test

Exercise the full request/response wiring with deterministic canned model
replies — no GPU, no model download:

```bash
# Linux / macOS:
GATEWAY_LLM=stub PYTHONPATH=serve:physic_pipeline/src:logic_pipeline/src \
  python -m pytest serve/tests -q
```

```powershell
# Windows PowerShell (note ';' path separator):
$env:GATEWAY_LLM="stub"
$env:PYTHONPATH="serve;physic_pipeline/src;logic_pipeline/src"
python -m pytest serve/tests -q
```

---

## 11. Running the pipelines standalone (development)

Both pipelines work independently of the gateway — useful for tuning and scoring.

### Logic pipeline (`logic_pipeline/`)
The generate→judge cascade, scored against gold:

```bash
cd logic_pipeline
export HF_TOKEN=hf_xxx        # Gemma repos are gated
./setup.sh                    # GPU check → venv → torch → deps → download models

python run_cascade.py --precision 4bit --show-gold --limit 20      # generate→judge
python run_cascade.py --mode vote --stages 4b,gemma8b,liquid8b --precision 4bit  # legacy vote
python run_cascade.py --backend stub --show-gold --limit 8         # no-GPU wiring test
```

Outputs land in `logic_pipeline/Result/` (per-record summary, verbatim model I/O,
machine-readable predictions, and the competition-format submission JSON). See
[logic_pipeline/README.md](logic_pipeline/README.md) for all flags.

### Physics pipeline (`physic_pipeline/`)
Solver-first; the LLM is optional (parsing + explanation rewrite only):

```powershell
cd physic_pipeline
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt; pip install -e .
pytest -q

# solver-only baseline (no LLM):
$env:LLM_BACKEND="none"; $env:EXACT_FAMA_CONFIG="configs/eval_solver_only.yaml"
python scripts\run_inference.py --input data\eval\fixed_smoke_physics_100.jsonl `
  --output artifacts\pred.jsonl
python scripts\evaluate.py --gold data\eval\fixed_smoke_physics_100.jsonl `
  --pred artifacts\pred.jsonl --report artifacts\eval.json
```

See [physic_pipeline/README.md](physic_pipeline/README.md) for data prep, fixed
benchmarks, and the parser/rewrite modes.

---

## 12. Compliance — the ≤ 8B rule

The Submission Guide (§6.3) limits the LLM params **loaded at the same moment** to
an 8B class (MoE counted by total params, Q2). The key interpretation we run under:
**only params resident in GPU VRAM count** — weights parked in CPU RAM (a slept
vLLM model) do not.

### Under the GPU-VRAM reading (how we run it), with `swap: true`
vLLM sleep level 1 **frees the slept group's GPU VRAM** (its weights are offloaded
to CPU RAM, which does not count). So only ONE group is ever on the GPU:

| Stage | On GPU (counts) | In CPU RAM (does not count) | GPU params |
|---|---|---|---|
| Stage 1 — generators | Qwen3.5-4B + Gemma-4-E2B | judge | **9.1B** ← peak |
| Stage 2 — judge | the 8B-class judge | both generators | 8.3B |

**Peak GPU ≈ 9.1B**, entirely from the two 4B generators being co-resident; the
judge stage is fine at 8.3B. To get the generator stage strictly ≤ 8B on GPU:

- **Counting convention** — Gemma-4-E2B is **2.3B effective** (5.1B only with
  embeddings); `4 + 2.3 = 6.3B` ≤ 8B. (See [SOLUTION.md](serve/submission/SOLUTION.md) §3.)
- **Swap the generators too** — run them one-at-a-time on the GPU (same sleep/wake),
  dropping GPU peak to ~5.1B at the cost of one extra swap per request.

### Why CPU-RAM sleep (not true unload) is used for the swap
Per-query *true* unload/reload would either keep weights in RAM (level 1 — fast,
~seconds) or reload from disk (level 2 — slow). A full vLLM cold start is ~30–90 s
vs. the ~60 s query timeout, so genuine load/unload per request is not viable.
Level 1 sleep gives a seconds-fast swap while freeing GPU VRAM — exactly what the
GPU-only reading needs. (`RESIDENCY_SLEEP_LEVEL=2` exists if you ever need the CPU
RAM freed too, but it costs disk-reload latency and isn't needed here.)

### The launch guard counts conservatively
`gateway/config.py` sums **total** params (17.4B for the default line-up, CPU+GPU)
against `max_resident_b` and logs a warning over 8B — a deliberately strict
check. `max_resident_b: 18` lets the default line-up launch; the GPU-resident peak
is the 9.1B above. Strictly-compliant line-ups (the 2×4B pair, or a single ≤8B
self-judge) are kept commented in `serve/logic_config.yaml` — set `max_resident_b: 8`
to switch.

### Non-LLM tools are free
The deterministic solvers, regex extractors, and unit verifiers are **0 params**
and don't count (Section 6.3).

`GET /v1/models` reports every resident model so the committee can verify the
line-up on the same host. Document your final choice in
[serve/submission/SOLUTION.md](serve/submission/SOLUTION.md) §3.

---

## 13. Submission artifacts

In `serve/submission/`:

| File | Purpose |
|---|---|
| `SOLUTION.md` | the one-page solution description — **export to `solution.pdf`** for the submission ZIP; fill in dataset counts and the model-size row you actually ran. |
| `urls.txt` | the live `PREDICT_URL` / `MODELS_URL` — **regenerated on every launch** (do not commit). |
| `notation_mapping.csv` | symbol/notation mapping reference. |

Before submitting: pick the matching line-up, accept Gemma licenses if used,
verify `/health` and `/v1/models`, and confirm `SOLUTION.md` §3 matches the
running models.

---

## 14. Troubleshooting

| Symptom | Fix |
|---|---|
| `WARNING: <repo> is gated and HF_TOKEN is not set` | Accept the license on huggingface.co and `export HF_TOKEN=hf_...`. Affects all `google/gemma-*` repos. |
| Launch aborts: `resident models total …B > the …B budget` | The line-up exceeds `max_resident_b`. Shrink the line-up or raise `max_resident_b` (a compliance decision). |
| Judge never comes up (swap mode) → launch exits | The judge is required in `swap: true`. Check `serve/logs/vllm_<port>.log`; ensure a recent vLLM; try `SWAP=0` to isolate. |
| `8bit`/FP8 fails to load | FP8 needs an Ada/Hopper/Blackwell GPU; use `4bit` on Ampere (A100/3090). |
| OOM with `swap: false` | Use `swap: true` (default), a smaller line-up, or `QUANTIZATION=4bit`. |
| No public URL printed | Cloudflare tunnel didn't start; check `serve/logs/cloudflared.log`, or set `CF_TUNNEL=0` and use vast.ai port mapping (see the note in `urls.txt`). |
| `ModuleNotFoundError: gateway` in tests | Set `PYTHONPATH` (use `;` on Windows, `:` on Linux) — see [§10](#10-no-gpu-wiring-test). |
| Switched `JUDGE_MODEL` but the old judge still loads | The gateway caches the line-up per process — fully relaunch (`bash serve/stop.sh` then `SKIP_INSTALL=1 bash setup.sh`). |

---

For the serving internals (the generate→judge flow in detail, fallbacks, and the
residency manager), see [serve/README.md](serve/README.md).
