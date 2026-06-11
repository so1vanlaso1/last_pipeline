# EXACT 2026 serving gateway

Wraps the two existing pipelines behind the **single competition `/predict`
endpoint**, served via **vLLM**, with one setup script for vast.ai.

```
  Evaluation server
        │  POST /predict { query_id, type, query, premises, options }
        ▼
  Gateway (FastAPI, :8000)
        ├── type1 → generate→judge flow over the resident vLLM line-up:
        │     Qwen3-4B (4B)          ┐ concurrent direct generators
        │     Qwen3-4B-Instruct (4B) ┘   {answer, premises_used, explanation}
        │     Gemma-4-E4B (8B)       the judge (thinking): rules on the candidates
        │     → deterministic code builds the result object
        ├── type2 → physic_pipeline ExactFamaPipeline (first model's vLLM)
        └── GET /v1/models ── aggregates every resident server's model list
        ▼
  [ { query_id, answer, unit, explanation, premises_used, reasoning } ]
```

## The Type 1 flow (mode: arbiter, default)

- **Stage 1 — generators.** Every `role: generator` model in
  `serve/logic_config.yaml` (default: `Qwen/Qwen3-4B` + `Qwen/Qwen3-4B-Instruct-2507`,
  4B each, **text-only**) answers **concurrently** (one vLLM server each), **direct
  (non-thinking)**, and must end its reply with `{"answer", "premises_used",
  "explanation"}` (the **last** balanced JSON object is parsed, so reasoning prose
  can't shadow it). Keep generators **text-only** (they must sleep clean) with combined
  size ≤ 8B — a VLM (e.g. Qwen3.5-4B) won't sleep-evict and breaks the judge phase.
- **Stage 2 — the judge.** The `role: judge` model (default:
  `google/gemma-4-E4B-it`, ~8B, thinking on) receives the original premises +
  question plus both candidates (marked *reference only*), decides the truly
  correct answer, and re-derives `premises_used` + the explanation in its own words.
- **Deterministic output.** Code — never the model — maps the canonical answer
  onto the exact option text, clamps `premises_used` to valid 0-based indices,
  and assembles the Section 4 result object. Fallbacks (judge unparseable →
  endorsed junior → constrained re-ask → uncertain option) keep the shape valid.

Line-ups without role tags keep the older behaviour (everything generates, the
highest-weight model arbitrates; a single model makes a strict + a skeptical
pass and self-judges). `mode: vote` switches to the cascade's weighted soft vote.

## Compliance (read before your slot)

The 3-model line-up **declares 16B total** (4 + 4 + 8; the committee counts MoE by
TOTAL params, Submission Guide §6.3/Q2) but is **never co-resident**. A residency
swap keeps only one group's weights on the GPU at any instant, so the
**loaded-and-running total is 8B at every moment** — `max(4+4, 8) = 8B` — which is
allowed per **Q3** (load/unload to stay ≤8B at any single moment). `max_resident_b:
16` is the launch guard's budget for the *total that exists*; the *momentary* limit
is enforced at runtime by the swap (sleep-before-wake, confirmed).

Verify it live during your slot: `curl http://<host>:8000/health` reports each
server's asleep state + VRAM and `params_loaded_running_b` (**8** at rest). Strictly
single-model line-ups (one ≤8B model that self-judges, or one 4B) are kept commented
in `serve/logic_config.yaml` — switching is a 30-second edit if you prefer every
`/v1/models` to sum to ≤8B with no swap to explain.

**Per-server `/v1/models` (§6.3).** The committee requires one `/v1/models` URL per
vLLM server. Each is exposed through the single tunnel at `/vllm/<port>/v1/models`
(e.g. `/vllm/8001/v1/models`), indexed by `GET /servers`, and written one-per-server
into `serve/submission/urls.txt`. `GET /v1/models` still aggregates all of them.
Every LLM call is local vLLM — no third-party inference API (§6.2 / Q5).

## Run it (vast.ai)

```bash
bash setup.sh          # installs, downloads the line-up, launches vLLMs + gateway + tunnel
```

The public URLs are written to `serve/submission/urls.txt`. Stop with
`bash serve/stop.sh`. Re-launch without reinstalling: `SKIP_INSTALL=1 bash setup.sh`.
Live Type 1 model calls are appended to `serve/logs/log.txt`; each entry includes
the stage, model, loaded-model list for that call, `nvidia-smi` VRAM snapshot,
system/user input, and raw model output.

GPU sizing for the default line-up:

* **`swap: true` (default) — RAM unload/reload (vLLM sleep level 1).** The two 4B
  generators stay co-resident and the 8B judge is **swapped in per query**: the
  inactive group's weights are **offloaded to CPU RAM** and copied back verbatim on
  wake (~1 s, lossless — required for FP8 weights). vLLM releases the weights'
  physical VRAM, so a slept model leaves only a small CUDA-context residual. Peak
  resident is `max(4+4, 8) = 8B params`, so only **8B of weights are on the GPU at any
  moment** (fits a 24 GB card). The judge boots first (alone, full card) and is slept
  so the generators load into the freed memory; `VLLM_SERVER_DEV_MODE=1` (run_server.sh
  sets it) enables the `/sleep` / `/wake_up` / `/is_sleeping` endpoints. Set
  `RESIDENCY_SLEEP_LEVEL=2` only for a 4bit/bf16 line-up (disk discard+reload; it
  re-quantizes and corrupts FP8 on wake). **Use only text-only models as generators**
  — a VLM (e.g. Qwen3.5-4B) will not sleep-evict its weights and keeps the judge phase
  above 8B.
* **`swap: false`** — all three resident at once: ~32+ GB in bf16 (4B ≈ 8 G + 4B ≈ 8 G
  + 8B ≈ 16 G + KV caches). **This breaks the ≤8B rule** (16B co-resident) — only for
  local debugging, never the graded slot. `GPU_MEM_UTIL` is split across the servers in
  proportion to each model's `params_b`.
* **`quantization: none | 8bit | 4bit`** (yaml or env `QUANTIZATION`, per-model
  override allowed) shrinks every model: `8bit` = online FP8 (~half VRAM, needs
  an Ada/Hopper/Blackwell GPU); `4bit` = bitsandbytes NF4 (~quarter VRAM, needs
  the `bitsandbytes` package). At 4bit even the all-resident line-up fits ~12 GB.

## Choosing the line-up (`serve/logic_config.yaml`)

`setup.sh` launches **one vLLM server per model** listed there (each with its
own `/v1/models`), downloads only those models, and **refuses to start if the
total-that-exists exceeds `max_resident_b`** (default 8; the shipped swap config
raises it to 16 explicitly — the *momentary* GPU load is held to 8B by the swap,
see the compliance section above). The default line-up (Qwen3-4B + Qwen3-4B-Instruct
+ Gemma-4) is **ungated** — no `HF_TOKEN` needed. Each model takes
`role: generator | judge` (Type 1 flow) plus
`params_b` and a vote `weight` (used by `mode: vote`). Each model also takes an
optional `quantization:` and `thinking:` (true/false) that override the global
`quantization:` / `thinking:` defaults — so you can, e.g., run the generators in
thinking mode at 4bit but give the judge a direct (no-think) full-precision
verdict. Helper JSON calls (premises_used, option pick) are always no-think.

## Key env vars

| Var | Default | Meaning |
|---|---|---|
| `LOGIC_CONFIG` | `serve/logic_config.yaml` | the resident model line-up + `mode:` |
| `LOGIC_MODE` | (yaml `mode:`, default `arbiter`) | `arbiter` (generate→judge) or `vote` |
| `LOGIC_THINK_TOKENS` | `1024` | max tokens per thinking generate/judge call |
| `MAX_RESIDENT_B` | (yaml `max_resident_b:`, default 8) | residency budget the launch guard enforces |
| `SWAP` | (yaml `swap:`, default `true`) | disk-swap the judge per query instead of holding it resident (24 GB-friendly) |
| `RESIDENCY_SLEEP_LEVEL` | `1` | `1` = offload slept weights to CPU RAM, copy back on wake (lossless, FP8-safe — default); `2` = discard + reload from disk (4bit/bf16 only; corrupts FP8) |
| `RESIDENCY_SLEEP_RETRIES` | `3` | times to retry+confirm a `/sleep` before the swap refuses to wake the next group (keeps GPU ≤ 8B) |
| `QUANTIZATION` | (yaml `quantization:`, default `none`) | `none`(bf16) / `8bit` (fp8) / `4bit` (bnb NF4) for every model |
| `THINKING` | (yaml `thinking:`, default `true`) | reasoning-call think mode for every model (per-model `thinking:` overrides) |
| `VLLM_VERSION` | `0.19.1` | pinned CUDA-12 vLLM (drivers ≤ CUDA 12.9); set empty for latest on CUDA-13 |
| `MODEL_ID` | `google/gemma-4-E4B-it` | fallback single model if the yaml is absent |
| `VLLM_BASE_PORT` / `GATEWAY_PORT` | `8001` / `8000` | first vLLM port (servers use base, base+1, …) / gateway port |
| `MAX_MODEL_LEN` / `GPU_MEM_UTIL` | `8192` / `0.90` | vLLM context / total GPU fraction (split ∝ params_b) |
| `MAX_NUM_SEQS` | `16` | max concurrent seqs/server — small cuts startup/sampler VRAM (the gateway is sequential) |
| `NGROK` / `NGROK_DOMAIN` | `1` / – | public tunnel via ngrok (default); pin a reserved static domain with `NGROK_DOMAIN` |
| `CF_TUNNEL` | `0` | legacy Cloudflare quick-tunnel fallback (only used if ngrok produced no URL) |
| `PHYSICS_LLM_FALLBACK` | `1` | LLM fills Type 2 answers only when the solver abstains |
| `GATEWAY_LLM` | `vllm` | set `stub` for the no-GPU wiring test |

## No-GPU wiring test

```bash
GATEWAY_LLM=stub PYTHONPATH=serve:physic_pipeline/src:logic_pipeline/src \
  python -m pytest serve/tests -q
```
