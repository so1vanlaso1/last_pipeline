# EXACT 2026 serving gateway

Wraps the two existing pipelines behind the **single competition `/predict`
endpoint**, served via **vLLM**, with one setup script for vast.ai.

```
  Evaluation server
        │  POST /predict { query_id, type, query, premises, options }
        ▼
  Gateway (FastAPI, :8000)
        ├── type1 → generate→judge flow over the resident vLLM line-up:
        │     Qwen3.5-4B  ┐ concurrent thinking generators
        │     Gemma-4-E2B ┘   {answer, premises_used, explanation}
        │     LFM2.5-8B-A1B   the judge: rules on the candidates
        │     → deterministic code builds the result object
        ├── type2 → physic_pipeline ExactFamaPipeline (first model's vLLM)
        └── GET /v1/models ── aggregates every resident server's model list
        ▼
  [ { query_id, answer, unit, explanation, premises_used, reasoning } ]
```

## The Type 1 flow (mode: arbiter, default)

- **Stage 1 — generators.** Every `role: generator` model in
  `serve/logic_config.yaml` (default: `Qwen/Qwen3.5-4B` + `google/gemma-4-E2B-it`)
  answers **concurrently** (one vLLM server each) with **thinking enabled**, and
  must end its reply with `{"answer", "premises_used", "explanation"}` (the
  **last** balanced JSON object is parsed, so reasoning prose can't shadow it).
- **Stage 2 — the judge.** The `role: judge` model (default:
  `LiquidAI/LFM2.5-8B-A1B`) receives the original premises + question plus both
  candidates (marked *reference only*), decides the truly correct answer, and
  re-derives `premises_used` + the explanation in its own words.
- **Deterministic output.** Code — never the model — maps the canonical answer
  onto the exact option text, clamps `premises_used` to valid 0-based indices,
  and assembles the Section 4 result object. Fallbacks (judge unparseable →
  endorsed junior → constrained re-ask → uncertain option) keep the shape valid.

Line-ups without role tags keep the older behaviour (everything generates, the
highest-weight model arbitrates; a single model makes a strict + a skeptical
pass and self-judges). `mode: vote` switches to the cascade's weighted soft vote.

## Compliance warning (read before your slot)

The default 3-model line-up totals **≈ 17.4B params resident** (committee counts
MoE by TOTAL params, Submission Guide §6.3/Q2) — that is **over the 8B-at-any-
moment limit** on a strict reading. `max_resident_b: 18` in the yaml is the
explicit acknowledgement that launches it anyway; the launcher and gateway log
loud warnings. Strictly-compliant line-ups (2×4B, or one ≤8B model that
self-judges) are kept commented in `serve/logic_config.yaml` — switching back is
a 30-second edit. Per-query load/unload is not an option (an 8B vLLM cold start
is ~30–90 s vs the ~60 s query timeout).

## Run it (vast.ai)

```bash
bash setup.sh          # installs, downloads the line-up, launches vLLMs + gateway + tunnel
```

The public URLs are written to `serve/submission/urls.txt`. Stop with
`bash serve/stop.sh`. Re-launch without reinstalling: `SKIP_INSTALL=1 bash setup.sh`.

GPU sizing for the default line-up:

* **`swap: true` (default)** — the two 4B generators stay co-resident and the 8B
  judge is **slept/woken per query** (vLLM sleep mode), so peak VRAM is
  `max(generators, judge) ≈ 18 GB` and the whole line-up **fits a 24 GB card**
  (4090 / 5070-class). Costs one sleep/wake swap per query (seconds). The judge
  boots first (alone, full card) and is slept immediately so the generators load
  into the freed memory; needs ~18 GB CPU RAM to hold the slept weights, and
  `VLLM_SERVER_DEV_MODE=1` (run_server.sh sets it) for the `/sleep` `/wake_up`
  endpoints.
* **`swap: false`** — all three resident at once: ~40+ GB total (4B ≈ 8 G +
  5.1B ≈ 11 G + 8.3B ≈ 17 G + KV caches), so a 48 GB card. `GPU_MEM_UTIL` is then
  split across the servers in proportion to each model's `params_b`.
* **`quantization: none | 8bit | 4bit`** (yaml or env `QUANTIZATION`, per-model
  override allowed) shrinks every model: `8bit` = online FP8 (~half VRAM, needs
  an Ada/Hopper/Blackwell GPU); `4bit` = bitsandbytes NF4 (~quarter VRAM, needs
  the `bitsandbytes` package). At 4bit even the all-resident line-up fits ~12 GB.

## Choosing the line-up (`serve/logic_config.yaml`)

`setup.sh` launches **one vLLM server per model** listed there (each with its
own `/v1/models`), downloads only those models, and **refuses to start if the
total exceeds `max_resident_b`** (default 8; the shipped config raises it to 18
explicitly — see the compliance warning above). Gated Gemma repos need
`HF_TOKEN`. Each model takes `role: generator | judge` (Type 1 flow) plus
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
| `SWAP` | (yaml `swap:`, default `true`) | sleep/wake-swap the judge instead of holding it resident (24 GB-friendly) |
| `QUANTIZATION` | (yaml `quantization:`, default `none`) | `none` / `8bit` (fp8) / `4bit` (bnb NF4) for every model |
| `THINKING` | (yaml `thinking:`, default `true`) | reasoning-call think mode for every model (per-model `thinking:` overrides) |
| `RESIDENCY_SLEEP_LEVEL` | `1` | vLLM sleep level (1 = offload weights to CPU RAM, fast wake) |
| `MODEL_ID` | `LiquidAI/LFM2.5-8B-A1B` | fallback single model if the yaml is absent |
| `VLLM_BASE_PORT` / `GATEWAY_PORT` | `8001` / `8000` | first vLLM port (servers use base, base+1, …) / gateway port |
| `MAX_MODEL_LEN` / `GPU_MEM_UTIL` | `8192` / `0.90` | vLLM context / total GPU fraction (split ∝ params_b) |
| `CF_TUNNEL` | `1` | auto Cloudflare quick tunnel for a public URL |
| `PHYSICS_LLM_FALLBACK` | `1` | LLM fills Type 2 answers only when the solver abstains |
| `GATEWAY_LLM` | `vllm` | set `stub` for the no-GPU wiring test |

## No-GPU wiring test

```bash
GATEWAY_LLM=stub PYTHONPATH=serve:physic_pipeline/src:logic_pipeline/src \
  python -m pytest serve/tests -q
```
