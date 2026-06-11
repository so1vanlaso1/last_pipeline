# EXACT 2026 — Solution Description

> One-page solution description (export this file to **solution.pdf** for the
> submission ZIP). Fill in the bracketed dataset sample-counts/examples with your
> finalized numbers before submitting.

## 1. System overview

A single public HTTP endpoint, `POST /predict`, accepts the unified competition
schema and routes each query by `type` to one of two internal pipelines:

```
POST /predict
  ├── type1 → Logic pipeline   (generate → judge: two 4B generators answer, an 8B
  │                              judge arbitrates and re-derives premises_used)
  └── type2 → Physics pipeline  (deterministic formula/template solver decides the
                                 numerical answer + unit; the LLM only canonicalizes
                                 phrasing and rewrites the explanation)
```

All LLM components are served by **vLLM** (OpenAI-compatible). The gateway also
reverse-proxies `GET /v1/models` for each server and exposes `GET /health` so the
committee can verify both the model identity and the live GPU residency.

- **Understanding.** Type 1: premises + question are rendered into a strict
  formal-logic examiner prompt (direct implication, contraposition, converse/inverse
  guards, quantifier and necessary/sufficient rules). Type 2: a deterministic
  extractor parses physical quantities (values + units, scientific notation, Greek
  symbols) from the problem text.
- **Reasoning.** Type 1 (`arbiter` flow): two **4B generators** answer concurrently
  (direct, non-thinking), each emitting `{answer, premises_used, explanation}`; an
  **8B judge** (thinking) then inspects the original premises + both candidates and
  decides the correct answer, independently re-deriving `premises_used`. Type 2:
  registered circuit / electrostatics solvers compute the result; the answer and
  unit come from the solver, not the LLM.
- **Explanation generation.** Type 1: the judge's justification plus the premise
  indices it used (`premises_used`). Type 2: an LLM rewrite of the verified solver
  evidence that is forbidden from changing the answer or unit.

`premises_used` (Type 1, 50% of the score) is taken from the judge's citation, with
a small dedicated JSON call as a fallback that never changes the answer. Type 2
returns the numerical `answer` and an ASCII `unit`, with `premises_used = []`.

## 2. Datasets used

| Dataset | Source / origin | Samples used | Notes |
|---|---|---|---|
| EXACT 2026 Logic (`Logic_Based_Educational_Queries.json`) | Official EXACT 2026 release | [N] | Type 1 prompt/parse tuning. Only premises-NL + questions are used; gold answers/FOL are never shown to the model. |
| EXACT 2026 Physics (`Physics_Problems_Text_Only.csv`) | Official EXACT 2026 release | [N] | Type 2 formula/template solver coverage. |
| [External / synthetic physics, if used] | [origin] | [N] | [one-line description + a couple of sample entries] |

Sample entries (fill in 2–3 short examples per dataset before submitting).

## 3. Model size calculation (≤ 8B loaded-and-running at any moment)

We declare **three** open-source LLMs, all served via vLLM. The total that *exists*
is 16B, but they are **never co-resident**: a residency swap keeps only one group's
weights on the GPU at any instant, so the **loaded-and-running total is 8B at every
moment** — within the limit per **Q3** ("load and unload so that at any single
moment the models resident and running on the GPU stay within 8B").

| Model | Role | Param count (counted) | On GPU during… |
|---|---|---|---|
| `Qwen/Qwen3.5-4B` | generator | 4B (dense) | generation |
| `Qwen/Qwen3-4B-Instruct-2507` | generator | 4B (dense) | generation |
| `google/gemma-4-E4B-it` | judge | **8B total** (MoE/Matformer — total, not the ~4B effective, per Q2) | arbitration |

**The invariant — exactly one group on the GPU:**

```
Generation phase :  Qwen3.5-4B (4B) + Qwen3-4B (4B) AWAKE   = 8B on GPU ; judge ASLEEP
Arbitration phase:  gemma-4-E4B (8B) AWAKE                  = 8B on GPU ; both gens ASLEEP
Peak resident     =  max(4 + 4, 8)                          = 8B  ✓
```

The swap uses **vLLM sleep mode, level 1**: the inactive group's weights are
**offloaded to CPU RAM** and copied back verbatim on wake (~1 s, lossless — required
for FP8 weights). vLLM's sleep releases the weights' physical VRAM (CUDA virtual-
memory unmapping), so a slept model leaves only a small CUDA-context residual on the
card, **not its weights**. Compliance is enforced, not just intended:

- **Sleep-before-wake is hard-enforced.** A group is woken only after the other group
  is *confirmed* asleep (synchronous `/sleep` + `/is_sleeping` check, with retries);
  if a sleep cannot be confirmed, the swap **refuses to wake** the next group and the
  query degrades — the GPU never exceeds 8B.
- **Live verification.** `GET /health` reports, per server, its role, params, whether
  it is asleep, and its current VRAM (via `nvidia-smi`), plus
  `params_loaded_running_b` (the awake total — **8** at rest). The committee can
  confirm ≤8B at any instant, including by inspecting GPU memory directly (§6.3).

Type 2 (physics) uses only the **first 4B generator** for its optional LLM calls, so
it is well within 8B. Non-LLM tools (the deterministic logic/physics solvers, regex
extractors, unit verifiers) are **0 params** and do not count (§6.3).

> **Strictly-single-model alternatives** (zero swap, every `/v1/models` sums to ≤8B)
> are kept one edit away in `serve/logic_config.yaml`: a single `gemma-4-E4B-it`
> (8B, two-pass self-judge) or a single `Qwen3.5-4B` (4B). Switching is a 30-second
> config change if a simpler footprint is preferred for the slot.

## 4. Serving & verification (vLLM)

Every LLM call hits a **local vLLM** OpenAI-compatible server — **no third-party
inference API** (Together / Fireworks / Groq / Replicate) is used anywhere, in
either pipeline (§6.2 / Q5). Each model runs in its own `vllm serve` process and is
verifiable independently:

- `…/vllm/8001/v1/models` → `Qwen/Qwen3.5-4B` (generator)
- `…/vllm/8002/v1/models` → `Qwen/Qwen3-4B-Instruct-2507` (generator)
- `…/vllm/8003/v1/models` → `google/gemma-4-E4B-it` (judge)

Each reports exactly the `id` declared in §3 and stays reachable even while its model
is swapped to sleep (the model list is metadata). These `/vllm/<port>/v1/models` paths
are a **read-only passthrough** of each real vLLM server's own response (the gateway
fetches it live), so they report the genuinely-loaded model without exposing vLLM's
swap-mode admin endpoints (`/sleep`, `/wake_up`) to the internet. The raw vLLM hosts
can additionally be published directly (`MODELS_URL_<port>_DIRECT` in `urls.txt`) when
the ports are exposed. All URLs are listed in `urls.txt` (one per server, §6.3);
`GET /servers` indexes them, `GET /v1/models` aggregates them, and `GET /health` shows
the live per-server asleep-state + VRAM. Both Type 1 and Type 2 use these same models.
