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
is 16B, but they are **never all co-resident**: a residency swap keeps only one group's
weights on the GPU at any instant, so the **loaded-and-running total is ≤ 8B at every
moment** — within the limit per **Q3** ("load and unload so that at any single
moment the models resident and running on the GPU stay within 8B").

| Model | Role | Param count (counted) | On GPU during… |
|---|---|---|---|
| `Qwen/Qwen3-4B` | generator | 4B (dense, text-only) | generation |
| `Qwen/Qwen3-4B-Instruct-2507` | generator | 4B (dense, text-only) | generation |
| `google/gemma-4-E4B-it` | judge | **8B total** (MoE/Matformer — total, not the ~4B effective, per Q2) | arbitration |

**The invariant — exactly one group on the GPU:**

```
Generation phase :  Qwen3-4B (4B) + Qwen3-4B-Instruct (4B) AWAKE = 8B on GPU ; judge ASLEEP
Arbitration phase:  gemma-4-E4B (8B) AWAKE                       = 8B on GPU ; both gens ASLEEP
Peak resident     =  max(4+4, 8)                                 = 8B  ✓
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
> (8B, two-pass self-judge) or a single `Qwen3-4B-Instruct-2507` (4B). Switching is a
> 30-second config change if a no-swap footprint is preferred for the slot.

## 4. Serving & verification (vLLM)

Every LLM call hits a **local vLLM** OpenAI-compatible server — **no third-party
inference API** (Together / Fireworks / Groq / Replicate) is used anywhere, in
either pipeline (§6.2 / Q5). Each model runs in its own `vllm serve` process and is
verifiable independently:

- `…/vllm/8001/v1/models` → `Qwen/Qwen3-4B` (generator)
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

## 5. Type 2 (physics) pipeline in detail

The Type 2 path reuses the `exact_fama.ExactFamaPipeline` verbatim behind the
gateway's `PhysicsAdapter`. A deterministic solver owns the answer and unit; the
shared 4B generator is invoked only as a guarded fallback when that solver abstains.

```
question text ──► LaTeX→ASCII normalize ──► quantity extractor (regex, SI normalization)
              ──► registered deterministic solvers (formula/template) ──► answer + unit
                     │  abstains ("Uncertain") / formula error / conf < 0.35
                     ▼
              rule-based PhysicsCanonicalizer ──► re-solve deterministically
                     │  still Uncertain
                     ▼
              single guarded CoT LLM (first generator) fills the value only
              ──► ASCII unit (to_ascii_unit) ──► {answer, unit, premises_used: []}
```

**Understanding.** The extractor (`physics/extractor.py`, `physics_solvers/common.py`)
parses physical quantities directly from the problem text with unit-aware regexes. It
handles scientific notation (`2×10^-6`, `2.0e-6`, `10^-9`), Unicode superscripts/
subscripts, Greek symbols (μ, Ω, π), and SI prefixes, converting every value to SI via
an explicit multiplier table (`mA→1e-3`, `kΩ→1e3`, `μF→1e-6`, `nC→1e-9`, …). Symbols are
inferred from both name (`voltage`, `current`, `capacitance`) and unit context (a bare
`C` in coulombs becomes charge `Q`; `N/C` or `V/m` becomes field `Efield`). Two layers
keep this robust to the committee's LaTeX: (1) the **Notation Mapping CSV**
(`notation_mapping.csv`) declares the ASCII forms our regexes expect, so the committee
rewrites their LaTeX into them before sending; (2) a defensive in-pipeline
`latex_to_ascii` pass (`gateway/units.py`) normalizes any remaining LaTeX up front
(`\times 10^{n}→e-notation`, `\mu F→uF`, `\Omega→ohm`, `R_1`/`R₁→R1`, numeric
`\frac{a}{b}→` decimal) — both verified against the real extractor, idempotent on
already-ASCII input.

**Reasoning (deterministic, code-not-LLM).** `solve_physics` runs the question through
an ordered registry of ~21 deterministic solvers (`physics_solvers/registry.py`); the
first to match returns the result. Coverage implemented in code spans: **DC circuits** —
Ohm's law (V=IR, I=V/R, R=V/I), series/parallel equivalent resistance, electric power
(P=VI, P=I²R, P=V²/R), battery terminal voltage (U=E−Ir), voltage dividers, parallel
bulbs, temperature-dependent resistance; **capacitors** — Q=CV, V=Q/C, energy E=½CV²,
series/parallel combinations and series voltage division, parallel-plate field and
capacitance; **electrostatics** — Coulomb's law F=k|Q₁Q₂|/r², point-charge field
E=k|Q|/r² and potential V=kQ/r, field-from-force E=F/Q, and full vector superposition
over triangle/line geometries; **AC / RLC** — reactances, impedance, RMS current, power
factor, operating-point; **LC resonance** — required C or L and f=1/(2π√(LC));
**magnetism/induction** — solenoid field B=μ₀NI/l, self-inductance, Faraday/self-induced
emf ε=−L·ΔI/Δt, flux; plus measurement-uncertainty (percentage error) and conceptual
templates. The chosen formula computes the value; output uses question-driven rounding
("two decimal places", "nearest integer") and the unit comes from the matched relation —
never from a language model.

**Explanation generation.** Each deterministic result carries solver evidence — the
selected formula, the extracted quantities, and a 4-step CoT (extract → select formula →
substitute → final answer). The optional `ExplanationGenerator.rewrite` turns that into
prose under hard constraints: it must copy `fixed_answer`/`fixed_unit` unchanged and may
not invent numbers, units, or premises (a banned-phrase guard strips internal terms).
This rewrite is **off by default** at the gateway (`PHYSICS_LLM_EXPLANATION=0`) to
protect the speed bonus, so the explanation is the solver's own draft. The unit is
finally rendered ASCII by `to_ascii_unit` (Ω→ohm, μ→u, superscripts→digits): `uF`,
`ohm`, `V/m`, `N/C`.

**Robustness / fallback.** When the deterministic solver abstains — answer `Uncertain`,
a `PHYSICS_FORMULA_ERROR`, or confidence < 0.35 — the **rule-based**
`PhysicsCanonicalizer` rewrites the question into one of 18 known canonical families
(preserving every original number; it never answers) and the solver runs again; the
rewrite is accepted only if it produces a confident (≥ 0.45) result and never overrides
a confident baseline. If the pipeline still returns `Uncertain`, the gateway makes a
**single guarded chain-of-thought call to the first 4B generator** (toggle
`PHYSICS_LLM_FALLBACK`) for a JSON `{answer, unit, steps}`; its value is used only
because there was nothing to preserve — it can never overwrite a confident solver
answer. For every Type 2 query, `premises_used` is always `[]`.
