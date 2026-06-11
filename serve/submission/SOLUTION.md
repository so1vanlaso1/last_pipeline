# EXACT 2026 — Solution Description

> One-page solution description (export this file to **solution.pdf** for the
> submission ZIP). Fill in the bracketed dataset sample-counts/examples with your
> finalized numbers before submitting.

## 1. System overview

A single public HTTP endpoint, `POST /predict`, accepts the unified competition
schema and routes each query by `type` to one of two internal pipelines:

```
POST /predict
  ├── type1 → Logic pipeline   (strict formal-logic examiner prompts + answer parsing,
  │                              served by the shared 8B LLM; premises_used extracted)
  └── type2 → Physics pipeline  (deterministic formula/template solver decides the
                                 numerical answer + unit; the LLM only canonicalizes
                                 phrasing and rewrites the explanation)
```

Both pipelines call **one shared LLM** served by **vLLM** (OpenAI-compatible). The
gateway also reverse-proxies `GET /v1/models` so the committee can verify the model
on the same host.

- **Understanding.** Type 1: premises + question are rendered into a strict
  formal-logic examiner prompt (direct implication, contraposition, converse/inverse
  guards, quantifier and necessary/sufficient rules). Type 2: a deterministic
  extractor parses physical quantities (values + units, scientific notation, Greek
  symbols) from the problem text.
- **Reasoning.** Type 1: the LLM answers in a strict `ANSWER: / WHY:` format that is
  parsed to a canonical label and mapped back to the exact option (or a free-form
  number/text answer when no options are given). Type 2: registered circuit /
  electrostatics formula solvers compute the result; the answer and unit come from
  the solver, not the LLM.
- **Explanation generation.** Type 1: the examiner's justification plus the premise
  indices it used (`premises_used`). Type 2: an LLM rewrite of the verified solver
  evidence that is forbidden from changing the answer or unit.

`premises_used` (Type 1, 50% of the score) is taken from the model's premise
citations, with a small dedicated JSON call as a fallback; this call never changes
the answer. Type 2 returns the numerical `answer` and an ASCII `unit`, with
`premises_used = []`.

## 2. Datasets used

| Dataset | Source / origin | Samples used | Notes |
|---|---|---|---|
| EXACT 2026 Logic (`Logic_Based_Educational_Queries.json`) | Official EXACT 2026 release | [N] | Type 1 prompt/parse tuning. Only premises-NL + questions are used; gold answers/FOL are never shown to the model. |
| EXACT 2026 Physics (`Physics_Problems_Text_Only.csv`) | Official EXACT 2026 release | [N] | Type 2 formula/template solver coverage. |
| [External / synthetic physics, if used] | [origin] | [N] | [one-line description + a couple of sample entries] |

Sample entries (fill in 2–3 short examples per dataset before submitting).

## 3. Model size calculation (≤ 8B at any moment)

The resident line-up is configurable (`serve/logic_config.yaml`); whichever you
submit, the parameters loaded and running at any single moment stay within 8B.
**Pick the one row that matches what you ran and delete the others:**

| Line-up | Models loaded & running | Params (effective / total) | Type 1 |
|---|---|---|---|
| **Two-judge vote** (default) | `Qwen/Qwen3.5-4B` + `google/gemma-4-E2B-it` | 4B + 2.3B = 6.3B eff (9.1B w/ embeddings) | cascade weighted vote |
| **Gen→judge (default)** | `Qwen/Qwen3.5-4B` + `google/gemma-4-E2B-it` + `google/gemma-4-E4B-it` (judge) | disk swap → peak ~9.1B on GPU | generate→judge |
| **Single Gemma-4-E4B** | `google/gemma-4-E4B-it` | 4.5B eff / **8B total** | single model |
| **Single Qwen3.5-4B** | `Qwen/Qwen3.5-4B` | 4B / 4B | single model |

All four are open-source and served via vLLM. Non-LLM tools (the deterministic
logic/physics solvers, regex extractors, unit verifiers) are **0 params** and
don't count (Section 6.3). The launcher refuses to start a line-up over the limit.

> State your parameter-counting convention explicitly. For the Gemma "E" models,
> Google reports both an **effective** count and a **with-embeddings** total
> (E2B = 2.3B / 5.1B; E4B = 4.5B / 8B). The default generate→judge line-up uses
> **disk swap** (vLLM sleep level 2) so only one group is on the GPU at a time —
> peak ~9.1B (the two generators); `gemma-4-E4B-it` alone is ≤ 8B by the strict
> with-embeddings total.

`GET /v1/models` reports every resident model, matching what is declared here.
Both Type 1 and Type 2 use these same resident model(s); the Type 2 physics solver
is deterministic and only calls the LLM for an optional fallback/explanation.
