# Full-LLM direct flow

This flow bypasses the solver-first pipeline.

It does **not** call:

```text
structured parser
logic solver
physics solver
answer verifier
explanation rewriter
```

It sends each question directly to Qwen and asks for final JSON output.

Use this only as an experiment/baseline comparison. The official project contract still prefers solver-first unless full-LLM direct evaluation is clearly better.

---

## Files added

```text
src/exact_fama/llm/direct_answerer.py
scripts/run_inference_full_llm.py
configs/eval_qwen_full_llm.yaml
README_FULL_LLM_FLOW.md
```

---

## Run on fixed logic benchmark

```powershell
$env:LLM_BACKEND="hf"
$env:MODEL_NAME="models/Qwen3-8B"
$env:LORA_ADAPTER_PATH=""
$env:LLM_TEMPERATURE="0.0"
$env:LLM_MAX_NEW_TOKENS="768"
$env:EXACT_FAMA_CONFIG="configs/eval_qwen_full_llm.yaml"
$env:EXACT_USE_PROVIDED_FOL="0"

python scripts\run_inference_full_llm.py `
  --input data\eval\fixed_smoke_logic_100.jsonl `
  --output artifacts\fixed_logic100_full_llm_predictions.jsonl `
  --log_every 5 `
  --write_every 20

python scripts\evaluate.py `
  --gold data\eval\fixed_smoke_logic_100.jsonl `
  --pred artifacts\fixed_logic100_full_llm_predictions.jsonl `
  --report artifacts\fixed_logic100_full_llm_eval.json
```

---

## Run on fixed physics benchmark

```powershell
$env:LLM_BACKEND="hf"
$env:MODEL_NAME="models/Qwen3-8B"
$env:LORA_ADAPTER_PATH=""
$env:LLM_TEMPERATURE="0.0"
$env:LLM_MAX_NEW_TOKENS="768"
$env:EXACT_FAMA_CONFIG="configs/eval_qwen_full_llm.yaml"
$env:EXACT_USE_PROVIDED_FOL="0"

python scripts\run_inference_full_llm.py `
  --input data\eval\fixed_smoke_physics_100.jsonl `
  --output artifacts\fixed_physics100_full_llm_predictions.jsonl `
  --log_every 5 `
  --write_every 20

python scripts\evaluate.py `
  --gold data\eval\fixed_smoke_physics_100.jsonl `
  --pred artifacts\fixed_physics100_full_llm_predictions.jsonl `
  --report artifacts\fixed_physics100_full_llm_eval.json
```

---

## Run on mixed benchmark

```powershell
python scripts\run_inference_full_llm.py `
  --input data\eval\fixed_smoke_mixed_200.jsonl `
  --output artifacts\fixed_mixed200_full_llm_predictions.jsonl `
  --log_every 5 `
  --write_every 20

python scripts\evaluate.py `
  --gold data\eval\fixed_smoke_mixed_200.jsonl `
  --pred artifacts\fixed_mixed200_full_llm_predictions.jsonl `
  --report artifacts\fixed_mixed200_full_llm_eval.json
```

---

## Recommended comparison table

| Benchmark | solver-only | parse+rewrite | full-LLM direct |
|---|---:|---:|---:|
| fixed logic 100 | | | |
| fixed physics 100 | | | |
| fixed mixed 200 | | | |

Only compare runs on the exact same fixed benchmark file.

---

## Output format

The direct LLM flow writes standard prediction rows:

```json
{
  "id": "sample-id",
  "answer": "A",
  "unit": null,
  "explanation": "Premise 1 and premise 3 support option A.",
  "fol": null,
  "cot": ["Identify relevant premises.", "Check each option.", "Select final answer."],
  "premises": [],
  "confidence": 0.72,
  "task_type": "logic",
  "used_modules": ["input_normalizer", "task_router", "full_llm_direct_answerer", "full_llm_output_validator"],
  "warnings": [],
  "debug": {
    "mode": "full_llm_direct",
    "raw_outputs": ["..."]
  }
}
```

---

## Notes

- For logic MCQ, the model is instructed to answer `A`, `B`, `C`, `D`, or `Unknown`.
- For logic yes/no, the model is instructed to answer `Yes`, `No`, or `Unknown`.
- For physics, the model is instructed to put the value in `answer` and the unit in `unit`.
- JSON parsing failures are retried once by default.
- Raw LLM outputs are stored in `debug.raw_outputs` by default for auditing.
- Use `--no_raw_debug` if artifact size becomes too large.
