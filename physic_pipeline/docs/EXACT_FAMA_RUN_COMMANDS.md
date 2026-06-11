# EXACT-FAMA — Command Cheat Sheet

PowerShell commands for the current official-data + preprocess + fixed-benchmark + solver-first + zero-shot parser/rewrite workflow.

Current decisions:

```text
Solver decides answer/unit.
Qwen may help with zero-shot structured parsing and explanation rewrite.
Parser is gated and must not hurt baseline.
Benchmarks must be fixed by ID/hash before comparing modes.
```

---

## 0. Setup

```powershell
cd C:\Users\admin\HCMUT\Project\EXACT-2026\exact-fama-qwen3-prototype
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

For local Qwen HF:

```powershell
pip install -r requirements-train.txt
```

Check:

```powershell
python scripts\check_environment.py
pytest -q
```

---

## 1. Put official data in place

```text
data\official\raw\Logic_Based_Educational_Queries.json
data\official\raw\Physics_Problems_Text_Only.csv
```

---

## 2. Preprocess official data

Recommended conservative clean:

```powershell
python scripts\preprocess_official_data.py `
  --logic data\official\raw\Logic_Based_Educational_Queries.json `
  --physics data\official\raw\Physics_Problems_Text_Only.csv `
  --out_dir data\official\clean
```

Expected outputs:

```text
data\official\clean\Logic_Based_Educational_Queries.clean.json
data\official\clean\Physics_Problems_Text_Only.clean.csv
data\official\clean\preprocess_report.json
data\official\clean\preprocess_suspicious.jsonl
```

Optional strict logic debugging set:

```powershell
python scripts\preprocess_official_data.py `
  --logic data\official\raw\Logic_Based_Educational_Queries.json `
  --physics data\official\raw\Physics_Problems_Text_Only.csv `
  --out_dir data\official\clean_strict `
  --drop_suspect_logic
```

Use strict mode only for internal debugging. For official-like evaluation, start with conservative clean.

---

## 3. Prepare project JSONL splits

Use cleaned files, not raw official files:

```powershell
python scripts\prepare_official_data.py `
  --logic data\official\clean\Logic_Based_Educational_Queries.clean.json `
  --physics data\official\clean\Physics_Problems_Text_Only.clean.csv `
  --out_dir data\raw `
  --train_ratio 0.80 `
  --dev_ratio 0.10 `
  --seed 2026 `
  --smoke_size 100
```

Expected outputs:

```text
data\raw\train.jsonl
data\raw\dev.jsonl
data\raw\blind.jsonl
data\raw\smoke_100.jsonl
data\raw\all_official.jsonl
data\raw\official_split_report.json
```

Important: `data\raw\smoke_100.jsonl` can change when preprocessing changes the source population. Do not compare scores across regenerated smoke files.

---

## 4. Create fixed logic/physics benchmarks

Create stable hash-by-id benchmarks from `all_official.jsonl`:

```powershell
python scripts\make_fixed_smoke_benchmarks.py `
  --input data\raw\all_official.jsonl `
  --out_dir data\eval `
  --logic_size 100 `
  --physics_size 100 `
  --seed 2026
```

Expected outputs:

```text
data\eval\fixed_smoke_logic_100.jsonl
data\eval\fixed_smoke_physics_100.jsonl
data\eval\fixed_smoke_mixed_200.jsonl
data\eval\fixed_smoke_benchmark_report.json
```

Use these fixed files for all mode comparisons:

```text
solver-only
rewrite-only
parse+rewrite
```

---

## 5. Solver-only baseline on fixed logic

```powershell
$env:LLM_BACKEND="none"
$env:LORA_ADAPTER_PATH=""
$env:EXACT_FAMA_CONFIG="configs/eval_solver_only.yaml"
$env:EXACT_USE_PROVIDED_FOL="0"

python scripts\run_inference.py `
  --input data\eval\fixed_smoke_logic_100.jsonl `
  --output artifacts\fixed_logic100_solver_only_predictions.jsonl `
  --log_every 10 `
  --write_every 20

python scripts\evaluate.py `
  --gold data\eval\fixed_smoke_logic_100.jsonl `
  --pred artifacts\fixed_logic100_solver_only_predictions.jsonl `
  --report artifacts\fixed_logic100_solver_only_eval.json
```

---

## 6. Solver-only baseline on fixed physics

```powershell
$env:LLM_BACKEND="none"
$env:LORA_ADAPTER_PATH=""
$env:EXACT_FAMA_CONFIG="configs/eval_solver_only.yaml"
$env:EXACT_USE_PROVIDED_FOL="0"

python scripts\run_inference.py `
  --input data\eval\fixed_smoke_physics_100.jsonl `
  --output artifacts\fixed_physics100_solver_only_predictions.jsonl `
  --log_every 10 `
  --write_every 20

python scripts\evaluate.py `
  --gold data\eval\fixed_smoke_physics_100.jsonl `
  --pred artifacts\fixed_physics100_solver_only_predictions.jsonl `
  --report artifacts\fixed_physics100_solver_only_eval.json
```

---

## 7. Solver-only baseline on mixed fixed benchmark

```powershell
$env:LLM_BACKEND="none"
$env:LORA_ADAPTER_PATH=""
$env:EXACT_FAMA_CONFIG="configs/eval_solver_only.yaml"
$env:EXACT_USE_PROVIDED_FOL="0"

python scripts\run_inference.py `
  --input data\eval\fixed_smoke_mixed_200.jsonl `
  --output artifacts\fixed_mixed200_solver_only_predictions.jsonl `
  --log_every 10 `
  --write_every 20

python scripts\evaluate.py `
  --gold data\eval\fixed_smoke_mixed_200.jsonl `
  --pred artifacts\fixed_mixed200_solver_only_predictions.jsonl `
  --report artifacts\fixed_mixed200_solver_only_eval.json
```

---

## 8. Qwen rewrite-only on fixed benchmarks

Logic:

```powershell
$env:LLM_BACKEND="hf"
$env:MODEL_NAME="models/Qwen3-8B"
$env:LORA_ADAPTER_PATH=""
$env:LLM_TEMPERATURE="0.1"
$env:LLM_MAX_NEW_TOKENS="512"
$env:EXACT_FAMA_CONFIG="configs/eval_qwen_rewrite.yaml"
$env:EXACT_USE_PROVIDED_FOL="0"

python scripts\run_inference.py `
  --input data\eval\fixed_smoke_logic_100.jsonl `
  --output artifacts\fixed_logic100_qwen_rewrite_predictions.jsonl `
  --log_every 5 `
  --write_every 20

python scripts\evaluate.py `
  --gold data\eval\fixed_smoke_logic_100.jsonl `
  --pred artifacts\fixed_logic100_qwen_rewrite_predictions.jsonl `
  --report artifacts\fixed_logic100_qwen_rewrite_eval.json

python scripts\evaluate_explanation_quality.py `
  --pred artifacts\fixed_logic100_qwen_rewrite_predictions.jsonl `
  --report artifacts\fixed_logic100_qwen_rewrite_expl_quality.json
```

Physics:

```powershell
python scripts\run_inference.py `
  --input data\eval\fixed_smoke_physics_100.jsonl `
  --output artifacts\fixed_physics100_qwen_rewrite_predictions.jsonl `
  --log_every 5 `
  --write_every 20

python scripts\evaluate.py `
  --gold data\eval\fixed_smoke_physics_100.jsonl `
  --pred artifacts\fixed_physics100_qwen_rewrite_predictions.jsonl `
  --report artifacts\fixed_physics100_qwen_rewrite_eval.json

python scripts\evaluate_explanation_quality.py `
  --pred artifacts\fixed_physics100_qwen_rewrite_predictions.jsonl `
  --report artifacts\fixed_physics100_qwen_rewrite_expl_quality.json
```

---

## 9. Zero-shot parser + rewrite on fixed logic

Parser is primarily intended for logic. Start with logic-only evaluation.

```powershell
$env:LLM_BACKEND="hf"
$env:MODEL_NAME="models/Qwen3-8B"
$env:LORA_ADAPTER_PATH=""
$env:LLM_TEMPERATURE="0.0"
$env:LLM_MAX_NEW_TOKENS="768"
$env:EXACT_FAMA_CONFIG="configs/eval_qwen_parse_rewrite.yaml"
$env:EXACT_USE_PROVIDED_FOL="0"

python scripts\run_inference.py `
  --input data\eval\fixed_smoke_logic_100.jsonl `
  --output artifacts\fixed_logic100_qwen_parse_rewrite_predictions.jsonl `
  --log_every 5 `
  --write_every 20

python scripts\evaluate.py `
  --gold data\eval\fixed_smoke_logic_100.jsonl `
  --pred artifacts\fixed_logic100_qwen_parse_rewrite_predictions.jsonl `
  --report artifacts\fixed_logic100_qwen_parse_rewrite_eval.json

python scripts\evaluate_explanation_quality.py `
  --pred artifacts\fixed_logic100_qwen_parse_rewrite_predictions.jsonl `
  --report artifacts\fixed_logic100_qwen_parse_rewrite_expl_quality.json
```

Only keep parser mode if it is at least as good as solver-only on the same fixed file.

---

## 10. Parser-only inspection

Use this to inspect raw parser output without solver, verifier, rewrite, or evaluator:

```powershell
$env:LLM_BACKEND="hf"
$env:MODEL_NAME="models/Qwen3-8B"
$env:LORA_ADAPTER_PATH=""
$env:LLM_TEMPERATURE="0.0"
$env:LLM_MAX_NEW_TOKENS="768"
$env:EXACT_FAMA_CONFIG="configs/eval_qwen_parse_rewrite.yaml"

python scripts\run_parser_only.py `
  --input data\eval\fixed_smoke_logic_100.jsonl `
  --output artifacts\fixed_logic100_parser_only.jsonl `
  --limit 10 `
  --pretty `
  --include_question `
  --include_premises
```

Print first examples:

```powershell
python scripts\run_parser_only.py `
  --input data\eval\fixed_smoke_logic_100.jsonl `
  --output artifacts\debug_parser_only.jsonl `
  --limit 3 `
  --print `
  --include_question `
  --include_premises
```

Inspect:

```text
artifacts\fixed_logic100_parser_only.pretty.json
artifacts\fixed_logic100_parser_only.summary.json
```

---

## 11. Filter solver-correct samples

Useful for judging explanation quality without solver mistakes.

```powershell
python scripts\filter_correct_predictions.py `
  --gold data\eval\fixed_smoke_logic_100.jsonl `
  --pred artifacts\fixed_logic100_solver_only_predictions.jsonl `
  --out_gold data\eval\fixed_logic100_solver_correct.jsonl

python scripts\run_inference.py `
  --input data\eval\fixed_logic100_solver_correct.jsonl `
  --output artifacts\fixed_logic100_solver_correct_qwen_rewrite_predictions.jsonl `
  --log_every 5 `
  --write_every 20

python scripts\evaluate_explanation_quality.py `
  --pred artifacts\fixed_logic100_solver_correct_qwen_rewrite_predictions.jsonl `
  --report artifacts\fixed_logic100_solver_correct_qwen_rewrite_expl_quality.json
```

---

## 12. Predict one

```powershell
python scripts\predict_one.py `
  --file data\eval\fixed_smoke_logic_100.jsonl `
  --index 0 `
  --show_pipeline `
  --show_trace `
  --output artifacts\debug_one.json
```

Inline physics:

```powershell
python scripts\predict_one.py `
  --json '{"type":"physics","question":"Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."}' `
  --show_pipeline `
  --show_trace
```

---

## 13. Interactive tools

Terminal UI:

```powershell
python scripts\predict_interactive.py
```

Gradio UI:

```powershell
python scripts\app_gradio.py
```

Open:

```text
http://127.0.0.1:7860
```

---

## 14. API

```powershell
uvicorn exact_fama.api:app --host 0.0.0.0 --port 8000 --reload
```

Test:

```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"type":"physics","question":"Calculate the current when V = 12 V and R = 4 ohms."}'
```

---

## 15. Docker solver-only API

```powershell
docker compose up --build
```

```powershell
curl http://localhost:8000/health
```

---

## 16. Quick sanity checks

```powershell
python -c "from exact_fama.config import load_settings; s=load_settings(); print(s.raw)"
python -c "from exact_fama.pipeline import ExactFamaPipeline; p=ExactFamaPipeline(); print(p.llm.backend, p.llm.model_name, p.explainer.use_llm, p.parser.use_llm)"
```

Check gold/pred id order before trusting an eval:

```powershell
python - <<'PY'
import json
gold_path = "data/eval/fixed_smoke_logic_100.jsonl"
pred_path = "artifacts/fixed_logic100_solver_only_predictions.jsonl"
with open(gold_path, encoding="utf-8") as fg, open(pred_path, encoding="utf-8") as fp:
    gold_ids = [json.loads(x)["id"] for x in fg if x.strip()]
    pred_ids = [json.loads(x)["id"] for x in fp if x.strip()]
print("gold rows:", len(gold_ids), "pred rows:", len(pred_ids))
print("same order:", gold_ids == pred_ids)
if gold_ids != pred_ids:
    raise SystemExit("Gold/pred mismatch. Re-run inference on the same input file.")
PY
```
