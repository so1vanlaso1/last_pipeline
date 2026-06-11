# EXACT-FAMA — Current Full Pipeline Setup

This document reflects the current project state after adding:

```text
official dataset
  -> conservative preprocess
  -> prepare jsonl
  -> fixed logic/physics benchmarks
  -> solver-only baseline
  -> zero-shot Qwen parser inspection
  -> gated parse+rewrite experiment
```

LoRA training and external datasets remain legacy experiments. Parser finetuning is intentionally deferred until zero-shot parser behavior is verified.

---

## 1. Pipeline contract

```text
Input
  -> task router
  -> optional structured parser
  -> baseline solver
  -> optional parser-assisted solver
  -> gated result selection
  -> verifier validates answer/unit
  -> optional Qwen rewrite writes explanation only
  -> output validator
```

Hard rules:

```text
Solver decides final answer/unit.
Qwen rewrite must not change answer/unit.
Parser output must be schema-valid, source-grounded, and option/question-clean.
Rejected parser output must not be passed into the solver.
```

---

## 2. Environment setup

Windows PowerShell:

```powershell
cd C:\Users\admin\HCMUT\Project\EXACT-2026\exact-fama-qwen3-prototype
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

For local Qwen HF inference:

```powershell
pip install -r requirements-train.txt
```

Verify:

```powershell
python scripts\check_environment.py
pytest -q
```

---

## 3. Official dataset placement

Put official files here:

```text
data\official\raw\Logic_Based_Educational_Queries.json
data\official\raw\Physics_Problems_Text_Only.csv
```

---

## 4. Preprocess official dataset

Conservative preprocess:

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

What preprocessing does:

```text
Physics:
  - drops rows with missing answer by default
  - drops QA-prefix rows with missing answer
  - detects translation/meta-text contamination
  - preserves clean scored rows

Logic:
  - preserves records by default
  - audits suspicious answer/explanation conflicts
  - optionally drops suspect records with --drop_suspect_logic
```

Strict logic debug mode:

```powershell
python scripts\preprocess_official_data.py `
  --logic data\official\raw\Logic_Based_Educational_Queries.json `
  --physics data\official\raw\Physics_Problems_Text_Only.csv `
  --out_dir data\official\clean_strict `
  --drop_suspect_logic
```

Use strict mode for internal debugging, not as the default official-like benchmark.

---

## 5. Prepare JSONL splits

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

Important: `smoke_100.jsonl` is not a stable benchmark if source rows change. Use fixed benchmarks for comparisons.

---

## 6. Create fixed benchmarks

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

Use the fixed files for fair comparisons:

```text
solver-only vs rewrite-only vs parse+rewrite
```

---

## 7. Configs

### Solver-only

```powershell
$env:LLM_BACKEND="none"
$env:LORA_ADAPTER_PATH=""
$env:EXACT_FAMA_CONFIG="configs/eval_solver_only.yaml"
$env:EXACT_USE_PROVIDED_FOL="0"
```

### Qwen rewrite-only

```powershell
$env:LLM_BACKEND="hf"
$env:MODEL_NAME="models/Qwen3-8B"
$env:LORA_ADAPTER_PATH=""
$env:LLM_TEMPERATURE="0.1"
$env:LLM_MAX_NEW_TOKENS="512"
$env:EXACT_FAMA_CONFIG="configs/eval_qwen_rewrite.yaml"
$env:EXACT_USE_PROVIDED_FOL="0"
```

### Parser-only / parse+rewrite

```powershell
$env:LLM_BACKEND="hf"
$env:MODEL_NAME="models/Qwen3-8B"
$env:LORA_ADAPTER_PATH=""
$env:LLM_TEMPERATURE="0.0"
$env:LLM_MAX_NEW_TOKENS="768"
$env:EXACT_FAMA_CONFIG="configs/eval_qwen_parse_rewrite.yaml"
$env:EXACT_USE_PROVIDED_FOL="0"
```

---

## 8. Run solver-only fixed baselines

Logic:

```powershell
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

Physics:

```powershell
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

Mixed:

```powershell
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

## 9. Parser-only inspection

Run parser alone without solver/rewrite/eval:

```powershell
python scripts\run_parser_only.py `
  --input data\eval\fixed_smoke_logic_100.jsonl `
  --output artifacts\fixed_logic100_parser_only.jsonl `
  --limit 10 `
  --pretty `
  --include_question `
  --include_premises
```

Inspect:

```text
artifacts\fixed_logic100_parser_only.pretty.json
artifacts\fixed_logic100_parser_only.summary.json
```

Look for:

```text
warning_count == 0
facts/rules only from premises
no option text parsed as fact
no question parsed as fact
negation preserved
AND conditions preserved
thresholds preserved
```

---

## 10. Parse + rewrite experiment

```powershell
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

Parser mode should be compared only against solver-only on the same fixed benchmark.

---

## 11. Evaluate rewrite on solver-correct samples only

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

## 12. Single-sample and interactive testing

Single sample from JSONL:

```powershell
python scripts\predict_one.py `
  --file data\eval\fixed_smoke_logic_100.jsonl `
  --index 0 `
  --show_pipeline `
  --show_trace `
  --output artifacts\debug_one.json
```

Interactive CLI:

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

## 13. Docker

Docker is intended mainly for solver-only API deployment. It does not install GPU/HF training dependencies by default.

```powershell
docker compose up --build
```

Health check:

```powershell
curl http://localhost:8000/health
```

Prediction:

```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"type":"physics","question":"Calculate the current when V = 12 V and R = 4 ohms."}'
```

---

## 14. Git hygiene

Do not commit:

```text
models/
data/official/
data/raw/
data/eval/
data/processed/
artifacts/
outputs/
runs/
wandb/
*.safetensors
*.arrow
```

Recommended tracked files:

```text
README.md
SETUP_FULL_PIPELINE.md
EXACT_FAMA_RUN_COMMANDS.md
README_PATCH_USAGE.md
README_FIXED_BENCHMARKS.md
README_PARSER_ONLY.md
requirements.txt
requirements-train.txt
pyproject.toml
Dockerfile
docker-compose.yml
Makefile
configs/
scripts/
src/
tests/
```

---

## 15. Current project decisions

- Keep zero-shot Qwen rewrite as the active explanation path.
- Keep parser zero-shot until parser-only inspection is satisfactory.
- Keep parser gated and solver-backed.
- Keep LoRA/external-data SFT scripts in `scripts/legacy/` unless intentionally revived.
- Evaluate parser primarily on fixed logic benchmark.
- Evaluate physics separately because parser does not fix formula coverage.
- Do not rely on hidden chain-of-thought; use public `cot`, `proof_steps`, `formula`, `quantities`, and final `explanation`.
