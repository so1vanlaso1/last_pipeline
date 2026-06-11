# EXACT-FAMA Qwen3-8B Prototype

Solver-first prototype for the EXACT 2026 educational QA challenge.

The current project state is:

- **Solver decides final `answer` and `unit`** for logic and physics.
- **Qwen3-8B local is optional** and used for:
  - zero-shot structured parsing, behind strict validation and solver-backed gating;
  - explanation rewriting, without changing answer/unit.
- **Parser finetuning is not active yet.** The current parser experiment is zero-shot first.
- **LoRA/external-data training is legacy for now.**
- **Benchmarks must be fixed by ID/hash** before comparing solver-only, rewrite-only, and parse+rewrite.

The core design:

```text
LLM proposes.
Parser is validated.
Solver decides.
Validator preserves answer/unit.
Rewrite improves explanation only.
```

---

## Current pipeline

```text
Input JSON
  -> PredictRequest validation
  -> Task router
  -> optional zero-shot structured parser
  -> baseline solver run
  -> optional parser-assisted solver run
  -> gated result selection
  -> answer/unit verifier
  -> optional Qwen explanation rewrite
  -> output validator
  -> EXACT-style JSON
```

The parser is not allowed to freely override answers. Failed or rejected parser output is not passed into the solver.

---

## Recommended modes

| Mode | Config | Purpose |
|---|---|---|
| Solver-only | `configs/eval_solver_only.yaml` | Fast baseline; no LLM loading |
| Qwen rewrite | `configs/eval_qwen_rewrite.yaml` | Solver answer + Qwen explanation rewrite |
| Parser-only inspection | `scripts/run_parser_only.py` | Inspect raw parser output without solver/rewrite |
| Parse + rewrite | `configs/eval_qwen_parse_rewrite.yaml` | Experimental zero-shot parser with gated solver-backed use |

---

## Repository layout

```text
exact-fama-qwen3-prototype/
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
    eval_solver_only.yaml
    eval_qwen_rewrite.yaml
    eval_qwen_parse_rewrite.yaml
  data/
    official/raw/                  # official source files, not committed
    official/clean/                # preprocessed official files, not committed
    raw/                           # generated train/dev/all jsonl, not committed
    eval/                          # fixed benchmark jsonl, not committed
  models/Qwen3-8B/                 # local model weights, not committed
  artifacts/                       # predictions/eval reports, not committed
  scripts/
    preprocess_official_data.py
    prepare_official_data.py
    make_fixed_smoke_benchmarks.py
    run_parser_only.py
    run_inference.py
    predict_one.py
    predict_interactive.py
    app_gradio.py
    evaluate.py
    evaluate_explanation_quality.py
    filter_correct_predictions.py
    check_rewrite_changed.py
  src/exact_fama/
    api.py
    pipeline.py
    router.py
    schemas.py
    explanation.py
    llm/
      qwen_client.py
      structured_parser.py
    logic/
    physics/
    utils/
  tests/
```

---

## Quick start

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
pytest -q
```

For local Qwen HF inference:

```powershell
pip install -r requirements-train.txt
```

Put official files here:

```text
data\official\raw\Logic_Based_Educational_Queries.json
data\official\raw\Physics_Problems_Text_Only.csv
```

---

## Preprocess official data

```powershell
python scripts\preprocess_official_data.py `
  --logic data\official\raw\Logic_Based_Educational_Queries.json `
  --physics data\official\raw\Physics_Problems_Text_Only.csv `
  --out_dir data\official\clean
```

This removes or quarantines known data issues, especially physics rows with missing answers and translation/meta-text artifacts. Suspect logic records are audited by default rather than automatically dropped.

---

## Prepare splits

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

Do not compare accuracy across regenerated `data\raw\smoke_100.jsonl` files. If preprocessing changes the population, smoke examples change too.

---

## Create fixed benchmarks

```powershell
python scripts\make_fixed_smoke_benchmarks.py `
  --input data\raw\all_official.jsonl `
  --out_dir data\eval `
  --logic_size 100 `
  --physics_size 100 `
  --seed 2026
```

Outputs:

```text
data\eval\fixed_smoke_logic_100.jsonl
data\eval\fixed_smoke_physics_100.jsonl
data\eval\fixed_smoke_mixed_200.jsonl
data\eval\fixed_smoke_benchmark_report.json
```

Use these files for stable comparisons.

---

## Run solver-only logic baseline

```powershell
$env:LLM_BACKEND="none"
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

## Run solver-only physics baseline

```powershell
$env:LLM_BACKEND="none"
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

## Inspect parser only

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

---

## Run parse + rewrite

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
```

Only keep parser mode if it does not reduce accuracy on the same fixed benchmark.

---

## Interactive use

CLI:

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

FastAPI:

```powershell
uvicorn exact_fama.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Submission notes

- Do not use closed-source LLMs in the submitted pipeline.
- Keep Qwen3-8B local/open-weight and within the <=8B constraint.
- Do not commit `models/`, generated `data/`, `artifacts/`, LoRA weights, or official raw dataset files.
- Qwen rewrite must not change `answer` or `unit`.
- Parser mode is experimental and gated. Use it only if it does not reduce correctness.
