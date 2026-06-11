# Fixed smoke benchmark scripts

Use `scripts/make_fixed_smoke_benchmarks.py` to create stable logic-only and physics-only benchmarks.

This avoids comparing scores across different randomly generated `smoke_100.jsonl` files.

---

## 1. Why fixed benchmarks are needed

`prepare_official_data.py` shuffles rows and takes a smoke subset. If preprocessing changes the source dataset, then the smoke set can change even with the same random seed.

That makes scores incomparable:

```text
old smoke_100.jsonl accuracy != new smoke_100.jsonl accuracy
```

The fixed benchmark script samples rows by stable hash of `id`, so the selected examples do not depend on input row order.

---

## 2. Create fixed logic/physics smoke sets

Run after `prepare_official_data.py` has produced:

```text
data/raw/all_official.jsonl
```

Command:

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
data/eval/fixed_smoke_logic_100.jsonl
data/eval/fixed_smoke_physics_100.jsonl
data/eval/fixed_smoke_mixed_200.jsonl
data/eval/fixed_smoke_benchmark_report.json
```

---

## 3. Run solver-only baseline on logic

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

## 4. Run solver-only baseline on physics

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

## 5. Run mixed benchmark

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

## 6. Compare parser fairly

Always run all modes on the same fixed file:

```text
fixed_smoke_logic_100.jsonl
fixed_smoke_physics_100.jsonl
fixed_smoke_mixed_200.jsonl
```

Recommended comparison table:

| Benchmark | solver-only | rewrite-only | parse+rewrite |
|---|---:|---:|---:|
| fixed logic 100 | | | |
| fixed physics 100 | | | |
| fixed mixed 200 | | | |

Parser should be judged primarily on `fixed_smoke_logic_100.jsonl`.

Physics should be evaluated separately because formula coverage belongs to the physics solver path, not the logic parser.

---

## 7. Check benchmark report

Open:

```text
data/eval/fixed_smoke_benchmark_report.json
```

Verify:

```text
logic.actual_size == requested logic size
physics.actual_size == requested physics size
logic_physics_id_overlap_count == 0
checksum_sha256 is recorded
first_ids are stable across repeated runs
```

---

## 8. Do not compare against regenerated smoke_100

Do not compare:

```text
old data/raw/smoke_100.jsonl
new data/raw/smoke_100.jsonl
```

because preprocessing can change source population and therefore change the smoke examples.

Use fixed benchmark files for all score comparisons.
