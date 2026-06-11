# Parser-only inspection script

Use `scripts/run_parser_only.py` when you want to inspect raw parser output without involving the solver.

This script calls only:

```text
PredictRequest validation
  -> route_task
  -> StructuredParser.parse
  -> JSONL dump
```

It does **not** call:

```text
logic solver
physics solver
answer verifier
explanation rewriter
evaluator
```

---

## 1. Required config

Use parser config:

```powershell
$env:LLM_BACKEND="hf"
$env:MODEL_NAME="models/Qwen3-8B"
$env:LORA_ADAPTER_PATH=""
$env:LLM_TEMPERATURE="0.0"
$env:LLM_MAX_NEW_TOKENS="768"
$env:EXACT_FAMA_CONFIG="configs/eval_qwen_parse_rewrite.yaml"
```

---

## 2. Run on fixed logic smoke

```powershell
python scripts\run_parser_only.py `
  --input data\eval\fixed_smoke_logic_100.jsonl `
  --output artifacts\fixed_logic100_parser_only.jsonl `
  --limit 10 `
  --pretty `
  --include_question `
  --include_premises
```

Outputs:

```text
artifacts\fixed_logic100_parser_only.jsonl
artifacts\fixed_logic100_parser_only.pretty.json
artifacts\fixed_logic100_parser_only.summary.json
```

---

## 3. Print first few outputs directly

```powershell
python scripts\run_parser_only.py `
  --input data\eval\fixed_smoke_logic_100.jsonl `
  --output artifacts\debug_parser_only.jsonl `
  --limit 3 `
  --print `
  --include_question `
  --include_premises
```

---

## 4. Run on a single custom JSONL

Create a small file:

```json
{"id":"debug-1","type":"logic","question":"Does John qualify?","premises-NL":["If a student completes all required courses, they are eligible for graduation.","John completes all required courses."],"answer":"Yes"}
```

Run:

```powershell
python scripts\run_parser_only.py `
  --input artifacts\debug_parser_input.jsonl `
  --output artifacts\debug_parser_output.jsonl `
  --pretty `
  --include_question `
  --include_premises
```

---

## 5. What to inspect

Open:

```text
artifacts\fixed_logic100_parser_only.pretty.json
artifacts\fixed_logic100_parser_only.summary.json
```

Check each row:

```text
parser_diagnostics.warning_count == 0
parser_diagnostics.fact_count is reasonable
parser_diagnostics.rule_count is reasonable
facts/rules are grounded in actual premises only
no option text is parsed as fact
no question/query is parsed as fact
negation is preserved
AND conditions are preserved
thresholds such as at least / greater than / fewer than are preserved
exceptions are preserved
premise ids match actual premise indices
```

---

## 6. Interpreting parser output

Good parser output should look like:

```json
{
  "facts": [
    {"id": 2, "original": "John completes all required courses.", "fact": "John completes all required courses"}
  ],
  "rules": [
    {
      "id": 1,
      "original": "If a student completes all required courses, they are eligible for graduation.",
      "if": ["a student completes all required courses"],
      "then": ["they are eligible for graduation"]
    }
  ],
  "query": "Does John qualify?"
}
```

Bad parser output includes:

```text
option A/B/C/D converted into facts
question converted into fact
fact contains a full if-then rule
negation removed
AND condition dropped
threshold dropped
parser output has warnings
```

---

## 7. Parser-only versus parse+rewrite

Parser-only is for inspection:

```text
run_parser_only.py
```

Parse+rewrite is for actual pipeline evaluation:

```text
run_inference.py with configs/eval_qwen_parse_rewrite.yaml
```

Only run parse+rewrite after parser-only output looks reasonable.
