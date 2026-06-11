PYTHON ?= python
SMOKE ?= data/raw/smoke_100.jsonl
ALL ?= data/raw/all_official.jsonl
ARTIFACTS ?= artifacts
CONFIG_SOLVER ?= configs/eval_solver_only.yaml
CONFIG_REWRITE ?= configs/eval_qwen_rewrite.yaml

.PHONY: install install-train test prepare-official smoke-solver smoke-qwen eval-smoke-qwen filter-smoke-correct all-solver api gradio docker-up docker-down clean

install:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	pip install -e .

install-train: install
	pip install -r requirements-train.txt

test:
	pytest -q

prepare-official:
	$(PYTHON) scripts/prepare_official_data.py \
		--logic data/official/raw/Logic_Based_Educational_Queries.json \
		--physics data/official/raw/Physics_Problems_Text_Only.csv \
		--out_dir data/raw \
		--train_ratio 0.80 \
		--dev_ratio 0.10 \
		--seed 2026 \
		--smoke_size 100

smoke-solver:
	LLM_BACKEND=none EXACT_FAMA_CONFIG=$(CONFIG_SOLVER) EXACT_USE_PROVIDED_FOL=0 \
	$(PYTHON) scripts/run_inference.py \
		--input $(SMOKE) \
		--output $(ARTIFACTS)/smoke100_solver_only_predictions.jsonl \
		--log_every 10 \
		--write_every 20
	$(PYTHON) scripts/evaluate.py \
		--gold $(SMOKE) \
		--pred $(ARTIFACTS)/smoke100_solver_only_predictions.jsonl \
		--report $(ARTIFACTS)/smoke100_solver_only_eval.json

smoke-qwen:
	LLM_BACKEND=hf MODEL_NAME=models/Qwen3-8B LORA_ADAPTER_PATH= LLM_TEMPERATURE=0.1 LLM_MAX_NEW_TOKENS=512 EXACT_FAMA_CONFIG=$(CONFIG_REWRITE) EXACT_USE_PROVIDED_FOL=0 \
	$(PYTHON) scripts/run_inference.py \
		--input $(SMOKE) \
		--output $(ARTIFACTS)/smoke100_qwen_rewrite_predictions.jsonl \
		--log_every 5 \
		--write_every 20

eval-smoke-qwen:
	$(PYTHON) scripts/evaluate.py \
		--gold $(SMOKE) \
		--pred $(ARTIFACTS)/smoke100_qwen_rewrite_predictions.jsonl \
		--report $(ARTIFACTS)/smoke100_qwen_rewrite_eval.json
	$(PYTHON) scripts/evaluate_explanation_quality.py \
		--pred $(ARTIFACTS)/smoke100_qwen_rewrite_predictions.jsonl \
		--report $(ARTIFACTS)/smoke100_qwen_rewrite_expl_quality.json
	$(PYTHON) scripts/check_rewrite_changed.py $(ARTIFACTS)/smoke100_qwen_rewrite_predictions.jsonl

filter-smoke-correct:
	$(PYTHON) scripts/filter_correct_predictions.py \
		--gold $(SMOKE) \
		--pred $(ARTIFACTS)/smoke100_solver_only_predictions.jsonl \
		--out_gold data/raw/smoke_100_solver_correct.jsonl

all-solver:
	LLM_BACKEND=none EXACT_FAMA_CONFIG=$(CONFIG_SOLVER) EXACT_USE_PROVIDED_FOL=0 \
	$(PYTHON) scripts/run_inference.py \
		--input $(ALL) \
		--output $(ARTIFACTS)/official_all_solver_only_predictions.jsonl \
		--log_every 50 \
		--write_every 100

api:
	LLM_BACKEND=none EXACT_FAMA_CONFIG=$(CONFIG_SOLVER) uvicorn exact_fama.api:app --host 0.0.0.0 --port 8000 --reload

gradio:
	$(PYTHON) scripts/app_gradio.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache **/__pycache__
