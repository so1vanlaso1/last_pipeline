#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# EXACT 2026 — one-shot setup for a vast.ai (Linux + NVIDIA GPU) machine.
#
# This is the ONLY command you need to run on the remote box:
#
#     bash setup.sh
#
# It will:
#   1. install system + Python deps into a local .venv,
#   2. install vLLM (brings the matching CUDA torch) + the gateway/physics deps,
#   3. download the models in serve/logic_config.yaml (generators + judge),
#   4. launch one vLLM server per model (each exposes /v1/models for verification),
#   5. launch the gateway      (the single competition /predict endpoint),
#   6. open a public URL (Cloudflare quick tunnel) and write it to
#      serve/submission/urls.txt.
#
# Servers run in the background (survive SSH disconnect). Stop with:
#     bash serve/stop.sh
#
# Overridable env vars (sensible defaults shown):
#   MODEL_ID=LiquidAI/LFM2.5-8B-A1B  # fallback single LLM if logic_config.yaml is absent
#   JUDGE_MODEL=liquid          # swap the Type-1 JUDGE: 'liquid'(default)|'gemma'|<full HF repo id>
#   JUDGE_PARAMS_B=8.3          # size the residency budget counts for the judge (default 8.3)
#   PREFETCH_JUDGES=1           # 1=pre-download BOTH judge candidates (liquid+gemma) for fast A/B
#   VLLM_PORT=8001              # vLLM OpenAI server port (internal)
#   GATEWAY_PORT=8000           # gateway /predict port (internal)
#   MAX_MODEL_LEN=8192          # vLLM context length
#   GPU_MEM_UTIL=0.90           # vLLM GPU memory fraction
#   CF_TUNNEL=1                 # 1=auto Cloudflare tunnel for a public URL; 0=off
#   PHYSICS_LLM_FALLBACK=1      # 1=LLM fills physics answers only when the solver abstains
#   HF_TOKEN=                   # only needed if you switch MODEL_ID to a gated repo
#   SKIP_INSTALL=0              # 1=skip pip install (just (re)launch the servers)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVE="$ROOT/serve"
cd "$ROOT"

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

MODEL_ID="${MODEL_ID:-LiquidAI/LFM2.5-8B-A1B}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

echo "=================================================================="
echo " EXACT 2026 setup  (logic line-up: serve/logic_config.yaml; fallback ${MODEL_ID})"
echo "=================================================================="

# ── 1. GPU sanity check ──────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "== GPU =="
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
else
    echo "[warn] nvidia-smi not found. vLLM needs an NVIDIA GPU; continuing anyway"
    echo "       (install the driver, or run a no-GPU wiring test with GATEWAY_LLM=stub)."
fi

# ── 2. System packages (best-effort; skip silently without sudo/apt) ─────────
if [ "$SKIP_INSTALL" != "1" ] && command -v apt-get >/dev/null 2>&1; then
    SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"
    echo "== Installing system packages (python venv, build tools, curl, git) =="
    $SUDO apt-get update -y || true
    $SUDO apt-get install -y python3 python3-venv python3-pip python3-dev build-essential git curl || true
fi

# ── 3. Python venv ───────────────────────────────────────────────────────────
PYTHON_BIN="$(command -v python3.11 || command -v python3 || command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "[error] No python found on PATH. Install python3 and re-run." >&2
    exit 1
fi
echo "Using $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
if [ ! -d "$ROOT/.venv" ]; then
    "$PYTHON_BIN" -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

# ── 4. Python deps ───────────────────────────────────────────────────────────
if [ "$SKIP_INSTALL" != "1" ]; then
    python -m pip install --upgrade pip wheel setuptools

    echo "== Installing vLLM (this pulls a CUDA-matched torch; can take a while) =="
    if [ -n "${VLLM_VERSION:-}" ]; then
        pip install "vllm==${VLLM_VERSION}"
    else
        pip install vllm
    fi

    echo "== Installing gateway + physics-pipeline requirements =="
    pip install -r "$SERVE/requirements.txt"
else
    echo "== SKIP_INSTALL=1 → skipping pip install =="
fi

# ── 4b. Pre-cache the swappable JUDGE candidates ─────────────────────────────
#   The ACTIVE judge (env JUDGE_MODEL, default Liquid) is fetched at launch by
#   run_server.sh anyway; this pre-pulls the OTHER candidate too, so you can flip
#   JUDGE_MODEL and relaunch (SKIP_INSTALL=1 bash setup.sh) with no 30-min wait.
#   Gated repos (Gemma) are skipped with a hint unless HF_TOKEN is set.
#   Disable with PREFETCH_JUDGES=0; customise the list with JUDGE_CANDIDATES.
PREFETCH_JUDGES="${PREFETCH_JUDGES:-1}"
JUDGE_CANDIDATES="${JUDGE_CANDIDATES:-LiquidAI/LFM2.5-8B-A1B google/gemma-4-E4B-it}"
if [ "$PREFETCH_JUDGES" = "1" ]; then
    echo "== Pre-caching judge candidates: ${JUDGE_CANDIDATES} =="
    for MID in $JUDGE_CANDIDATES; do
        case "$MID" in
            google/*|*gemma*)
                if [ -z "${HF_TOKEN:-}" ]; then
                    echo "[setup] skip ${MID} — gated; accept its license on huggingface.co and 'export HF_TOKEN=hf_...' to pre-cache it."
                    continue
                fi ;;
        esac
        echo "[setup] downloading ${MID} (if not already cached)…"
        HF_TOKEN="${HF_TOKEN:-}" python - "$MID" <<'PY' || echo "[setup] ${MID}: prefetch skipped (vLLM will fetch it at launch)."
import os, sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], token=os.environ.get("HF_TOKEN") or None)
print("  cached:", sys.argv[1])
PY
    done
fi

# ── 5. The resident model line-up is read from serve/logic_config.yaml and each
#       model is downloaded by run_server.sh just before its vLLM server starts. ─

# ── 6. Quick import sanity check (catches a broken physics install early) ─────
echo "== Sanity check: gateway + physics imports =="
PYTHONPATH="$SERVE:$ROOT/physic_pipeline/src:$ROOT/logic_pipeline/src" python - <<'PY' || echo "[warn] import sanity check reported an issue (see above)."
import importlib
for m in ("prompts", "schema", "cascade", "exact_fama.pipeline", "gateway.app"):
    importlib.import_module(m)
    print(f"  ok: {m}")
PY

# Validate the resident line-up against the configured residency budget
# (max_resident_b in serve/logic_config.yaml; run_server.sh enforces this too).
echo "== Resident model line-up (must fit max_resident_b; committee limit is 8B) =="
PYTHONPATH="$SERVE:$ROOT/physic_pipeline/src:$ROOT/logic_pipeline/src" python -m gateway.config >/dev/null || {
    echo "[error] serve/logic_config.yaml exceeds its residency budget. Fix it before launching." >&2
    exit 2
}

# ── 7. Launch the servers + tunnel ───────────────────────────────────────────
export MODEL_ID
chmod +x "$SERVE/run_server.sh" "$SERVE/stop.sh" 2>/dev/null || true
bash "$SERVE/run_server.sh"

echo
echo "Setup complete. The endpoint is live and will stay up after you disconnect."
echo "Re-launch later without reinstalling:  SKIP_INSTALL=1 bash setup.sh"
