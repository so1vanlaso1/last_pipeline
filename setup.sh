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
#   6. open a public URL (ngrok tunnel — set NGROK_AUTHTOKEN=<your token>) and
#      write it to serve/submission/urls.txt.
#
# Servers run in the background (survive SSH disconnect). Stop with:
#     bash serve/stop.sh
#
# Overridable env vars (sensible defaults shown):
#   MODEL_ID=google/gemma-4-E4B-it  # fallback single LLM if logic_config.yaml is absent
#   JUDGE_MODEL=google/gemma-4-E4B-it  # the Type-1 JUDGE repo (default Gemma-4-E4B; any HF id)
#   JUDGE_PARAMS_B=8            # size the residency budget counts for the judge (default 8)
#   VLLM_VERSION=               # vLLM version. DEFAULT auto: empty (=latest) on a CUDA>=13
#                               # box (driver >= 580, Blackwell sm_120 — THIS droplet), and
#                               # pinned 0.19.1 (CUDA-12 wheel) only when nvidia-smi reports
#                               # CUDA 12. Override to force a specific version.
#   VLLM_PORT=8001              # vLLM OpenAI server port (internal)
#   GATEWAY_PORT=8000           # gateway /predict port (internal)
#   MAX_MODEL_LEN=8192          # vLLM context length
#   GPU_MEM_UTIL=0.90           # vLLM GPU memory fraction
#   MAX_NUM_SEQS=16             # max concurrent seqs per server (small = less startup VRAM; gateway is sequential)
#   QUANTIZATION=none           # precision for every model: none(bf16) | 8bit | 4bit
#   NGROK_AUTHTOKEN=            # REQUIRED for a public URL — your ngrok agent authtoken
#                               # (https://dashboard.ngrok.com). setup.sh downloads the ngrok
#                               # binary and runs `ngrok config add-authtoken` with it.
#   NGROK_DOMAIN=               # OPTIONAL — pin a reserved static ngrok domain (else the
#                               # account's default assigned domain, stable across restarts).
#   CF_TUNNEL=0                 # legacy Cloudflare quick-tunnel fallback (only if ngrok gave
#                               # no URL); ngrok is the default public tunnel now.
#   PHYSICS_LLM_FALLBACK=1      # 1=LLM fills physics answers only when the solver abstains
#   HF_TOKEN=                   # OPTIONAL — the default line-up (Qwen + Gemma-4) is ungated
#   SKIP_INSTALL=0              # 1=skip pip install (just (re)launch the servers)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVE="$ROOT/serve"
cd "$ROOT"

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

MODEL_ID="${MODEL_ID:-google/gemma-4-E4B-it}"
# Choose the vLLM version from the box's CUDA. vLLM ships ONE CUDA build per release:
# the LATEST wheel targets CUDA 13 / Blackwell sm_120 (driver >= 580 — THIS droplet);
# the 0.19.1 wheel is a CUDA-12 build for older drivers (up to CUDA 12.9). Auto-detect
# the runtime CUDA major from nvidia-smi and default accordingly (operator override wins).
if [ -z "${VLLM_VERSION:-}" ]; then
    CUDA_MAJOR="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' | head -n1)"
    if [ -z "$CUDA_MAJOR" ]; then
        CUDA_MAJOR="$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -n1 | cut -d. -f1 | tr -d '[:space:]')"
    fi
    if ! [[ "$CUDA_MAJOR" =~ ^[0-9]+$ ]]; then
        CUDA_MAJOR=""
    fi
    if [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -lt 13 ] 2>/dev/null; then
        VLLM_VERSION="0.19.1"   # CUDA-12 box → pinned cu12 wheel
        echo "[setup] nvidia-smi CUDA ${CUDA_MAJOR} (<13) → pinning vLLM ${VLLM_VERSION} (CUDA-12 wheel)"
    else
        VLLM_VERSION=""         # CUDA>=13 (or unknown) → latest wheel (Blackwell/CUDA-13)
        echo "[setup] nvidia-smi CUDA ${CUDA_MAJOR:-unknown} → installing latest vLLM (CUDA-13/Blackwell wheel)"
    fi
fi
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

    # vLLM ships ONE CUDA build per release and pulls its OWN matching torch, so we
    # install them together as a consistent stack. Do NOT swap only torch's CUDA:
    # vLLM's compiled extension (vllm._C) would then mismatch
    # (ImportError: libcudart.so.<N>). VLLM_VERSION is auto-selected above from the
    # box's CUDA: empty (latest, CUDA-13/Blackwell wheel — THIS droplet's RTX 5090)
    # on CUDA>=13, or pinned 0.19.1 (CUDA-12 wheel) on CUDA 12. The line-up
    # (Qwen3-4B + Qwen3-4B-Instruct-2507 generators + gemma-4-E4B-it judge) loads on both.
    echo "== Installing vLLM (+ its matching torch; can take a while) =="
    if [ -n "${VLLM_VERSION:-}" ]; then
        pip install "vllm==${VLLM_VERSION}"
    else
        pip install vllm
    fi

    echo "== Installing gateway + physics-pipeline requirements =="
    pip install -r "$SERVE/requirements.txt"

    # Fail fast with a clear message if torch can't reach the GPU — on vast.ai this
    # is almost always "host driver too old for this vLLM/torch CUDA build".
    python - <<'PY' || { echo "[setup] ERROR: torch cannot use the GPU. Your NVIDIA driver is likely too old for this vLLM build's CUDA. Fix: use a box whose 'nvidia-smi' CUDA Version >= the wheel's CUDA (a recent vLLM needs CUDA 13 → driver >= 580), or pin an older VLLM_VERSION built for your driver's CUDA." >&2; exit 3; }
import torch, sys
print(f"  torch {torch.__version__}, cuda_available={torch.cuda.is_available()}")
sys.exit(0 if torch.cuda.is_available() else 1)
PY
else
    echo "== SKIP_INSTALL=1 → skipping pip install =="
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

# ── 6b. ngrok public tunnel: fetch the binary + register the authtoken ────────
# The live droplet's public URL is an ngrok tunnel. run_server.sh only starts ngrok
# if the serve/ngrok binary exists AND an authtoken is configured, so a fresh box
# needs both set up here. NGROK_AUTHTOKEN is the ONE secret the operator must supply
# (get it at https://dashboard.ngrok.com); without it the box still runs but exposes
# only the raw public-IP:port (no stable public URL). Set NGROK=0 to skip entirely.
if [ "${NGROK:-1}" != "0" ]; then
    NGROK_BIN="$SERVE/ngrok"
    if [ ! -x "$NGROK_BIN" ]; then
        case "$(uname -m)" in
            x86_64|amd64) NG_ARCH=amd64 ;;
            aarch64|arm64) NG_ARCH=arm64 ;;
            *) NG_ARCH=amd64 ;;
        esac
        echo "== Downloading ngrok ($NG_ARCH) =="
        curl -fsSL "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-${NG_ARCH}.tgz" -o /tmp/ngrok.tgz \
            && tar -xzf /tmp/ngrok.tgz -C "$SERVE" ngrok \
            && chmod +x "$NGROK_BIN" \
            && echo "[setup] ngrok installed → $NGROK_BIN" \
            || echo "[setup][warn] ngrok download failed — the box will come up without a public URL." >&2
    fi
    if [ -x "$NGROK_BIN" ] && [ -n "${NGROK_AUTHTOKEN:-}" ]; then
        "$NGROK_BIN" config add-authtoken "$NGROK_AUTHTOKEN" \
            && echo "[setup] ngrok authtoken registered." \
            || echo "[setup][warn] could not register the ngrok authtoken." >&2
    elif [ -x "$NGROK_BIN" ] && [ ! -f "${HOME}/.config/ngrok/ngrok.yml" ]; then
        echo "[setup][warn] NGROK_AUTHTOKEN not set and no ~/.config/ngrok/ngrok.yml — the public URL will be SKIPPED. Export NGROK_AUTHTOKEN=<token> and re-run, or run: $NGROK_BIN config add-authtoken <token>" >&2
    fi
fi

# ── 7. Launch the servers + tunnel ───────────────────────────────────────────
export MODEL_ID
chmod +x "$SERVE/run_server.sh" "$SERVE/stop.sh" 2>/dev/null || true
bash "$SERVE/run_server.sh"

echo
echo "Setup complete. The endpoint is live and will stay up after you disconnect."
echo "Re-launch later without reinstalling:  SKIP_INSTALL=1 bash setup.sh"
