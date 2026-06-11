#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Quickstart setup for the cascade NL-QA pipeline.
#
# Target: a Linux box (Ubuntu 22.04+/WSL2) with an NVIDIA RTX 5070 (12 GB,
# Blackwell sm_120) and the NVIDIA driver already installed (check `nvidia-smi`).
# Blackwell needs CUDA 12.8 wheels — that's the default here.
#
# What it does:
#   1. sanity-checks the GPU + Python,
#   2. creates .venv and installs torch (cu128) + all requirements,
#   3. DOWNLOADS the 3 cascade models into the HF cache (so runs are offline),
#   4. verifies torch + each model's architecture actually load.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# The two Gemma repos are GATED — accept the license on huggingface.co and export
# a token first:    export HF_TOKEN=hf_xxx
#
# Overridable env vars (defaults are the repo ids the pipeline ships with):
#   QWEN_ID        4B judge A            (default: Qwen/Qwen3.5-4B)
#   GEMMA_SMALL_ID 4B judge B            (default: google/gemma-4-E2B-it)
#   GEMMA_BIG_ID   Gemma 8B              (default: google/gemma-4-E4B-it)
#   LIQUID_ID      Liquid 8B (MoE)       (default: LiquidAI/LFM2.5-8B-A1B)
#   CUDA_WHL       torch wheel channel   (default: cu128; use cu126/cu121 on older drivers)
#   HF_TOKEN       HF access token (required for the gated Gemma repos)
#
# If a Gemma download 404s, the current equivalents are google/gemma-3n-E2B-it
# and google/gemma-3n-E4B-it — re-run with GEMMA_SMALL_ID/GEMMA_BIG_ID set to those.
# NOTE: LiquidAI/LFM2.5-8B-A1B needs transformers>=5.0 (pinned in requirements.txt);
# it is NOT gated. You only need all four models if you run all three stages —
# download just the ones for the --stages you intend to use.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

# Force UTF-8 for the Python heredocs below (HF cache paths / model names may be
# non-ASCII); harmless on a UTF-8 locale.
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

export QWEN_ID=${QWEN_ID:-Qwen/Qwen3.5-4B}
export GEMMA_SMALL_ID=${GEMMA_SMALL_ID:-google/gemma-4-E2B-it}
export GEMMA_BIG_ID=${GEMMA_BIG_ID:-google/gemma-4-E4B-it}
export LIQUID_ID=${LIQUID_ID:-LiquidAI/LFM2.5-8B-A1B}
export CUDA_WHL=${CUDA_WHL:-cu128}

# 1. GPU sanity check.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found — install the NVIDIA driver before running this." >&2
    echo "(You can still run the no-GPU wiring test: python run_cascade.py --backend stub)" >&2
    exit 1
fi
echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 2. Python: prefer 3.11, else any python3.
PYTHON_BIN=$(command -v python3.11 || command -v python3 || true)
if [ -z "$PYTHON_BIN" ]; then
    echo "No python3 on PATH. Install it, e.g.:" >&2
    echo "  sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip build-essential git" >&2
    exit 1
fi
echo "Using $PYTHON_BIN ($($PYTHON_BIN --version))"
if ! "$PYTHON_BIN" -c "import ensurepip, venv" >/dev/null 2>&1; then
    echo "Python is missing venv/pip. Install: sudo apt-get install -y python3-venv python3-pip" >&2
    exit 1
fi

# 3. venv.
if [ ! -d .venv ]; then
    "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 4. torch + torchvision first (right CUDA wheels for Blackwell), then the rest.
# IMPORTANT: install torchvision from the SAME ${CUDA_WHL} channel as torch.
# requirements.txt pulls timm, which depends on torchvision; if pip is left to
# resolve torchvision from the default PyPI index it grabs a wheel built for a
# DIFFERENT CUDA than torch. That mismatch makes transformers' lazy torchvision
# import fail at model-load time ("PyTorch and torchvision were compiled with
# different CUDA major versions"), so every model silently fails to load and
# accuracy comes out 0%. Pinning both to ${CUDA_WHL} here, before
# requirements.txt, means timm finds a matching torchvision already installed.
pip install --upgrade pip wheel
echo "== Installing torch + torchvision from the ${CUDA_WHL} channel =="
pip install --index-url "https://download.pytorch.org/whl/${CUDA_WHL}" "torch>=2.7" torchvision
echo "== Installing the rest of requirements.txt =="
pip install -r requirements.txt

# Guard: if a transitive dep (or a previous broken run) left a torchvision whose
# CUDA major version doesn't match torch, force it back onto the ${CUDA_WHL}
# channel. Keeps re-runs idempotent and repairs an already-broken env in place.
echo "== Verifying torch/torchvision CUDA match =="
if ! python - <<'PY'
import re, sys
try:
    import torch, torchvision
except Exception as exc:  # a bad torchvision build can throw on import itself
    print(f"[warn] torchvision import failed ({exc}); will reinstall.")
    sys.exit(1)
m = re.search(r"\+cu(\d+)", torchvision.__version__ or "")
tv = m.group(1)[:2] if m else ""                       # torchvision CUDA major
tc = (torch.version.cuda or "").replace(".", "")[:2]   # torch CUDA major
print(f"torch CUDA={torch.version.cuda}  torchvision={torchvision.__version__}")
sys.exit(0 if tv and tc and tv == tc else 1)
PY
then
    echo "   mismatch detected — reinstalling torchvision from ${CUDA_WHL}"
    pip install --index-url "https://download.pytorch.org/whl/${CUDA_WHL}" \
        --force-reinstall --no-deps torchvision
fi

# 5. Download the 3 models into the HF cache. Gemma repos are gated → HF_TOKEN.
echo "== Downloading models (first run can take a while) =="
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[warn] HF_TOKEN is not set. The Gemma repos are gated and will fail to"
    echo "       download without it. Accept the license on huggingface.co, then:"
    echo "         export HF_TOKEN=hf_xxx   &&   ./setup.sh"
fi
python - <<'PY'
import os
from huggingface_hub import snapshot_download
ids = [os.environ["QWEN_ID"], os.environ["GEMMA_SMALL_ID"],
       os.environ["GEMMA_BIG_ID"], os.environ["LIQUID_ID"]]
token = os.environ.get("HF_TOKEN") or None
for mid in ids:
    try:
        print(f"Fetching {mid} …")
        path = snapshot_download(repo_id=mid, token=token)
        print(f"  cached at: {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [error] could not fetch {mid}: {exc}")
        print(f"          if it's a 404, set GEMMA_*_ID to the gemma-3n-* repo and re-run;")
        print(f"          if it's gated/401, accept the license + export HF_TOKEN.")
PY

# 6. Import + architecture check.
python - <<'PY'
import os, torch
print(f"torch    {torch.__version__}  (CUDA {torch.version.cuda})")
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print(f"GPU      {torch.cuda.get_device_name(0)}  (sm_{cc[0]}{cc[1]})")
else:
    print("GPU      (none visible to torch)")
# torchvision must match torch's CUDA major or transformers' lazy import of it
# crashes every model load (the silent-0% failure). Report it explicitly.
import re
try:
    import torchvision
    m = re.search(r"\+cu(\d+)", torchvision.__version__ or "")
    tv = m.group(1)[:2] if m else ""
    tc = (torch.version.cuda or "").replace(".", "")[:2]
    ok = bool(tv) and bool(tc) and tv == tc
    print(f'torchvision {torchvision.__version__}  '
          f'[{"OK" if ok else "MISMATCH"} vs torch CUDA {torch.version.cuda}]')
    if not ok:
        whl = os.environ.get("CUDA_WHL", "cu128")
        print("[warn] torchvision CUDA != torch CUDA — models will fail to load. Fix:")
        print(f"         pip install --index-url https://download.pytorch.org/whl/{whl} "
              f"--force-reinstall --no-deps torchvision")
except Exception as exc:  # noqa: BLE001
    print(f"[warn] torchvision import/check failed: {exc}")
import transformers, accelerate, bitsandbytes
print(f"transformers {transformers.__version__}  accelerate {accelerate.__version__}  "
      f"bitsandbytes {bitsandbytes.__version__}")
from transformers import AutoConfig
for mid in (os.environ["QWEN_ID"], os.environ["GEMMA_SMALL_ID"],
            os.environ["GEMMA_BIG_ID"], os.environ["LIQUID_ID"]):
    try:
        cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)
        print(f'model    {mid}  (model_type={getattr(cfg, "model_type", "?")})  OK')
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {mid}: {exc}")
        print('       if the model_type is unknown, upgrade transformers:')
        print('         pip install -U "git+https://github.com/huggingface/transformers"')
PY

echo
echo "Environment ready. Try:"
echo "  python run_cascade.py --backend stub --stages 4b,gemma8b,liquid8b --show-gold --limit 8  # no-GPU wiring test"
echo "  python run_cascade.py --stages 4b,gemma8b --precision 4bit --show-gold --limit 20         # 4B judges + Gemma 8B"
echo "  python run_cascade.py --stages 4b,gemma8b,liquid8b --precision 4bit --show-gold            # all three stages"
