#!/usr/bin/env bash
# Relaunch the vLLM server(s) + gateway without reinstalling. setup.sh calls this;
# you can also run it directly after a reboot. The resident model line-up is read
# from serve/logic_config.yaml (one vLLM server per model; the total must fit the
# configured residency budget `max_resident_b` — the committee's value is 8B).
#
# Env overrides: GATEWAY_PORT, VLLM_BASE_PORT, MAX_MODEL_LEN, GPU_MEM_UTIL,
#                CF_TUNNEL, PHYSICS_LLM_FALLBACK, LOGIC_CONFIG, HF_TOKEN.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # <repo>/serve
ROOT="$(cd "$HERE/.." && pwd)"                            # <repo>
LOGDIR="$HERE/logs"
mkdir -p "$LOGDIR"

GATEWAY_PORT="${GATEWAY_PORT:-8000}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# The gateway drives each server SEQUENTIALLY, so a small max concurrent-sequence
# count is plenty — and it slashes vLLM's sampler-warmup memory (256 dummy seqs ×
# a big vocab can OOM an 8B bf16 model on a 24 GB card right at startup). Raise it
# only if you actually batch many requests at once.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
CF_TUNNEL="${CF_TUNNEL:-0}"   # legacy Cloudflare quick tunnel OFF by default; ngrok is the public tunnel

export VLLM_BASE_PORT GPU_MEM_UTIL
export GATEWAY_LLM="${GATEWAY_LLM:-vllm}"
export PHYSICS_LLM_FALLBACK="${PHYSICS_LLM_FALLBACK:-1}"
# Sleep/wake swap level. DEFAULT 1 = RAM offload (the slept group's already-quantized
# weights are parked in CPU RAM and copied back verbatim on wake). This is REQUIRED for
# the FP8 (8bit) line-up: level 2 (discard + reload-from-disk) RE-QUANTIZES on wake and
# produces CORRUPT generations (streams of "!!!!") for these Qwen/Gemma FP8 models.
# Level 1 also wakes faster (RAM copy, ~1s) and still moves weights OFF the GPU, so the
# "≤8B loaded on GPU at any moment" rule and the OOM budget both still hold. Set 2 only
# if you switch the whole line-up to 4bit/bf16 (which survive a disk reload).
export RESIDENCY_SLEEP_LEVEL="${RESIDENCY_SLEEP_LEVEL:-1}"
export PYTHONPATH="$HERE:$ROOT/physic_pipeline/src:$ROOT/logic_pipeline/src${PYTHONPATH:+:$PYTHONPATH}"

if [ -d "$ROOT/.venv" ]; then
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
fi

# ── Resolve the resident model line-up (enforces the budget; exits 2 if over) ─
# Plan stdout: a "#swap\t{0|1}" meta line, then one
#   id \t port \t gpu_frac \t role \t quant_flags
# row per model (quant_flags '-' = full precision). See gateway/config.py.
if ! PLAN="$(python -m gateway.config 2>"$LOGDIR/config.err")"; then
    echo "[run] line-up is not compliant:" >&2
    cat "$LOGDIR/config.err" >&2
    exit 2
fi
cat "$LOGDIR/config.err" >&2     # show the "[config] line-up: …" summary

# ── Parse the plan into parallel arrays + the swap flag ───────────────────────
SWAP=0
MIDS=(); PORTS=(); FRACS=(); ROLES=(); QUANTS=()
while IFS=$'\t' read -r F1 F2 F3 F4 F5; do
    [ -z "${F1:-}" ] && continue
    if [ "$F1" = "#swap" ]; then SWAP="${F2:-0}"; continue; fi
    case "$F1" in \#*) continue ;; esac
    MIDS+=("$F1"); PORTS+=("$F2"); FRACS+=("$F3"); ROLES+=("${F4:-}"); QUANTS+=("${F5:-}")
done <<< "$PLAN"

# Sleep/wake swap needs vLLM's admin endpoints (/sleep, /wake_up) — dev-mode only.
if [ "$SWAP" = "1" ]; then
    export VLLM_SERVER_DEV_MODE=1
    echo "[run] SWAP on: generators stay co-resident; the judge is loaded alone first, then slept."
fi

# ── helpers ──────────────────────────────────────────────────────────────────
server_up() { curl -fsS "http://localhost:$1/v1/models" >/dev/null 2>&1; }

download_model() {
    local MID="$1"
    # The default line-up (Qwen3-4B + Qwen3-4B-Instruct-2507 + gemma-4-E4B-it) is
    # ungated — no HF_TOKEN needed. For a gated repo, `export HF_TOKEN=hf_...` and it
    # is passed through.
    echo "[run] downloading ${MID} (if not cached)…"
    HF_TOKEN="${HF_TOKEN:-}" python - "$MID" <<'PY' || echo "[run] (download will fall back to vLLM's own fetch)"
import os, sys
from huggingface_hub import snapshot_download
try:
    p = snapshot_download(repo_id=sys.argv[1], token=os.environ.get("HF_TOKEN") or None)
    print(f"  cached: {p}")
except Exception as e:
    raise SystemExit(f"  warn: {e}")
PY
}

start_one() {                                    # MID PORT FRAC QUANT
    local MID="$1" PORT="$2" FRAC="$3" QUANT="$4"
    if server_up "$PORT"; then
        echo "[run] vLLM for ${MID} already up on :${PORT}"; return 0
    fi
    download_model "$MID"
    local QARGS=(); [ -n "$QUANT" ] && [ "$QUANT" != "-" ] && read -ra QARGS <<< "$QUANT"
    local SLEEPF=(); [ "$SWAP" = "1" ] && SLEEPF=(--enable-sleep-mode)
    # ENFORCE_EAGER=1 disables CUDA-graph capture (saves VRAM; slightly slower
    # inference). DEFAULT ON here: the 2×4B + 8B sleep/wake swap keeps THREE vLLM
    # processes coexisting, and a slept process's GPU residual shrinks (~4.7→~4.1 GiB)
    # without a cudagraph pool — that headroom is what keeps the awake group from
    # OOMing the 32 GB card. Set ENFORCE_EAGER=0 to restore graphs if VRAM allows.
    local EAGERF=(); [ "${ENFORCE_EAGER:-1}" = "1" ] && EAGERF=(--enforce-eager)
    # VIT_ATTN_BACKEND pins the multimodal vision-tower attention backend. On Blackwell
    # GPUs (RTX 50xx) with an older driver, the prebuilt flash-attn ViT kernel can fail
    # with cudaErrorUnsupportedPtxVersion; set VIT_ATTN_BACKEND=TORCH_SDPA to avoid it.
    local VITF=(); [ -n "${VIT_ATTN_BACKEND:-}" ] && VITF=(--mm-encoder-attn-backend "$VIT_ATTN_BACKEND")
    # The logic + physics flow is TEXT-ONLY. The judge (gemma-4-E4B-it) ships as a
    # vision-language checkpoint; left alone, vLLM loads its
    # vision encoder AND memory-profiles it with a max-size dummy image — a ~7 GB
    # peak that pushed the co-resident generator on :8001 to "Available KV cache
    # memory: -7.3 GiB" and a hard ValueError on its tight gpu_frac slice. The
    # --language-model-only flag drops the encoder entirely (no vision profiling).
    # Set LANGUAGE_MODEL_ONLY=0 to restore vision for an image line-up.
    local LMOF=(); [ "${LANGUAGE_MODEL_ONLY:-1}" = "1" ] && LMOF=(--language-model-only)
    echo "[run] starting vLLM ${MID} on :${PORT} (gpu_frac=${FRAC}, quant='${QUANT}', log: $LOGDIR/vllm_${PORT}.log)"
    if command -v vllm >/dev/null 2>&1; then
        nohup vllm serve "$MID" \
            --host 0.0.0.0 --port "$PORT" --served-model-name "$MID" \
            --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$FRAC" --dtype auto \
            --max-num-seqs "$MAX_NUM_SEQS" \
            "${SLEEPF[@]}" "${EAGERF[@]}" "${VITF[@]}" "${LMOF[@]}" "${QARGS[@]}" \
            > "$LOGDIR/vllm_${PORT}.log" 2>&1 &
    else
        nohup python -m vllm.entrypoints.openai.api_server \
            --model "$MID" --host 0.0.0.0 --port "$PORT" --served-model-name "$MID" \
            --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$FRAC" --dtype auto \
            --max-num-seqs "$MAX_NUM_SEQS" \
            "${SLEEPF[@]}" "${EAGERF[@]}" "${VITF[@]}" "${LMOF[@]}" "${QARGS[@]}" \
            > "$LOGDIR/vllm_${PORT}.log" 2>&1 &
    fi
    echo $! > "$LOGDIR/vllm_${PORT}.pid"
}

wait_ready() {                                   # PORT MAX_SECS  -> 0 if up
    local PORT="$1" SECS="$2"
    local pidfile="$LOGDIR/vllm_${PORT}.pid" pid
    for _ in $(seq 1 "$SECS"); do
        if server_up "$PORT"; then return 0; fi
        # Quick-fail: if the vLLM process has already exited, stop waiting now
        # instead of polling a dead port for the full timeout.
        pid="$(cat "$pidfile" 2>/dev/null || true)"
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            echo "[run] vLLM for :${PORT} (pid $pid) exited early — see $LOGDIR/vllm_${PORT}.log" >&2
            return 1
        fi
        sleep 1
    done
    return 1
}

sleep_server() { curl -fsS -X POST "http://localhost:$1/sleep?level=${RESIDENCY_SLEEP_LEVEL:-1}" >/dev/null 2>&1 || true; }

# ── 1. Launch the vLLM servers ───────────────────────────────────────────────
GEN_PORTS=()
if [ "$SWAP" = "1" ]; then
    # (a) The JUDGE boots ALONE on the card (its gpu_frac assumes the whole GPU),
    #     then is slept so its VRAM is freed for the generators.
    for idx in "${!MIDS[@]}"; do
        [ "${ROLES[$idx]}" = "judge" ] || continue
        start_one "${MIDS[$idx]}" "${PORTS[$idx]}" "${FRACS[$idx]}" "${QUANTS[$idx]}"
    done
    for idx in "${!MIDS[@]}"; do
        [ "${ROLES[$idx]}" = "judge" ] || continue
        echo "[run] waiting for JUDGE :${PORTS[$idx]} (first run downloads weights; up to 30 min)…"
        if wait_ready "${PORTS[$idx]}" 1800; then
            echo "[run] judge :${PORTS[$idx]} ready — sleeping it so generators get the VRAM."
            sleep_server "${PORTS[$idx]}"
        else
            echo "[run] ERROR: judge :${PORTS[$idx]} did not come up. See $LOGDIR/vllm_${PORTS[$idx]}.log" >&2
            exit 1
        fi
    done
    # (b) The GENERATORS boot into the freed memory and stay AWAKE (resting state).
    #     Bring them up ONE AT A TIME — wait for each to finish vLLM's memory
    #     profiling before launching the next. Two generators profiling the shared
    #     card at once makes free-VRAM move under each other's feet; when one trips
    #     it releases memory mid-profile and the sibling dies with "Error in memory
    #     profiling … current free memory <larger>", cascading the whole launch.
    for idx in "${!MIDS[@]}"; do
        [ "${ROLES[$idx]}" = "judge" ] && continue
        start_one "${MIDS[$idx]}" "${PORTS[$idx]}" "${FRACS[$idx]}" "${QUANTS[$idx]}"
        GEN_PORTS+=("${PORTS[$idx]}")
        if wait_ready "${PORTS[$idx]}" 1800; then
            echo "[run] generator :${PORTS[$idx]} ready."
        else
            echo "[run] WARNING: generator :${PORTS[$idx]} did not come up — see $LOGDIR/vllm_${PORTS[$idx]}.log" >&2
        fi
    done
else
    # No swap: every model resident at once — launch them all together.
    for idx in "${!MIDS[@]}"; do
        start_one "${MIDS[$idx]}" "${PORTS[$idx]}" "${FRACS[$idx]}" "${QUANTS[$idx]}"
    done
    GEN_PORTS=("${PORTS[@]}")
fi

# The PRIMARY (first generator) is required — physics + premises fallbacks use it.
# The rest download in parallel; SKIP any that don't come up (the flow degrades).
PRIMARY="${GEN_PORTS[0]}"
echo "[run] waiting for PRIMARY vLLM :${PRIMARY}/v1/models (first run downloads weights; up to 30 min)…"
if ! wait_ready "$PRIMARY" 1800; then
    echo "[run] ERROR: primary vLLM on :${PRIMARY} did not come up. See $LOGDIR/vllm_${PRIMARY}.log" >&2
    exit 1
fi
echo "[run] primary vLLM :${PRIMARY} ready."
for PORT in "${GEN_PORTS[@]:1}"; do
    if wait_ready "$PORT" 600; then
        echo "[run] vLLM :${PORT} ready."
    else
        echo "[run] WARNING: vLLM on :${PORT} did not come up — that model is SKIPPED; the flow degrades. See $LOGDIR/vllm_${PORT}.log" >&2
    fi
done

# ── 2. Gateway (the /predict endpoint) ───────────────────────────────────────
if curl -fsS "http://localhost:${GATEWAY_PORT}/health" >/dev/null 2>&1; then
    echo "[run] gateway already up on :${GATEWAY_PORT}"
else
    echo "[run] starting gateway on :${GATEWAY_PORT} (log: $LOGDIR/gateway.log)"
    nohup uvicorn gateway.app:app --host 0.0.0.0 --port "$GATEWAY_PORT" --workers 1 \
        > "$LOGDIR/gateway.log" 2>&1 &
    echo $! > "$LOGDIR/gateway.pid"
fi
for _ in $(seq 1 120); do
    curl -fsS "http://localhost:${GATEWAY_PORT}/health" >/dev/null 2>&1 && break
    sleep 1
done
echo "[run] gateway ready."

# ── 3. Public URL (ngrok primary; Cloudflare quick tunnel legacy fallback) ────
PUBLIC_URL=""

# ngrok (default ON) — the public static tunnel. Reads the agent authtoken stored by
# `ngrok config add-authtoken <TOKEN>`. Set NGROK_DOMAIN=<reserved-domain> to pin a
# specific static domain (custom names need a PAID plan; on Free the account's single
# assigned domain is used automatically and is stable across restarts). NGROK=0 disables.
NGROK="${NGROK:-1}"
NGROK_BIN="$HERE/ngrok"
if [ "$NGROK" = "1" ] && [ -x "$NGROK_BIN" ]; then
    if [ ! -f "${HOME}/.config/ngrok/ngrok.yml" ] && [ -z "${NGROK_AUTHTOKEN:-}" ]; then
        echo "[run] ngrok: no authtoken configured — skipping (run: $NGROK_BIN config add-authtoken <TOKEN>)" >&2
    else
        pkill -f 'ngrok http' 2>/dev/null || true        # one agent session per account
        sleep 1
        NGROK_URLF=(); [ -n "${NGROK_DOMAIN:-}" ] && NGROK_URLF=(--url="https://${NGROK_DOMAIN}")
        echo "[run] starting ngrok → http://localhost:${GATEWAY_PORT}${NGROK_DOMAIN:+ (domain ${NGROK_DOMAIN})}"
        nohup "$NGROK_BIN" http "$GATEWAY_PORT" "${NGROK_URLF[@]}" --log=stdout --log-format=logfmt \
            > "$LOGDIR/ngrok.log" 2>&1 &
        echo $! > "$LOGDIR/ngrok.pid"
        for _ in $(seq 1 40); do
            PUBLIC_URL="$(grep -oE 'url=https://[A-Za-z0-9.:/-]+' "$LOGDIR/ngrok.log" 2>/dev/null | head -n1 | sed 's/^url=//')"
            [ -n "$PUBLIC_URL" ] && break
            grep -q 'ERR_NGROK' "$LOGDIR/ngrok.log" 2>/dev/null && { echo "[run] ngrok failed — see $LOGDIR/ngrok.log" >&2; break; }
            sleep 1
        done
        [ -n "$PUBLIC_URL" ] && echo "[run] ngrok URL: $PUBLIC_URL"
    fi
fi

# Cloudflare quick tunnel — legacy fallback, used only if ngrok produced no URL and
# CF_TUNNEL=1. Gives an EPHEMERAL *.trycloudflare.com URL that changes every restart.
if [ -z "$PUBLIC_URL" ] && [ "$CF_TUNNEL" = "1" ]; then
    CF_BIN="$HERE/cloudflared"
    if [ ! -x "$CF_BIN" ]; then
        echo "[run] downloading cloudflared…"
        case "$(uname -m)" in
            x86_64|amd64) CF_ARCH=amd64 ;;
            aarch64|arm64) CF_ARCH=arm64 ;;
            *) CF_ARCH=amd64 ;;
        esac
        curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CF_ARCH}" -o "$CF_BIN" || true
        chmod +x "$CF_BIN" 2>/dev/null || true
    fi
    if [ -x "$CF_BIN" ]; then
        echo "[run] starting Cloudflare tunnel → http://localhost:${GATEWAY_PORT}"
        nohup "$CF_BIN" tunnel --no-autoupdate --url "http://localhost:${GATEWAY_PORT}" \
            > "$LOGDIR/cloudflared.log" 2>&1 &
        echo $! > "$LOGDIR/cloudflared.pid"
        for _ in $(seq 1 40); do
            PUBLIC_URL="$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$LOGDIR/cloudflared.log" 2>/dev/null | head -n1 || true)"
            [ -n "$PUBLIC_URL" ] && break
            sleep 1
        done
    fi
fi

PUBLIC_IP="${PUBLIC_IPADDR:-$(curl -fsS https://api.ipify.org 2>/dev/null || true)}"
URLS_FILE="$HERE/submission/urls.txt"
mkdir -p "$HERE/submission"
if [ -n "$PUBLIC_URL" ]; then
    BASE="$PUBLIC_URL"
else
    BASE="http://${PUBLIC_IP:-<PUBLIC_IP>}:${GATEWAY_PORT}"
fi
{
    echo "# EXACT 2026 — submission URLs (generated $(date -u '+%Y-%m-%dT%H:%M:%SZ'))"
    echo "PREDICT_URL=${BASE}/predict"
    echo "#"
    echo "# One /v1/models per vLLM server (Submission Guide §6.3 — 'a /v1/models URL"
    echo "# for each vLLM server'). Proxied through this single host, each reachable"
    echo "# even while that model is swapped to sleep (the list is metadata)."
    for idx in "${!MIDS[@]}"; do
        echo "MODELS_URL_${PORTS[$idx]}=${BASE}/vllm/${PORTS[$idx]}/v1/models   # ${MIDS[$idx]} (${ROLES[$idx]:-voter})"
    done
    echo "#"
    echo "# Aggregated list of every resident model (convenience):"
    echo "MODELS_URL=${BASE}/v1/models"
    echo "# Live GPU-residency proof (<=8B loaded-and-running at any instant):"
    echo "HEALTH_URL=${BASE}/health"
    if [ -z "$PUBLIC_URL" ]; then
        echo "#"
        echo "# (No tunnel — using vast.ai port mapping? replace host:port with the"
        echo "#  external mapping vast.ai shows for internal port ${GATEWAY_PORT}.)"
    fi
    # OPTIONAL: DIRECT vLLM host /v1/models (the literal §6.2 '<your-vllm-host>/v1/models',
    # hitting each raw vLLM server instead of the gateway proxy). Only emitted if you
    # actually expose the vLLM ports — set VLLM_PUBLIC_BASE=<public-host-or-ip>, and for
    # any port vast.ai REMAPS to a different external port, override the whole URL with
    # VLLM_PUBLIC_<port>=http://host:extport.  ⚠ SECURITY: with SWAP on, vLLM runs in dev
    # mode and the same port also serves /sleep & /wake_up — only expose these to a
    # trusted committee; otherwise prefer the read-only /vllm/<port>/v1/models proxy above.
    if [ -n "${VLLM_PUBLIC_BASE:-}" ] || compgen -A variable | grep -q '^VLLM_PUBLIC_[0-9]'; then
        echo "#"
        echo "# DIRECT raw-vLLM-host /v1/models (§6.2):"
        for idx in "${!MIDS[@]}"; do
            p="${PORTS[$idx]}"; ov="VLLM_PUBLIC_${p}"
            if [ -n "${!ov:-}" ]; then
                echo "MODELS_URL_${p}_DIRECT=${!ov%/}/v1/models   # ${MIDS[$idx]}"
            elif [ -n "${VLLM_PUBLIC_BASE:-}" ]; then
                echo "MODELS_URL_${p}_DIRECT=http://${VLLM_PUBLIC_BASE}:${p}/v1/models   # ${MIDS[$idx]} (override w/ VLLM_PUBLIC_${p} if vast.ai remaps this port)"
            fi
        done
    fi
} > "$URLS_FILE"

echo
echo "=================================================================="
echo " EXACT 2026 gateway is up."
echo "   resident models : $(echo "$PLAN" | awk -F'\t' '$1 !~ /^#/ {printf "%s(:%s) ", $1, $2}')"
echo "   gateway (local) : http://localhost:${GATEWAY_PORT}/predict"
echo "   models  (local) : http://localhost:${GATEWAY_PORT}/v1/models"
if [ -n "$PUBLIC_URL" ]; then
    echo "   PUBLIC predict  : ${PUBLIC_URL}/predict"
    echo "   PUBLIC models   : ${PUBLIC_URL}/v1/models"
else
    echo "   public IP       : ${PUBLIC_IP:-<unknown>} (map port ${GATEWAY_PORT} on vast.ai)"
fi
echo "   urls written    : ${URLS_FILE}"
echo "   logs            : ${LOGDIR}/{vllm_*,gateway,cloudflared}.log"
echo "   stop            : bash ${HERE}/stop.sh"
echo "=================================================================="
