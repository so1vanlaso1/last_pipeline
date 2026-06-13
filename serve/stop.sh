#!/usr/bin/env bash
# Stop the gateway, vLLM server, and tunnel started by setup.sh / run_server.sh.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$HERE/logs"

for pidfile in "$LOGDIR"/ngrok.pid "$LOGDIR"/cloudflared.pid "$LOGDIR"/gateway.pid "$LOGDIR"/vllm_*.pid; do
    [ -f "$pidfile" ] || continue
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
        echo "[stop] killing $(basename "$pidfile" .pid) (pid $pid)"
        kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
done
# Belt-and-suspenders: catch a child that outlived its launcher pid.
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "gateway.app:app" 2>/dev/null || true
# ngrok is the DEFAULT public tunnel (run_server.sh writes ngrok.pid); kill any
# agent that the pidfile loop above missed, so `stop` does not leave the public
# URL fronting a dead gateway (and frees the one-session-per-account slot).
pkill -f 'ngrok http' 2>/dev/null || true
echo "[stop] done."
