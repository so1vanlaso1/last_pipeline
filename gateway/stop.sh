#!/usr/bin/env bash
# Stop the gateway, vLLM server, and tunnel started by setup.sh / run_server.sh.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$HERE/logs"

for pidfile in "$LOGDIR"/cloudflared.pid "$LOGDIR"/gateway.pid "$LOGDIR"/vllm_*.pid; do
    [ -f "$pidfile" ] || continue
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
        echo "[stop] killing $(basename "$pidfile" .pid) (pid $pid)"
        kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
done
# Belt-and-suspenders: catch a vLLM child that outlived its launcher pid.
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "gateway.app:app" 2>/dev/null || true
echo "[stop] done."
