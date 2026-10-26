#!/bin/bash
# Start vLLM in the background and optionally wait for /health.
# Usage: ./start.sh [--wait]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/lib/common.sh"

LOG_DIR="${LOG_DIR:-$HOME/.local/state/qwen38-flash-next}"
PID_FILE="$LOG_DIR/vllm.pid"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    die "already running (PID $(cat "$PID_FILE")). Use ./stop.sh first."
fi

log "starting vLLM"
nohup "$SCRIPT_DIR/scripts/serve.sh" >>"$LOG_DIR/vllm.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
sleep 1
kill -0 "$PID" 2>/dev/null || die "server exited immediately; see $LOG_DIR/vllm.log"
log "PID $PID, log $LOG_DIR/vllm.log"

if [[ "${1:-}" == "--wait" ]]; then
    HOST="$BIND_HOST"; [[ "$HOST" == "0.0.0.0" ]] && HOST="127.0.0.1"
    log "waiting for http://$HOST:$PORT/health"
    for i in $(seq 1 240); do
        if curl -fsS --max-time 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
            log "ready after $((i * 5))s"
            exit 0
        fi
        sleep 5
    done
    tail -n 80 "$LOG_DIR/vllm.log" >&2 || true
    die "health check timed out"
fi