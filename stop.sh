#!/bin/bash
# Gracefully stop the PID started by start.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/lib/common.sh"

LOG_DIR="${LOG_DIR:-$HOME/.local/state/qwen38-flash-next}"
PID_FILE="$LOG_DIR/vllm.pid"

if [[ ! -f "$PID_FILE" ]]; then
    if command -v systemctl >/dev/null 2>&1; then
        systemctl --user stop vllm-qwen38-tp3 2>/dev/null || true
    fi
    log "no PID file; nothing to stop"
    exit 0
fi

PID="$(cat "$PID_FILE")"
if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    log "process $PID is gone; removed stale PID file"
    exit 0
fi

log "SIGINT $PID (graceful)"
kill -INT "$PID" 2>/dev/null || true
for _ in $(seq 1 60); do
    kill -0 "$PID" 2>/dev/null || { rm -f "$PID_FILE"; log "stopped"; exit 0; }
    sleep 3
done

warn "still alive after 180s; SIGTERM"
kill -TERM "$PID" 2>/dev/null || true
for _ in $(seq 1 20); do
    kill -0 "$PID" 2>/dev/null || { rm -f "$PID_FILE"; log "stopped"; exit 0; }
    sleep 3
done

warn "still alive after SIGTERM; SIGKILL"
kill -KILL "$PID" 2>/dev/null || true
rm -f "$PID_FILE"