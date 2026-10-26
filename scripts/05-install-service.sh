#!/bin/bash
# Render and install the systemd user unit.
# The host used for development has no systemd; start.sh is the fallback.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

if ! command -v systemctl >/dev/null 2>&1; then
    warn "systemctl not found; use ./start.sh and ./stop.sh instead"
    exit 0
fi
if [[ "$(ps -p 1 -o comm= 2>/dev/null || true)" != "systemd" ]]; then
    warn "PID 1 is not systemd; use ./start.sh and ./stop.sh instead"
    exit 0
fi

UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/vllm-qwen38-tp3.service"
mkdir -p "$UNIT_DIR"
sed -e "s|@REPO_DIR@|$REPO_DIR|g" -e "s|@VENV@|${VENV:-$HOME/vllm-venv}|g" \
    "$REPO_DIR/services/vllm-qwen38-tp3.service.in" >"$UNIT"

systemctl --user daemon-reload
systemctl --user enable --now vllm-qwen38-tp3
systemctl --user --no-pager status vllm-qwen38-tp3 || true
log "installed $UNIT"