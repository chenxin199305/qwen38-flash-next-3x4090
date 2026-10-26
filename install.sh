#!/bin/bash
# One-shot installer: preflight -> venv -> checkpoint -> FP8 hybrid -> patches
# -> FP8 MTP overlay -> service. Every step is idempotent.
#
# Usage:
#   ./install.sh [--skip-download] [--skip-venv] [--skip-fp8hybrid]
#                [--skip-patches] [--skip-mtp-fp8] [--skip-service] [--skip-preflight]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/lib/common.sh"

SKIP_DOWNLOAD=0 SKIP_VENV=0 SKIP_FP8HYBRID=0 SKIP_PATCHES=0 SKIP_MTP=0 SKIP_SERVICE=0 SKIP_PREFLIGHT=0
for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=1 ;;
        --skip-venv) SKIP_VENV=1 ;;
        --skip-fp8hybrid) SKIP_FP8HYBRID=1 ;;
        --skip-patches) SKIP_PATCHES=1 ;;
        --skip-mtp-fp8) SKIP_MTP=1 ;;
        --skip-service) SKIP_SERVICE=1 ;;
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        *) die "unknown flag: $arg" ;;
    esac
done

[[ "$SKIP_PREFLIGHT" == "1" ]] || "$SCRIPT_DIR/scripts/00-preflight.sh"
[[ "$SKIP_VENV" == "1" ]] || "$SCRIPT_DIR/scripts/01-setup-venv.sh"
if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    log "skip checkpoint download"
else
    "$SCRIPT_DIR/scripts/02-download-model.sh"
fi

if [[ "$SKIP_FP8HYBRID" == "1" ]]; then
    log "skip FP8-hybrid build"
elif [[ -f "$FP8HYBRID_DIR/fp8hybrid-manifest.json" ]]; then
    log "FP8-hybrid overlay already exists; verifying"
    "$(vllm_python)" "$SCRIPT_DIR/scripts/02a-build-fp8hybrid.py" --check
elif [[ -f "$FP8HYBRID_DIR/model.safetensors.index.json" ]]; then
    warn "FP8-hybrid overlay exists without a manifest; skipping rebuild"
    warn "delete $FP8HYBRID_DIR and re-run if you want a clean 156-tensor overlay"
else
    "$(vllm_python)" "$SCRIPT_DIR/scripts/02a-build-fp8hybrid.py"
fi

if [[ "$SKIP_PATCHES" == "1" ]]; then
    log "skip vLLM patches"
else
    "$SCRIPT_DIR/scripts/03-apply-patches.sh"
fi

if [[ "$SKIP_MTP" == "1" ]]; then
    log "skip FP8 MTP overlay"
else
    "$SCRIPT_DIR/scripts/04-build-mtp-fp8.sh"
fi

if [[ "$SKIP_SERVICE" == "1" ]]; then
    log "skip service install"
else
    "$SCRIPT_DIR/scripts/05-install-service.sh"
fi

cat <<EOF

==> install complete

Start:
  ./start.sh --wait
  ./scripts/healthcheck.sh
  ./scripts/smoke.sh

Logs:
  ${LOG_DIR:-$HOME/.local/state/qwen38-flash-next}/vllm.log

If systemd is unavailable, ./start.sh and ./stop.sh are the supported path.
EOF