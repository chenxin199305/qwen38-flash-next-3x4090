#!/bin/bash
# Build the FP8 MTP overlay. Skipped when MTP_K=0.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

if [[ "${MTP_K:-2}" == "0" ]]; then
    log "MTP_K=0; skipping FP8 MTP overlay"
    exit 0
fi

need_dir "$FP8HYBRID_DIR"
PY="$(vllm_python)"
log "build FP8 MTP overlay"
"$PY" "$REPO_DIR/patches/02m-mtp-fp8.py" \
    --source "$FP8HYBRID_DIR" \
    --dest "$MTP_FP8_DIR"

need_file "$MTP_FP8_DIR/model-fp8-mtp.safetensors"
log "FP8 MTP overlay ready: $MTP_FP8_DIR"