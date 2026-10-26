#!/bin/bash
# Apply the vLLM 0.30.0 patches. Idempotent; each script keeps a .bak file.
# Usage: ./scripts/03-apply-patches.sh [--check-only]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

CHECK_ONLY=0
[[ "${1:-}" == "--check-only" ]] && CHECK_ONLY=1
PY="$(vllm_python)"
VLLM_ROOT="$(vllm_root)"
check_vllm_version

run_patch() {
    local script="$1"; shift
    if [[ "$CHECK_ONLY" == "1" ]]; then
        "$PY" "$REPO_DIR/patches/$script" "$@" --check
    else
        "$PY" "$REPO_DIR/patches/$script" "$@"
    fi
}

log "02f FLA shared-memory gate"
run_patch 02f-fla-shmem.py --vllm-root "$VLLM_ROOT"

log "02h mamba block-size prefix-cache fix"
run_patch 02h-mamba-block-size.py --vllm-root "$VLLM_ROOT"

log "02i TP3+EP3 model patch"
run_patch 02i-tp3-ep.py --vllm-root "$VLLM_ROOT"

log "02k MARLIN EP top-k fix"
run_patch 02k-marlin-ep.py --vllm-root "$VLLM_ROOT"

if [[ "${ENABLE_QSA_SPEC_CACHE:-1}" == "1" ]]; then
    log "02g QSA fused-draft metadata patch"
    QSA="$VLLM_ROOT/models/qwen4_exp/common/qsa_cache.py"
    [[ -f "$QSA" ]] || die "missing $QSA"
    if [[ "$CHECK_ONLY" == "1" ]]; then
        "$PY" "$REPO_DIR/patches/02g-qsa-spec-cache.py" "$QSA" >/dev/null
    else
        "$PY" "$REPO_DIR/patches/02g-qsa-spec-cache.py" "$QSA"
    fi
fi

log "patch status"
"$PY" "$REPO_DIR/patches/02f-fla-shmem.py" --vllm-root "$VLLM_ROOT" --check
"$PY" "$REPO_DIR/patches/02h-mamba-block-size.py" --vllm-root "$VLLM_ROOT" --check
"$PY" "$REPO_DIR/patches/02i-tp3-ep.py" --vllm-root "$VLLM_ROOT" --check
"$PY" "$REPO_DIR/patches/02k-marlin-ep.py" --vllm-root "$VLLM_ROOT" --check