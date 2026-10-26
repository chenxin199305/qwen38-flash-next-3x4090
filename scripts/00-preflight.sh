#!/bin/bash
# Check host, GPUs, driver, RAM, disk and ports before installing.
# Usage: ./scripts/00-preflight.sh [--strict]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

STRICT=0
[[ "${1:-}" == "--strict" ]] && STRICT=1
FAIL=0
WARN=0

fail() { printf 'FAIL  %s\n' "$*" >&2; FAIL=$((FAIL + 1)); }
warn() { printf 'WARN  %s\n' "$*" >&2; WARN=$((WARN + 1)); }
ok()   { printf 'OK    %s\n' "$*"; }

log "host"
[[ "$(uname -s)" == "Linux" ]] && ok "Linux" || fail "Linux required"
[[ "$(uname -m)" == "x86_64" ]] && ok "x86_64" || warn "x86_64 expected; this recipe targets sm_89 RTX 4090"

log "GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found"
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader)
    printf '%s\n' "${GPUS[@]}"
    NGPU="${#GPUS[@]}"
    [[ "$NGPU" -ge 3 ]] && ok "$NGPU GPUs visible" || fail "3 GPUs required, found $NGPU"
    for g in "${GPUS[@]:0:3}"; do
        mem="$(awk -F', ' '{print $3}' <<<"$g" | tr -dc '0-9')"
        [[ -n "$mem" && "$mem" -ge 46000 ]] && ok "GPU memory ${mem} MiB" \
            || fail "GPU memory ${mem:-unknown} MiB < 46 GiB; 24GB cards are not supported"
    done
    DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    DRIVER_MAJOR="${DRIVER%%.*}"
    if [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ && "$DRIVER_MAJOR" -ge 580 ]]; then
        ok "driver $DRIVER (PyPI vLLM 0.30 wheel path)"
    else
        warn "driver $DRIVER < 580: PyPI vLLM 0.30 wheels link CUDA 13; build from source or use the documented container"
    fi
fi

EXPECTED_GPUS="$(awk -F, '{print NF}' <<<"${GPU_IDS:-0,1,2}")"
[[ "$EXPECTED_GPUS" == "${TP_SIZE:-3}" ]] && ok "GPU_IDS has ${TP_SIZE:-3} entries" \
    || fail "GPU_IDS=${GPU_IDS:-} does not match TP_SIZE=${TP_SIZE:-3}"

log "memory and disk"
RAM_GB="$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
if [[ "$RAM_GB" -ge 256 ]]; then
    ok "${RAM_GB} GiB RAM"
elif [[ "$RAM_GB" -ge 128 ]]; then
    warn "${RAM_GB} GiB RAM: 51.2 GiB PLE table is pinned; 256 GiB recommended"
else
    fail "${RAM_GB} GiB RAM: at least 128 GiB required"
fi

CHECK_DIR="${MODEL_DIR:-$HOME/models}"
mkdir -p "$(dirname "$CHECK_DIR")" 2>/dev/null || true
FREE_GB="$(df -Pk "$(dirname "$CHECK_DIR")" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024/1024}')"
if [[ -n "$FREE_GB" && "$FREE_GB" -ge 200 ]]; then
    ok "${FREE_GB} GiB free next to $CHECK_DIR"
elif [[ -n "$FREE_GB" && "$FREE_GB" -ge 150 ]]; then
    warn "${FREE_GB} GiB free: base checkpoint + overlays need ~150-200 GiB"
else
    warn "${FREE_GB:-unknown} GiB free: check disk before downloading"
fi

log "commands"
for cmd in python3 curl git; do
    command -v "$cmd" >/dev/null 2>&1 && ok "$cmd" || fail "$cmd missing"
done
if [[ "${SKIP_VLLM_INSTALL:-0}" != "1" ]]; then
    for cmd in uv gcc; do
        command -v "$cmd" >/dev/null 2>&1 && ok "$cmd" || fail "$cmd missing (needed for venv install)"
    done
fi

log "port"
if port_in_use "${PORT:-12345}"; then
    warn "port ${PORT:-12345} is already listening"
else
    ok "port ${PORT:-12345} is free"
fi

log "model"
if [[ -f "${MODEL_DIR:-}/model.safetensors.index.json" ]]; then
    ok "base checkpoint index found: $MODEL_DIR"
else
    warn "base checkpoint not found at ${MODEL_DIR:-}; install will download it"
fi
if [[ -f "${FP8HYBRID_DIR:-}/fp8hybrid-manifest.json" || -f "${FP8HYBRID_DIR:-}/model.safetensors.index.json" ]]; then
    ok "FP8-hybrid overlay found: $FP8HYBRID_DIR"
else
    warn "FP8-hybrid overlay not built yet: $FP8HYBRID_DIR"
fi

printf '\npreflight: %d failure(s), %d warning(s)\n' "$FAIL" "$WARN"
if [[ "$FAIL" -gt 0 || ( "$STRICT" == 1 && "$WARN" -gt 0 ) ]]; then
    exit 1
fi