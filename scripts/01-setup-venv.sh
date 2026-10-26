#!/bin/bash
# Create a venv and install the pinned vLLM version.
# Driver >=580: PyPI wheel path.
# Driver 570 (CUDA 12.8): build vLLM from source or use a CUDA 12.8 wheel;
# see docs/compatibility.md. Set SKIP_VLLM_INSTALL=1 if the venv already exists.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

VENV="${VENV:-$HOME/vllm-venv}"
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

if [[ "${SKIP_VLLM_INSTALL:-0}" == "1" ]]; then
    [[ -x "$VENV/bin/vllm" ]] || die "SKIP_VLLM_INSTALL=1 but $VENV/bin/vllm is missing"
    "$VENV/bin/python" -c "import vllm; print('vLLM', vllm.__version__)"
    exit 0
fi

log "dependencies"
for cmd in uv gcc python3; do require_cmd "$cmd"; done
[[ -x "$CUDA_HOME/bin/nvcc" ]] || warn "nvcc not found under $CUDA_HOME; source builds will fail"

DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
DRIVER_MAJOR="${DRIVER%%.*}"
if [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ && "$DRIVER_MAJOR" -lt 580 && "${FORCE_PYPI_VLLM:-0}" != "1" ]]; then
    cat >&2 <<EOF
!! driver $DRIVER is older than 580.
!! vLLM 0.30 PyPI wheels link CUDA 13 and fail at import on this driver.
!! Options:
!!   1) build vLLM 0.30.0 from source with CUDA 12.8 (docs/compatibility.md)
!!   2) use the CUDA 12.8 container from the same document
!!   3) if you already built a +cu128 wheel: VLLM_WHEEL=/path/to.whl ./scripts/01-setup-venv.sh
!! Set FORCE_PYPI_VLLM=1 only if you know the wheel works.
EOF
    exit 1
fi

if [[ -x "$VENV/bin/vllm" ]]; then
    log "venv already has vLLM: $VENV"
    "$VENV/bin/python" -c "import vllm; print('vLLM', vllm.__version__)"
    exit 0
fi

log "create venv: $VENV"
uv venv "$VENV" --python "$(command -v python3)"

log "install vLLM $VLLM_VERSION"
if [[ -n "${VLLM_WHEEL:-}" ]]; then
    UV_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}" \
        uv pip install --python "$VENV/bin/python" "$VLLM_WHEEL" ninja
else
    UV_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}" \
        uv pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION" ninja
fi

"$VENV/bin/python" - <<'PY'
import torch
import vllm
print("vLLM", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY