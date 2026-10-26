#!/bin/bash
# Collect environment information for bug reports. Redact keys before posting.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

echo "=== date ==="; date -Is
echo "=== uname ==="; uname -a
echo "=== GPU ==="; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
echo "=== memory ==="; free -h || true
echo "=== disk ==="; df -h "${MODEL_DIR:-$HOME}" || true
echo "=== vLLM ==="
"${VENV:-$HOME/vllm-venv}/bin/python" - <<'PY' 2>/dev/null || true
import torch, vllm
print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY
echo "=== patches ==="
if [[ -x "${VENV:-$HOME/vllm-venv}/bin/python" ]]; then
    for p in 02f-fla-shmem 02h-mamba-block-size 02i-tp3-ep 02k-marlin-ep; do
        "${VENV:-$HOME/vllm-venv}/bin/python" "$REPO_DIR/patches/$p.py" --check 2>&1 | tail -1 || true
    done
fi
echo "=== model ==="
ls -lh "${MODEL_DIR:-}" 2>/dev/null | head -5 || true
ls -lh "${FP8HYBRID_DIR:-}" 2>/dev/null | head -5 || true
ls -lh "${MTP_FP8_DIR:-}" 2>/dev/null | head -5 || true
echo "=== port ==="
ss -ltnp 2>/dev/null | grep ":${PORT:-12345}" || echo "not listening"
echo "=== log tail ==="
tail -n 40 "${LOG_DIR:-$HOME/.local/state/qwen38-flash-next}/vllm.log" 2>/dev/null || true