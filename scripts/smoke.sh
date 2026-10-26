#!/bin/bash
# Functional smoke test against a running server.
# Usage: ./scripts/smoke.sh [--long-context] [--prefix-cache]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

HOST="${BIND_HOST:-127.0.0.1}"
[[ "$HOST" == "0.0.0.0" ]] && HOST="127.0.0.1"
API_KEY=""
if [[ -n "${API_KEY_FILE:-}" && -r "$API_KEY_FILE" ]]; then
    API_KEY="$(head -n1 "$API_KEY_FILE")"
elif [[ -n "${VLLM_API_KEY:-}" ]]; then
    API_KEY="$VLLM_API_KEY"
fi

exec python3 "$REPO_DIR/benchmarks/smoke.py" \
    --base-url "http://$HOST:${PORT:-12345}/v1" \
    --model "${SERVED_MODEL_NAME:-qwen38-flash-next-nvfp4}" \
    --api-key "$API_KEY" "$@"