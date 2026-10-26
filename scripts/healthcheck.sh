#!/bin/bash
# Health + model list. Usage: ./scripts/healthcheck.sh [--wait]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

HOST="${BIND_HOST:-127.0.0.1}"
[[ "$HOST" == "0.0.0.0" ]] && HOST="127.0.0.1"
URL="http://$HOST:${PORT:-12345}"
AUTH=()
if [[ -n "${API_KEY_FILE:-}" && -r "$API_KEY_FILE" ]]; then
    AUTH=(-H "Authorization: Bearer $(head -n1 "$API_KEY_FILE")")
elif [[ -n "${VLLM_API_KEY:-}" ]]; then
    AUTH=(-H "Authorization: Bearer $VLLM_API_KEY")
fi

if [[ "${1:-}" == "--wait" ]]; then
    for _ in $(seq 1 240); do
        curl -fsS --max-time 2 "$URL/health" >/dev/null 2>&1 && break
        sleep 5
    done
fi

log "GET $URL/health"
curl -fsS "$URL/health" && printf '\n'
log "GET $URL/v1/models"
curl -fsS "${AUTH[@]}" "$URL/v1/models" | python3 -m json.tool