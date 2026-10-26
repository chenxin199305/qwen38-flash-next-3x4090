#!/bin/bash
# Minimal OpenAI-compatible request.
set -euo pipefail
BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:12345/v1}"
MODEL="${MODEL_ID:-qwen38-flash-next-nvfp4}"
AUTH=()
[[ -n "${VLLM_API_KEY:-}" ]] && AUTH=(-H "Authorization: Bearer $VLLM_API_KEY")

curl -sS "$BASE_URL/chat/completions" \
    -H 'Content-Type: application/json' \
    "${AUTH[@]}" \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Explain TP3+EP3 in two sentences.\"}],
        \"max_tokens\": 128,
        \"temperature\": 0
    }" | python3 -m json.tool