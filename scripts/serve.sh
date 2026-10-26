#!/bin/bash
# Foreground launcher. systemd and start.sh both call this.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

VENV="${VENV:-$HOME/vllm-venv}"
MODEL_PATH="${MODEL_PATH:-$MTP_FP8_DIR}"
GPU_IDS="${GPU_IDS:-0,1,2}"
TP_SIZE="${TP_SIZE:-3}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
PORT="${PORT:-12345}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen38-flash-next-nvfp4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.96}"
YARN_FACTOR="${YARN_FACTOR:-2.0}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
MTP_K="${MTP_K:-2}"

need_file "$VENV/bin/vllm"
need_dir "$MODEL_PATH"
[[ -f "$MODEL_PATH/config.json" ]] || die "missing $MODEL_PATH/config.json"

IFS=',' read -r -a GPU_ARR <<<"$GPU_IDS"
[[ "${#GPU_ARR[@]}" == "$TP_SIZE" ]] || die "GPU_IDS has ${#GPU_ARR[@]} entries, TP_SIZE=$TP_SIZE"
[[ "$TP_SIZE" == "3" ]] || die "this recipe is TP3-only (TP_SIZE=3)"

for gpu in "${GPU_ARR[@]}"; do
    uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu")" || die "GPU $gpu not found"
    if [[ "${DRY_RUN:-0}" != "1" ]] && nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$uuid"; then
        die "GPU $gpu is busy; stop the other process or change GPU_IDS"
    fi
done

API_KEY=""
if [[ -n "${API_KEY_FILE:-}" ]]; then
    [[ -r "$API_KEY_FILE" ]] || die "API_KEY_FILE is not readable: $API_KEY_FILE"
    API_KEY="$(head -n1 "$API_KEY_FILE")"
elif [[ -n "${VLLM_API_KEY:-}" ]]; then
    API_KEY="$VLLM_API_KEY"
fi
if [[ "$BIND_HOST" != "127.0.0.1" && "$BIND_HOST" != "localhost" && -z "$API_KEY" && "${ALLOW_INSECURE:-0}" != "1" ]]; then
    die "refusing to bind $BIND_HOST without an API key; set API_KEY_FILE or ALLOW_INSECURE=1"
fi

case "$MTP_K" in
    0) SPEC_ARGS=() ;;
    1 | 2 | 3) SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K}") ;;
    *) die "MTP_K must be 0, 1, 2, or 3" ;;
esac

if (( MAX_MODEL_LEN > 262144 )); then
    HF_JSON="$(printf '{"text_config":{"make_ngram_vocab_size_divisible_by":384,"rope_parameters":{"rope_type":"yarn","factor":%s,"original_max_position_embeddings":262144,"rope_theta":10000000,"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25}}}' "$YARN_FACTOR")"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
else
    HF_JSON='{"text_config":{"make_ngram_vocab_size_divisible_by":384}}'
fi

API_ARGS=()
[[ -n "$API_KEY" ]] && API_ARGS=(--api-key "$API_KEY")

log "vLLM TP3+EP3 on GPUs $GPU_IDS, port $PORT, MTP=$MTP_K, context=$MAX_MODEL_LEN"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export VLLM_PLE_CPU_OFFLOAD="${VLLM_PLE_CPU_OFFLOAD:-1}"
export VLLM_FP8_HYBRID="${VLLM_FP8_HYBRID:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

CMD=(
    "$VENV/bin/vllm" serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$BIND_HOST" --port "$PORT"
    --tensor-parallel-size "$TP_SIZE" --enable-expert-parallel --language-model-only
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-num-seqs "$MAX_NUM_SEQS" --watermark 0.05 --async-scheduling
    --long-prefill-token-threshold 4096 --enable-prefix-caching
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --enable-auto-tool-choice --reasoning-parser qwen3 --tool-call-parser step3p5
    --attention-config '{"indexer_kv_dtype":"fp8"}'
    --hf-overrides "$HF_JSON"
    "${SPEC_ARGS[@]}" "${API_ARGS[@]}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi
exec "${CMD[@]}"