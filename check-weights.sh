#!/bin/bash
# Verify the base checkpoint and both overlays without loading the model.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/lib/common.sh"

MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-NVFP4}" \
FP8HYBRID_DIR="${FP8HYBRID_DIR:-$HOME/models/Qwen3.8-Flash-Next-NVFP4-fp8hybrid}" \
MTP_FP8_DIR="${MTP_FP8_DIR:-$HOME/models/Qwen3.8-Flash-Next-NVFP4-mtp-fp8hybrid}" \
python3 - <<'PY'
import json, os
from pathlib import Path

base = Path(os.environ["MODEL_DIR"])
hybrid = Path(os.environ["FP8HYBRID_DIR"])
mtp = Path(os.environ["MTP_FP8_DIR"])

def check_base() -> None:
    required = ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"]
    missing = [f for f in required if not (base / f).is_file()]
    if missing:
        raise SystemExit(f"base checkpoint missing: {missing}")
    index = json.loads((base / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing = [s for s in shards if not (base / s).exists()]
    if missing:
        raise SystemExit(f"base checkpoint missing shards: {missing[:5]}")
    print(f"OK base: {len(index['weight_map'])} tensors, {len(shards)} shards")

def check_hybrid() -> None:
    manifest = hybrid / "fp8hybrid-manifest.json"
    if not manifest.is_file():
        print(f"SKIP hybrid: {hybrid} (not built)")
        return
    data = json.loads(manifest.read_text())
    expected = data.get("target_tensors")
    if expected not in (156, 300):
        raise SystemExit(f"unexpected hybrid target count: {expected}")
    index = json.loads((hybrid / "model.safetensors.index.json").read_text())
    scales = [k for k in index["weight_map"] if k.endswith(".weight_scale_inv")]
    if len(scales) != expected:
        raise SystemExit(f"hybrid index has {len(scales)} scale tensors, expected {expected}")
    missing = sorted({s for s in index["weight_map"].values() if not (hybrid / s).exists()})
    if missing:
        raise SystemExit(f"hybrid missing shards: {missing[:5]}")
    print(f"OK hybrid: {expected} targets, {len(index['weight_map'])} tensors, worst rel err {data.get('worst_rel_err')}")

def check_mtp() -> None:
    f = mtp / "model-fp8-mtp.safetensors"
    if not f.is_file():
        print(f"SKIP MTP: {mtp} (not built)")
        return
    from safetensors import safe_open
    with safe_open(f, framework="pt", device="cpu") as st:
        n = len(st.keys())
    if n != 512 * 6:
        raise SystemExit(f"MTP overlay has {n} tensors, expected 3072")
    index = json.loads((mtp / "model.safetensors.index.json").read_text())
    print(f"OK MTP: {n} tensors, {len(index['weight_map'])} index keys")

check_base()
check_hybrid()
check_mtp()
PY