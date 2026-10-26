#!/bin/bash
# Download the base NVFP4 checkpoint.
# ModelScope is the default; HF is supported for users with access.
# The checkpoint is ~126 GiB. Resumable; existing files are verified/skipped.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

MODEL_ID="${MODEL_ID:-}"
[[ -n "$MODEL_ID" ]] || die "set MODEL_ID in .env to the checkpoint you are authorized to use"
[[ "$MODEL_ID" != "<your-gated-checkpoint>" ]] || die "edit MODEL_ID in .env; the sample is a placeholder"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-master}"
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-modelscope}"
PAR="${PAR:-6}"

mkdir -p "$MODEL_DIR"

if [[ "$DOWNLOAD_SOURCE" == "hf" ]]; then
    log "Hugging Face download: $MODEL_ID"
    require_cmd python3
    if command -v hf >/dev/null 2>&1; then
        HF_ARGS=(download "$MODEL_ID" --local-dir "$MODEL_DIR")
        [[ -n "${HF_TOKEN:-}" ]] && HF_ARGS+=(--token "$HF_TOKEN")
        hf "${HF_ARGS[@]}"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_ARGS=(download "$MODEL_ID" --local-dir "$MODEL_DIR")
        [[ -n "${HF_TOKEN:-}" ]] && HF_ARGS+=(--token "$HF_TOKEN")
        huggingface-cli "${HF_ARGS[@]}"
    else
        die "install huggingface_hub (pip install 'huggingface_hub[cli]') or set DOWNLOAD_SOURCE=modelscope"
    fi
else
    log "ModelScope download: $MODEL_ID"
    require_cmd curl
    API="https://modelscope.cn/api/v1/models/${MODEL_ID}/repo/files?Revision=${MODEL_REVISION}&Recursive=true"
    BASE="https://modelscope.cn/models/${MODEL_ID}/resolve/${MODEL_REVISION}"
    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' EXIT

    curl -fsSL --max-time 120 "$API" -o "$TMP/files.json"

    MODEL_DIR="$MODEL_DIR" TMP="$TMP" python3 - <<'PY'
import json, os
from pathlib import Path

tmp = Path(os.environ["TMP"])
dest = Path(os.environ["MODEL_DIR"])
data = json.loads((tmp / "files.json").read_text())
files = data["Data"]["Files"]
manifest, small = {}, []
for f in files:
    p = f["Path"]
    if p.endswith(".gitattributes"):
        continue
    # Skip publisher-side per-shard audit JSON; inference does not need it.
    if p.startswith("layer-") and p.endswith(".json"):
        continue
    if p.endswith(".safetensors"):
        manifest[p] = f["Sha256"]
    else:
        small.append(p)
(tmp / "manifest.json").write_text(json.dumps({"manifest": manifest, "small": small}))
print(f"manifest: {len(manifest)} safetensors + {len(small)} small files")
PY

    MODEL_DIR="$MODEL_DIR" BASE="$BASE" TMP="$TMP" python3 - <<'PY'
import json, os, subprocess
from pathlib import Path

tmp = Path(os.environ["TMP"])
dest = Path(os.environ["MODEL_DIR"])
base = os.environ["BASE"]
d = json.loads((tmp / "manifest.json").read_text())
required = {"config.json", "generation_config.json", "model.safetensors.index.json",
            "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
for p in d["small"]:
    out = dest / p
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"[cached] {p}")
        continue
    r = subprocess.run(["curl", "-fsSL", "--retry", "5", "-o", str(out), f"{base}/{p}"])
    print(f"[{'OK' if r.returncode == 0 else 'FAIL'}] {p}")
    if r.returncode != 0 and p in required:
        raise SystemExit(f"required file failed: {p}")
PY

    MODEL_DIR="$MODEL_DIR" BASE="$BASE" TMP="$TMP" PAR="$PAR" python3 - <<'PY'
import concurrent.futures as cf
import hashlib, json, os, subprocess
from pathlib import Path

tmp = Path(os.environ["TMP"])
dest = Path(os.environ["MODEL_DIR"])
base = os.environ["BASE"]
par = int(os.environ.get("PAR", "6"))
d = json.loads((tmp / "manifest.json").read_text())
files = sorted(d["manifest"].items())

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(item):
    name, expected = item
    path = dest / name
    if path.exists() and sha256(path) == expected:
        return name, True, "cached"
    for _ in range(40):
        r = subprocess.run(["curl", "-sL", "--retry", "5", "-C", "-",
                            "--speed-limit", "51200", "--speed-time", "60",
                            "-o", str(path), f"{base}/{name}"], capture_output=True)
        if r.returncode == 0 and sha256(path) == expected:
            return name, True, "ok"
        if r.returncode not in (0, 33, 36):
            path.unlink(missing_ok=True)
    return name, False, "fail"

with cf.ThreadPoolExecutor(par) as ex:
    for i, (name, ok, how) in enumerate(ex.map(fetch, files), 1):
        print(f"[{'OK' if ok else 'FAIL'}:{how}] {i}/{len(files)} {name}", flush=True)
        if not ok:
            raise SystemExit(f"download failed: {name}")
print("ALL_DONE")
PY
fi

log "verify key files"
for f in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json; do
    [[ -s "$MODEL_DIR/$f" ]] || die "missing $MODEL_DIR/$f"
done
du -sh "$MODEL_DIR"