# Quick start

## 0. Check the host

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
```

You need **3x RTX 4090 48GB** (about 49,140 MiB each). 24GB cards are not
supported: the model needs ~28 GiB/card before KV.

```bash
git clone https://github.com/chenxin199305/qwen38-flash-next-3x4090.git
cd qwen38-flash-next-3x4090
cp .env.sample .env
```

Edit `.env`:

- `MODEL_DIR`, `FP8HYBRID_DIR`, `MTP_FP8_DIR`: where the checkpoints live;
- `GPU_IDS=0,1,2`;
- `PORT` if 12345 is taken;
- `BIND_HOST=127.0.0.1` for local use. For LAN access, set `0.0.0.0` and an
  `API_KEY_FILE`, or `ALLOW_INSECURE=1` on a trusted network.

```bash
./scripts/00-preflight.sh
```

## 1. Install vLLM

```bash
./scripts/01-setup-venv.sh
```

Driver >= 580: the PyPI wheel path works. Driver 570 / CUDA 12.8: the script
stops and points to [compatibility.md](compatibility.md). Build vLLM 0.30.0
from source with CUDA 12.8, or set `VLLM_WHEEL=/path/to/vllm-0.30.0+cu128.whl`.

If the venv already exists:

```bash
SKIP_VLLM_INSTALL=1 ./scripts/01-setup-venv.sh
```

## 2. Download the base checkpoint

```bash
./scripts/02-download-model.sh
```

ModelScope is the default. For Hugging Face:

```bash
DOWNLOAD_SOURCE=hf HF_TOKEN="$(cat ~/.hf_token)" ./scripts/02-download-model.sh
```

The checkpoint is ~126 GiB. The script verifies SHA256 and skips existing
files. It does not download the publisher's per-shard audit JSON.

## 3. Build the FP8-hybrid overlay

```bash
./scripts/02a-build-fp8hybrid.py --dry-run   # expect 156 targets, 4 shards
./scripts/02a-build-fp8hybrid.py
```

This creates `FP8HYBRID_DIR` and leaves the source untouched. It converts 156
GDN/QSA side-layer tensors to blockwise FP8. Add `--include-shared-experts`
for the 300-tensor layout (experimental). Check:

```bash
./scripts/02a-build-fp8hybrid.py --check
```

## 4. Apply the vLLM patches

```bash
./scripts/03-apply-patches.sh
```

This applies, in order:

- `02f` FLA shared-memory gate;
- `02h` mamba block-size prefix-cache fix;
- `02i` TP3+EP3 model patch;
- `02k` MARLIN EP top-k fix;
- `02g` QSA fused-draft metadata (MTP).

Each patch is idempotent and keeps a `.bak-*` file. `--check` verifies the
anchors without writing.

## 5. Build the FP8 MTP overlay

```bash
./scripts/04-build-mtp-fp8.sh
```

This writes ~2.34 GiB and verifies expert 0 cosine similarity. If `MTP_K=0`,
the script skips it and `MODEL_PATH` should point at `FP8HYBRID_DIR`.

## 6. Start and verify

```bash
./start.sh --wait
./scripts/healthcheck.sh
./scripts/smoke.sh
```

`smoke.sh` checks health, models, arithmetic, tool calls and streaming. With
`--prefix-cache` it also checks that a repeated prompt gets faster and does
not change the greedy answer. With `--long-context` it runs a ~288K needle
test (slow).

Logs:

```bash
tail -f "$HOME/.local/state/qwen38-flash-next/vllm.log"
```

## 7. Use the API

```bash
curl http://127.0.0.1:12345/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next-nvfp4","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

OpenAI Python client:

```bash
pip install openai
python examples/openai-python.py
```

## Profiles

Append the profile to `.env` (later values win):

```bash
cat config/profiles/capacity.env >> .env      # MTP=0, max KV
# or
cat config/profiles/throughput.env >> .env    # max-num-seqs=32
```

Then restart:

```bash
./stop.sh
./start.sh --wait
```

## systemd (optional)

```bash
./scripts/05-install-service.sh
systemctl --user status vllm-qwen38-tp3
journalctl --user -u vllm-qwen38-tp3 -f
```

The reference host has no systemd; `start.sh` / `stop.sh` is the tested path.
The unit is a template and should be reviewed on your host.

## Updating

The patch anchors are pinned to vLLM 0.30.0. Do not upgrade vLLM without
re-running:

```bash
./scripts/03-apply-patches.sh --check-only
```

A failed anchor is a signal to port the patch, not to skip it.