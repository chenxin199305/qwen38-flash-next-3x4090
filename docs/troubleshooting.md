# Troubleshooting

Run `./scripts/doctor.sh` first. It collects the GPU, driver, vLLM, patch,
model and port state in one page.

## Install

### `vllm` imports with `libcudart.so.13` / version errors

Cause: vLLM 0.30 PyPI wheels link CUDA 13, but the host driver is < 580
(e.g. 570 / CUDA 12.8).

Fix: build vLLM 0.30.0 from source with CUDA 12.8 and
`TORCH_CUDA_ARCH_LIST=8.9`, or use the CUDA 12.8 container described in
[compatibility.md](compatibility.md). Do not force the PyPI wheel unless you
have verified it on your driver.

### `uv` or `gcc` missing

`./scripts/01-setup-venv.sh` needs `uv`, `gcc` and the CUDA toolkit. Install
them first, or set `SKIP_VLLM_INSTALL=1` if the venv already has vLLM.

### `02f`/`02h`/`02i`/`02k` says "expected one anchor"

Cause: the installed vLLM is not exactly 0.30.0, or a previous patch changed
the anchor.

Fix: check `vllm.__version__`. The scripts intentionally fail on a different
version. Reinstall 0.30.0, or port the anchor to your version and update the
script.

## Startup

### `GPU 0 is busy`

`serve.sh` refuses to start when another process is using a target GPU. This
is deliberate: a second vLLM would OOM and could corrupt the running service.

```bash
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv
```

Stop the other process, or change `GPU_IDS` (but this recipe requires 3 GPUs).

### OOM during model load

Check that you are on 48GB cards (`nvidia-smi` reports ~49,140 MiB). A 24GB
card cannot hold ~28 GiB of weights before KV.

If you are on 48GB cards:

- confirm `MODEL_PATH` points to the FP8-hybrid/MTP overlay, not the raw base;
- lower `GPU_MEM_UTIL` from 0.96 to 0.94;
- lower `MAX_NUM_SEQS`;
- lower `MAX_MODEL_LEN` (e.g. 262144);
- set `MTP_K=0` and use the capacity profile.

### KV pool is much smaller than expected

- `MAX_MODEL_LEN` above 262K requires YaRN; check the startup log for
  `rope_parameters` / `factor`.
- `MTP_K=2/3` reserves draft state; MTP=0 gives the largest pool.
- CUDA-graph capture for large `MAX_NUM_SEQS` costs KV. Try 16 before 32.
- `--kv-cache-dtype fp8_e4m3` must be active; if the log shows BF16 KV, the
  flag or patch did not apply.

## Wrong output

### Arithmetic is wrong / repeated tokens

Run:

```bash
./scripts/smoke.sh
```

If the arithmetic checks fail, verify `02k`:

```bash
${VENV:-$HOME/vllm-venv}/bin/python patches/02k-marlin-ep.py --check
```

Without `02k`, EP3 can produce fluent but numerically wrong output. Re-apply
the patch and restart.

### Output changes when the same prompt is repeated

This is the prefix-cache/Mamba bug. Verify `02h`:

```bash
${VENV:-$HOME/vllm-venv}/bin/python patches/02h-mamba-block-size.py --check
```

Run `python benchmarks/smoke.py --prefix-cache`. The second request should be
faster and should not change the greedy answer.

### MTP log says "falling back"

`02g` is missing or did not apply. Check:

```bash
${VENV:-$HOME/vllm-venv}/bin/python patches/02g-qsa-spec-cache.py --selftest
```

### Tool calls are not parsed

The recipe uses `--tool-call-parser step3p5` and `--reasoning-parser qwen3`.
If your client sends a different chat template, the parser may not match.
Test with `benchmarks/smoke.py`; it sends a minimal `get_weather` tool.

## PLE / memory

### Host RAM is exhausted

The PLE table is 51.2 GB of pinned memory. Plus the OS, page cache and the
model process, 256 GiB is recommended. 128 GiB can work but leaves little
headroom. Check:

```bash
free -h
grep -E 'MemAvailable|MemFree' /proc/meminfo
```

### `VLLM_PLE_CPU_OFFLOAD=1` has no effect

Check that `serve.sh` exports it and that the model's PLE tensors are FP8.
The startup log should mention PLE CPU offload. If you run vLLM directly,
export the variable yourself.

## API / security

### `401 Unauthorized`

You set an API key but the client does not send it. Use:

```bash
curl -H "Authorization: Bearer $VLLM_API_KEY" ...
```

### `serve.sh` refuses to bind `0.0.0.0`

This is intentional. Set `API_KEY_FILE` (mode 600) or `ALLOW_INSECURE=1` only
on a trusted network. For remote access, use a TLS reverse proxy; do not
expose port 12345 directly.

### Port already in use

```bash
ss -ltnp | grep 12345
```

Change `PORT` in `.env` or stop the other service.

## Performance

### Single-stream is much slower than 124 tok/s

- Check that MTP is active: the startup log should show
  `num_speculative_tokens=2`.
- Check that the FP8-hybrid overlay is active (`VLLM_FP8_HYBRID=1`).
- Check that the GPU is not shared.
- Check the driver / clock throttle with `nvidia-smi -q -d PERFORMANCE`.
- Run `benchmarks/bench.py --runs 3` after a warmup.

### 16-way throughput is lower than 8-way

This can happen with MTP=3 or a KV-limited pool. Use MTP=2 and
`MAX_NUM_SEQS=16`; verify the KV pool in the startup log. If you need 32
slots, measure the KV cost first.

### Prefix cache hit rate is low

The mamba alignment granularity is large (1600 tokens in some versions).
Short repeated prefixes may not hit. Use a prefix longer than one block and
check the second TTFT. This is expected, not a bug.

## Logs

```bash
tail -f "$LOG_DIR/vllm.log"
# or
journalctl --user -u vllm-qwen38-tp3 -f
```

Include the startup KV line and the first error in an issue. Redact keys and
IPs.