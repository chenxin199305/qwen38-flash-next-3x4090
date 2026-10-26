# Qwen3.8-Flash-Next NVFP4 on 3x RTX 4090 48GB

**TP3 + EP3, no MoE padding, 524K context, 1.01M KV tokens, 124 tok/s single-stream, 635 tok/s at 16 streams.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-0.30.0-green.svg)](https://github.com/vllm-project/vllm)
[![GPU](https://img.shields.io/badge/GPU-3x%20RTX%204090%2048GB-orange.svg)](#requirements)

> **AI-generated content.** The scripts and notes in this repository were
> assembled by an AI assistant from a real deployment process. They may contain
> errors or omit environment-specific details. Issues and corrections are
> welcome. The repository does not distribute model weights.

## Why this recipe

Most 3-card attempts either fall back to TP2 (wasting a card) or pad the MoE
intermediate size so TP3 can split it. This recipe does neither.

| Advantage | How | Result |
|---|---|---|
| **Use all 3 cards** | TP3 + EP3, 3x 48GB | No fourth card, no 2-card compromise |
| **No MoE 640 → 768 padding** | EP3 keeps complete experts per GPU: **171 / 171 / 170** | Saves ~**4.5 GiB/card** vs padded MoE768; no padding-induced quality risk |
| **Correct EP output** | MARLIN EP top-k multiplication fixed (`02k`) | Relative error **0.47–1.16 → 0.0016**; 20/20 arithmetic regression |
| **524K context with real concurrency** | YaRN factor 2.0 + FP8 KV + FP8 indexer | KV pool **1,011,875 tokens** = **1.93x** a 524K request |
| **Low model footprint** | Blockwise-FP8 side layers (`02a`) + FP8 MTP experts (`02m`) | **27.96 GiB/card** model load, **15.67 GiB/card** KV |
| **Useful throughput** | MTP=2 speculative decoding + QSA fused-draft metadata (`02g`) | **124.3 tok/s** single, **387.4 / 617.6 / 635.3 tok/s** at 4/8/16 streams |
| **PLE table does not eat VRAM** | vLLM native pinned CPU offload (`VLLM_PLE_CPU_OFFLOAD=1`) | 51.2 GB n-gram table lives in host RAM, 8 lookups/token over UVA |
| **Reproducible** | vLLM 0.30.0 pinned, patch `--check`, benchmark JSON | Every claim has a command and a recorded config |

The headline is not just "it runs". It is: **three 48GB cards, a 180B-class
NVFP4 MoE, 524K context, and ~1M KV tokens without adding experts or a fourth
GPU.**

## Requirements

| Item | Requirement |
|---|---|
| GPU | **3x RTX 4090 48GB** (reports 49,140 MiB). 24GB cards cannot hold the ~28 GiB/card model load. |
| CPU / RAM | x86_64, **256 GiB RAM recommended** (51.2 GiB PLE table is pinned; 128 GiB is the minimum) |
| Disk | ~200 GiB free (126 GiB base + ~13 GiB FP8-hybrid + ~6 GiB MTP overlay) |
| OS | Linux |
| Driver | 570.153.02 + CUDA 12.8 (source-built vLLM) or >= 580 (PyPI wheel path) |
| vLLM | **0.30.0** — patch anchors are version-specific |
| torch | 2.13.0+cu126 on the reference host |
| Model | `<your-gated-checkpoint>` — NVFP4, ~176B total, ~6B active, 512 experts, 48 layers, 1 MTP layer |
| Model access | Obtain a compatible checkpoint yourself; this repository does not distribute weights. |

> **Checkpoint license.** The checkpoint's model card may describe it as a
> private candidate release and may refer to the source model's license. This
> repository does not redistribute weights or generated overlays. Obtain the
> checkpoint from its publisher and comply with its terms.

## Quick start

```bash
git clone https://github.com/chenxin199305/qwen38-flash-next-3x4090.git
cd qwen38-flash-next-3x4090

cp .env.sample .env
# Edit MODEL_DIR / FP8HYBRID_DIR / MTP_FP8_DIR / GPU_IDS / PORT.
# Keep BIND_HOST=127.0.0.1 unless you also configure an API key.

./install.sh          # preflight -> venv -> model -> FP8 hybrid -> patches -> FP8 MTP
./start.sh --wait
./scripts/healthcheck.sh
./scripts/smoke.sh
```

Already have the checkpoint and a vLLM 0.30.0 venv?

```bash
./install.sh --skip-download --skip-venv
```

Choose a tuning profile by appending it to `.env` (later values win):

```bash
cat config/profiles/capacity.env >> .env      # MTP=0, max KV
# or
cat config/profiles/throughput.env >> .env    # max-num-seqs=32
```

Minimal API call:

```bash
curl http://127.0.0.1:12345/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next-nvfp4","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Architecture

```text
OpenAI-compatible client
          |
          v
  vLLM 0.30.0  TP3 + EP3  (GPU 0/1/2)
          |
          |-- NVFP4 routed experts, EP3: 171 / 171 / 170 complete experts
          |-- FP8 hybrid side layers: GDN in/out, QSA q/k/v/o, shared expert
          |-- FP8 KV + FP8 indexer: 1,011,875 tokens at 524K
          |-- MTP=2 with FP8 experts: +52,428 KV tokens vs BF16 MTP
          '-- PLE 51.2 GB pinned host RAM (UVA, 8 lookups/token)
```

- **TP3** splits attention / dense layers across 3 GPUs.
- **EP3** puts whole routed experts on a GPU; no intermediate padding.
- **PLE** is an n-gram embedding table. It is a lookup, not a matmul; keeping
  it in pinned host memory leaves VRAM for KV.
- **FP8 KV/indexer** is what makes 524K x 1.93 concurrency possible on 48GB
  cards.

More detail: [docs/architecture.md](docs/architecture.md).

## Deployment details that matter

These are the problems that took the longest to solve. The full symptom →
cause → fix table is in [docs/troubleshooting.md](docs/troubleshooting.md).

| Problem | What you see | Root cause | Fix |
|---|---|---|---|
| TP3 cannot split MoE 640 | Need to pad intermediate to 768, +4.5 GiB/card | 640 is not 128-aligned for 3-way TP | **EP3** keeps complete experts: 171/171/170; no padding |
| EP output duplicated / wrong | Garbage or repeated tokens | MARLIN multiplies top-k weights inside the kernel even with `expert_map` | `02k` moves the multiplication outside; error 0.47–1.16 → 0.0016 |
| Prefix cache returns wrong output | Greedy output changes on a cache hit | Mamba state slot seeded with the minimum KV block size | `02h` uses `mamba_block_size`; verify with repeat-prefix test |
| MTP gives no speedup | Draft metadata rebuilt every step | QSA backend lacks fused-draft metadata support | `02g` ports upstream PR #58449; fallback log disappears |
| Model does not fit / KV too small | OOM or tiny KV pool | BF16 side layers and BF16 MTP experts | `02a` converts 156 GDN/QSA side tensors to blockwise FP8 (add `--include-shared-experts` for 300); `02m` converts 512 MTP experts |
| FP8 KV rejected | Startup assertion | QSA dtype gate / indexer config | vLLM 0.30.0 + `--kv-cache-dtype fp8_e4m3` + `attention-config.indexer_kv_dtype=fp8` |
| PLE table eats VRAM | OOM at load | 51.2 GB table treated as GPU memory | `VLLM_PLE_CPU_OFFLOAD=1` pinned host memory |
| PyPI wheel import fails on driver 570 | `libcudart.so.13` / version errors | vLLM 0.30 wheels link CUDA 13 | Build from source with CUDA 12.8 or use the documented container; driver >= 580 avoids this |
| YaRN silently ignored | Context still 262K | `--rope-scaling` removed in newer vLLM | Use `--hf-overrides` with `text_config.rope_parameters` |
| 24GB 4090 fails | OOM at load | Model needs ~28 GiB/card | 48GB cards required; use the 2-card recipe for 24GB cards |

## Measured performance

Reference host: 3x RTX 4090 48GB, driver 570.153.02, vLLM 0.30.0, torch
2.13.0+cu126, 256 GiB RAM. Output fixed at 256 tokens with `ignore_eos`;
token counts from `usage.completion_tokens`, not SSE chunks.

| Config | Model load / card | KV / card | KV pool | Single | 4-way | 8-way | 16-way |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 MTP=2 | 28.74 GiB | 14.89 GiB | 959,447 | 123.4 | 369.8 | 608.6 | 632.3 |
| **FP8 MTP=2 (default)** | **27.96 GiB** | **15.67 GiB** | **1,011,875** | **124.3** | **387.4** | **617.6** | **635.3** |

FP8 MTP vs BF16 MTP is roughly throughput-neutral in this measurement; the
real gain is **+52,428 KV tokens (+5.5%)**. Do not read the small tok/s delta
as a speedup.

**Reproducibility check.** On the reference host, the default `02a` (156
targets) plus `02m` reproduced the production overlays byte-for-byte:
`model-fp8-mtp.safetensors` and `model-bf16-00011-mtp-fp8.safetensors` both
matched their SHA-256. The 300-target `--include-shared-experts` layout is
available but is not the measured default.

MTP depth trade-off (BF16, same host):

| `MTP_K` | KV pool | 524K concurrency | Single | 4-way | 8-way | 16-way |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | **1,319,176** | **2.52x** | 86.7 | 240.4 | 345.5 | 559.0 |
| 1 | 1,016,145 | 1.94x | 117.9 | 244.2 | 440.9 | 636.4 |
| 2 | 959,447 | 1.83x | **129.7** | 310.8 | 479.5 | **644.5** |
| 3 | 912,462 | 1.74x | 126.7 | 328.0 | 453.2 | 541.1 |

MTP=2 is the default. MTP=3 loses at 16-way. MTP=0 is the capacity profile.
Methodology and raw JSON: [docs/benchmarks.md](docs/benchmarks.md).

## Tuning profiles

| Profile | MTP | Context | KV dtype | Max seqs | Use case |
|---|---:|---:|---|---:|---|
| `balanced.env` | 2 | 524K | FP8 | 16 | Default |
| `capacity.env` | 0 | 524K | FP8 | 16 | Longest context per request, max KV |
| `throughput.env` | 2 | 524K | FP8 | 32 | Many concurrent agent sessions |
| `native-262k.env` | 2 | 262K | FP8 | 16 | No YaRN; more KV headroom |

CUDA-graph capture for 32 slots costs some KV; measure before production.
`--max-num-seqs` is a scheduling slot limit, not a promise that every slot
fits a 524K request.

## Repository layout

```text
install.sh                  one-shot installer
start.sh / stop.sh          background lifecycle (no systemd required)
check-weights.sh            verify base + overlays
.env.sample                 all knobs
config/profiles/            runtime profiles
patches/                    02a/02f/02g/02h/02i/02k/02m + FP8 hybrid shim
scripts/                    preflight, venv, download, serve, health, doctor, smoke
services/                   systemd user unit template
benchmarks/                 single-stream, concurrency, smoke, results
tests/                      quantization/config/secret-scan tests
docs/                       architecture, design notes, troubleshooting, FAQ
```

## Security

- Default bind is `127.0.0.1`.
- Non-loopback bind without an API key is refused unless `ALLOW_INSECURE=1`.
- For LAN or public access, put the server behind a TLS reverse proxy with
  authentication. This repository deliberately contains **no** tunnel, FRP or
  public-endpoint configuration.
- Keep `.env` and the key file at mode `600`.
- Never paste keys, IPs or logs with credentials into an issue.

See [SECURITY.md](SECURITY.md).

## Known limitations

- **Text-only verified.** Vision is not validated in this recipe.
- **48GB cards only.** 3x24GB cannot hold ~28 GiB/card.
- **FP8 MTP tested with K=2.** K=0/1/3 are available but not re-validated with
  the FP8 MTP overlay.
- **vLLM 0.30.0 only.** Patches are exact-anchor edits; a different vLLM
  version will fail `--check` by design.
- **systemd unit is provided but was not validated on the reference host**
  (that host has no systemd). `start.sh` / `stop.sh` are the tested path.
- **Throughput numbers are one host, one run.** Re-measure on your cards.

## Documentation

| Document | Contents |
|---|---|
| [docs/quickstart.md](docs/quickstart.md) | Step-by-step install and first request |
| [docs/architecture.md](docs/architecture.md) | TP3/EP3, PLE, FP8 KV, MTP |
| [docs/design-notes.md](docs/design-notes.md) | Why MoE768 was rejected, EP bug, patch history |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Symptom → cause → fix |
| [docs/benchmarks.md](docs/benchmarks.md) | Methodology and raw results |
| [docs/compatibility.md](docs/compatibility.md) | Driver / vLLM / model matrix |
| [docs/faq.md](docs/faq.md) | Common questions |
| [docs/provenance.md](docs/provenance.md) | Source, credits, AI disclaimer |

## License and credits

Code: [Apache-2.0](LICENSE). Model weights: their own license, not included.

The TP3/EP3 work builds on vLLM and the Qwen3.8-Flash-Next community. In
particular, the mamba prefix-cache fix is ported from
[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX), and
the blockwise-FP8 side-layer pattern and hybrid shim are derived from
[Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound).
See [NOTICE](NOTICE) for details.

If this recipe saves you a week of debugging, a star helps others find it.
If it breaks, please open an issue with `./scripts/doctor.sh` output.