# Compatibility

## Tested combination

| Component | Version |
|---|---|
| GPU | 3x RTX 4090 48GB (49,140 MiB) |
| Driver | 570.153.02 |
| CUDA | 12.8 (source-built vLLM) |
| vLLM | 0.30.0 |
| torch | 2.13.0+cu126 |
| OS | Linux x86_64 |
| Model | `<your-gated-checkpoint>` (NVFP4 layout below) |

## Driver matrix

| Driver | Path | Notes |
|---|---|---|
| >= 580 | PyPI `vllm==0.30.0` | Wheels link CUDA 13; should import |
| 570.x / CUDA 12.8 | Source build or `+cu128` wheel | The reference host; build with `TORCH_CUDA_ARCH_LIST=8.9` |
| < 570 | Not tested | Upgrade the driver or build against your CUDA |

The source build is not automated here because it depends on the host's CUDA
toolkit and PyTorch index. The key settings:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=8.9
# build vLLM 0.30.0 from source, then:
uv pip install --python "$VENV/bin/python" dist/vllm-0.30.0+cu128-*.whl
```

Do not mix a CUDA 13 wheel with a CUDA 12.8 driver. The import error is
usually `libcudart.so.13` or a version-node mismatch.

## vLLM patch matrix

| Patch | Purpose | Version guard | Critical? |
|---|---|---|---|
| `02f-fla-shmem.py` | 102400 → 101376 FLA shared-memory gate | 0.30.0 | Performance |
| `02h-mamba-block-size.py` | Prefix-cache Mamba state fix | 0.30.0 | **Correctness** |
| `02i-tp3-ep.py` | TP3+EP3 model adaptation | 0.30.0 | **Required** |
| `02k-marlin-ep.py` | EP top-k multiplication fix | 0.30.0 | **Correctness** |
| `02g-qsa-spec-cache.py` | QSA fused-draft metadata | 0.30.0 | MTP performance |
| `02a-build-fp8hybrid.py` | 156 side-layer FP8 conversion (300 with `--include-shared-experts`) | Model layout | Capacity |
| `02m-mtp-fp8.py` | 512 MTP-expert FP8 overlay | Model layout | Capacity |
| `vllm_fp8_hybrid_modelopt.py` | Runtime FP8-hybrid dispatch | 0.30.0 | Required for `02a` |

A different vLLM version will fail the `--check` paths. That is intentional:
the patches are exact-anchor edits, not a fork.

## Model variants

This recipe targets an NVFP4 checkpoint with:
- NVFP4 routed experts (ModelOpt, group 16);
- BF16 dense side layers;
- FP8 PLE table;
- 1 MTP layer.

Other checkpoints may have different tensor names, side-layer layouts or PLE
dtypes. Run `scripts/02a-build-fp8hybrid.py --dry-run` first; it expects 156
target tensors by default (300 with `--include-shared-experts`). Do not force a
mismatch.

## What is not covered

- 2x or 4x GPU configurations.
- 24GB RTX 4090.
- DGX Spark / GB10 (aarch64, sm_121, unified memory).
- Multi-node tensor parallelism.
- Vision input validation.
- SGLang or other engines.

Those belong in separate recipes. Keeping this repository narrow is what makes
the patch set testable.