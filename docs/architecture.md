# Architecture

## Model

Qwen3.8-Flash-Next is a hybrid-attention multimodal MoE:

- 48 decoder layers.
- 36 GDN (linear attention) layers + 12 QSA (sparse full attention) layers.
- 512 routed experts per MoE layer, top-10 routing, intermediate size 640.
- ~176B total parameters (125B main + 51B PLE n-gram table), ~6B active/token.
- 1 MTP layer.
- Native context 262,144; this recipe serves 524,288 with YaRN factor 2.0.

The NVFP4 checkpoint quantizes only the routed experts. Dense side layers
(GDN projections, QSA q/k/v/o, shared expert) remain BF16. The PLE n-gram
table is FP8 and is a pure lookup table.

## Why TP3 + EP3

Naive TP3 splits the MoE intermediate dimension 640 into 214/213/213, which is
not 128-aligned for the NVFP4/MARLIN kernels. The first working approach
padded 640 → 768, costing ~4.5 GiB per card and changing the expert shape.

EP3 avoids that: every routed expert stays whole on one GPU. The 512 experts
are distributed 171 / 171 / 170 across the three cards. Attention and dense
layers are still TP3.

```text
GPU0: experts 0..170     + TP3 shard of attention/dense
GPU1: experts 171..341   + TP3 shard of attention/dense
GPU2: experts 342..511   + TP3 shard of attention/dense
```

The `02i` patch adapts vLLM's Qwen4Exp model for TP3:

- GDN layers run with local TP1 (full conv/qkvz/ba/out projections and
  `A_log`/`dt_bias` replicated).
- QSA loads all 24 Q heads and 2 KV heads on every rank.
- Shared experts are replicated when TP3 cannot align them.
- `embed_tokens` / `lm_head` use `padding_size=192`.
- PLE divisor is 384; the checkpoint's original 128-aligned shards are read
  with tail padding.

## EP correctness fix

vLLM 0.30's MARLIN MoE path multiplies top-k router weights inside the kernel
even when an `expert_map` is present. With EP3, that double-applies the
weights and produces duplicated or wrong output. `02k` disables the in-kernel
multiplication for the EP path and applies it outside the kernel.

Measured against an FP32 reference, the relative error dropped from
0.47–1.16 to about 0.0016. This is why `scripts/smoke.sh` includes arithmetic
prompts: they catch the regression quickly.

## Memory layout

Per card at the production profile:

| Item | Size |
|---|---:|
| Model weights (NVFP4 experts + FP8 side layers + FP8 MTP) | 27.96 GiB |
| KV cache | 15.67 GiB |
| KV pool | 1,011,875 tokens |

The 51.2 GB PLE table is **not** in VRAM. vLLM's native PLE CPU offload keeps
it in pinned host memory; the Triton kernel reads rows over UVA. This is the
main reason a 3x48GB box can run a 180B-class model with a useful KV pool.

## FP8 side layers

The dense side layers are read in full on every decoded token and dominate
decode bandwidth. `scripts/02a-build-fp8hybrid.py` converts 156 tensors by
default (36 GDN x 3 + 12 QSA x 4); `--include-shared-experts` adds the 144
shared-expert tensors for 300 total. The measured production numbers use the
156-tensor layout. Blockwise FP8 layout:

```text
weight           float8_e4m3fn  [out, in]
weight_scale_inv float32        [out/128, in/128]
```

The source checkpoint is untouched. Unchanged shards and small files are
symlinked; only the 13 shards containing target tensors are rewritten. The
runtime shim `patches/vllm_fp8_hybrid_modelopt.py` scans the safetensors
metadata and routes those tensors to vLLM's blockwise `Fp8Config`, leaving the
NVFP4 MoE path alone.

Worst per-tensor relative error is about 3.54%; model load drops from 28.74 to
27.96 GiB/card, and KV rises by 52,428 tokens.

## FP8 KV and indexer

`--kv-cache-dtype fp8_e4m3` plus
`--attention-config '{"indexer_kv_dtype":"fp8"}'` halves the KV bytes. QSA
reads/writes through its own Triton kernel, so this is not an FA-only feature.
The result is 1,011,875 tokens at 524K context, or 1.93x concurrency.

## MTP

MTP=2 adds a draft head that proposes two tokens per step; the target model
verifies them. vLLM 0.30's QSA backend initially rebuilt attention metadata
for every draft step, making MTP a net loss. `02g` ports the fused-draft
metadata support from upstream PR #58449. With that patch, MTP=2 gives the
best balance: 124.3 tok/s single-stream and 635.3 tok/s at 16 streams.

MTP=3 is slower at 16-way on this host. MTP=0 is the capacity profile.

## Prefix caching

Hybrid GDN/QSA models have multiple KV groups with different block sizes.
vLLM's engine overwrites `cache_config.block_size` with the smallest group
size, but the mamba state seed and the chunk split need the mamba block size.
Without `02h`, a prefix-cache hit restores an all-zero Mamba state: the server
does not crash, it just returns wrong output. The patch makes both sides use
the mamba block size.

This is the most dangerous class of bug in the recipe: a silent correctness
failure that only appears on cache hits. Run the repeat-prefix check in
`benchmarks/smoke.py --prefix-cache` after any vLLM change.

## Data flow

```text
prompt
  -> tokenizer
  -> PLE lookup (host pinned memory, 8 rows/token)
  -> 48 layers: 36 GDN + 12 QSA, EP3 routed experts
  -> MTP draft (2 tokens) -> target verification
  -> detokenizer -> SSE / JSON
```

The OpenAI API, tool parser (`step3p5`) and reasoning parser (`qwen3`) are
served by vLLM on port 12345 by default.