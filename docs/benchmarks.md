# Benchmarks

## Method

- Hardware: 3x RTX 4090 48GB (49,140 MiB), driver 570.153.02, 256 GiB RAM.
- Software: vLLM 0.30.0, torch 2.13.0+cu126, CUDA 12.8.
- Model: `<your-gated-checkpoint>` + FP8-hybrid + FP8 MTP overlay.
- Output: fixed 256 tokens, `ignore_eos=true`.
- Token count: `usage.completion_tokens`, **not** SSE chunks. This matters
  with speculative decoding, where one chunk can contain several tokens.
- Warmup: at least one request per configuration before measuring.
- Concurrency: `1,4,8,16` streams, each stream a separate request.
- TTFT: time to first token; decode tok/s: completion tokens divided by
  (last token time − first token time).

The numbers below are one host, one run. Treat small deltas as noise. The
FP8-vs-BF16 MTP comparison is throughput-neutral within measurement noise;
the real difference is KV capacity.

## FP8 MTP=2 vs BF16 MTP=2

| Metric | BF16 MTP=2 | FP8 MTP=2 |
|---|---:|---:|
| Model load / card | 28.74 GiB | **27.96 GiB** |
| KV / card | 14.89 GiB | **15.67 GiB** |
| KV pool | 959,447 | **1,011,875** |
| 524K concurrency | 1.83x | **1.93x** |
| Single-stream | 123.4 tok/s | **124.3 tok/s** |
| 4-way aggregate | 369.8 | **387.4** |
| 8-way aggregate | 608.6 | **617.6** |
| 16-way aggregate | 632.3 | **635.3** |
| Draft accept length | ~2.0–2.2 | ~2.0–2.1 |

## MTP depth (BF16)

| `MTP_K` | KV pool | 524K concurrency | Single | 4-way | 8-way | 16-way |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | **1,319,176** | **2.52x** | 86.7 | 240.4 | 345.5 | 559.0 |
| 1 | 1,016,145 | 1.94x | 117.9 | 244.2 | 440.9 | 636.4 |
| 2 | 959,447 | 1.83x | **129.7** | 310.8 | 479.5 | **644.5** |
| 3 | 912,462 | 1.74x | 126.7 | 328.0 | 453.2 | 541.1 |

The FP8 MTP overlay was validated with K=2 only. K=0/1/3 remain available
but are not re-benchmarked with the overlay.

## Reproduce

Start the server, then:

```bash
# single stream, 3 runs after 1 warmup
python benchmarks/bench.py --runs 3 --max-tokens 256 --ignore-eos

# concurrency sweep
python benchmarks/bench-concurrency.py --levels 1,4,8,16 --max-tokens 256 --ignore-eos
```

For a fixed-length comparison, `--ignore-eos` is important; without it, the
model may stop early and the tok/s will not be comparable.

To record a result:

```bash
python benchmarks/bench-concurrency.py \
  --levels 1,4,8,16 --max-tokens 256 --ignore-eos \
  --output benchmarks/results/$(date +%F)-3x4090.json
```

## Raw result

The recorded production result is in
[`benchmarks/results/2026-10-25-3x4090-fp8-mtp2.json`](../benchmarks/results/2026-10-25-3x4090-fp8-mtp2.json).
It includes the hardware, software, flags and KV pool so a future run can be
compared fairly.

## What not to compare

- Do not compare a speculative run counted by SSE chunks with a non-spec run
  counted by `usage.completion_tokens`.
- Do not compare a 262K native run with a 524K YaRN run without labeling the
  context; KV per request is different.
- Do not compare a cold first request with a prefix-cache hit.
- Do not quote 16-way aggregate as per-user speed. At 16 streams the
  per-stream decode is lower; aggregate is higher.