# Changelog

## v1.0.0 — 2026-10-26

Initial public release.

- Qwen3.8-Flash-Next NVFP4 on 3x RTX 4090 48GB, vLLM 0.30.0.
- TP3 + EP3: 171/171/170 complete experts per GPU, no 640→768 MoE padding.
- MARLIN EP top-k fix (`02k`), TP3 model patch (`02i`), QSA fused-draft
  metadata (`02g`), mamba prefix-cache block-size fix (`02h`), FLA shared
  memory gate (`02f`).
- Blockwise-FP8 side-layer overlay (`02a`) and FP8 MTP overlay (`02m`).
- Measured on the reference host: 1,011,875 KV tokens at 524K context
  (1.93x), 124.3 tok/s single-stream, 635.3 tok/s at 16 concurrent streams.
- Known limitations: text-only (vision unverified), 48GB cards required,
  FP8 MTP tested with K=2 only, systemd unit not validated on the reference
  host (no systemd).