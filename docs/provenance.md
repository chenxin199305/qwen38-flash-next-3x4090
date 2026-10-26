# Provenance

## Source

This repository is a clean extraction of the 3x RTX 4090 TP3+EP3 deployment
from a private research repository. The private history and exact source
revision are intentionally withheld.

The public repository starts with a fresh history. It does not contain the
private repository's other platform branches, internal hostnames, tunnel or
scheduler configuration, or personal client setup.

## What was removed

- Internal IP addresses, hostnames, usernames and the exact source revision.
- Tunnel, scheduler and service-manager configuration.
- Other model deployments and platform branches.
- Personal client configuration and MCP paths.
- Model weights, generated overlays and logs.

## What was added

- `scripts/02a-build-fp8hybrid.py`: a non-destructive overlay builder for the
  156 GDN/QSA FP8 side-layer tensors (300 with shared experts, opt-in).
- `patches/02f-fla-shmem.py` and `patches/02h-mamba-block-size.py`: scripted
  versions of fixes that were previously manual.
- `install.sh`, `start.sh`, `stop.sh`, `check-weights.sh`, `doctor.sh`,
  `smoke.sh`, `benchmarks/` and CI.
- Public documentation focused on this single recipe.

## AI disclaimer

The README, docs, scripts and configuration were assembled by an AI assistant
from the deployment process and the measured results. The numbers come from
the reference host; the patches were checked against vLLM 0.30.0. The AI may
have introduced errors in prose, defaults or edge-case handling. If you find a
problem, please open an issue with the output of `./scripts/doctor.sh`.

## Third-party credits

- vLLM — Apache-2.0.
- blazux/qwen3.8-Flash-DGX — Apache-2.0 — mamba prefix-cache fix.
- Saren-Arterius/qwen3.8-Flash-DGX-AutoRound — Apache-2.0 — blockwise-FP8
  side-layer conversion pattern and hybrid dispatch shim.
- vLLM PR #58449 — QSA fused-draft metadata.
- vLLM issue #48895 — MARLIN EP top-k report.

See [NOTICE](../NOTICE) for the full list.