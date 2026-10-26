# Contributing

Thanks for testing this recipe.

## Before opening an issue

Run:

```bash
./scripts/doctor.sh
./check-weights.sh
```

Include the output (redact API keys, IPs and usernames), plus:

- GPU model and memory (`nvidia-smi -L`)
- driver and CUDA version
- vLLM and torch versions
- the exact `.env` values that differ from `.env.sample`
- the failing request and the relevant log lines

## Pull requests

- Keep the recipe narrow: Qwen3.8-Flash-Next NVFP4, 3x RTX 4090 48GB, vLLM
  0.30.0. Other hardware belongs in a separate repository or branch.
- Do not add model weights, generated overlays, logs or secrets.
- Run `bash -n` on shell scripts and `python -m py_compile` on Python files.
- If you change a patch anchor, update the corresponding `--check` path and
  document the vLLM version it applies to.
- Performance claims need a reproducible command and a JSON result under
  `benchmarks/results/`.

## AI-generated content

Much of this repository was assembled by an AI assistant from a real
deployment log. If a step is wrong or outdated, a precise issue is more useful
than a rewrite.