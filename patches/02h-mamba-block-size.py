#!/usr/bin/env python3
"""Fix prefix caching for hybrid GDN/QSA models (mamba align mode).

EngineCore overwrites ``cache_config.block_size`` with the smallest KV-group
block size.  The mamba state-slot seed and the block-aligned prefill split
must use the mamba block size, not that minimum.  Without this fix a prefix
cache hit restores an all-zero Mamba state (wrong outputs), and chunk
boundaries rarely land on cacheable mamba boundaries (near-zero hit rate).

Ported from blazux/qwen3.8-Flash-DGX (Apache-2.0), adapted to vLLM 0.30.0.
Idempotent; backups: mamba_hybrid.py.bak-mamba, scheduler.py.bak-mamba.
"""

from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path

SEED_OLD = (
    "                (new_req_data.num_computed_tokens - 1) "
    "// self.cache_config.block_size\n"
)
SEED_NEW = (
    "                (new_req_data.num_computed_tokens - 1)\n"
    "                // (self.cache_config.mamba_block_size or self.cache_config.block_size)\n"
)
SCHED_OLD = (
    "        block_size = self.cache_config.block_size\n"
    "        # The last block-aligned position whose state can be cached."
)
SCHED_NEW = (
    "        block_size = self.block_size  # scheduler block size (LCM of groups) == mamba block size\n"
    "        # The last block-aligned position whose state can be cached."
)


def patch_file(path: Path, replacements: list[tuple[str, str]], check: bool, suffix: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing file: {path}")
    original = path.read_text()
    patched = original
    for old, new in replacements:
        if new in patched:
            continue
        if patched.count(old) != 1:
            raise SystemExit(f"expected one anchor in {path}: {old[:60]!r}")
        patched = patched.replace(old, new, 1)
    ast.parse(patched, filename=str(path))
    if patched == original:
        print(f"OK {path}")
        return
    if check:
        raise SystemExit(f"not patched: {path}")
    backup = path.with_name(path.name + suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(patched)
    print(f"PATCHED {path} (backup: {backup})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.vllm_root
    if root is None:
        import vllm

        if vllm.__version__ != "0.30.0":
            raise SystemExit(f"unsupported vLLM version: {vllm.__version__}; expected 0.30.0")
        root = Path(vllm.__file__).parent
    patch_file(
        root / "v1/worker/gpu/model_states/mamba_hybrid.py",
        [(SEED_OLD, SEED_NEW)],
        args.check,
        ".bak-mamba",
    )
    patch_file(
        root / "v1/core/sched/scheduler.py",
        [(SCHED_OLD, SCHED_NEW)],
        args.check,
        ".bak-mamba",
    )


if __name__ == "__main__":
    main()