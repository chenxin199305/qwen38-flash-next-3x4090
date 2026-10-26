#!/usr/bin/env python3
"""Lower the FLA shared-memory gate for sm_89 (RTX 4090).

vLLM's flash-linear-attention kernels fall back to small GDN tiles because the
gate asks for 102400 bytes while sm_89 exposes 101376 bytes (99 KiB) of
opt-in shared memory.  Lowering DEFAULT to 101376 lets the big-tile GDN path
run.  This is a performance patch, not a correctness patch.

Idempotent; backup: utils.py.bak-fla.
"""

from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path

OLD = "    DEFAULT = 102400"
NEW = "    DEFAULT = 101376  # 99 KiB opt-in shared memory (sm_89/sm_121)"
MARKER = "DEFAULT = 101376"


def patch_file(path: Path, check: bool) -> None:
    if not path.is_file():
        raise SystemExit(f"missing file: {path}")
    original = path.read_text()
    if MARKER in original:
        print(f"OK {path}")
        return
    if original.count(OLD) != 1:
        raise SystemExit(f"expected one FLA shmem anchor in {path}")
    if check:
        raise SystemExit(f"not patched: {path}")
    patched = original.replace(OLD, NEW, 1)
    ast.parse(patched, filename=str(path))
    backup = path.with_name(path.name + ".bak-fla")
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
    patch_file(root / "third_party/flash_linear_attention/ops/utils.py", args.check)


if __name__ == "__main__":
    main()