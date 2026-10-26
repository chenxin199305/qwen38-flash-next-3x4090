#!/usr/bin/env python3
"""Fix NVFP4 MARLIN expert-parallel execution in vLLM 0.30.0."""

import argparse
import ast
import shutil
from pathlib import Path


MARLIN_REPLACEMENTS = [
    (
        """        top_k=1,
        mul_topk_weights=not apply_router_weight_on_input,
        b_q_type=quant_type,""",
        """        top_k=1,
        mul_topk_weights=(
            not apply_router_weight_on_input and expert_map is None
        ),
        b_q_type=quant_type,""",
    ),
    (
        """        is_zp_float=False,
    )

    return output""",
        """        is_zp_float=False,
    )

    if not apply_router_weight_on_input and expert_map is not None:
        output.mul_(topk_weights.reshape(-1, 1).to(output.dtype))

    return output""",
    ),
]


def patch_file(path: Path, replacements: list[tuple[str, str]], check: bool) -> None:
    original = path.read_text()
    patched = original
    for old, new in replacements:
        if new in patched:
            continue
        if patched.count(old) != 1:
            raise RuntimeError(f"expected one patch anchor in {path}: {old[:80]!r}")
        patched = patched.replace(old, new, 1)
    ast.parse(patched, filename=str(path))
    if patched == original:
        print(f"OK {path}")
        return
    if check:
        raise RuntimeError(f"not patched: {path}")
    backup = path.with_name(path.name + ".bak-ep")
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
        root / "model_executor/layers/fused_moe/experts/marlin_moe.py",
        MARLIN_REPLACEMENTS,
        args.check,
    )


if __name__ == "__main__":
    main()
