#!/usr/bin/env python3
"""Build a non-destructive blockwise-FP8 overlay for Qwen3.8-Flash-Next.

The NVFP4 checkpoint keeps the routed experts in NVFP4 but leaves the
dense side layers in BF16.  Those layers are read in full on every decoded
token and dominate decode bandwidth.  By default this script converts the 156
GDN + QSA side-layer weights (36 GDN x 3 + 12 QSA x 4) to DeepSeek-style
blockwise FP8-e4m3.  `--include-shared-experts` adds the 144 shared-expert
weights (48 x 3), for 300 total.  The production A/B numbers in this repository
were measured with the 156-tensor layout; the 300-tensor layout is available
for experiments.

    weight          float8_e4m3fn   [out, in]
    weight_scale_inv float32         [out/128, in/128]

The source checkpoint is never modified.  Unchanged shards and small files are
symlinked into the destination; only shards containing target tensors are
rewritten.  The runtime shim in patches/vllm_fp8_hybrid_modelopt.py routes
those tensors to vLLM's blockwise Fp8Config while keeping the NVFP4 MoE path.

Usage:
    python 02a-build-fp8hybrid.py --source ... --dest ...
    python 02a-build-fp8hybrid.py --check
    python 02a-build-fp8hybrid.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

BLOCK = 128
FP8_MAX = 448.0
MANIFEST = "fp8hybrid-manifest.json"

LAYER = r"model\.language_model\.layers\.\d+\."
CORE_TARGETS = re.compile(
    LAYER
    + r"(?:linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)"
    + r"|self_attn\.(?:q_proj|k_proj|v_proj|o_proj))\.weight$"
)
SHARED_TARGETS = re.compile(
    LAYER + r"mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj)\.weight$"
)


def default_source() -> Path:
    return Path(os.environ.get(
        "MODEL_DIR", Path.home() / "models/Qwen3.8-Flash-Next-NVFP4"
    ))


def default_dest() -> Path:
    return Path(os.environ.get(
        "FP8HYBRID_DIR", Path.home() / "models/Qwen3.8-Flash-Next-NVFP4-fp8hybrid"
    ))


def quantize_block(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    out_features, in_features = weight.shape
    if out_features % BLOCK or in_features % BLOCK:
        raise ValueError(f"weight shape {tuple(weight.shape)} is not {BLOCK}-aligned")
    blocks = weight.float().reshape(
        out_features // BLOCK, BLOCK, in_features // BLOCK, BLOCK
    )
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_MAX
    quantized = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    restored = quantized.float() * scale
    rel = ((restored - blocks).abs().amax() / blocks.abs().amax()).item()
    return (
        quantized.reshape(out_features, in_features).contiguous(),
        scale.squeeze(1, 3).to(torch.float32).contiguous(),
        rel,
    )


def load_index(source: Path) -> dict:
    path = source / "model.safetensors.index.json"
    if not path.is_file():
        raise SystemExit(f"missing index: {path}")
    return json.loads(path.read_text())


def target_map(index: dict, include_shared: bool = False) -> dict[str, str]:
    out = {}
    for name, shard in index["weight_map"].items():
        if CORE_TARGETS.search(name) or (include_shared and SHARED_TARGETS.search(name)):
            out[name] = shard
    return out


def expected_count(include_shared: bool) -> int:
    return 300 if include_shared else 156


def link_or_keep(source: Path, dest: Path) -> None:
    target = dest / source.name
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise SystemExit(f"existing symlink points elsewhere: {target}")
        return
    if target.exists():
        raise SystemExit(f"unexpected existing file: {target}")
    target.symlink_to(source)


def build(source: Path, dest: Path, expect_targets: int, force: bool, include_shared: bool) -> None:
    index = load_index(source)
    targets = target_map(index, include_shared)
    if len(targets) != expect_targets:
        raise SystemExit(
            f"found {len(targets)} target tensors, expected {expect_targets}. "
            "Check the checkpoint layout or pass --expect-targets."
        )
    by_file: dict[str, list[str]] = {}
    for name, shard in targets.items():
        by_file.setdefault(shard, []).append(name)

    manifest_path = dest / MANIFEST
    if dest.exists() and any(dest.iterdir()) and not manifest_path.exists() and not force:
        raise SystemExit(
            f"destination is non-empty and has no {MANIFEST}: {dest}. "
            "Choose another directory or pass --force."
        )
    if force and dest.exists():
        for entry in list(dest.iterdir()):
            if entry.is_symlink() or entry.is_file():
                entry.unlink()
            else:
                raise SystemExit(f"refusing to remove directory: {entry}")
    dest.mkdir(parents=True, exist_ok=True)

    # Small files are shared with the source.
    for entry in source.iterdir():
        if entry.name == "model.safetensors.index.json" or entry.name.endswith(".safetensors"):
            continue
        link_or_keep(entry, dest)

    weight_map = dict(index["weight_map"])
    total_scale_bytes = 0
    worst = 0.0
    for i, (shard, names) in enumerate(sorted(by_file.items()), 1):
        src_path = source / shard
        if not src_path.is_file():
            raise SystemExit(f"missing shard: {src_path}")
        tensors = load_file(src_path)
        for name in names:
            if name not in tensors:
                raise SystemExit(f"missing tensor {name} in {shard}")
            weight = tensors.pop(name)
            quantized, scale, rel = quantize_block(weight)
            scale_name = name.replace(".weight", ".weight_scale_inv")
            tensors[name] = quantized
            tensors[scale_name] = scale
            weight_map[name] = shard
            weight_map[scale_name] = shard
            total_scale_bytes += scale.numel() * scale.element_size()
            worst = max(worst, rel)
        out_path = dest / shard
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        save_file(tensors, tmp_path, metadata={"format": "pt"})
        tmp_path.replace(out_path)
        print(f"[{i}/{len(by_file)}] {shard}: {len(names)} tensors")

    # Shards without target tensors are shared.
    for shard in sorted(set(index["weight_map"].values()) - set(by_file)):
        link_or_keep(source / shard, dest)

    metadata = dict(index.get("metadata", {}))
    if "total_size" in metadata:
        metadata["total_size"] = int(metadata["total_size"]) + total_scale_bytes
    new_index = {"metadata": metadata, "weight_map": weight_map}
    (dest / "model.safetensors.index.json").write_text(
        json.dumps(new_index, ensure_ascii=False, indent=2) + "\n"
    )

    manifest = {
        "source": str(source),
        "dest": str(dest),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_tensors": len(targets),
        "include_shared_experts": include_shared,
        "target_shards": len(by_file),
        "worst_rel_err": worst,
        "block": BLOCK,
        "dtype": "float8_e4m3fn",
        "scale_dtype": "float32",
        "scale_name": "weight_scale_inv",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"overlay ready: {dest} (worst rel err {worst:.4f})")


def check(source: Path, dest: Path, expect_targets: int, include_shared: bool) -> None:
    manifest_path = dest / MANIFEST
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("target_tensors") != expect_targets:
        raise SystemExit(f"manifest target count {manifest.get('target_tensors')} != {expect_targets}")
    manifest_shared = manifest.get("include_shared_experts")
    if manifest_shared is None:
        manifest_shared = manifest.get("target_tensors") == 300
    if bool(manifest_shared) != include_shared:
        raise SystemExit(
            "manifest layout mismatch: built with "
            f"include_shared_experts={manifest_shared}, "
            f"check requested {include_shared}"
        )

    index = json.loads((dest / "model.safetensors.index.json").read_text())
    targets = target_map(index, include_shared)
    if len(targets) != expect_targets:
        raise SystemExit(f"index has {len(targets)} targets, expected {expect_targets}")
    for name in targets:
        scale = name.replace(".weight", ".weight_scale_inv")
        if scale not in index["weight_map"]:
            raise SystemExit(f"missing scale entry: {scale}")
    missing = sorted({shard for shard in index["weight_map"].values() if not (dest / shard).exists()})
    if missing:
        raise SystemExit(f"missing shards in overlay: {missing[:5]}")
    print(
        f"OK {dest}: {len(targets)} targets, {manifest['target_shards']} rewritten shards, "
        f"worst rel err {manifest['worst_rel_err']:.4f}"
    )


def dry_run(source: Path, expect_targets: int, include_shared: bool) -> None:
    index = load_index(source)
    targets = target_map(index, include_shared)
    by_file: dict[str, int] = {}
    for shard in targets.values():
        by_file[shard] = by_file.get(shard, 0) + 1
    print(f"source: {source}")
    print(f"targets: {len(targets)} (expected {expect_targets})")
    print(f"rewritten shards: {len(by_file)}")
    for shard, count in sorted(by_file.items()):
        print(f"  {shard}: {count}")
    if len(targets) != expect_targets:
        raise SystemExit("target count mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--dest", type=Path, default=default_dest())
    parser.add_argument("--include-shared-experts", action="store_true",
                        help="also convert 144 shared-expert tensors (300 total); experimental")
    parser.add_argument("--expect-targets", type=int, default=None,
                        help="override the expected count (default: 156, or 300 with shared experts)")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    dest = args.dest.expanduser().resolve()
    if source == dest:
        raise SystemExit("source and destination must differ")
    if not (source / "model.safetensors.index.json").is_file():
        raise SystemExit(f"not a checkpoint: {source}")

    include_shared = args.include_shared_experts
    expect_targets = args.expect_targets or expected_count(include_shared)
    if args.dry_run:
        dry_run(source, expect_targets, include_shared)
    elif args.check:
        check(source, dest, expect_targets, include_shared)
    else:
        build(source, dest, expect_targets, args.force, include_shared)


if __name__ == "__main__":
    main()