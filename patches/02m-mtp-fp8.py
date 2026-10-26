#!/usr/bin/env python3
"""Quantize Qwen3.8 MTP experts to 128x128 block FP8 and build an overlay checkpoint.

The source is the FP8-hybrid checkpoint (scripts/02a-build-fp8hybrid.py).
The destination is a new overlay; the source is never modified.

Usage:
    python 02m-mtp-fp8.py --source ... --dest ...
"""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SOURCE = Path(os.environ.get(
    "FP8HYBRID_DIR", Path.home() / "models/Qwen3.8-Flash-Next-NVFP4-fp8hybrid"
))
DEST = Path(os.environ.get(
    "MTP_FP8_DIR", Path.home() / "models/Qwen3.8-Flash-Next-NVFP4-mtp-fp8hybrid"
))
MTP_FILE = "model-fp8-mtp.safetensors"
PRUNED_FILE = "model-bf16-00011-mtp-fp8.safetensors"
PACKED_KEYS = (
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "mtp.layers.0.mlp.experts.down_proj",
)
NUM_EXPERTS = 512
BLOCK = 128
FP8_MAX = 448.0


def quantize_block(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = weight.shape
    if out_features % BLOCK or in_features % BLOCK:
        raise ValueError(f"weight shape {weight.shape} is not {BLOCK}-aligned")
    blocks = weight.float().reshape(
        out_features // BLOCK, BLOCK, in_features // BLOCK, BLOCK
    )
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True)
    scale = (amax / FP8_MAX).clamp_min(1e-12)
    quantized = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quantized.reshape(out_features, in_features), scale.squeeze(1, 3).to(
        torch.bfloat16
    )


def dequantize_block(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return weight.float() * scale.float().repeat_interleave(BLOCK, 0).repeat_interleave(
        BLOCK, 1
    )


def expert_tensor_names(expert: int) -> dict[str, tuple[str, str]]:
    prefix = f"mtp.layers.0.mlp.experts.{expert}"
    return {
        "gate": (f"{prefix}.gate_proj.weight", f"{prefix}.gate_proj.weight_scale_inv"),
        "up": (f"{prefix}.up_proj.weight", f"{prefix}.up_proj.weight_scale_inv"),
        "down": (f"{prefix}.down_proj.weight", f"{prefix}.down_proj.weight_scale_inv"),
    }


def packed_shard_name(source: Path) -> str:
    index = json.loads((source / "model.safetensors.index.json").read_text())
    packed_files = {index["weight_map"][key] for key in PACKED_KEYS}
    if len(packed_files) != 1:
        raise RuntimeError(f"MTP packed tensors span unexpected shards: {packed_files}")
    return packed_files.pop()


def write_pruned_shard(source: Path, output: Path) -> int:
    packed_path = source / packed_shard_name(source)
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(packed_path, framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            if name not in PACKED_KEYS:
                tensors[name] = checkpoint.get_tensor(name)
    if any(name in tensors for name in PACKED_KEYS):
        raise RuntimeError("packed MTP tensors remain in pruned shard")
    temp_output = output.with_name(output.name + ".tmp")
    if temp_output.exists():
        temp_output.unlink()
    save_file(tensors, temp_output, metadata={"format": "pt"})
    temp_output.replace(output)
    print(
        f"wrote {output} ({output.stat().st_size / 2**30:.3f} GiB, "
        f"{len(tensors)} tensors)",
        flush=True,
    )
    return len(tensors)


def quantize_experts(source: Path, output: Path) -> int:
    packed_path = source / packed_shard_name(source)

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(packed_path, framework="pt", device="cpu") as checkpoint:
        gate_up = checkpoint.get_slice(PACKED_KEYS[0])
        down = checkpoint.get_slice(PACKED_KEYS[1])
        for expert in range(NUM_EXPERTS):
            names = expert_tensor_names(expert)
            packed = gate_up[expert]
            for part, weight in (("gate", packed[:640]), ("up", packed[640:])):
                weight_name, scale_name = names[part]
                quantized, scale = quantize_block(weight)
                tensors[weight_name] = quantized
                tensors[scale_name] = scale
            weight_name, scale_name = names["down"]
            quantized, scale = quantize_block(down[expert])
            tensors[weight_name] = quantized
            tensors[scale_name] = scale
            if expert % 64 == 0 or expert == NUM_EXPERTS - 1:
                print(f"quantized expert {expert}/{NUM_EXPERTS - 1}", flush=True)

    names = expert_tensor_names(0)
    with safe_open(packed_path, framework="pt", device="cpu") as checkpoint:
        packed = checkpoint.get_slice(PACKED_KEYS[0])[0]
        down = checkpoint.get_slice(PACKED_KEYS[1])[0]
    checks = (
        ("gate", packed[:640]),
        ("up", packed[640:]),
        ("down", down),
    )
    for part, original in checks:
        weight_name, scale_name = names[part]
        restored = dequantize_block(tensors[weight_name], tensors[scale_name])
        cosine = torch.nn.functional.cosine_similarity(
            restored.flatten(), original.float().flatten(), dim=0
        ).item()
        if cosine < 0.999:
            raise RuntimeError(f"expert 0 {part} FP8 cosine too low: {cosine}")
        print(f"verified expert 0 {part}: cosine={cosine:.6f}", flush=True)

    temp_output = output.with_name(output.name + ".tmp")
    if temp_output.exists():
        temp_output.unlink()
    save_file(tensors, temp_output, metadata={"format": "pt"})
    temp_output.replace(output)
    print(f"wrote {output} ({output.stat().st_size / 2**30:.3f} GiB)", flush=True)
    return len(tensors)


def build_quant_config(source_config: dict) -> tuple[dict, dict]:
    original = source_config["quantization_config"]
    ignore = [
        pattern
        for pattern in original["ignore"]
        if pattern not in {"mtp.*", "model.mtp.*"}
    ]
    target_experts = [
        f"model.language_model.layers.{layer}.mlp.experts" for layer in range(48)
    ]
    quantized_layers = {
        name: {"quant_algo": "NVFP4", "group_size": 16}
        for name in target_experts
    }
    quantized_layers["mtp.layers.0.mlp.experts"] = {
        "quant_algo": "FP8_PB_WO",
        "group_size": 128,
    }
    config = {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": False,
                    "group_size": 16,
                    "num_bits": 4,
                    "type": "float",
                },
                "targets": target_experts,
                "weights": {
                    "dynamic": False,
                    "group_size": 16,
                    "num_bits": 4,
                    "type": "float",
                },
            },
            "group_1": {
                "input_activations": {
                    "dynamic": True,
                    "group_size": 128,
                    "num_bits": 8,
                    "type": "float",
                },
                "targets": ["mtp.layers.0.mlp.experts"],
                "weights": {
                    "dynamic": False,
                    "group_size": 128,
                    "num_bits": 8,
                    "type": "float",
                },
            },
        },
        "ignore": ignore,
        "producer": original["producer"],
        "quant_algo": "MIXED_PRECISION",
        "quant_method": "modelopt",
        "quantized_layers": quantized_layers,
    }
    hf_quant = {
        "producer": original["producer"],
        "quantization": {
            "exclude_modules": ignore,
            "group_size": 16,
            "kv_cache_quant_algo": None,
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": quantized_layers,
        },
    }
    return config, hf_quant


def prepare_overlay(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    old_shard = packed_shard_name(source)
    reserved = {
        "config.json",
        "hf_quant_config.json",
        "model.safetensors.index.json",
        MTP_FILE,
        PRUNED_FILE,
        old_shard,
    }
    legacy_link = dest / old_shard
    if legacy_link.is_symlink():
        legacy_link.unlink()
    for entry in source.iterdir():
        if entry.name in reserved:
            continue
        target = dest / entry.name
        if target.is_symlink():
            if target.resolve() != entry.resolve():
                raise RuntimeError(f"unexpected existing symlink: {target}")
            continue
        if target.exists():
            raise RuntimeError(f"unexpected existing file: {target}")
        target.symlink_to(entry)


def build_index(source: Path, tensor_count: int) -> dict:
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    old_shard = packed_shard_name(source)
    packed_size = 0
    for key in PACKED_KEYS:
        if key not in weight_map:
            raise RuntimeError(f"missing packed MTP tensor: {key}")
        del weight_map[key]
    for key, shard in weight_map.items():
        if shard == old_shard:
            weight_map[key] = PRUNED_FILE
    packed_size += (512 * 1280 * 2560 + 512 * 2560 * 640) * 2
    for expert in range(NUM_EXPERTS):
        for weight_name, scale_name in expert_tensor_names(expert).values():
            weight_map[weight_name] = MTP_FILE
            weight_map[scale_name] = MTP_FILE
    quantized_size = 512 * 3 * 640 * 2560
    scale_size = 512 * 3 * 100 * 2
    if "metadata" in index and "total_size" in index["metadata"]:
        index["metadata"]["total_size"] = (
            index["metadata"]["total_size"] - packed_size + quantized_size + scale_size
        )
    if len([key for key in weight_map if key.startswith("mtp.")]) != tensor_count + 29:
        raise RuntimeError("unexpected MTP tensor count in index")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()
    source, dest = args.source.resolve(), args.dest.resolve()
    if source == dest:
        raise SystemExit("source and destination must differ")

    prepare_overlay(source, dest)
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"], hf_quant = build_quant_config(config)

    output = dest / MTP_FILE
    if output.exists():
        with safe_open(output, framework="pt", device="cpu") as checkpoint:
            tensor_count = len(checkpoint.keys())
        if tensor_count != NUM_EXPERTS * 6:
            raise RuntimeError(f"incomplete existing {output}: {tensor_count} tensors")
        print(f"reusing existing {output}", flush=True)
    else:
        tensor_count = quantize_experts(source, output)

    source_index = json.loads((source / "model.safetensors.index.json").read_text())
    old_shard = packed_shard_name(source)
    expected_pruned = (
        sum(shard == old_shard for shard in source_index["weight_map"].values())
        - len(PACKED_KEYS)
    )
    pruned = dest / PRUNED_FILE
    if pruned.exists():
        with safe_open(pruned, framework="pt", device="cpu") as checkpoint:
            keys = set(checkpoint.keys())
        if len(keys) != expected_pruned or keys.intersection(PACKED_KEYS):
            raise RuntimeError(f"incomplete existing {pruned}")
        print(f"reusing existing {pruned}", flush=True)
    else:
        write_pruned_shard(source, pruned)

    index = build_index(source, tensor_count)
    (dest / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    (dest / "hf_quant_config.json").write_text(
        json.dumps(hf_quant, ensure_ascii=False, indent=2) + "\n"
    )
    (dest / "model.safetensors.index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"overlay checkpoint ready: {dest}")


if __name__ == "__main__":
    main()