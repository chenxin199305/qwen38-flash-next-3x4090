import importlib.util
from pathlib import Path

import pytest


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fp8_block_roundtrip():
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    module = load(Path(__file__).resolve().parents[1] / "scripts/02a-build-fp8hybrid.py")
    weight = torch.randn(256, 512, dtype=torch.bfloat16)
    quantized, scale, rel = module.quantize_block(weight)
    assert quantized.dtype == torch.float8_e4m3fn
    assert tuple(scale.shape) == (2, 4)
    restored = module.quantize_block(weight)[0].float() * scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    cosine = torch.nn.functional.cosine_similarity(
        restored.flatten(), weight.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999
    assert rel < 0.1


def test_mtp_block_roundtrip():
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    module = load(Path(__file__).resolve().parents[1] / "patches/02m-mtp-fp8.py")
    weight = torch.randn(256, 512, dtype=torch.bfloat16)
    quantized, scale = module.quantize_block(weight)
    restored = module.dequantize_block(quantized, scale)
    cosine = torch.nn.functional.cosine_similarity(
        restored.flatten(), weight.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999