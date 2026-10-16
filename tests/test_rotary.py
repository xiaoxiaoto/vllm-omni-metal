# SPDX-License-Identifier: Apache-2.0
"""Scope and cache contracts for optional vpipe NeoX RoPE."""

import os
from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding, RotaryEmbeddingBase

from vllm_omni_metal import rotary


@pytest.mark.parametrize(
    "worker", [None, "vllm_omni_metal.native_worker.MetalARWorker", "other.Worker"]
)
def test_rotary_cpu_and_unavailable_artifact_retain_native(worker, monkeypatch):
    monkeypatch.setattr(
        rotary,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(parallel_config=SimpleNamespace(worker_cls=worker)),
    )
    with set_current_vllm_config(VllmConfig()):
        layer = rotary.MetalRotaryEmbedding(8, 8, 16, 10000, True, torch.float32)
    positions = torch.tensor([2, 1])
    q = torch.randn(2, 16)
    k = torch.randn(2, 8)
    expected = RotaryEmbedding.forward_native(layer, positions, q, k)
    actual = layer.forward_oot(positions, q, k)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_rotary_registration_is_oot_not_in_tree():
    from vllm.model_executor.custom_op import op_registry, op_registry_oot

    assert op_registry["rotary_embedding"] is RotaryEmbeddingBase
    assert op_registry_oot["RotaryEmbedding"] is rotary.MetalRotaryEmbedding
    assert rotary.MetalRotaryEmbedding.name == RotaryEmbedding.name


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="explicit MPS integration required",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_qk_adapter_preserves_scaled_cache_and_position_bounds(monkeypatch, dtype):
    from vllm_omni_metal import vpipe_kernels

    library = torch.mps.compile_shader(
        f"#define VPIPE_ELT {vpipe_kernels._SCALARS[dtype]}\n" + vpipe_kernels._SOURCE.read_text()
    )
    monkeypatch.setattr(vpipe_kernels, "_library", lambda _: library)
    monkeypatch.setattr(
        rotary,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(
                worker_cls="vllm_omni_metal.native_worker.MetalARWorker"
            )
        ),
    )
    with set_current_vllm_config(VllmConfig()):
        layer = rotary.MetalRotaryEmbedding(64, 64, 32, 10000, True, dtype)
    layer.cos_sin_cache = (layer.cos_sin_cache * 0.75).to("mps")
    positions = torch.tensor([3, 0, 3, 15], device="mps")
    q = torch.randn(4, 256, dtype=dtype, device="mps")
    k = torch.randn(4, 64, dtype=dtype, device="mps")
    expected = RotaryEmbedding.forward_native(layer, positions, q, k)
    actual = layer.forward_oot(positions, q, k)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)
    for invalid in (-1, 32):
        with pytest.raises((IndexError, RuntimeError)):
            layer.forward_oot(torch.tensor([invalid]), q[:1], k[:1])
