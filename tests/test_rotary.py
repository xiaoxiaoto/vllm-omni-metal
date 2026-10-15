# SPDX-License-Identifier: Apache-2.0
"""Scope and cache contracts for optional vpipe NeoX RoPE."""

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
