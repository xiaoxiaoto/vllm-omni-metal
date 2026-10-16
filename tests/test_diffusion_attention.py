# SPDX-License-Identifier: Apache-2.0
"""Small real-MPS diffusion SDPA regressions; no model downloads."""

import importlib
import os

import pytest
import torch
from vllm_omni.diffusion.attention.backends import abstract

from vllm_omni_metal.platform import MetalOmniPlatform

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1"
        or not torch.backends.mps.is_available(),
        reason="explicit MPS integration opt-in required",
    ),
]


@pytest.fixture(autouse=True)
def metal_platform(monkeypatch):
    # Exercise the actual OOT dispatcher, independently of process plugin discovery.
    monkeypatch.setattr(abstract, "current_omni_platform", MetalOmniPlatform())


def _implementation(heads, kv_heads, head_size=8, scale=0.37, causal=False):
    path = MetalOmniPlatform.get_diffusion_attn_backend_cls(None, head_size)
    module_name, class_name = path.rsplit(".", 1)
    backend = getattr(importlib.import_module(module_name), class_name)
    return backend.get_impl_cls()(
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_size=head_size,
        softmax_scale=scale,
        causal=causal,
    )


def _inputs(dtype, heads, kv_heads, noncontiguous=False):
    generator = torch.Generator(device="cpu").manual_seed(173)
    width = 16 if noncontiguous else 8
    tensors = []
    for length, count in ((3, heads), (5, kv_heads), (5, kv_heads)):
        tensor = torch.randn(2, length, count, width, generator=generator).to(dtype).to("mps")
        if noncontiguous:
            tensor = tensor[..., ::2]
            assert not tensor.is_contiguous()
        tensors.append(tensor)
    return tuple(tensors)


def _mask(kind, dtype):
    if kind == "none":
        return None
    keep = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    if kind == "bool":
        return keep.to("mps")
    # A full-rank additive mask must remain additive, not become a boolean mask.
    bias = torch.arange(15, dtype=torch.float32).reshape(1, 1, 3, 5) * 0.07
    bias = bias.expand(2, 1, 3, 5).clone()
    return bias.masked_fill(~keep[:, None, None, :], float("-inf")).to(dtype).to("mps")


def _reference(query, key, value, mask, scale, causal):
    """Independent CPU FP64 score/softmax/value calculation in BSHD layout."""
    query, key, value = (
        tensor.detach().cpu().double().permute(0, 2, 1, 3) for tensor in (query, key, value)
    )
    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    scores = (query @ key.transpose(-2, -1)) * scale
    if mask is not None:
        mask = mask.detach().cpu()
        if mask.ndim == 2:
            mask = mask[:, None, None, :]
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask.double()
    if causal:
        allowed = torch.ones(query.shape[2], key.shape[2], dtype=torch.bool).tril()
        scores = scores.masked_fill(~allowed, float("-inf"))
    return (scores.softmax(dim=-1) @ value).permute(0, 2, 1, 3)


@pytest.mark.parametrize(
    "dtype,heads,kv_heads,mask_kind,causal,noncontiguous",
    [
        pytest.param(torch.float32, 2, 2, "none", False, False, id="fp32-mha"),
        pytest.param(torch.float16, 4, 2, "bool", False, True, id="fp16-gqa-padding-strided"),
        pytest.param(torch.bfloat16, 2, 2, "additive", False, False, id="bf16-mha-additive"),
        pytest.param(torch.float32, 4, 1, "none", True, True, id="fp32-gqa-causal-strided"),
        pytest.param(torch.float16, 2, 2, "none", True, False, id="fp16-mha-causal"),
        pytest.param(torch.bfloat16, 4, 2, "bool", False, True, id="bf16-gqa-padding-strided"),
    ],
)
def test_platform_forward_matches_independent_reference(
    dtype, heads, kv_heads, mask_kind, causal, noncontiguous
):
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

    query, key, value = _inputs(dtype, heads, kv_heads, noncontiguous)
    mask = _mask(mask_kind, dtype)
    metadata = AttentionMetadata(attn_mask=mask) if mask is not None else None
    tensors = (query, key, value) + ((mask,) if mask is not None else ())
    saved = [tensor.cpu().clone() for tensor in tensors]
    expected = _reference(query, key, value, mask, scale=0.37, causal=causal)
    impl = _implementation(heads, kv_heads, causal=causal)

    # Before the fix this public call raises "No forward implementation for platform".
    actual = impl.forward(query, key, value, metadata)

    assert actual.shape == query.shape
    assert actual.device == query.device
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all().item()
    tolerance = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(actual.cpu().double(), expected, atol=tolerance, rtol=tolerance)
    for tensor, original in zip(tensors, saved, strict=True):
        torch.testing.assert_close(tensor.cpu(), original, atol=0, rtol=0)

    if mask is not None:
        # Changing only padding must have no effect, for boolean AND additive masks.
        changed_key, changed_value = key.clone(), value.clone()
        changed_key[0, 3:] = 50
        changed_value[0, 3:] = -40
        changed_key[1, 4:] = -50
        changed_value[1, 4:] = 40
        changed = impl.forward(query, changed_key, changed_value, metadata)
        torch.testing.assert_close(changed.cpu(), actual.cpu(), atol=tolerance, rtol=tolerance)
        for tensor, original in zip(tensors, saved, strict=True):
            torch.testing.assert_close(tensor.cpu(), original, atol=0, rtol=0)


def test_platform_forward_rejects_nondivisible_gqa():
    query, key, value = _inputs(torch.float32, heads=3, kv_heads=2)
    # Some Omni releases validate here; others let Torch reject the head geometry.
    with pytest.raises((ValueError, RuntimeError), match=r"(?i)head|divis|multiple|gqa"):
        _implementation(3, 2).forward(query, key, value)


def test_metal_backend_preserves_upstream_dispatch_and_registry():
    from vllm_omni.diffusion.attention.backends.abstract import AttentionImpl
    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl

    upstream_forward = SDPAImpl.forward
    registry_paths = {backend: backend.get_path() for backend in DiffusionAttentionBackendEnum}
    query, key, value = _inputs(torch.float32, heads=2, kv_heads=2)
    impl = _implementation(2, 2)
    output = impl.forward(query, key, value)
    assert torch.isfinite(output).all().item()

    assert SDPAImpl.forward is upstream_forward
    assert SDPAImpl.forward is AttentionImpl.forward
    assert SDPABackend.get_impl_cls() is SDPAImpl
    assert DiffusionAttentionBackendEnum.TORCH_SDPA.get_class() is SDPABackend
    assert {
        backend: backend.get_path() for backend in DiffusionAttentionBackendEnum
    } == registry_paths
    # Metal's local subclass must not make unmodified upstream SDPA dispatch as CUDA.
    upstream = SDPAImpl(num_heads=2, num_kv_heads=2, head_size=8, softmax_scale=0.37)
    with pytest.raises(NotImplementedError, match="No forward implementation for platform"):
        upstream.forward(query, key, value)
