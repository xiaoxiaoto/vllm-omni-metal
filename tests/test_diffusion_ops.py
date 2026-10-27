# SPDX-License-Identifier: Apache-2.0
"""Diffusion math/cast contracts; isolated shaders are not production AOT evidence."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.normalization import RMSNorm

from vllm_omni_metal.diffusion import ops
from vllm_omni_metal.ops import aot_kernels as kernels

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
MPS = pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="explicit MPS integration opt-in required",
)


def reference_rotary(x, freqs):
    pairs = x.float().reshape(*x.shape[:-1], -1, 2).contiguous()
    return (
        torch.view_as_real(torch.view_as_complex(pairs) * freqs.unsqueeze(-2))
        .flatten(-2)
        .to(x.dtype)
    )


def reference_rms(x, weight, eps):
    module = RMSNorm(x.shape[-1], eps, elementwise_affine=weight is not None)
    if weight is not None:
        module.weight = torch.nn.Parameter(weight, requires_grad=weight.requires_grad)
    return module(x)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
def test_cpu_separate_silu_rounds_before_multiply(dtype, strided):
    torch.manual_seed(291)
    gate = torch.randn(2, 3, 34, dtype=dtype)
    up = torch.randn_like(gate)
    if strided:
        gate, up = gate[..., ::2], up[..., ::2]
    before = gate.clone(), up.clone()
    actual = ops.silu_mul(gate, up, _validated=True)
    torch.testing.assert_close(actual, F.silu(gate) * up, atol=0, rtol=0)
    for value, saved in zip((gate, up), before, strict=True):
        torch.testing.assert_close(value, saved, atol=0, rtol=0)
    assert actual.dtype == dtype and actual.device == gate.device


def test_silu_broadcast_and_mixed_dtype_keep_torch_semantics():
    gate = torch.randn(2, 7, dtype=torch.float16)
    up = torch.randn(7, dtype=torch.float32)
    torch.testing.assert_close(ops.silu_mul(gate, up), F.silu(gate) * up, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batched", [False, True])
def test_cpu_rotary_adjacent_pairs_and_strided_inputs(dtype, batched):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 2, 28, dtype=dtype)[..., ::2]
    angles = torch.randn((2, 3, 14) if batched else (3, 14))
    freqs = torch.polar(torch.ones_like(angles), angles)[..., ::2]
    saved = x.clone(), freqs.clone()
    actual = ops.rotary(x, freqs)
    torch.testing.assert_close(actual, reference_rotary(x, freqs), atol=0, rtol=0)
    for value, before in zip((x, freqs), saved, strict=True):
        torch.testing.assert_close(value, before, atol=0, rtol=0)
    assert actual.shape == x.shape and actual.dtype == dtype


def test_interleaved_is_not_neox_and_uses_fp32_complex_products():
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    freqs = torch.tensor([[1j, 1j]], dtype=torch.complex64)
    torch.testing.assert_close(ops.rotary(x, freqs), torch.tensor([[[[-2.0, 1.0, -4.0, 3.0]]]]))
    neox = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])
    assert not torch.equal(ops.rotary(x, freqs), neox)
    with pytest.raises(ValueError, match="complex frequencies"):
        ops.rotary(x, torch.ones(1, 4))
    with pytest.raises(ValueError, match="even"):
        ops.rotary(x[..., :3], freqs)
    torch.manual_seed(714)
    low = torch.randn(1, 3, 2, 18, dtype=torch.bfloat16)
    angles = torch.randn(3, 9)
    freqs = torch.polar(torch.ones_like(angles), angles)
    actual = ops.rotary(low, freqs)
    real, imag = low[..., ::2], low[..., 1::2]
    cosine, sine = freqs.real[None, :, None], freqs.imag[None, :, None]
    rounded = torch.stack(
        (
            (real * cosine).to(low.dtype) - (imag * sine).to(low.dtype),
            (imag * cosine).to(low.dtype) + (real * sine).to(low.dtype),
        ),
        dim=-1,
    ).flatten(-2)
    assert not torch.equal(actual, rounded)
    torch.testing.assert_close(actual, reference_rotary(low, freqs), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("weight_dtype", [None, *DTYPES])
def test_cpu_rms_matches_diffusers_cast_contract(dtype, weight_dtype):
    torch.manual_seed(771)
    x = torch.randn(2, 3, 38, dtype=dtype)[..., ::2]
    weight = None if weight_dtype is None else torch.randn(38, dtype=weight_dtype)[::2]
    saved = x.clone(), None if weight is None else weight.clone()
    actual = ops.rms_norm(x, weight, 1e-6)
    expected = reference_rms(x, weight, 1e-6)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(x, saved[0], atol=0, rtol=0)
    if weight is not None:
        torch.testing.assert_close(weight, saved[1], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_weight_cast_cannot_be_moved_after_affine(dtype):
    torch.manual_seed(859)
    x = torch.randn(2, 37, dtype=dtype)
    weight = torch.randn(37, dtype=dtype)
    normalized = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    actual = ops.rms_norm(x, weight, 1e-6)
    torch.testing.assert_close(actual, normalized.to(dtype) * weight, atol=0, rtol=0)
    assert not torch.equal(actual, (normalized * weight.float()).to(dtype))
    fp32_weight = weight.float()
    mixed = ops.rms_norm(x, fp32_weight, 1e-6)
    assert mixed.dtype == torch.float32
    torch.testing.assert_close(mixed, normalized * fp32_weight, atol=0, rtol=0)


def test_rms_eps_none_retains_diffusers_error_and_broadcast_weights():
    x = torch.randn(2, 7, dtype=torch.float16)
    weight = torch.tensor([1.75], dtype=torch.float32)
    torch.testing.assert_close(ops.rms_norm(x, weight, 1e-6), reference_rms(x, weight, 1e-6))
    with pytest.raises(TypeError):
        ops.rms_norm(x, None, None)


@pytest.mark.parametrize("operation", ["silu", "rotary", "rms"])
def test_cpu_gradients_match_original_expressions(operation):
    torch.manual_seed(611)
    x = torch.randn(2, 3, 2, 12, requires_grad=True)
    other = torch.randn_like(x, requires_grad=True)
    if operation == "silu":
        actual, expected = ops.silu_mul(x, other), F.silu(x) * other
        inputs = (x, other)
    elif operation == "rotary":
        freqs = torch.randn(3, 6, dtype=torch.complex64, requires_grad=True)
        actual, expected = ops.rotary(x, freqs), reference_rotary(x, freqs)
        inputs = (x, freqs)
    else:
        weight = torch.randn(12, requires_grad=True)
        actual = ops.rms_norm(x, weight, 1e-6)
        variance = x.float().square().mean(-1, keepdim=True)
        expected = x * torch.rsqrt(variance + 1e-6) * weight
        inputs = (x, weight)
    actual_grad = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), inputs)
    for value, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)


def test_metrics_and_nested_baseline_scope(monkeypatch):
    def forbidden(*args):
        pytest.fail("baseline must not probe optimized libraries")

    monkeypatch.setattr(ops, "_supports_separate_silu", forbidden)
    monkeypatch.setattr(ops, "_supports_fused_rms", forbidden)
    monkeypatch.setattr(kernels, "supports_interleaved_rotary", forbidden)
    x = torch.ones(1, 2, 1, 4)
    freqs = torch.ones(2, 2, dtype=torch.complex64)
    ops.reset_metrics()
    with ops.use_optimizations(False):
        with ops.use_optimizations(True):
            assert ops._ENABLED.get()
        assert not ops._ENABLED.get()
        ops.silu_mul(x, x)
        ops.rotary(x, freqs)
        ops.rms_norm(x, None, 1e-6)
    assert ops._ENABLED.get()
    assert ops.metrics_snapshot() == {
        "silu_mul:torch": 1,
        "rotary:torch_complex": 1,
        "rms_norm:torch": 1,
    }
    snapshot = ops.metrics_snapshot()
    snapshot.clear()
    assert ops.metrics_snapshot()
    ops.reset_metrics()
    assert ops.metrics_snapshot() == {}


def test_missing_rotary_aot_does_not_compile(monkeypatch, tmp_path):
    monkeypatch.setattr(kernels, "_diffusion_artifact_path", lambda _: tmp_path / "absent.metallib")
    monkeypatch.setattr(torch.mps, "compile_shader", lambda *a: pytest.fail("runtime JIT"))
    monkeypatch.setattr(torch.mps, "load_metallib", lambda *a: pytest.fail("missing artifact load"))
    kernels._diffusion_library.cache_clear()
    try:
        assert kernels._diffusion_library(torch.float16) is None
    finally:
        kernels._diffusion_library.cache_clear()


def test_explicit_diffusion_builder_missing_toolchain_is_actionable(monkeypatch):
    monkeypatch.setattr(
        kernels.psutil, "virtual_memory", lambda: SimpleNamespace(available=4 * 1024**3)
    )
    monkeypatch.setattr(kernels.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024**3))
    monkeypatch.setattr(kernels.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="already installed Apple Metal compiler"):
        kernels.build_diffusion_kernels()


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("weight_dtype", [None, *DTYPES])
def test_mps_rms_fused_matches_diffusers_not_vllm(dtype, weight_dtype):
    torch.manual_seed(771)
    x = torch.randn(2, 3, 257, dtype=dtype, device="mps")
    weight = None if weight_dtype is None else torch.randn(257, dtype=weight_dtype, device="mps")
    before = x.clone()
    expected = reference_rms(x, weight, 1e-6)
    ops.reset_metrics()
    actual = ops.rms_norm(x, weight, 1e-6)
    assert ops.metrics_snapshot() == {"rms_norm:torch_mps_fused": 1}
    assert actual.dtype == expected.dtype
    tolerance = 2e-6 if actual.dtype == torch.float32 else 2 * torch.finfo(actual.dtype).eps
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(x, before, atol=0, rtol=0)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", DTYPES)
def test_mps_missing_aot_strides_and_grad_preserve_torch(monkeypatch, dtype):
    monkeypatch.setattr(kernels, "_library", lambda _: None)
    monkeypatch.setattr(kernels, "_diffusion_library", lambda _: None)
    monkeypatch.setattr(torch.mps, "compile_shader", lambda *a: pytest.fail("runtime JIT"))
    torch.manual_seed(292)
    for strided, grad in [(False, False), (True, False), (False, True)]:
        x = torch.randn(2, 3, 2, 28, dtype=dtype, device="mps", requires_grad=grad)
        if strided:
            x = x[..., ::2]
        up = torch.randn_like(x)
        angles = torch.randn(3, x.shape[-1] // 2, device="mps")
        freqs = torch.polar(torch.ones_like(angles), angles)
        weight = torch.randn(x.shape[-1], dtype=dtype, device="mps", requires_grad=grad)
        ops.reset_metrics()
        actual = ops.silu_mul(x, up)
        rotated = ops.rotary(x, freqs)
        normalized = ops.rms_norm(x, weight, 1e-6)
        torch.testing.assert_close(actual, F.silu(x) * up, atol=0, rtol=0)
        torch.testing.assert_close(rotated, reference_rotary(x, freqs), atol=0, rtol=0)
        rms_tol = 2e-6 if dtype == torch.float32 else 2 * torch.finfo(dtype).eps
        torch.testing.assert_close(
            normalized, reference_rms(x, weight, 1e-6), atol=rms_tol, rtol=rms_tol
        )
        assert ops.metrics_snapshot()["silu_mul:torch"] == 1
        assert ops.metrics_snapshot()["rotary:torch_complex"] == 1
        if strided or grad:
            assert ops.metrics_snapshot()["rms_norm:torch"] == 1
        if grad:
            (actual.float().sum() + rotated.float().sum() + normalized.float().sum()).backward()
            assert x.grad is not None and weight.grad is not None


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", DTYPES)
def test_mps_large_separate_silu_reuses_mlx_without_packing(monkeypatch, dtype):
    monkeypatch.setattr(kernels, "_library", lambda _: None)
    monkeypatch.setattr(torch, "cat", lambda *a, **k: pytest.fail("unnecessary packed activation"))
    gate = torch.randn(2, 128, 1024, dtype=dtype, device="mps")
    up = torch.randn_like(gate)
    before = gate.clone(), up.clone()
    expected = F.silu(gate) * up
    ops.reset_metrics()
    actual = ops.silu_mul(gate, up)
    assert ops.metrics_snapshot() == {"silu_mul:mlx": 1}
    tolerance = 2e-6 if dtype == torch.float32 else 2 * torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for value, saved in zip((gate, up), before, strict=True):
        torch.testing.assert_close(value, saved, atol=0, rtol=0)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", [6, 14, 128])
def test_explicit_test_shaders_separate_silu_and_interleaved_rotary(monkeypatch, dtype, head_dim):
    # Explicit isolated numerical test only; this is NOT the production AOT loader.
    elementwise = torch.mps.compile_shader(
        f"#define ELEMENTWISE_ELT {kernels._SCALARS[dtype]}\n" + kernels._SOURCE.read_text()
    )
    rotary_library = torch.mps.compile_shader(
        f"#define DIFFUSION_ELT {kernels._SCALARS[dtype]}\n"
        + kernels._diffusion_source().read_text()
    )
    monkeypatch.setattr(kernels, "_library", lambda _: elementwise)
    monkeypatch.setattr(kernels, "_diffusion_library", lambda _: rotary_library)
    torch.manual_seed(73)
    x = torch.randn(2, 3, 2, head_dim, dtype=dtype, device="mps")
    up = torch.randn_like(x)
    before = x.clone(), up.clone()
    ops.reset_metrics()
    actual = ops.silu_mul(x, up)
    tolerance = 2e-6 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual, F.silu(x) * up, atol=tolerance, rtol=tolerance)
    for shape in [(3, head_dim // 2), (1, 3, head_dim // 2), (2, 3, head_dim // 2)]:
        angles = torch.randn(shape, device="mps")
        freqs = torch.polar(torch.ones_like(angles), angles)
        saved_freqs = freqs.clone()
        # Torch producer and consumer share the same stream with the AOT-style launch.
        result = ops.rotary(x * 0.75, freqs) + 1
        expected = reference_rotary(x * 0.75, freqs) + 1
        torch.testing.assert_close(result, expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(freqs, saved_freqs, atol=0, rtol=0)
    assert ops.metrics_snapshot() == {"silu_mul:aot": 1, "rotary:aot": 3}
    for value, saved in zip((x, up), before, strict=True):
        torch.testing.assert_close(value, saved, atol=0, rtol=0)


@pytest.mark.integration
@MPS
def test_mps_conjugate_and_strided_frequencies_never_use_raw_storage(monkeypatch):
    monkeypatch.setattr(
        kernels, "_diffusion_library", lambda _: pytest.fail("unsupported view must not load AOT")
    )
    x = torch.randn(1, 3, 2, 14, device="mps")
    angles = torch.randn(3, 14, device="mps")
    frequencies = torch.polar(torch.ones_like(angles), angles)
    for freqs in (frequencies[:, :7].contiguous().conj(), frequencies[:, ::2]):
        actual = ops.rotary(x, freqs)
        torch.testing.assert_close(actual, reference_rotary(x, freqs), atol=0, rtol=0)
