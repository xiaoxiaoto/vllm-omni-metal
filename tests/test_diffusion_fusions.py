# SPDX-License-Identifier: Apache-2.0
"""Independent numerical contracts for explicit single-dispatch diffusion fusions."""

import os

import pytest
import torch

from vllm_omni_metal.diffusion import ops
from vllm_omni_metal.ops import aot_kernels as kernels

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MPS = pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="explicit MPS integration opt-in required",
)


def adaln_reference(residual, branch, gate, weight, shift, scale, eps):
    summed = (residual.float() + branch.float() * gate.float()).to(residual.dtype)
    variance = summed.float().square().mean(-1, keepdim=True)
    norm = (summed.float() * torch.rsqrt(variance + eps)).to(residual.dtype)
    affine = (norm * weight).to(residual.dtype)
    factor = (1 + scale).to(residual.dtype)
    return ((affine * factor).to(residual.dtype) + shift).to(residual.dtype), summed


def qk_reference(x, weight, freqs, eps):
    norm = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    if weight is None:
        norm = norm.to(x.dtype)
    else:
        if weight.dtype in (torch.float16, torch.bfloat16):
            norm = norm.to(weight.dtype)
        norm = norm * weight
    pairs = norm.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2).contiguous()
    complex_result = torch.view_as_complex(pairs) * freqs.unsqueeze(-2)
    return torch.view_as_real(complex_result).flatten(-2).to(norm.dtype)


def adaln_inputs(dtype, device="cpu", width=37, broadcast=True):
    x = torch.randn(2, 3, width, dtype=dtype, device=device)
    branch = torch.randn_like(x)
    shapes = ((), (2, 1, width), (1, 3, 1)) if broadcast else (x.shape,) * 3
    gate, shift, scale = (torch.randn(shape, dtype=dtype, device=device) for shape in shapes)
    weight = torch.randn(width, dtype=dtype, device=device)
    return x, branch, gate, weight, shift, scale, 1e-6


def qk_inputs(dtype, device="cpu", width=30, batched=False):
    q = torch.randn(2, 3, 4, width, dtype=dtype, device=device)
    k = torch.randn(2, 3, 2, width, dtype=dtype, device=device)
    qw = torch.randn(width, dtype=dtype, device=device)
    kw = torch.randn_like(qw)
    angles = torch.randn((2, 3, width // 2) if batched else (3, width // 2), device=device)
    freqs = torch.polar(torch.ones_like(angles), angles)
    return q, k, qw, kw, freqs, 1e-6, 1e-5


def assert_unchanged(args, saved):
    for value, before in zip(args, saved, strict=True):
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, before, atol=0, rtol=0)


def save_inputs(args):
    return [x.clone() if isinstance(x, torch.Tensor) else x for x in args]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("broadcast", [False, True])
def test_adaln_cpu_rounding_broadcast_and_no_mutation(dtype, broadcast):
    torch.manual_seed(823)
    args = adaln_inputs(dtype, broadcast=broadcast)
    before = save_inputs(args)
    actual = ops.gated_residual_rms_adaln(*args)
    expected = adaln_reference(*args)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)
        assert value.dtype == dtype
    assert_unchanged(args, before)
    if dtype != torch.float32:
        x, branch, gate, _, _, _, _ = args
        unrounded = x.float() + branch.float() * gate.float()
        assert not torch.equal(unrounded.square().mean(-1), actual[1].float().square().mean(-1))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("weight_dtype", [None, *DTYPES])
def test_qk_cpu_gqa_independent_cast_contract(dtype, batched, weight_dtype):
    torch.manual_seed(418)
    q, k, qw, kw, freqs, qe, ke = qk_inputs(dtype, batched=batched)
    qw = None if weight_dtype is None else qw.to(weight_dtype)
    # Independent K weight dtype and epsilon, not Q's settings.
    kw = kw.float()
    args = q, k, qw, kw, freqs, qe, ke
    before = save_inputs(args)
    actual = ops.qk_norm_rotary(*args)
    expected = qk_reference(q, qw, freqs, qe), qk_reference(k, kw, freqs, ke)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)
        assert value.dtype == reference.dtype
    assert_unchanged(args, before)


@pytest.mark.parametrize("operator", ["adaln", "qk"])
@pytest.mark.parametrize("layout", ["offset", "strided", "grad"])
def test_cpu_views_and_gradients_match_oracle(operator, layout):
    torch.manual_seed(91)
    if operator == "adaln":
        args = list(adaln_inputs(torch.float32, width=12))
        indices = (0, 1, 3)
        entry, reference = ops.gated_residual_rms_adaln, adaln_reference
    else:
        args = list(qk_inputs(torch.float32, width=12))
        indices = (0, 1, 2, 3)
        entry = ops.qk_norm_rotary

        def reference(q, k, qw, kw, f, qe, ke):
            return qk_reference(q, qw, f, qe), qk_reference(k, kw, f, ke)

    for index in indices:
        x = args[index]
        if layout == "offset":
            args[index] = torch.cat((torch.zeros_like(x[:1]), x), dim=0)[1:]
        elif layout == "strided":
            args[index] = torch.stack((x, x), dim=-1)[..., 0]
        else:
            args[index] = x.requires_grad_()
    actual, expected = entry(*args), reference(*args)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
    if layout == "grad":
        inputs = [args[index] for index in indices]
        ag = torch.autograd.grad(sum(x.square().sum() for x in actual), inputs, retain_graph=True)
        eg = torch.autograd.grad(sum(x.square().sum() for x in expected), inputs)
        for a, e in zip(ag, eg, strict=True):
            torch.testing.assert_close(a, e, atol=0, rtol=0)


@pytest.mark.parametrize("operator", ["adaln", "qk"])
@pytest.mark.parametrize("width", [2, 65538])
def test_zero_rows_and_superwide_fallback(operator, width):
    if operator == "adaln":
        x = torch.empty(0, width)
        result = ops.gated_residual_rms_adaln(
            x, x, torch.tensor(1.0), torch.ones(width), torch.tensor(0.0), torch.tensor(0.0), 1e-6
        )
    else:
        q, k = torch.empty(0, 3, 4, width), torch.empty(0, 3, 2, width)
        result = ops.qk_norm_rotary(
            q, k, None, None, torch.ones(3, width // 2, dtype=torch.complex64)
        )
    assert all(x.numel() == 0 for x in result)
    if width > 65536:
        if operator == "adaln":
            args = adaln_inputs(torch.float32, width=width)
            actual, expected = ops.gated_residual_rms_adaln(*args), adaln_reference(*args)
        else:
            args = qk_inputs(torch.float32, width=width)
            q, k, qw, kw, f, qe, ke = args
            actual = ops.qk_norm_rotary(*args)
            expected = qk_reference(q, qw, f, qe), qk_reference(k, kw, f, ke)
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, e, atol=0, rtol=0)


@pytest.mark.parametrize("eps", [-1, float("inf"), float("nan"), None])
def test_fusions_reject_invalid_epsilon(eps):
    args = list(adaln_inputs(torch.float32))
    args[-1] = eps
    with pytest.raises(ValueError, match="epsilon"):
        ops.gated_residual_rms_adaln(*args)
    args = list(qk_inputs(torch.float32))
    args[-1] = eps
    with pytest.raises(ValueError, match="epsilon"):
        ops.qk_norm_rotary(*args)


@pytest.mark.parametrize("case", ["branch", "weight", "broadcast", "dtype", "integer", "width"])
def test_adaln_rejects_illegal_inputs(case):
    args = list(adaln_inputs(torch.float32))
    if case == "branch":
        args[1] = args[1][:1]
    elif case == "weight":
        args[3] = args[3][:1]
    elif case == "broadcast":
        args[2] = torch.ones(4, 2, 3, 37)
    elif case == "dtype":
        args[4] = args[4].half()
    elif case == "integer":
        args[0] = args[0].long()
    else:
        args[0] = args[0][..., :0]
    with pytest.raises(ValueError):
        ops.gated_residual_rms_adaln(*args)


@pytest.mark.parametrize(
    "case", ["sequence", "heads", "odd", "weight", "freqs", "freq_dtype", "dtype"]
)
def test_qk_rejects_illegal_inputs(case):
    args = list(qk_inputs(torch.float32))
    if case == "sequence":
        args[1] = args[1][:, :1]
    elif case == "heads":
        args[1] = args[1][:, :, :0]
    elif case == "odd":
        args[0], args[1] = args[0][..., :-1], args[1][..., :-1]
    elif case == "weight":
        args[2] = args[2][None]
    elif case == "freqs":
        args[4] = args[4][:1]
    elif case == "freq_dtype":
        args[4] = args[4].real
    else:
        args[1] = args[1].half()
    with pytest.raises(ValueError):
        ops.qk_norm_rotary(*args)


def test_fusion_baseline_metrics_and_no_library_probe(monkeypatch):
    def forbidden(*args):
        pytest.fail("baseline must not probe AOT")

    monkeypatch.setattr(kernels, "supports_gated_residual_rms_adaln", forbidden)
    monkeypatch.setattr(kernels, "supports_qk_norm_rotary", forbidden)
    ops.reset_metrics()
    with ops.use_optimizations(False):
        ops.gated_residual_rms_adaln(*adaln_inputs(torch.float32))
        ops.qk_norm_rotary(*qk_inputs(torch.float32))
    assert ops.metrics_snapshot() == {
        "gated_residual_rms_adaln:torch": 1,
        "qk_norm_rotary:torch": 1,
    }


@pytest.fixture
def explicit_fusion_shader(monkeypatch):
    libraries = {}

    def library(dtype):
        if dtype not in libraries:
            libraries[dtype] = torch.mps.compile_shader(
                f"#define DIFFUSION_ELT {kernels._SCALARS[dtype]}\n"
                + kernels._diffusion_source().read_text()
            )
        return libraries[dtype]

    monkeypatch.setattr(kernels, "_diffusion_library", library)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", [14, 257, 1536])
@pytest.mark.parametrize("broadcast", [False, True])
def test_single_dispatch_adaln_numerics_and_call_isolation(
    explicit_fusion_shader, dtype, width, broadcast
):
    torch.manual_seed(401)
    args = adaln_inputs(dtype, "mps", width=width, broadcast=broadcast)
    saved = save_inputs(args)
    ops.reset_metrics()
    first = ops.gated_residual_rms_adaln(*args)
    expected = adaln_reference(*args)
    second_args = adaln_inputs(dtype, "mps", width=width + 3)
    ops.gated_residual_rms_adaln(*second_args)
    assert ops.metrics_snapshot() == {"gated_residual_rms_adaln:aot": 2}
    # FP32 reduction trees differ; low precision permits at most its existing
    # RMSNorm tolerance, never a model-specific relaxed accuracy threshold.
    tolerance = 2e-6 if dtype == torch.float32 else 2 * torch.finfo(dtype).eps
    torch.testing.assert_close(first[0], expected[0], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(first[1], expected[1], atol=0, rtol=0)
    assert_unchanged(args, saved)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", [14, 96, 1536])
@pytest.mark.parametrize("batched", [False, True])
def test_single_dispatch_qk_gqa_numerics_and_isolation(
    explicit_fusion_shader, dtype, width, batched
):
    torch.manual_seed(992)
    args = qk_inputs(dtype, "mps", width=width, batched=batched)
    saved = save_inputs(args)
    q, k, qw, kw, f, qe, ke = args
    ops.reset_metrics()
    first = ops.qk_norm_rotary(*args)
    expected = qk_reference(q, qw, f, qe), qk_reference(k, kw, f, ke)
    ops.qk_norm_rotary(q, k, None, None, f, qe, ke)
    assert ops.metrics_snapshot() == {"qk_norm_rotary:aot": 2}
    tolerance = 2e-6 if dtype == torch.float32 else 2 * torch.finfo(dtype).eps
    for a, e in zip(first, expected, strict=True):
        torch.testing.assert_close(a, e, atol=tolerance, rtol=tolerance)
    assert_unchanged(args, saved)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ["missing", "offset", "strided", "grad", "subnormal_eps", "wide"])
def test_mps_fusion_legal_fallbacks_never_compile(monkeypatch, dtype, case):
    monkeypatch.setattr(torch.mps, "compile_shader", lambda *a: pytest.fail("request JIT"))

    def unavailable(_):
        if case != "missing":
            pytest.fail("unsupported inputs must not probe the AOT library")
        return None

    monkeypatch.setattr(kernels, "_diffusion_library", unavailable)
    width = 65538 if case == "wide" else 14
    aa = list(adaln_inputs(dtype, "mps", width=width))
    qa = list(qk_inputs(dtype, "mps", width=width))
    if case in ("offset", "strided", "grad"):
        for args in (aa, qa):
            x = args[0]
            if case == "offset":
                args[0] = torch.cat((x[:1], x), dim=0)[1:]
            elif case == "strided":
                args[0] = torch.stack((x, x), dim=-1)[..., 0]
            else:
                args[0] = x.requires_grad_()
    if case == "subnormal_eps":
        aa[-1], qa[-1] = 1e-40, 1e-40
    ops.reset_metrics()
    actual = ops.gated_residual_rms_adaln(*aa)
    for a, e in zip(actual, adaln_reference(*aa), strict=True):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
    q, k, qw, kw, f, qe, ke = qa
    actual = ops.qk_norm_rotary(*qa)
    for a, e in zip(actual, (qk_reference(q, qw, f, qe), qk_reference(k, kw, f, ke)), strict=True):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
    assert ops.metrics_snapshot() == {
        "gated_residual_rms_adaln:torch": 1,
        "qk_norm_rotary:torch": 1,
    }


@pytest.mark.parametrize("operator", ["adaln", "qk"])
def test_fusions_reject_cross_device_inputs(operator):
    if operator == "adaln":
        args = list(adaln_inputs(torch.float32))
        args[2] = torch.empty((), device="meta")
        entry = ops.gated_residual_rms_adaln
    else:
        args = list(qk_inputs(torch.float32))
        args[4] = torch.empty(args[4].shape, dtype=torch.complex64, device="meta")
        entry = ops.qk_norm_rotary
    with pytest.raises(ValueError, match="device"):
        entry(*args)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ["mixed_weight", "strided_frequency", "conjugate_frequency"])
def test_mps_qk_preserves_legal_frequency_and_weight_fallback(monkeypatch, dtype, case):
    monkeypatch.setattr(torch.mps, "compile_shader", lambda *a: pytest.fail("request JIT"))

    def forbidden(_):
        pytest.fail("unsupported tensors must not probe AOT")

    monkeypatch.setattr(kernels, "_diffusion_library", forbidden)
    q, k, qw, kw, freqs, qe, ke = qk_inputs(dtype, "mps")
    if case == "mixed_weight":
        qw = qw.to(torch.float16 if dtype == torch.float32 else torch.float32)
    elif case == "strided_frequency":
        freqs = torch.stack((freqs, freqs), dim=-1)[..., 0]
    else:
        freqs = freqs.conj()
    actual = ops.qk_norm_rotary(q, k, qw, kw, freqs, qe, ke)
    expected = qk_reference(q, qw, freqs, qe), qk_reference(k, kw, freqs, ke)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, atol=0, rtol=0)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", DTYPES)
def test_adaln_threadgroup_staging_width_boundary(explicit_fusion_shader, dtype, monkeypatch):
    limit = 7168 if dtype == torch.float32 else 14336
    torch.manual_seed(717)
    args = adaln_inputs(dtype, "mps", width=limit)
    ops.reset_metrics()
    actual = ops.gated_residual_rms_adaln(*args)
    expected = adaln_reference(*args)
    assert ops.metrics_snapshot() == {"gated_residual_rms_adaln:aot": 1}
    tolerance = 2e-6 if dtype == torch.float32 else 2 * torch.finfo(dtype).eps
    torch.testing.assert_close(actual[0], expected[0], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)

    def forbidden(_):
        pytest.fail("over-width row must not probe the AOT library")

    monkeypatch.setattr(kernels, "_diffusion_library", forbidden)
    args = adaln_inputs(dtype, "mps", width=limit + 1)
    ops.reset_metrics()
    actual = ops.gated_residual_rms_adaln(*args)
    expected = adaln_reference(*args)
    assert ops.metrics_snapshot() == {"gated_residual_rms_adaln:torch": 1}
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)
