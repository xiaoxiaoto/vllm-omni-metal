# SPDX-License-Identifier: Apache-2.0
"""Explicit diffusion operators; no model patching or runtime Metal compilation.

RoPE uses adjacent real/imaginary pairs and FP32 complex multiplication, not
NeoX's split-half rotation or its low-precision product rounding. RMSNorm follows
diffusers' FP32 variance and weight-dependent cast, not vLLM's unconditional cast.
"""

import math
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.nn.functional as F

from vllm_omni_metal.ops import aot_kernels

_COUNTS = Counter()
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MIN_COMPILED_ELEMENTS = 262144
_ENABLED = ContextVar("diffusion_ops_enabled", default=True)


def metrics_snapshot():
    """Snapshot successful calls by ``operator:implementation`` in this process."""
    return dict(_COUNTS)


def reset_metrics():
    """Clear only these explicit operator entrypoints' process-local counters."""
    _COUNTS.clear()


@contextmanager
def use_optimizations(enabled):
    """Locally select optimized or exact Torch math for an explicit A/B scope."""
    token = _ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _ENABLED.reset(token)


def _record(operator, implementation, result):
    _COUNTS[f"{operator}:{implementation}"] += 1
    return result


def _supports_separate_silu(gate, up):
    return (
        gate.device.type == "mps"
        and gate.dtype in _DTYPES
        and up.device == gate.device
        and up.dtype == gate.dtype
        and up.shape == gate.shape
        and gate.is_contiguous()
        and up.is_contiguous()
        and not gate.is_neg()
        and not up.is_neg()
        and not gate.requires_grad
        and not up.requires_grad
        and 0 < gate.numel() < 2**31
    )


def silu_mul(gate, up, *, _validated=False):
    """Return ``F.silu(gate) * up`` without packing or mutating either input.

    An existing AOT artifact takes priority. Otherwise the existing MLX compiled
    operation is used for large contiguous inputs. ``_validated=True`` lets an
    external calibrator opt smaller inputs into that bridge; it never bypasses
    dtype, layout, device, shape, or autograd checks. Broadcasting and mixed
    dtypes retain the original Torch expression on the original device.
    """
    if _ENABLED.get() and _supports_separate_silu(gate, up):
        if aot_kernels.supports_separate_silu(gate, up):
            result = aot_kernels.separate_silu_mul(gate, up)
            return _record("silu_mul", "aot", result)
        if _validated or gate.numel() >= _MIN_COMPILED_ELEMENTS:
            import mlx.core as mx
            from vllm_metal.tensor_bridge import mlx_to_torch, torch_to_mlx

            from vllm_omni_metal.ops.activation import _compiled_silu_and_mul

            # Retain both borrowed Torch allocations until the lazy operation completes.
            shared_gate, shared_up = torch_to_mlx(gate), torch_to_mlx(up)
            result = _compiled_silu_and_mul()(shared_gate, shared_up)
            mx.eval(result)
            return _record("silu_mul", "mlx", mlx_to_torch(result, "mps"))
    return _record("silu_mul", "torch", F.silu(gate) * up)


def rotary(x, freqs):
    """Interleaved complex RoPE, returning x's dtype without changing its inputs.

    For BSHD activations, frequencies have shape [S,D/2] or [B,S,D/2].
    More general broadcast-compatible frequency shapes use the same Torch
    complex expression. Real cos/sin tables are deliberately not accepted.
    Strided inputs and gradients are supported by the original Torch math.
    """
    if not freqs.is_complex():
        raise ValueError("rotary requires complex frequencies, not a NeoX cos/sin cache")
    if x.ndim == 0 or x.shape[-1] % 2:
        raise ValueError("rotary requires an even last dimension")
    if _ENABLED.get() and aot_kernels.supports_interleaved_rotary(x, freqs):
        return _record("rotary", "aot", aot_kernels.interleaved_rotary(x, freqs))
    return _record("rotary", "torch_complex", _torch_rotary(x, freqs))


def _torch_rotary(x, freqs):
    pairs = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    # view_as_complex requires even outer strides, including for FP32 views.
    if pairs.stride(-1) != 1 or any(stride % 2 for stride in pairs.stride()[:-1]):
        pairs = pairs.contiguous()
    rotated = torch.view_as_complex(pairs) * freqs.unsqueeze(-2)
    result = torch.view_as_real(rotated).flatten(-2).to(x.dtype)
    return result


def _supports_fused_rms(x, weight, eps):
    return (
        x.device.type == "mps"
        and x.dtype in _DTYPES
        and x.ndim > 0
        and x.numel() > 0
        and x.is_contiguous()
        and not x.requires_grad
        and aot_kernels._supports_add_rms_norm_epsilon(eps)
        and (
            weight is None
            or (
                weight.device == x.device
                and weight.dtype in _DTYPES
                and weight.shape == (x.shape[-1],)
                and weight.is_contiguous()
                and not weight.requires_grad
            )
        )
    )


def rms_norm(x, weight, eps):
    """Diffusers RMSNorm (no bias): FP32 variance, conditional cast, then weight.

    Low-precision weights cause a cast BEFORE the learned multiplication; FP32
    weights keep the normalized activation in FP32 even when x is FP16/BF16.
    No-weight results return x's dtype. The MPS fused normalization is called
    WITHOUT a learned weight so its internal affine cast order cannot leak into
    this contract. Other legal inputs retain the decomposed Torch semantics.
    """
    if _ENABLED.get() and _supports_fused_rms(x, weight, eps):
        # Only normalize in x's dtype when that is the required pre-affine dtype.
        normalized_dtype = x.dtype if weight is None else weight.dtype
        source = x if normalized_dtype == x.dtype else x.float()
        normalized = F.rms_norm(source, (x.shape[-1],), None, eps)
        if weight is not None:
            if weight.dtype in (torch.float16, torch.bfloat16):
                normalized = normalized.to(weight.dtype)
            normalized = normalized * weight
        return _record("rms_norm", "torch_mps_fused", normalized)
    return _record("rms_norm", "torch", _torch_rms_norm(x, weight, eps))


def _torch_rms_norm(x, weight, eps):
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    if weight is not None:
        if weight.dtype in (torch.float16, torch.bfloat16):
            normalized = normalized.to(weight.dtype)
        normalized = normalized * weight
    else:
        normalized = normalized.to(x.dtype)
    return normalized


def _validate_epsilon(eps):
    if not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps < 0:
        raise ValueError("epsilon must be a finite nonnegative number")


def _validate_tensor(value, reference, name):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.dtype not in _DTYPES or value.device != reference.device:
        raise ValueError(f"{name} must have a supported floating dtype on the input device")


def gated_residual_rms_adaln(residual, branch, gate, weight, shift, scale, eps):
    """Return (AdaLN result, rounded residual), with no input mutation.

    ``residual`` and ``branch`` have identical nonempty last dimensions and
    dtype/device. Weight is [D]; gate/shift/scale use right-aligned Torch
    broadcasting, without expanding the residual shape. All share its dtype.
    Zero rows are legal. Gradients and strided/offset views retain Torch math.

    First compute ``(residual.float() + branch.float() * gate.float()).to(dtype)``.
    Variance is FP32 over that ROUNDED residual, unlike vLLM add+RMSNorm.
    Normalize in FP32, cast to source dtype, multiply weight and round again.
    ``1 + scale`` is rounded to source dtype; its product with the affine
    result is rounded before adding shift, then the sum is rounded once more.
    These are ordinary Torch operation boundaries, not fused FP32 affine math.
    AOT retains the rounded row in 28 KiB of threadgroup storage, with no
    global-memory reread. Its maximum width is 7168 for FP32, 14336 for
    FP16/BF16; larger legal widths use the exact Torch path.
    """
    if not isinstance(residual, torch.Tensor) or residual.dtype not in _DTYPES:
        raise ValueError("residual must have FP16, BF16 or FP32 dtype")
    if residual.ndim == 0 or residual.shape[-1] == 0:
        raise ValueError("residual requires a positive last dimension")
    _validate_epsilon(eps)
    for name, value in (
        ("branch", branch),
        ("gate", gate),
        ("weight", weight),
        ("shift", shift),
        ("scale", scale),
    ):
        _validate_tensor(value, residual, name)
        if value.dtype != residual.dtype:
            raise ValueError(f"{name} must have residual dtype")
    if branch.shape != residual.shape or weight.shape != (residual.shape[-1],):
        raise ValueError("branch must match residual shape and weight must be [D]")
    for value in (gate, shift, scale):
        try:
            shape = torch.broadcast_shapes(value.shape, residual.shape)
        except RuntimeError as error:
            raise ValueError("gate/shift/scale must broadcast to residual") from error
        if shape != residual.shape:
            raise ValueError("gate/shift/scale must not expand residual shape")
    args = residual, branch, gate, weight, shift, scale, eps
    if _ENABLED.get() and aot_kernels.supports_gated_residual_rms_adaln(*args):
        return _record(
            "gated_residual_rms_adaln", "aot", aot_kernels.gated_residual_rms_adaln(*args)
        )
    rounded = (residual.float() + branch.float() * gate.float()).to(residual.dtype)
    inverse = torch.rsqrt(rounded.float().square().mean(-1, keepdim=True) + eps)
    normalized = (rounded.float() * inverse).to(residual.dtype) * weight
    result = normalized * (1 + scale) + shift
    return _record("gated_residual_rms_adaln", "torch", (result, rounded))


def qk_norm_rotary(query, key, query_weight, key_weight, freqs, query_eps=1e-6, key_eps=1e-6):
    """RMSNorm then interleaved complex RoPE for BSHD Q/K, including GQA.

    Q/K share B,S,D and dtype/device, with independently positive head counts.
    Each optional weight is [D], independently FP16/BF16/FP32 on that device.
    Each RMSNorm follows :func:`rms_norm`'s conditional pre-weight cast;
    its output dtype is retained by :func:`rotary`, which computes complex
    products in FP32 and finally casts back to that norm output dtype.
    Frequencies are complex64 [S,D/2], [1,S,D/2] or [B,S,D/2].
    This is not NeoX. Legal views and autograd use the exact Torch path.
    """
    if not isinstance(query, torch.Tensor) or query.dtype not in _DTYPES:
        raise ValueError("query must have FP16, BF16 or FP32 dtype")
    _validate_tensor(key, query, "key")
    if (
        query.ndim != 4
        or key.ndim != 4
        or query.dtype != key.dtype
        or query.shape[:2] != key.shape[:2]
        or query.shape[-1] != key.shape[-1]
        or query.shape[2] == 0
        or key.shape[2] == 0
        or query.shape[-1] == 0
        or query.shape[-1] % 2
    ):
        raise ValueError("Q/K require matching BSHD geometry, positive heads and even D")
    for weight in (query_weight, key_weight):
        if weight is not None:
            _validate_tensor(weight, query, "weight")
            if weight.shape != (query.shape[-1],):
                raise ValueError("Q/K weights must be [D]")
    _validate_epsilon(query_eps)
    _validate_epsilon(key_eps)
    b, s, _, d = query.shape
    if (
        not isinstance(freqs, torch.Tensor)
        or freqs.device != query.device
        or freqs.dtype != torch.complex64
        or freqs.shape not in ((s, d // 2), (1, s, d // 2), (b, s, d // 2))
    ):
        raise ValueError("frequencies must be complex64 [S,D/2] or [B,S,D/2] on Q/K device")
    args = query, key, query_weight, key_weight, freqs, query_eps, key_eps
    if _ENABLED.get() and aot_kernels.supports_qk_norm_rotary(*args):
        return _record("qk_norm_rotary", "aot", aot_kernels.qk_norm_rotary(*args))
    # Do not use inner dispatch or metrics in the exact decomposed oracle.
    result = (
        _torch_rotary(_torch_rms_norm(query, query_weight, query_eps), freqs),
        _torch_rotary(_torch_rms_norm(key, key_weight, key_eps), freqs),
    )
    return _record("qk_norm_rotary", "torch", result)
