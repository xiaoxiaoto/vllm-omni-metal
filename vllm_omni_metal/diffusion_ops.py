# SPDX-License-Identifier: Apache-2.0
"""Explicit diffusion operators; no model patching or runtime Metal compilation.

RoPE uses adjacent real/imaginary pairs and FP32 complex multiplication, not
NeoX's split-half rotation or its low-precision product rounding. RMSNorm follows
diffusers' FP32 variance and weight-dependent cast, not vLLM's unconditional cast.
"""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.nn.functional as F

from vllm_omni_metal import vpipe_kernels

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
        if vpipe_kernels.supports_separate_silu(gate, up):
            result = vpipe_kernels.separate_silu_mul(gate, up)
            return _record("silu_mul", "aot", result)
        if _validated or gate.numel() >= _MIN_COMPILED_ELEMENTS:
            import mlx.core as mx
            from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch, torch_to_mlx

            from vllm_omni_metal.activation import _compiled_silu_and_mul

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
    if _ENABLED.get() and vpipe_kernels.supports_interleaved_rotary(x, freqs):
        return _record("rotary", "aot", vpipe_kernels.interleaved_rotary(x, freqs))
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    # view_as_complex requires even outer strides, including for FP32 views.
    if pairs.stride(-1) != 1 or any(stride % 2 for stride in pairs.stride()[:-1]):
        pairs = pairs.contiguous()
    rotated = torch.view_as_complex(pairs) * freqs.unsqueeze(-2)
    result = torch.view_as_real(rotated).flatten(-2).to(x.dtype)
    return _record("rotary", "torch_complex", result)


def _supports_fused_rms(x, weight, eps):
    return (
        x.device.type == "mps"
        and x.dtype in _DTYPES
        and x.ndim > 0
        and x.numel() > 0
        and x.is_contiguous()
        and not x.requires_grad
        and vpipe_kernels._supports_add_rms_norm_epsilon(eps)
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
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    if weight is not None:
        if weight.dtype in (torch.float16, torch.bfloat16):
            normalized = normalized.to(weight.dtype)
        normalized = normalized * weight
    else:
        normalized = normalized.to(x.dtype)
    return _record("rms_norm", "torch", normalized)
