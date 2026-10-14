# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm-metal, commit fb9fbc3bd857bee3f7a74f59ee46cc068fa7267c.
"""Use upstream's PyTorch fused MPS RMSNorm adapter through vLLM IR."""

import torch
from vllm import ir


def _supports_rms_norm(x, weight, epsilon, variance_size=None):
    return (
        x.device.type == "mps"
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.ndim > 0
        and x.numel() > 0
        and weight is not None
        and weight.shape == (x.shape[-1],)
        and weight.dtype == x.dtype
        and weight.device == x.device
        and variance_size is None
    )


@ir.ops.rms_norm.register_impl("omni_mps", supports_args=_supports_rms_norm)
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> torch.Tensor:
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, epsilon)


def _supports_add_rms_norm(x, x_residual, weight, epsilon, variance_size=None):
    return (
        _supports_rms_norm(x, weight, epsilon, variance_size)
        and x_residual.dtype == x.dtype
        and x_residual.device == x.device
        and x_residual.shape == x.shape
    )


@ir.ops.fused_add_rms_norm.register_impl("omni_mps", supports_args=_supports_add_rms_norm)
def fused_add_rms_norm(
    x: torch.Tensor,
    x_residual: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert weight is not None
    # vLLM requires variance from the unrounded FP32 sum and cast-before-weight ordering.
    summed = x.float() + x_residual.float()
    normalized = torch.nn.functional.rms_norm(
        summed,
        (summed.shape[-1],),
        torch.ones_like(weight, dtype=torch.float32),
        epsilon,
    )
    return normalized.to(x.dtype) * weight, summed.to(x.dtype)
