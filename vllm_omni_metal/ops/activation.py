# SPDX-License-Identifier: Apache-2.0
"""Generic OOT SiLU×gate using optional Torch-stream kernels or the official MLX bridge."""

from functools import lru_cache

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.activation import SiluAndMul

from vllm_omni_metal.ops import vpipe_kernels

# shortcut: bridge overhead loses on small decode inputs; remeasure before lowering this gate.
MIN_COMPILED_ELEMENTS = 262144


def _supports_compiled_silu(x: torch.Tensor) -> bool:
    return (
        x.device.type == "mps"
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and not x.requires_grad
        and x.ndim in (1, 2, 3)
        and x.shape[-1] % 2 == 0
        and x.is_contiguous()
        and x.numel() // 2 >= MIN_COMPILED_ELEMENTS
    )


@lru_cache(maxsize=1)
def _compiled_silu_and_mul():
    import mlx.core as mx
    import mlx.nn as nn

    def operation(gate, up):
        # Match Torch SiLU's FP32 compute and dtype rounding before the separate multiply.
        return nn.silu(gate.astype(mx.float32)).astype(gate.dtype) * up

    return mx.compile(operation, shapeless=True)


def _silu_and_mul_mps(x: torch.Tensor) -> torch.Tensor:
    if vpipe_kernels.supports_silu(x):
        return vpipe_kernels.silu_and_mul(x)

    import mlx.core as mx
    from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch, torch_to_mlx

    shared = torch_to_mlx(x)
    width = x.shape[-1] // 2
    result = _compiled_silu_and_mul()(shared[..., :width], shared[..., width:])
    # Keep the borrowed input alive through eval; never donate Torch's activation storage.
    mx.eval(result)
    return mlx_to_torch(result, "mps")


@SiluAndMul.register_oot
class MetalSiluAndMul(SiluAndMul):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        config = get_current_vllm_config_or_none()
        worker = getattr(getattr(config, "parallel_config", None), "worker_cls", "")
        if (
            isinstance(worker, str)
            and worker.startswith("vllm_omni_metal.native.worker.")
            and _supports_compiled_silu(x)
        ):
            return _silu_and_mul_mps(x)
        return self.forward_native(x)


# OOT lookup uses the class name; enable/disable directives must keep the original op name.
MetalSiluAndMul.name = SiluAndMul.name
