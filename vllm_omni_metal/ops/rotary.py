# SPDX-License-Identifier: Apache-2.0
"""OOT cached-table NeoX RoPE; other upstream RoPE contracts remain unchanged."""

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

from vllm_omni_metal.ops import vpipe_kernels


@RotaryEmbedding.register_oot
class MetalRotaryEmbedding(RotaryEmbedding):
    def forward_oot(self, positions, query, key=None):
        config = get_current_vllm_config_or_none()
        worker = getattr(getattr(config, "parallel_config", None), "worker_cls", "")
        tensors = (query,) if key is None else (query, key)
        supported = (
            isinstance(worker, str)
            and worker.startswith("vllm_omni_metal.native.worker.")
            and self.is_neox_style
            and 0 < self.rotary_dim <= self.head_size <= 1024
            and self.rotary_dim % 2 == 0
            and positions.dtype in (torch.int32, torch.int64)
            and positions.numel() > 0
            and all(
                t.device.type == "mps"
                and t.device == query.device
                and t.dtype == query.dtype
                and t.dtype in (torch.float32, torch.float16, torch.bfloat16)
                and not t.requires_grad
                and t.ndim in (2, 3)
                and t.is_contiguous()
                and 0 < t.numel() < 2**31
                and t.numel() % (positions.numel() * self.head_size) == 0
                for t in tensors
            )
        )
        if not supported or vpipe_kernels._library(query.dtype) is None:
            return self.forward_native(positions, query, key)
        # Preserve upstream cache construction/scaling, dtype matching and index validation.
        cache = self._match_cos_sin_cache_dtype(query)
        selected = cache.index_select(0, positions.flatten())
        if (
            key is not None
            and self.rotary_dim == self.head_size
            and query.dtype in (torch.float16, torch.bfloat16)
            and selected.dtype == query.dtype
            and selected.shape == (positions.numel(), self.head_size)
            and selected.is_contiguous()
        ):
            return vpipe_kernels.cached_neox_rope_qk(query, key, selected, self.head_size)
        cos, sin = (table.float().contiguous() for table in selected.chunk(2, dim=-1))
        rotated = [
            vpipe_kernels.cached_neox_rope(
                tensor.view(positions.numel(), -1, self.head_size), cos, sin, self.rotary_dim
            ).reshape(tensor.shape)
            for tensor in tensors
        ]
        return rotated[0], rotated[1] if key is not None else None


MetalRotaryEmbedding.name = RotaryEmbedding.name
