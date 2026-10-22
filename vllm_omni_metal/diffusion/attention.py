# SPDX-License-Identifier: Apache-2.0
"""Use upstream Torch SDPA without its in-tree-only platform dispatch."""

import torch
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl


class MetalSDPABackend(SDPABackend):
    @staticmethod
    def get_impl_cls() -> type["MetalSDPAImpl"]:
        return MetalSDPAImpl


class MetalSDPAImpl(SDPAImpl):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        # The shared implementation preserves upstream masks, GQA, scale and layout.
        return self._forward_impl(query, key, value, attn_metadata)
