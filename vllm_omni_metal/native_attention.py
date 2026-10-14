# SPDX-License-Identifier: Apache-2.0
"""Dense paged KV with upstream Metal kernels and a Torch SDPA reference path."""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    MultipleOf,
)


@dataclass
class MetalAttentionMetadata(AttentionMetadata):
    query_start_loc: list[int]
    seq_lens: list[int]
    block_tables: list[list[int]]


class MetalAttentionMetadataBuilder(AttentionMetadataBuilder):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return MetalAttentionMetadata(
            common_attn_metadata.query_start_loc_cpu.tolist(),
            common_attn_metadata.seq_lens.cpu().tolist(),
            common_attn_metadata.block_table_tensor.cpu().tolist(),
        )


class MetalAttentionBackend(AttentionBackend):
    accept_output_buffer = True

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return MetalAttentionImpl

    @staticmethod
    def get_builder_cls():
        return MetalAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(1)]

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        return (2, num_blocks, block_size, num_kv_heads, head_size)


class MetalAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        **kwargs,
    ):
        if min(num_heads, num_kv_heads, head_size) <= 0:
            raise ValueError("attention head counts and head size must be positive")
        if not math.isfinite(scale):
            raise ValueError("attention scale must be finite")
        if sliding_window is not None and (
            not isinstance(sliding_window, int) or sliding_window <= 0
        ):
            raise ValueError("sliding window must be a positive integer")
        if num_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if alibi_slopes is not None or logits_soft_cap or kv_sharing_target_layer_name:
            raise NotImplementedError(
                "Metal SDPA does not implement ALiBi, soft caps or shared KV layers"
            )
        if kv_cache_dtype not in ("auto", "float16", "bfloat16", "float32"):
            raise NotImplementedError("Metal attention requires unquantized KV")
        if attn_type != "decoder":
            raise NotImplementedError("Metal paged attention currently requires decoder attention")
        if kwargs.get("sinks") is not None:
            raise NotImplementedError("Metal SDPA does not support attention sinks")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = scale
        self.sliding_window = sliding_window

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Metal attention does not support quantized output")
        if attn_metadata is None:
            raise ValueError("Metal attention requires request metadata")
        meta = attn_metadata
        starts = meta.query_start_loc
        lengths = meta.seq_lens
        tables = meta.block_tables
        for tensor, heads in (
            (query, self.num_heads),
            (key, self.num_kv_heads),
            (value, self.num_kv_heads),
        ):
            expected = (heads * self.head_size,) if tensor.ndim == 2 else (heads, self.head_size)
            if tensor.ndim not in (2, 3) or tensor.shape[1:] != expected:
                raise ValueError("Q/K/V shape does not match attention heads")
        tensors = (query, key, value, kv_cache)
        if query.device.type not in ("cpu", "mps") or any(
            t.device != query.device for t in tensors
        ):
            raise ValueError("Q/K/V and KV cache must be on the same CPU or MPS device")
        if query.dtype not in (torch.float32, torch.float16, torch.bfloat16) or any(
            t.dtype != query.dtype for t in tensors
        ):
            raise ValueError("Q/K/V and KV cache must share an unquantized floating dtype")
        q = query.reshape(-1, self.num_heads, self.head_size)
        k = key.reshape(-1, self.num_kv_heads, self.head_size)
        v = value.reshape(-1, self.num_kv_heads, self.head_size)
        if output is not None and (
            output.shape != query.shape
            or output.dtype != query.dtype
            or output.device != query.device
        ):
            raise ValueError("output shape/device/dtype must match query")
        if not lengths or len(starts) != len(lengths) + 1 or len(tables) != len(lengths):
            raise ValueError("inconsistent request/block metadata")
        if starts[0] != 0 or starts[-1] != len(q) or len(k) != len(q) or len(v) != len(q):
            raise ValueError("query boundaries do not match Q/K/V rows")
        if (
            kv_cache.ndim != 5
            or kv_cache.shape[0] != 2
            or kv_cache.shape[3:] != (self.num_kv_heads, self.head_size)
            or min(kv_cache.shape) <= 0
        ):
            raise ValueError("expected paged KV shape [2, blocks, block_size, heads, head_size]")
        span = 1
        for stride, size in sorted(zip(kv_cache.stride(), kv_cache.shape, strict=True)):
            if size > 1:
                if stride < span:
                    raise ValueError("KV cache strides must not overlap")
                span += (size - 1) * stride
        block_size = kv_cache.shape[2]
        used_blocks = set()
        # Validate the whole batch before mutating any shared cache page.
        for i, length in enumerate(lengths):
            if type(length) is not int or length <= 0 or length >= 2**31:
                raise ValueError("sequence length must be a positive int32")
            if type(starts[i]) is not int or type(starts[i + 1]) is not int:
                raise ValueError("query boundaries must be integers")
            count = starts[i + 1] - starts[i]
            pages = (length + block_size - 1) // block_size
            table = tables[i][:pages]
            if count <= 0 or count > length or len(table) != pages:
                raise ValueError("invalid query length or insufficient block table")
            if any(
                type(block) is not int or block < 0 or block >= kv_cache.shape[1] for block in table
            ):
                raise ValueError("block index outside KV allocation")
            if len(set(table)) != len(table) or used_blocks.intersection(table):
                raise ValueError("KV blocks must not alias within or across requests")
            used_blocks.update(table)
        from vllm_omni_metal.metal_attention import paged_attention, supports_paged_attention

        if supports_paged_attention(q, kv_cache):
            result = paged_attention(q, k, v, kv_cache, meta, self.scale, self.sliding_window)
            if output is not None:
                output.copy_(result.reshape_as(output))
                return output
            return result
        results = []
        for i, length in enumerate(lengths):
            begin, end = starts[i : i + 2]
            count = end - begin
            prefix = length - count
            positions = torch.arange(prefix, length, device=k.device)
            table = torch.tensor(tables[i], dtype=torch.long, device=k.device)
            physical_blocks = table[positions // block_size]
            page_offsets = positions % block_size
            kv_cache[0, physical_blocks, page_offsets] = k[begin:end]
            kv_cache[1, physical_blocks, page_offsets] = v[begin:end]
            past_positions = torch.arange(length, device=k.device)
            past_blocks = table[past_positions // block_size]
            past_offsets = past_positions % block_size
            all_k = kv_cache[0, past_blocks, past_offsets]
            all_v = kv_cache[1, past_blocks, past_offsets]
            repeats = self.num_heads // self.num_kv_heads
            all_k = all_k.repeat_interleave(repeats, dim=1).transpose(0, 1).unsqueeze(0)
            all_v = all_v.repeat_interleave(repeats, dim=1).transpose(0, 1).unsqueeze(0)
            mask = past_positions.unsqueeze(0) <= positions.unsqueeze(1)
            if self.sliding_window is not None:
                mask &= past_positions.unsqueeze(0) > positions.unsqueeze(1) - self.sliding_window
            attended = F.scaled_dot_product_attention(
                q[begin:end].transpose(0, 1).unsqueeze(0),
                all_k,
                all_v,
                attn_mask=mask,
                scale=self.scale,
            )
            results.append(attended.squeeze(0).transpose(0, 1).reshape(count, -1))
        result = torch.cat(results, dim=0)
        if output is not None:
            output.copy_(result.reshape_as(output))
            return output
        return result
