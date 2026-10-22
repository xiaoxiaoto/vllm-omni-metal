# SPDX-License-Identifier: Apache-2.0
"""Reuse vllm-metal 0.31's published paged kernels, not a second implementation."""


def supports_paged_attention(query, cache):
    return (
        query.device.type == "mps"
        and query.shape[-1] in (64, 96, 128, 256, 512)
        and cache.shape[2] in (8, 16, 32)
        and cache.stride(4) == 1
        and cache.stride(3) == cache.shape[4]
        and cache.stride(2) == cache.shape[3] * cache.shape[4]
    )


def paged_attention(query, key, value, cache, metadata, scale, sliding_window):
    """The caller validates shapes, dtype, strides and the whole batch before cache writes."""
    rows, heads, head_size = query.shape
    max_length = max(metadata.seq_lens)
    # Use the smallest upstream partition (GQA P256), not just the default P512.
    partitions = (max_length + 255) // 256
    scratch_bytes = rows * heads * partitions * (head_size * query.element_size() + 8)
    if scratch_bytes > 256 * 1024**2:
        raise ValueError("Metal attention scratch estimate exceeds the 256 MiB per-call budget")

    width = max((length + cache.shape[2] - 1) // cache.shape[2] for length in metadata.seq_lens)
    tables = []
    slots = []
    for request, length in enumerate(metadata.seq_lens):
        begin, end = metadata.query_start_loc[request : request + 2]
        pages = (length + cache.shape[2] - 1) // cache.shape[2]
        table = metadata.block_tables[request][:pages]
        tables.append(table + [0] * (width - pages))
        slots.extend(
            table[pos // cache.shape[2]] * cache.shape[2] + pos % cache.shape[2]
            for pos in range(length - (end - begin), length)
        )
    import torch

    if query.dtype in (torch.float32, torch.float16, torch.bfloat16):
        from vllm_omni_metal.ops.mps_attention import get_mps_ops

        native = get_mps_ops()
        if native is not None:
            if (
                max(cache.stride(1), cache.stride(3), heads * head_size, key.shape[1] * head_size)
                >= 2**31
            ):
                raise ValueError("Torch-stream attention strides must fit int32")
            result = torch.empty_like(query, memory_format=torch.contiguous_format)
            sequences = len(metadata.seq_lens)
            table_elements = sequences * width
            # One asynchronous transfer for the int32 metadata; keep copies on the Torch stream.
            packed = torch.tensor(
                [block for table in tables for block in table]
                + metadata.seq_lens
                + metadata.query_start_loc,
                dtype=torch.int32,
            ).to(query.device, non_blocking=True)
            native.forward(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                cache[0],
                cache[1],
                torch.tensor(slots, dtype=torch.int64).to(query.device, non_blocking=True),
                packed[:table_elements].view(sequences, width),
                packed[table_elements : table_elements + sequences],
                packed[table_elements + sequences :],
                result,
                max_length,
                scale,
                sliding_window if sliding_window is not None else -1,
                0.0,
                1,
            )
            return result.reshape(rows, -1)

    import mlx.core as mx
    from vllm_metal.metal import get_ops
    from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch, torch_to_mlx

    ops = get_ops()
    # The upstream bridge waits for Torch writes; mx.eval completes MLX writes before Torch reuse.
    q, k, v, kc, vc = (
        torch_to_mlx(tensor)
        for tensor in (query.contiguous(), key.contiguous(), value.contiguous(), cache[0], cache[1])
    )
    kc, vc = ops.reshape_and_cache(k, v, kc, vc, mx.array(slots, dtype=mx.int64))
    result = mx.array(0)
    ops.paged_attention_primitive(
        q,
        kc,
        vc,
        key.shape[1],
        scale,
        0.0,
        mx.array(tables, dtype=mx.int32),
        mx.array(metadata.seq_lens, dtype=mx.int32),
        mx.array(metadata.query_start_loc, dtype=mx.int32),
        cache.shape[2],
        max_length,
        sliding_window if sliding_window is not None else -1,
        result,
    )
    mx.eval(result)
    return mlx_to_torch(result, query.device).reshape(rows, -1)
