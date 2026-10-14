# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch.nn import functional as F

from vllm_omni_metal.native_attention import MetalAttentionImpl, MetalAttentionMetadata


@pytest.mark.parametrize("chunk_sizes", [[7], [2, 3, 2], [5, 1, 1]])
def test_paged_gqa_matches_full_causal_attention(chunk_sizes):
    torch.manual_seed(5)
    q = torch.randn(7, 4, 8)
    k = torch.randn(7, 2, 8)
    v = torch.randn(7, 2, 8)
    impl = MetalAttentionImpl(4, 8, 8**-0.5, 2)
    cache = torch.full((2, 5, 2, 2, 8), float("nan"))
    # Non-contiguous physical blocks; the unused block must not leak into output.
    blocks = [[3, 0, 4, 1]]
    reference = (
        F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0),
            v.repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0),
            is_causal=True,
        )
        .squeeze(0)
        .transpose(0, 1)
        .reshape(7, -1)
    )
    offset = 0
    for size in chunk_sizes:
        end = offset + size
        metadata = MetalAttentionMetadata([0, size], [end], blocks)
        actual = impl.forward(None, q[offset:end], k[offset:end], v[offset:end], cache, metadata)
        torch.testing.assert_close(actual, reference[offset:end])
        offset = end


def test_requests_do_not_share_kv():
    impl = MetalAttentionImpl(2, 4, 0.5, 1)
    cache = torch.full((2, 4, 2, 1, 4), float("nan"))
    q = torch.randn(2, 2, 4)
    k = torch.randn(2, 1, 4)
    v = torch.randn(2, 1, 4)
    metadata = MetalAttentionMetadata([0, 1, 2], [1, 1], [[0], [3]])
    out = impl.forward(None, q, k, v, cache, metadata)
    torch.testing.assert_close(out[0].view(2, 4), v[0].expand(2, 4))
    torch.testing.assert_close(out[1].view(2, 4), v[1].expand(2, 4))
    # Reusing a physical block for a new request must ignore stale tail positions.
    metadata = MetalAttentionMetadata([0, 1], [1], [[0]])
    new_v = v[:1] + 10
    out = impl.forward(None, q[:1], k[:1], new_v, cache, metadata)
    torch.testing.assert_close(out.view(2, 4), new_v[0].expand(2, 4))


@pytest.mark.parametrize("blocks", [[[9]], [[]], [[-1]]])
def test_bad_block_tables_rejected_before_cache_write(blocks):
    impl = MetalAttentionImpl(2, 4, 0.5, 1)
    cache = torch.zeros(2, 2, 2, 1, 4)
    metadata = MetalAttentionMetadata([0, 1], [1], blocks)
    with pytest.raises(ValueError, match="block"):
        impl.forward(
            None, torch.ones(1, 2, 4), torch.ones(1, 1, 4), torch.ones(1, 1, 4), cache, metadata
        )
    assert not cache.any()


@pytest.mark.parametrize(
    "case",
    ["q_shape", "dtype", "cache_shape", "overlap", "duplicate", "shared", "output", "length"],
)
def test_invalid_attention_inputs_rejected_before_cache_write(case):
    impl = MetalAttentionImpl(2, 4, 0.5, 1)
    q = torch.ones(2, 2, 4)
    k = v = torch.ones(2, 1, 4)
    cache = torch.zeros(2, 4, 2, 1, 4)
    metadata = MetalAttentionMetadata([0, 1, 2], [1, 1], [[0], [3]])
    output = None
    if case == "q_shape":
        q = torch.ones(2, 1, 8)
    elif case == "dtype":
        k = k.half()
    elif case == "cache_shape":
        cache = torch.zeros(2, 4, 2, 2, 4)
    elif case == "overlap":
        cache = cache[:1].expand(2, -1, -1, -1, -1)
    elif case == "duplicate":
        metadata = MetalAttentionMetadata([0, 2], [3], [[0, 0]])
    elif case == "shared":
        metadata.block_tables = [[0], [0]]
    elif case == "output":
        output = torch.empty(2, 2, 3)
    elif case == "length":
        metadata.seq_lens[1] = 2**31
    with pytest.raises(ValueError):
        impl.forward(None, q, k, v, cache, metadata, output=output)
    assert not cache.any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_heads": 0},
        {"num_kv_heads": 0},
        {"head_size": 0},
        {"scale": float("nan")},
        {"sliding_window": 0},
    ],
)
def test_invalid_attention_parameters(kwargs):
    with pytest.raises(ValueError):
        MetalAttentionImpl(
            **({"num_heads": 2, "head_size": 4, "scale": 0.5, "num_kv_heads": 1} | kwargs)
        )


def test_native_scratch_budget_covers_small_gqa_partitions_before_initialization(monkeypatch):
    import sys
    from types import SimpleNamespace

    from vllm_omni_metal.metal_attention import paged_attention

    monkeypatch.setitem(sys.modules, "mlx", None)
    query = torch.empty(1, 32, 128)
    metadata = SimpleNamespace(seq_lens=[5_000_000])
    with pytest.raises(ValueError, match="256 MiB"):
        paged_attention(query, None, None, None, metadata, 1.0, None)


def test_cpu_and_unsupported_geometry_do_not_select_native_kernels():
    from vllm_omni_metal.metal_attention import supports_paged_attention

    cache = torch.zeros(2, 4, 16, 1, 64)
    assert not supports_paged_attention(torch.ones(1, 2, 64), cache)
