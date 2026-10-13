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
