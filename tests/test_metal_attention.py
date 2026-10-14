# SPDX-License-Identifier: Apache-2.0
"""Small real-MPS operator tests; no model downloads."""

import os

import pytest
import torch

from vllm_omni_metal.native_attention import MetalAttentionImpl, MetalAttentionMetadata

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1"
        or not torch.backends.mps.is_available(),
        reason="explicit MPS integration opt-in required",
    ),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "head_size,window", [(128, None), (64, 17), (96, None), (256, 31), (512, None)]
)
def test_metal_paged_attention_matches_independent_reference(dtype, head_size, window, monkeypatch):
    torch.manual_seed(41)
    lengths = [1031, 19]
    counts = [3, 2]
    heads, kv_heads, block_size = 4, 2, 16
    pages = [(length + block_size - 1) // block_size for length in lengths]
    total_blocks = sum(pages) + 1
    order = torch.randperm(total_blocks).tolist()
    tables = [order[: pages[0]], order[pages[0] : pages[0] + pages[1]]]
    shape = (2, total_blocks, block_size, kv_heads, head_size)
    page_elements = block_size * kv_heads * head_size
    # Matches worker allocations: interleaved K/V pages, padded block strides, nonzero offset.
    stride = 2 * page_elements + 32
    backing = torch.full((total_blocks * stride + 64,), float("nan"), dtype=dtype, device="mps")
    cache = torch.as_strided(
        backing, shape, (page_elements, stride, kv_heads * head_size, head_size, 1), 32
    )
    q = torch.randn(sum(counts), heads, head_size).to(dtype).to("mps")
    keys = [torch.randn(length, kv_heads, head_size).to(dtype) for length in lengths]
    values = [torch.randn(length, kv_heads, head_size).to(dtype) for length in lengths]
    for request, length in enumerate(lengths):
        for pos in range(length - counts[request]):
            block, offset = tables[request][pos // block_size], pos % block_size
            cache[0, block, offset] = keys[request][pos].to("mps")
            cache[1, block, offset] = values[request][pos].to("mps")
    k = torch.cat([key[-count:] for key, count in zip(keys, counts, strict=True)]).to("mps")
    v = torch.cat([value[-count:] for value, count in zip(values, counts, strict=True)]).to("mps")
    expected = []
    row = 0
    for request, length in enumerate(lengths):
        for pos in range(length - counts[request], length):
            start = max(0, pos - window + 1) if window is not None else 0
            head_outputs = []
            for head in range(heads):
                kv_head = head // (heads // kv_heads)
                key = keys[request][start : pos + 1, kv_head].double()
                value = values[request][start : pos + 1, kv_head].double()
                scores = key @ q[row, head].cpu().double() * head_size**-0.5
                head_outputs.append(scores.softmax(0) @ value)
            expected.append(torch.stack(head_outputs))
            row += 1
    expected = torch.stack(expected).reshape(sum(counts), -1).to(dtype)

    def forbidden_sdpa(*args, **kwargs):
        raise AssertionError("MPS paged path must not gather KV into SDPA")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", forbidden_sdpa)
    monkeypatch.setattr(torch.Tensor, "repeat_interleave", forbidden_sdpa)
    saved_q, saved_k, saved_v = q.clone(), k.clone(), v.clone()
    metadata = MetalAttentionMetadata([0, counts[0], sum(counts)], lengths, tables)
    impl = MetalAttentionImpl(heads, head_size, head_size**-0.5, kv_heads, sliding_window=window)
    output = torch.empty_like(q)
    actual = impl.forward(None, q, k, v, cache, metadata, output=output)
    assert actual.data_ptr() == output.data_ptr()
    tolerance = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(
        actual.cpu().reshape_as(expected), expected, atol=tolerance, rtol=tolerance
    )
    assert torch.isfinite(actual).all()
    for tensor, saved in ((q, saved_q), (k, saved_k), (v, saved_v)):
        torch.testing.assert_close(tensor, saved)
    for request, length in enumerate(lengths):
        for pos in range(length - counts[request], length):
            block, offset = tables[request][pos // block_size], pos % block_size
            torch.testing.assert_close(cache[0, block, offset].cpu(), keys[request][pos])
            torch.testing.assert_close(cache[1, block, offset].cpu(), values[request][pos])
    assert backing[:32].isnan().all() and backing[-32:].isnan().all()


@pytest.mark.parametrize("heads,kv_heads,block_size", [(4, 4, 8), (8, 1, 32)])
def test_native_prefill_decode_and_shared_storage_lifetime(
    heads, kv_heads, block_size, monkeypatch
):
    torch.manual_seed(12)
    head_size = 64
    impl = MetalAttentionImpl(heads, head_size, head_size**-0.5, kv_heads)
    cache = torch.full(
        (2, 16, block_size, kv_heads, head_size), float("nan"), dtype=torch.float16, device="mps"
    )
    tables = [list(range(1, 8)), list(range(8, 15))]
    histories = [[], []]
    held = []

    def forbidden(*args, **kwargs):
        raise AssertionError("supported geometry must use upstream paged kernels")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", forbidden)
    for counts in ([39, 2], [1, 1], [3, 2]):
        q = torch.randn(sum(counts), heads, head_size, dtype=torch.float16)
        k = torch.randn(sum(counts), kv_heads, head_size, dtype=torch.float16)
        v = torch.randn_like(k)
        expected = []
        row = 0
        for request, count in enumerate(counts):
            for _ in range(count):
                histories[request].append((k[row], v[row]))
                keys = torch.stack([item[0] for item in histories[request]]).double()
                values = torch.stack([item[1] for item in histories[request]]).double()
                heads_out = []
                for head in range(heads):
                    kv_head = head // (heads // kv_heads)
                    scores = keys[:, kv_head] @ q[row, head].double() * head_size**-0.5
                    heads_out.append(scores.softmax(0) @ values[:, kv_head])
                expected.append(torch.stack(heads_out))
                row += 1
        meta = MetalAttentionMetadata(
            [0, counts[0], sum(counts)], [len(h) for h in histories], tables
        )
        actual = impl.forward(None, q.to("mps"), k.to("mps"), v.to("mps"), cache, meta)
        expected = torch.stack(expected).reshape(sum(counts), -1).half()
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-3, rtol=2e-3)
        held.append((actual, actual.clone()))
        for previous, saved in held:
            torch.testing.assert_close(previous, saved)


def test_metal_paged_attention_large_logits_and_noncontiguous_inputs():
    impl = MetalAttentionImpl(4, 64, 64**-0.5, 1)
    q = torch.full((2, 4, 128), 100.0, device="mps")[..., ::2]
    k = torch.full((2, 1, 128), 100.0, device="mps")[..., ::2]
    v = torch.randn(2, 1, 128, device="mps")[..., ::2]
    cache = torch.full((2, 4, 16, 1, 64), float("nan"), device="mps")
    meta = MetalAttentionMetadata([0, 2], [2], [[3]])
    actual = impl.forward(None, q, k, v, cache, meta).reshape(2, 4, 64)
    expected = torch.stack([v[0, 0], v[:, 0].mean(0)]).unsqueeze(1).expand_as(actual)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
