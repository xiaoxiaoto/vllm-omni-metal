# SPDX-License-Identifier: Apache-2.0
"""Bounded, synchronized comparison with the previous MPS gather + SDPA path."""

import argparse
import json
import statistics
import subprocess
from time import perf_counter
from unittest.mock import patch

import psutil
import torch

from vllm_omni_metal.metal_attention import supports_paged_attention
from vllm_omni_metal.native_attention import MetalAttentionImpl, MetalAttentionMetadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-size", type=int, choices=[64, 96, 128, 256, 512], default=128)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--require-speedup", action="store_true")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS is required; no CPU fallback")
    if not (
        1 <= args.batch_size <= 32 and 1 <= args.seq_len <= 16384 and 3 <= args.iterations <= 100
    ):
        parser.error("budget: batch 1..32, sequence 1..16384, iterations 3..100")
    if args.heads <= 0 or args.kv_heads <= 0 or args.heads % args.kv_heads:
        parser.error("query heads must be a positive multiple of KV heads")
    dtype = getattr(torch, args.dtype)
    pages = (args.seq_len + 15) // 16
    itemsize = torch.empty((), dtype=dtype).element_size()
    # Include cache, reference GQA expansion, QKV/output and native split scratch.
    estimated_bytes = (
        (
            2 * args.batch_size * pages * 16 * args.kv_heads * args.head_size
            + 2 * args.batch_size * args.seq_len * args.heads * args.head_size
            + args.batch_size * (2 * args.heads + 2 * args.kv_heads) * args.head_size
        )
        * itemsize
        + args.batch_size
        * args.heads
        * ((args.seq_len + 255) // 256)
        * (args.head_size * itemsize + 8)
        + 4 * args.batch_size * args.heads * args.seq_len
    )
    if (
        estimated_bytes > 256 * 1024**2
        or psutil.virtual_memory().available < 3 * 1024**3 + estimated_bytes
    ):
        parser.error("benchmark exceeds 256 MiB budget or 3 GiB system headroom")
    torch.manual_seed(17)
    q = torch.randn(args.batch_size, args.heads, args.head_size, dtype=dtype, device="mps")
    k = torch.randn(args.batch_size, args.kv_heads, args.head_size, dtype=dtype, device="mps")
    v = torch.randn_like(k)
    cache = torch.randn(
        2, args.batch_size * pages, 16, args.kv_heads, args.head_size, dtype=dtype, device="mps"
    )
    assert supports_paged_attention(q, cache)
    metadata = MetalAttentionMetadata(
        list(range(args.batch_size + 1)),
        [args.seq_len] * args.batch_size,
        [list(range(i * pages, (i + 1) * pages)) for i in range(args.batch_size)],
    )
    impl = MetalAttentionImpl(args.heads, args.head_size, args.head_size**-0.5, args.kv_heads)

    def step(native):
        # Keep exactly the old path available as a benchmark reference, not a runtime flag.
        with patch("vllm_omni_metal.metal_attention.supports_paged_attention", return_value=native):
            return impl.forward(None, q, k, v, cache, metadata)

    tolerance = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(step(True), step(False), atol=tolerance, rtol=tolerance)
    for _ in range(5):
        step(False)
        step(True)
    samples = {False: [], True: []}
    for _ in range(args.iterations):
        for native in (False, True):
            torch.mps.synchronize()
            start = perf_counter()
            result = step(native)
            torch.mps.synchronize()
            samples[native].append((perf_counter() - start) * 1000)
    assert torch.isfinite(result).all()
    reference_ms, paged_ms = (statistics.median(samples[native]) for native in (False, True))
    chip = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout.strip()
    print(
        json.dumps(
            {
                "device": chip,
                "dtype": args.dtype,
                "shape": {
                    "batch": args.batch_size,
                    "seq_len": args.seq_len,
                    "heads": args.heads,
                    "kv_heads": args.kv_heads,
                    "head_size": args.head_size,
                },
                "iterations": args.iterations,
                "reference_median_ms": reference_ms,
                "paged_median_ms": paged_ms,
                "speedup": reference_ms / paged_ms,
                "estimated_tensor_mib": estimated_bytes / 1024**2,
            },
            indent=2,
        )
    )
    if args.require_speedup and paged_ms >= reference_ms:
        raise SystemExit("native paged attention did not beat the reference on this shape")


if __name__ == "__main__":
    main()
