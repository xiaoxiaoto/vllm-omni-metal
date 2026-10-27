# SPDX-License-Identifier: Apache-2.0
"""Bounded, synchronized operator benchmarks; test-only shader compilation is explicit."""

import argparse
import json
import statistics
from time import perf_counter

import psutil
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

from vllm_omni_metal.ops import aot_kernels, rotary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=["silu", "rope"], default="silu")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--compile-for-test", action="store_true")
    parser.add_argument("--require-speedup", action="store_true")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS required")
    if not (
        1 <= args.rows <= 128 and 128 <= args.hidden_size <= 16384 and 3 <= args.iterations <= 100
    ):
        parser.error("budget: rows 1..128, hidden size 128..16384, iterations 3..100")
    if args.hidden_size % 128:
        parser.error("hidden size must be divisible by 128")
    dtype = getattr(torch, args.dtype)
    estimated = args.rows * args.hidden_size * 32 + 128 * 128 * 4
    if estimated > 64 * 1024**2 or psutil.virtual_memory().available < 3 * 1024**3 + estimated:
        parser.error("benchmark exceeds 64 MiB or 3 GiB headroom budget")
    if args.compile_for_test:
        library = torch.mps.compile_shader(
            f"#define ELEMENTWISE_ELT {aot_kernels._SCALARS[dtype]}\n"
            + aot_kernels._SOURCE.read_text()
        )
        aot_kernels._library = lambda _: library
    elif aot_kernels._library(dtype) is None:
        parser.error("explicitly build AOT metallibs first; no worker JIT fallback")
    torch.manual_seed(67)
    config = VllmConfig()
    config.parallel_config.worker_cls = "vllm_omni_metal.native.worker.MetalARWorker"
    config.compilation_config.custom_ops = ["none", "+rotary_embedding"]
    with set_current_vllm_config(config):
        if args.operator == "silu":
            x = torch.randn(args.rows, 2 * args.hidden_size, device="mps", dtype=dtype)
            functions = [
                lambda: SiluAndMul.forward_native(x),
                lambda: aot_kernels.silu_and_mul(x),
            ]
        else:
            layer = RotaryEmbedding(128, 128, 128, 10000, True, dtype).to("mps")
            assert isinstance(layer, rotary.MetalRotaryEmbedding)
            positions = torch.arange(args.rows, device="mps")
            q = torch.randn(args.rows, args.hidden_size, device="mps", dtype=dtype)
            k = torch.randn(args.rows, args.hidden_size // 4, device="mps", dtype=dtype)
            functions = [
                lambda: layer.forward_native(positions, q, k),
                lambda: layer.forward_oot(positions, q, k),
            ]
        tolerance = 2e-6 if dtype == torch.float32 else 0
        reference, actual = functions[0](), functions[1]()
        if args.operator == "silu":
            torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
        else:
            for a, b in zip(actual, reference, strict=True):
                torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
        for _ in range(10):
            for function in functions:
                function()
        samples = [[], []]
        for _ in range(args.iterations):
            for i, function in enumerate(functions):
                torch.mps.synchronize()
                start = perf_counter()
                result = function()
                torch.mps.synchronize()
                samples[i].append((perf_counter() - start) * 1000)
        tensors = (result,) if args.operator == "silu" else result
        assert all(torch.isfinite(tensor).all() for tensor in tensors)
    original, fused = (statistics.median(sample) for sample in samples)
    print(
        json.dumps(
            {
                "operator": args.operator,
                "dtype": args.dtype,
                "rows": args.rows,
                "hidden_size": args.hidden_size,
                "iterations": args.iterations,
                "reference_median_ms": original,
                "fused_median_ms": fused,
                "speedup": original / fused,
                "test_only_compilation": args.compile_for_test,
                "estimated_mib": estimated / 1024**2,
            },
            indent=2,
        )
    )
    if args.require_speedup and fused >= original:
        raise SystemExit("no speedup on this shape; do not claim decode/model throughput gain")


if __name__ == "__main__":
    main()
