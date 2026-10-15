# SPDX-License-Identifier: Apache-2.0
"""Bounded SiLU×gate benchmark including OOT dispatch and Torch/MLX synchronization."""

import argparse
import json
import statistics
import subprocess
from time import perf_counter

import psutil
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul

from vllm_omni_metal.activation import MetalSiluAndMul, _supports_compiled_silu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--require-speedup", action="store_true")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS is required; no CPU fallback")
    if not (
        1 <= args.rows <= 1024 and 1 <= args.hidden_size <= 16384 and 3 <= args.iterations <= 100
    ):
        parser.error("budget: rows 1..1024, hidden size 1..16384, iterations 3..100")
    estimated_bytes = 8 * args.rows * args.hidden_size * 4
    if (
        estimated_bytes > 64 * 1024**2
        or psutil.virtual_memory().available < 3 * 1024**3 + estimated_bytes
    ):
        parser.error("benchmark exceeds 64 MiB budget or 3 GiB system headroom")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(29)
    x = torch.randn(args.rows, 2 * args.hidden_size, dtype=dtype, device="mps")
    saved = x.clone()
    config = VllmConfig()
    config.parallel_config.worker_cls = "vllm_omni_metal.native_worker.MetalARWorker"
    config.compilation_config.custom_ops = ["none", "+silu_and_mul"]
    samples = {"native": [], "optimized": []}
    with set_current_vllm_config(config):
        layer = SiluAndMul(compile_native=False)
        assert isinstance(layer, MetalSiluAndMul)
        assert layer._forward_method.__func__ is MetalSiluAndMul.forward_oot
        expected = SiluAndMul.forward_native(x)
        actual = layer(x)
        tol = {torch.float32: 2e-6, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
        assert actual.device.type == "mps" and actual.dtype == dtype
        assert torch.isfinite(actual).all()
        for provider, fn in (("native", SiluAndMul.forward_native), ("optimized", layer)):
            for _ in range(10):
                fn(x)
            for _ in range(args.iterations):
                torch.mps.synchronize()
                start = perf_counter()
                actual = fn(x)
                torch.mps.synchronize()
                samples[provider].append((perf_counter() - start) * 1000)
    torch.testing.assert_close(x, saved, atol=0, rtol=0)
    native_ms, optimized_ms = (statistics.median(samples[p]) for p in ("native", "optimized"))
    speedup = native_ms / optimized_ms
    compiled = _supports_compiled_silu(x)
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
                "rows": args.rows,
                "hidden_size": args.hidden_size,
                "iterations": args.iterations,
                "estimated_tensor_mib": estimated_bytes / 1024**2,
                "dispatch": "compiled_mlx_with_bridge" if compiled else "native",
                "native_median_ms": native_ms,
                "optimized_median_ms": optimized_ms,
                "speedup": speedup,
            },
            indent=2,
        )
    )
    if args.require_speedup and (not compiled or speedup <= 1):
        raise SystemExit("compiled activation did not beat native on this shape")


if __name__ == "__main__":
    main()
