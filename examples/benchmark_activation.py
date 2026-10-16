# SPDX-License-Identifier: Apache-2.0
"""Bounded SiLU×gate A/B including OOT dispatch, bridge costs and final synchronization."""

import argparse
import json
import statistics
import subprocess
from time import perf_counter

import psutil
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul

from vllm_omni_metal import vpipe_kernels
from vllm_omni_metal.activation import MetalSiluAndMul, _supports_compiled_silu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--compile-for-test", action="store_true")
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
    original_library = vpipe_kernels._library
    original_supports = vpipe_kernels.supports_silu
    if args.compile_for_test:
        library = torch.mps.compile_shader(
            f"#define VPIPE_ELT {vpipe_kernels._SCALARS[dtype]}\n"
            + vpipe_kernels._SOURCE.read_text()
        )
        vpipe_kernels._library = lambda _: library
    torch.manual_seed(29)
    x = torch.randn(args.rows, 2 * args.hidden_size, dtype=dtype, device="mps")
    saved = x.clone()
    config = VllmConfig()
    config.parallel_config.worker_cls = "vllm_omni_metal.native_worker.MetalARWorker"
    config.compilation_config.custom_ops = ["none", "+silu_and_mul"]
    compiled = _supports_compiled_silu(x)
    torch_stream = compiled and original_supports(x)
    routes = ("native", "previous_mps", "optimized")
    samples = {route: [] for route in routes}
    try:
        with set_current_vllm_config(config):
            layer = SiluAndMul(compile_native=False)
            assert isinstance(layer, MetalSiluAndMul)
            expected = SiluAndMul.forward_native(x)
            for iteration in range(args.iterations + 10):
                order = routes if iteration % 2 == 0 else tuple(reversed(routes))
                for route in order:
                    vpipe_kernels.supports_silu = (
                        (lambda unused: False) if route == "previous_mps" else original_supports
                    )
                    fn = SiluAndMul.forward_native if route == "native" else layer
                    torch.mps.synchronize()
                    start = perf_counter()
                    actual = fn(x)
                    torch.mps.synchronize()
                    elapsed = (perf_counter() - start) * 1000
                    if iteration >= 10:
                        samples[route].append(elapsed)
                    if iteration == 0:
                        tolerance = 2e-6 if dtype == torch.float32 else 0
                        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                        assert actual.device.type == "mps" and actual.dtype == dtype
                        assert torch.isfinite(actual).all()
    finally:
        vpipe_kernels._library = original_library
        vpipe_kernels.supports_silu = original_supports
    torch.testing.assert_close(x, saved, atol=0, rtol=0)
    medians = {route: statistics.median(values) for route, values in samples.items()}
    speedup = medians["native"] / medians["optimized"]
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
                "balanced_order": True,
                "test_only_compilation": args.compile_for_test,
                "estimated_tensor_mib": estimated_bytes / 1024**2,
                "dispatch": "torch_stream"
                if torch_stream
                else "compiled_mlx_with_bridge"
                if compiled
                else "native",
                "native_median_ms": medians["native"],
                "previous_mps_median_ms": medians["previous_mps"],
                "optimized_median_ms": medians["optimized"],
                "speedup": speedup,
                "speedup_over_previous_mps": medians["previous_mps"] / medians["optimized"],
            },
            indent=2,
        )
    )
    if args.require_speedup and (not compiled or speedup <= 1):
        raise SystemExit("optimized activation did not beat native on this shape")


if __name__ == "__main__":
    main()
