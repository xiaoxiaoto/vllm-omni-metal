# SPDX-License-Identifier: Apache-2.0
"""Bounded, balanced RMSNorm A/B against native and the preceding MPS decomposition."""

import argparse
import json
import statistics
import subprocess
from time import perf_counter

import psutil
import torch
from vllm import ir

import vllm_omni_metal.ops.normalization  # noqa: F401
from vllm_omni_metal.ops import vpipe_kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
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
    estimated_bytes = 8 * args.rows * args.hidden_size * 4 + 4 * args.hidden_size * 4
    if (
        estimated_bytes > 64 * 1024**2
        or psutil.virtual_memory().available < 3 * 1024**3 + estimated_bytes
    ):
        parser.error("benchmark exceeds 64 MiB budget or 3 GiB system headroom")
    dtype = getattr(torch, args.dtype)
    original_library = vpipe_kernels._library
    if args.compile_for_test:
        library = torch.mps.compile_shader(
            f"#define VPIPE_ELT {vpipe_kernels._SCALARS[dtype]}\n"
            + vpipe_kernels._SOURCE.read_text()
        )
        vpipe_kernels._library = lambda _: library
    original_supports = vpipe_kernels.supports_add_rms_norm
    torch.manual_seed(17)
    results = {}
    try:
        for fused in (False, True):
            op = ir.ops.fused_add_rms_norm if fused else ir.ops.rms_norm
            inputs = op.generate_inputs(
                num_tokens=args.rows, hidden_size=args.hidden_size, dtype=dtype, device="mps"
            )
            saved = [item.clone() for item in inputs if isinstance(item, torch.Tensor)]
            expected = op.impls["native"].impl_fn(*inputs)
            routes = ("native", "previous_mps", "omni_mps") if fused else ("native", "omni_mps")
            samples = {route: [] for route in routes}
            single_kernel = fused and original_supports(*inputs[:3])
            for iteration in range(args.iterations + 10):
                order = routes if iteration % 2 == 0 else tuple(reversed(routes))
                for route in order:
                    vpipe_kernels.supports_add_rms_norm = (
                        (lambda *unused: False) if route == "previous_mps" else original_supports
                    )
                    provider = "native" if route == "native" else "omni_mps"
                    with op.set_priority(
                        [provider, "native"] if provider != "native" else [provider]
                    ):
                        torch.mps.synchronize()
                        start = perf_counter()
                        actual = op(*inputs)
                        torch.mps.synchronize()
                        elapsed = (perf_counter() - start) * 1000
                    if iteration >= 10:
                        samples[route].append(elapsed)
                    if iteration == 0:
                        expected_items = expected if fused else (expected,)
                        actual_items = actual if fused else (actual,)
                        for result, reference in zip(actual_items, expected_items, strict=True):
                            torch.testing.assert_close(result, reference, **op.get_tolerance(dtype))
                            assert result.device.type == "mps" and result.dtype == dtype
                            assert torch.isfinite(result).all()
            for item, before in zip(
                (item for item in inputs if isinstance(item, torch.Tensor)), saved, strict=True
            ):
                torch.testing.assert_close(item, before, atol=0, rtol=0)
            medians = {route: statistics.median(values) for route, values in samples.items()}
            results["residual_rms_norm" if fused else "rms_norm"] = {
                "native_median_ms": medians["native"],
                "optimized_median_ms": medians["omni_mps"],
                "speedup": medians["native"] / medians["omni_mps"],
                "single_kernel": bool(single_kernel),
                **(
                    {
                        "previous_mps_median_ms": medians["previous_mps"],
                        "speedup_over_previous_mps": medians["previous_mps"] / medians["omni_mps"],
                    }
                    if fused
                    else {}
                ),
            }
    finally:
        vpipe_kernels._library = original_library
        vpipe_kernels.supports_add_rms_norm = original_supports
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
                "results": results,
            },
            indent=2,
        )
    )
    if args.require_speedup and any(result["speedup"] <= 1 for result in results.values()):
        raise SystemExit("an optimized norm did not beat native on this shape")


if __name__ == "__main__":
    main()
