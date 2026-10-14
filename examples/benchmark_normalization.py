# SPDX-License-Identifier: Apache-2.0
"""Bounded, synchronized RMSNorm microbenchmark against vLLM's native decomposition."""

import argparse
import json
import statistics
import subprocess
from time import perf_counter

import psutil
import torch
from vllm import ir

import vllm_omni_metal.normalization  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--iterations", type=int, default=30)
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
    torch.manual_seed(17)
    results = {}
    for fused in (False, True):
        op = ir.ops.fused_add_rms_norm if fused else ir.ops.rms_norm
        inputs = op.generate_inputs(
            num_tokens=args.rows, hidden_size=args.hidden_size, dtype=dtype, device="mps"
        )
        saved = [item.clone() for item in inputs if isinstance(item, torch.Tensor)]
        expected = op.impls["native"].impl_fn(*inputs)
        samples = {"native": [], "omni_mps": []}
        for provider in samples:
            with op.set_priority([provider, "native"] if provider != "native" else ["native"]):
                assert op.dispatch(*inputs).provider == provider
                actual = op(*inputs)
                expected_items = expected if fused else (expected,)
                actual_items = actual if fused else (actual,)
                for result, reference in zip(actual_items, expected_items, strict=True):
                    torch.testing.assert_close(result, reference, **op.get_tolerance(dtype))
                    assert result.device.type == "mps" and result.dtype == dtype
                    assert torch.isfinite(result).all()
                for _ in range(10):
                    op(*inputs)
                for _ in range(args.iterations):
                    torch.mps.synchronize()
                    start = perf_counter()
                    actual = op(*inputs)
                    torch.mps.synchronize()
                    samples[provider].append((perf_counter() - start) * 1000)
        for item, before in zip(
            (item for item in inputs if isinstance(item, torch.Tensor)), saved, strict=True
        ):
            torch.testing.assert_close(item, before, atol=0, rtol=0)
        native_ms, optimized_ms = (statistics.median(samples[p]) for p in ("native", "omni_mps"))
        results["residual_rms_norm" if fused else "rms_norm"] = {
            "native_median_ms": native_ms,
            "optimized_median_ms": optimized_ms,
            "speedup": native_ms / optimized_ms,
        }
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
                "results": results,
            },
            indent=2,
        )
    )
    if args.require_speedup and any(result["speedup"] <= 1 for result in results.values()):
        raise SystemExit("an optimized norm did not beat native on this shape")


if __name__ == "__main__":
    main()
