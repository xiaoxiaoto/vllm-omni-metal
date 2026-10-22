# SPDX-License-Identifier: Apache-2.0
"""IR dispatch, upstream precision contract, and scoped Metal configuration."""

import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm import ir
from vllm.config import ParallelConfig
from vllm.config.kernel import IrOpPriorityConfig, KernelConfig

from vllm_omni_metal.execution.compat import _patch_native_ir_priority
from vllm_omni_metal.ops import normalization  # noqa: F401


def test_ir_defaults_are_scoped_idempotent_and_preserve_user_priority(monkeypatch):
    calls = []

    class Platform:
        @classmethod
        def get_default_ir_op_priority(cls, config):
            calls.append(config)
            return IrOpPriorityConfig.with_default(["native"])

    _patch_native_ir_priority(Platform)
    patched = Platform.get_default_ir_op_priority.__func__
    _patch_native_ir_priority(Platform)
    assert Platform.get_default_ir_op_priority.__func__ is patched
    ordinary = SimpleNamespace(parallel_config=SimpleNamespace(worker_cls="auto"))
    assert Platform.get_default_ir_op_priority(ordinary).rms_norm == ["native"]
    assert Platform.get_default_ir_op_priority(None).rms_norm == ["native"]
    assert calls == [ordinary, None]
    native = SimpleNamespace(
        parallel_config=ParallelConfig(worker_cls="vllm_omni_metal.native.worker.MetalARWorker")
    )
    defaults = Platform.get_default_ir_op_priority(native)
    assert defaults.rms_norm == defaults.fused_add_rms_norm == ["omni_mps", "native"]
    assert defaults.gelu_and_mul_sparse == ["native"]
    monkeypatch.setattr("vllm.platforms.current_platform", Platform)
    native.model_config = None
    kernel = KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=["native"]))
    kernel.set_platform_defaults(native)
    assert kernel.ir_op_priority.rms_norm == ["native", "omni_mps"]
    assert kernel.ir_op_priority.fused_add_rms_norm == ["omni_mps", "native"]


@pytest.mark.parametrize("fused", [False, True])
def test_cpu_arguments_keep_native_provider(fused):
    op = ir.ops.fused_add_rms_norm if fused else ir.ops.rms_norm
    args = op.generate_inputs(num_tokens=2, hidden_size=128, dtype=torch.float16, device="cpu")
    with op.set_priority(["omni_mps", "native"]):
        assert op.dispatch(*args).provider == "native"
        actual = op(*args)
    expected = op.impls["native"].impl_fn(*args)
    if fused:
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference, atol=0, rtol=0)
    else:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fresh_native_worker_registers_provider_without_mlx():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['mlx'] = None; "
            "import vllm_omni_metal.native.worker; from vllm import ir; "
            "assert 'omni_mps' in ir.ops.rms_norm.impls; "
            "assert 'omni_mps' in ir.ops.fused_add_rms_norm.impls; "
            "assert sys.modules['mlx'] is None",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )


@pytest.mark.parametrize(
    "args,available",
    [
        (["--rows", "0"], 8 * 1024**3),
        (["--hidden-size", "0"], 8 * 1024**3),
        (["--iterations", "1"], 8 * 1024**3),
        (["--rows", "1024", "--hidden-size", "16384"], 8 * 1024**3),
        ([], 2 * 1024**3),
    ],
)
def test_norm_benchmark_budget_stops_before_allocating(monkeypatch, args, available):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=available))
    monkeypatch.setattr(sys, "argv", ["normalization.py", *args])

    def forbidden(*args, **kwargs):
        raise AssertionError("unsafe benchmark must not allocate GPU tensors")

    monkeypatch.setattr(torch, "randn", forbidden)
    script = Path(__file__).resolve().parents[1] / "examples/benchmarks/normalization.py"
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2


MPS = pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="explicit MPS integration opt-in required",
)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("hidden_size", [128, 4096])
def test_mps_norm_preserves_fp32_sum_and_cast_before_weight(dtype, fused, hidden_size):
    torch.manual_seed(17)
    op = ir.ops.fused_add_rms_norm if fused else ir.ops.rms_norm
    args = op.generate_inputs(num_tokens=4, hidden_size=hidden_size, dtype=dtype, device="cpu")
    args[-2].copy_(torch.tensor([1.25, 1.75, 0.75, 1.5], dtype=dtype).repeat(hidden_size // 4))
    if fused:
        args[0][0].fill_(1)
        args[1][0] = (
            torch.tensor([0.125, 0.375, 0.625, 0.875], dtype=dtype).repeat(hidden_size // 4)
            * torch.finfo(dtype).eps
        )
    expected = op.impls["native"].impl_fn(*args)
    gpu_args = tuple(item.to("mps") if isinstance(item, torch.Tensor) else item for item in args)
    saved = [item.clone() for item in gpu_args if isinstance(item, torch.Tensor)]
    with op.set_priority(["omni_mps", "native"]):
        assert op.dispatch(*gpu_args).provider == "omni_mps"
        actual = op(*gpu_args)
    for item, before in zip(
        (item for item in gpu_args if isinstance(item, torch.Tensor)), saved, strict=True
    ):
        torch.testing.assert_close(item, before, atol=0, rtol=0)
    if fused:
        actual, residual = actual
        expected, reference_residual = expected
        torch.testing.assert_close(residual.cpu(), reference_residual, atol=0, rtol=0)
        torch.testing.assert_close(actual[0].cpu(), expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual.cpu(), expected, **op.get_tolerance(dtype))


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fused", [False, True])
def test_noncontiguous_multidimensional_norm_and_maybe_inplace(dtype, fused):
    torch.manual_seed(23)
    x = torch.randn(2, 3, 514, dtype=dtype, device="mps")[..., ::2]
    residual = torch.randn_like(x)
    weight = torch.randn(257, dtype=dtype, device="mps")
    saved_x, saved_residual = x.clone(), residual.clone()
    op = ir.ops.fused_add_rms_norm if fused else ir.ops.rms_norm
    args = (x, residual, weight, 1e-6) if fused else (x, weight, 1e-6)
    cpu_args = tuple(item.cpu() if isinstance(item, torch.Tensor) else item for item in args)
    expected = op.impls["native"].impl_fn(*cpu_args)
    with op.set_priority(["omni_mps", "native"]):
        assert op.dispatch(*args).provider == "omni_mps"
        actual = op.maybe_inplace(*args) if fused else op(*args)
    if fused:
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result.cpu(), reference, **op.get_tolerance(dtype))
    else:
        torch.testing.assert_close(actual.cpu(), expected, **op.get_tolerance(dtype))
    torch.testing.assert_close(x, saved_x, atol=0, rtol=0)
    torch.testing.assert_close(residual, saved_residual, atol=0, rtol=0)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize(
    "case", ["fp32", "no_weight", "variance", "mixed_weight", "broadcast_weight", "empty"]
)
def test_unsupported_mps_norm_arguments_use_native_without_cpu_transfer(case):
    x = torch.randn(2, 128, dtype=torch.float16, device="mps")
    weight = torch.ones(128, dtype=x.dtype, device="mps")
    variance = None
    if case == "fp32":
        x, weight = x.float(), weight.float()
    elif case == "no_weight":
        weight = None
    elif case == "variance":
        variance = 64
    elif case == "mixed_weight":
        weight = weight.float()
    elif case == "broadcast_weight":
        weight = weight[:1]
    elif case == "empty":
        x = x[:0]
    op = ir.ops.rms_norm
    with op.set_priority(["omni_mps", "native"]):
        assert op.dispatch(x, weight, 1e-6, variance).provider == "native"
        actual = op(x, weight, 1e-6, variance)
    assert actual.device.type == "mps" and actual.dtype == x.dtype
    torch.testing.assert_close(
        actual, op.impls["native"].impl_fn(x, weight, 1e-6, variance), atol=0, rtol=0
    )


@pytest.fixture
def explicit_norm_shader(monkeypatch):
    from vllm_omni_metal.ops import vpipe_kernels

    libraries = {}

    def library(dtype):
        if dtype not in libraries:
            # Isolated test opt-in only; production remains load_metallib-only.
            libraries[dtype] = torch.mps.compile_shader(
                f"#define VPIPE_ELT {vpipe_kernels._SCALARS[dtype]}\n"
                + vpipe_kernels._SOURCE.read_text()
            )
        return libraries[dtype]

    monkeypatch.setattr(vpipe_kernels, "_library", library)
    return vpipe_kernels


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [128, 4096])
def test_ir_add_rms_norm_uses_single_kernel_with_upstream_rounding(
    explicit_norm_shader, monkeypatch, dtype, hidden_size
):
    kernels = explicit_norm_shader
    torch.manual_seed(17)
    op = ir.ops.fused_add_rms_norm
    args = op.generate_inputs(num_tokens=4, hidden_size=hidden_size, dtype=dtype, device="cpu")
    args[0][0].fill_(1)
    args[1][0] = (
        torch.tensor([0.125, 0.375, 0.625, 0.875], dtype=dtype).repeat(hidden_size // 4)
        * torch.finfo(dtype).eps
    )
    args[2].copy_(torch.tensor([1.25, 1.75, 0.75, 1.5], dtype=dtype).repeat(hidden_size // 4))
    expected = op.impls["native"].impl_fn(*args)
    gpu_args = tuple(item.to("mps") if isinstance(item, torch.Tensor) else item for item in args)
    saved = [item.clone() for item in gpu_args if isinstance(item, torch.Tensor)]
    calls = []
    add_rms_norm = kernels.add_rms_norm

    def fused(*inputs):
        calls.append(True)
        return add_rms_norm(*inputs)

    monkeypatch.setattr(kernels, "add_rms_norm", fused)
    # A successful fused path must not allocate the fallback's ones vector.
    monkeypatch.setattr(torch, "ones_like", lambda *a, **k: pytest.fail("unfused normalization"))
    with op.set_priority(["omni_mps", "native"]):
        assert op.dispatch(*gpu_args).provider == "omni_mps"
        actual, summed = op(*gpu_args)
    assert calls == [True]
    torch.testing.assert_close(actual.cpu(), expected[0], **op.get_tolerance(dtype))
    torch.testing.assert_close(actual[0].cpu(), expected[0][0], atol=0, rtol=0)
    torch.testing.assert_close(summed.cpu(), expected[1], atol=0, rtol=0)
    for item, before in zip(gpu_args[:3], saved, strict=True):
        torch.testing.assert_close(item, before, atol=0, rtol=0)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "case", ["missing_aot", "strided_x", "strided_residual", "strided_weight", "epsilon"]
)
def test_ir_add_rms_norm_keeps_omni_mps_fallback_without_jit(monkeypatch, dtype, case):
    from vllm_omni_metal.ops import vpipe_kernels

    torch.manual_seed(73)
    x = torch.randn(2, 3, 514, dtype=dtype, device="mps")
    residual = torch.randn_like(x)
    weight = torch.randn(514, dtype=dtype, device="mps")
    epsilon = 1e-6
    if case == "strided_x":
        x, residual, weight = x[..., ::2], residual[..., :257].contiguous(), weight[:257]
    elif case == "strided_residual":
        x, residual, weight = x[..., :257].contiguous(), residual[..., ::2], weight[:257]
    elif case == "strided_weight":
        x, residual, weight = (
            x[..., :257].contiguous(),
            residual[..., :257].contiguous(),
            weight[::2],
        )
    elif case == "epsilon":
        epsilon = 1e-40
    saved = [item.clone() for item in (x, residual, weight)]
    expected = ir.ops.fused_add_rms_norm.impls["native"].impl_fn(
        x.cpu(), residual.cpu(), weight.cpu(), epsilon
    )
    monkeypatch.setattr(torch.mps, "compile_shader", lambda _: pytest.fail("worker JIT forbidden"))
    monkeypatch.setattr(
        vpipe_kernels,
        "_library",
        lambda _: None if case == "missing_aot" else pytest.fail("unsupported shader load"),
    )
    monkeypatch.setattr(
        vpipe_kernels, "add_rms_norm", lambda *a: pytest.fail("unsupported fused dispatch")
    )
    op = ir.ops.fused_add_rms_norm
    with op.set_priority(["omni_mps", "native"]):
        assert op.dispatch(x, residual, weight, epsilon).provider == "omni_mps"
        actual = op(x, residual, weight, epsilon)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result.cpu(), reference, **op.get_tolerance(dtype))
    for item, before in zip((x, residual, weight), saved, strict=True):
        torch.testing.assert_close(item, before, atol=0, rtol=0)
