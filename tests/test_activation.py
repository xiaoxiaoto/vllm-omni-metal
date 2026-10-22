# SPDX-License-Identifier: Apache-2.0
"""Scoped OOT activation dispatch, precision, and bounded microbenchmark."""

import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul

from vllm_omni_metal.ops import activation


def native_config(worker="vllm_omni_metal.native.worker.MetalARWorker"):
    config = VllmConfig()
    config.parallel_config.worker_cls = worker
    config.compilation_config.custom_ops = ["none", "+silu_and_mul"]
    return config


def test_fresh_worker_registers_operator_not_models_or_original_methods():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['mlx'] = None; "
            "from vllm.model_executor.layers.activation import SiluAndMul; "
            "from vllm.model_executor.custom_op import op_registry, op_registry_oot; "
            "from vllm.model_executor.models import ModelRegistry; "
            "models = dict(ModelRegistry.models); ops = dict(op_registry); "
            "original = SiluAndMul.forward_native; "
            "import vllm_omni_metal.native.worker; "
            "from vllm_omni_metal.ops.activation import MetalSiluAndMul; "
            "assert op_registry_oot['SiluAndMul'] is MetalSiluAndMul; "
            "assert all(op_registry[k] is v for k, v in ops.items()); "
            "assert ModelRegistry.models == models; "
            "assert SiluAndMul.forward_native is original; "
            "assert sys.modules['mlx'] is None",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )


@pytest.mark.parametrize(
    "worker", ["auto", "external.Worker", "vllm_omni_metal.native.worker.MetalARWorker"]
)
def test_operator_scope_and_cpu_fallback(monkeypatch, worker):
    config = native_config(worker)
    x = torch.randn(2, 16)
    calls = []
    monkeypatch.setattr(activation, "_supports_compiled_silu", lambda x: True)
    monkeypatch.setattr(
        activation,
        "_silu_and_mul_mps",
        lambda x: calls.append(x) or SiluAndMul.forward_native(x),
    )
    with set_current_vllm_config(config):
        layer = SiluAndMul(compile_native=False)
        assert isinstance(layer, activation.MetalSiluAndMul)
        actual = layer.forward_oot(x)
    torch.testing.assert_close(actual, SiluAndMul.forward_native(x), atol=0, rtol=0)
    assert len(calls) == (1 if worker.startswith("vllm_omni_metal.native.worker.") else 0)
    with set_current_vllm_config(config):
        monkeypatch.setattr(activation, "_supports_compiled_silu", lambda x: False)
        layer.forward_oot(x)
    assert len(calls) == (1 if worker.startswith("vllm_omni_metal.native.worker.") else 0)
    layer.forward_oot(x)  # No active engine configuration must not select the bridge.
    assert len(calls) == (1 if worker.startswith("vllm_omni_metal.native.worker.") else 0)


@pytest.mark.parametrize(
    "change,expected",
    [
        ({}, True),
        ({"device": torch.device("cpu")}, False),
        ({"dtype": torch.float64}, False),
        ({"requires_grad": True}, False),
        ({"ndim": 0}, False),
        ({"ndim": 4}, False),
        ({"shape": (32, 16385)}, False),
        ({"is_contiguous": lambda: False}, False),
        ({"numel": lambda: 2 * (262144 - 1)}, False),
        ({"numel": lambda: 0}, False),
    ],
)
def test_compilation_gate_is_parameter_based(change, expected):
    metadata = dict(
        device=torch.device("mps"),
        dtype=torch.bfloat16,
        requires_grad=False,
        ndim=2,
        shape=(32, 16384),
        is_contiguous=lambda: True,
        numel=lambda: 2 * 262144,
    )
    metadata.update(change)
    assert activation._supports_compiled_silu(SimpleNamespace(**metadata)) is expected


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
def test_activation_benchmark_budget_stops_before_allocating(monkeypatch, args, available):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=available))
    monkeypatch.setattr(sys, "argv", ["activation.py", *args])

    def forbidden(*args, **kwargs):
        raise AssertionError("unsafe benchmark must not allocate GPU tensors")

    monkeypatch.setattr(torch, "randn", forbidden)
    script = Path(__file__).resolve().parents[1] / "examples/benchmarks/activation.py"
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2


MPS = pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="explicit MPS integration opt-in required",
)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(524288,), (32, 16384), (2, 16, 16384)])
def test_mps_compiled_activation_parity_without_mutation_or_donation(dtype, shape):
    torch.manual_seed(29)
    x = torch.randn(shape, dtype=dtype, device="mps")
    before = x.clone()
    expected = SiluAndMul.forward_native(x.cpu())
    with set_current_vllm_config(native_config()):
        layer = SiluAndMul(compile_native=False)
        assert activation._supports_compiled_silu(x)
        actual = layer(x)
    assert actual.device.type == "mps" and actual.dtype == dtype
    assert actual.shape == expected.shape and torch.isfinite(actual).all()
    tol = {torch.float32: 2e-6, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(actual.cpu(), expected, atol=tol, rtol=tol)
    torch.testing.assert_close(x, before, atol=0, rtol=0)
    held = actual.clone()
    with set_current_vllm_config(native_config()):
        x.fill_(0)
        second = layer(x)
    torch.testing.assert_close(second, torch.zeros_like(second), atol=0, rtol=0)
    del x, second
    torch.mps.synchronize()
    torch.testing.assert_close(actual, held, atol=0, rtol=0)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_torch_stream_activation_producer_consumer_without_mlx(monkeypatch, dtype):
    from vllm_omni_metal.ops import vpipe_kernels

    config = native_config()

    library = torch.mps.compile_shader(
        f"#define VPIPE_ELT {vpipe_kernels._SCALARS[dtype]}\n" + vpipe_kernels._SOURCE.read_text()
    )
    monkeypatch.setattr(vpipe_kernels, "_library", lambda _: library)
    monkeypatch.setitem(sys.modules, "mlx", None)
    torch.manual_seed(29)
    x = torch.randn(32, 16384, dtype=dtype, device="mps")
    before = x.cpu()
    expected = (SiluAndMul.forward_native(x * 0.5) + 1).cpu()
    with set_current_vllm_config(config):
        layer = SiluAndMul(compile_native=False)
        actual = layer(x * 0.5) + 1
    tolerance = 2e-6 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(x.cpu(), before, atol=0, rtol=0)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_activation_cast_before_gate_is_discriminating(monkeypatch, dtype):
    gate = torch.linspace(-3, 3, 257).to(dtype)
    up = torch.tensor([1.25, 1.75, 0.75, 1.5], dtype=dtype).repeat(65)[:257]
    expected = torch.nn.functional.silu(gate) * up
    unrounded = (torch.nn.functional.silu(gate.float()) * up.float()).to(dtype)
    mask = expected != unrounded
    assert mask.sum() > 50
    # Exercise the bridge even when an optional Torch-stream artifact is installed.
    monkeypatch.setattr(activation.vpipe_kernels, "supports_silu", lambda _: False)
    actual = activation._silu_and_mul_mps(torch.cat([gate, up]).to("mps")).cpu()
    torch.testing.assert_close(actual[mask], expected[mask], atol=0, rtol=0)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("case", ["small", "noncontiguous", "gradient", "disabled", "empty", "odd"])
def test_mps_activation_fallback_keeps_upstream_semantics(monkeypatch, case):
    config = native_config()
    x = torch.randn(32, 16384, device="mps")
    if case == "small":
        x = x[:1]
    elif case == "noncontiguous":
        x = x.T
    elif case == "gradient":
        x.requires_grad_(True)
    elif case == "disabled":
        config.compilation_config.custom_ops = ["none", "-silu_and_mul"]
    elif case == "empty":
        x = x[:0]
    elif case == "odd":
        x = x[:, :16383].contiguous()

    def forbidden(*args, **kwargs):
        raise AssertionError("fallback must not call the cross-framework kernel")

    monkeypatch.setattr(activation, "_silu_and_mul_mps", forbidden)
    with set_current_vllm_config(config):
        layer = SiluAndMul(compile_native=False)
        if case == "odd":
            with pytest.raises(RuntimeError):
                layer(x)
            return
        actual = layer(x)
    assert actual.device.type == "mps" and actual.dtype == x.dtype
    torch.testing.assert_close(actual, SiluAndMul.forward_native(x), atol=0, rtol=0)
    if case == "gradient":
        actual.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
