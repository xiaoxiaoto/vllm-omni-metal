# SPDX-License-Identifier: Apache-2.0
"""AOT kernels must preserve Torch rounding, cache tables and input storage."""

import math
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm_omni_metal.ops import aot_kernels


@pytest.mark.parametrize("args", [["--rows", "0"], ["--rows", "128", "--hidden-size", "16384"]])
def test_benchmark_budget_rejects_before_allocating(monkeypatch, args):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        aot_kernels.psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 1024**3)
    )
    monkeypatch.setattr(sys, "argv", ["aot_kernels.py", *args, "--compile-for-test"])
    monkeypatch.setattr(
        torch.mps, "compile_shader", lambda _: pytest.fail("budget gate must precede compilation")
    )
    script = Path(__file__).resolve().parents[1] / "examples/benchmarks/aot_kernels.py"
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2


def test_missing_artifact_never_compiles_or_creates_cache(tmp_path, monkeypatch):
    path = tmp_path / "absent" / "ops.metallib"
    monkeypatch.setattr(aot_kernels, "_artifact_path", lambda _: path)
    monkeypatch.setattr(torch.mps, "compile_shader", lambda _: pytest.fail("worker JIT forbidden"))
    aot_kernels._library.cache_clear()
    try:
        assert aot_kernels._library(torch.float16) is None
        assert not path.parent.exists()
    finally:
        aot_kernels._library.cache_clear()


@pytest.mark.parametrize("case", ["symlink", "public_directory", "public_file"])
def test_unsafe_aot_artifact_fails_explicitly(tmp_path, monkeypatch, case):
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    path = directory / "ops.metallib"
    path.write_bytes(b"invalid")
    if case == "symlink":
        real = directory / "real.metallib"
        path.rename(real)
        path.symlink_to(real)
    elif case == "public_directory":
        directory.chmod(0o755)
    else:
        path.chmod(0o666)
    monkeypatch.setattr(aot_kernels, "_artifact_path", lambda _: path)
    aot_kernels._library.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="private|unsafe"):
            aot_kernels._library(torch.float16)
    finally:
        aot_kernels._library.cache_clear()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cpu_never_uses_elementwise_shader(dtype):
    assert not aot_kernels.supports_silu(torch.ones(1, 128, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cpu_add_rms_norm_never_loads_shader(monkeypatch, dtype):
    monkeypatch.setattr(aot_kernels, "_library", lambda _: pytest.fail("CPU shader load"))
    x = torch.ones(2, 128, dtype=dtype)
    assert not aot_kernels.supports_add_rms_norm(x, x, x[0])


def _norm_metadata(shape=(2, 128), dtype=torch.float16, **overrides):
    metadata = dict(
        shape=shape,
        dtype=dtype,
        device=torch.device("mps"),
        ndim=len(shape),
        requires_grad=False,
        numel=lambda: math.prod(shape),
        is_contiguous=lambda: True,
    )
    metadata.update(overrides)
    return SimpleNamespace(**metadata)


@pytest.mark.parametrize(
    "shape,accepted",
    [
        ((), False),
        ((0, 128), False),
        ((2, 0), False),
        ((1,), True),
        ((2, 257), True),
        ((1, 65537), True),
        ((1, 2**31 - 1), True),
        ((1, 2**31), False),
        ((2, 2**30), False),
    ],
)
def test_add_rms_norm_shape_integer_and_width_domain(monkeypatch, shape, accepted):
    monkeypatch.setattr(aot_kernels, "_library", lambda _: object())
    x = _norm_metadata(shape)
    weight = _norm_metadata((shape[-1],) if shape else (1,))
    assert aot_kernels.supports_add_rms_norm(x, x, weight) is accepted


@pytest.mark.parametrize("input_index", [0, 1, 2])
@pytest.mark.parametrize("case", ["grad", "strided", "dtype", "device", "shape"])
def test_add_rms_norm_rejects_unsupported_tensor_contract(monkeypatch, input_index, case):
    monkeypatch.setattr(aot_kernels, "_library", lambda _: object())
    args = [_norm_metadata(), _norm_metadata(), _norm_metadata((128,))]
    target = args[input_index]
    if case == "grad":
        target.requires_grad = True
    elif case == "strided":
        target.is_contiguous = lambda: False
    elif case == "dtype":
        target.dtype = torch.float32
    elif case == "device":
        target.device = torch.device("cpu")
    else:
        target.shape = (3, 64)
    assert not aot_kernels.supports_add_rms_norm(*args)


@pytest.mark.parametrize(
    "epsilon,accepted",
    [
        (0.0, True),
        (1e-6, True),
        (1e-38, False),
        (-1.0, False),
        (float("nan"), False),
        (float("inf"), False),
        (1e39, False),
        (10**400, False),
    ],
)
def test_add_rms_norm_epsilon_cannot_silently_narrow(epsilon, accepted):
    assert aot_kernels._supports_add_rms_norm_epsilon(epsilon) is accepted
    if not accepted:
        with pytest.raises(ValueError, match="epsilon"):
            aot_kernels.add_rms_norm(None, None, None, epsilon)


def test_add_rms_norm_missing_artifact_rejects_without_compilation(monkeypatch):
    monkeypatch.setattr(aot_kernels, "_library", lambda _: None)
    monkeypatch.setattr(torch.mps, "compile_shader", lambda _: pytest.fail("worker JIT forbidden"))
    x, weight = _norm_metadata(), _norm_metadata((128,))
    assert not aot_kernels.supports_add_rms_norm(x, x, weight)
    with pytest.raises(ValueError):
        aot_kernels.add_rms_norm(x, x, weight, 1e-6)


def test_elementwise_build_missing_compiler_creates_nothing(tmp_path, monkeypatch):
    path = tmp_path / "absent" / "ops.metallib"
    monkeypatch.setattr(aot_kernels, "_artifact_path", lambda _: path)
    monkeypatch.setattr(aot_kernels.shutil, "which", lambda _: "/usr/bin/xcrun")
    monkeypatch.setattr(
        aot_kernels.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1)
    )
    with pytest.raises(RuntimeError, match="compiler unavailable"):
        aot_kernels.build_elementwise_kernels()
    assert not path.parent.exists()


@pytest.mark.parametrize("available,disk", [(2 * 1024**3, 1024**3), (8 * 1024**3, 128 * 1024**2)])
def test_elementwise_build_budget_stops_before_compiler(tmp_path, monkeypatch, available, disk):
    path = tmp_path / "absent" / "ops.metallib"
    monkeypatch.setattr(aot_kernels, "_artifact_path", lambda _: path)
    monkeypatch.setattr(
        aot_kernels.psutil, "virtual_memory", lambda: SimpleNamespace(available=available)
    )
    monkeypatch.setattr(aot_kernels.shutil, "disk_usage", lambda _: SimpleNamespace(free=disk))
    with pytest.raises(RuntimeError, match="budget"):
        aot_kernels.build_elementwise_kernels()
    assert not path.parent.exists()


MPS = pytest.mark.skipif(
    os.getenv("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="explicit MPS integration required",
)


@pytest.fixture
def explicit_test_shaders(monkeypatch):
    # Test-only compilation: production workers may only load explicitly built metallibs.
    libraries = {}

    def library(dtype):
        if dtype not in libraries:
            scalar = aot_kernels._SCALARS[dtype]
            libraries[dtype] = torch.mps.compile_shader(
                f"#define ELEMENTWISE_ELT {scalar}\n" + aot_kernels._SOURCE.read_text()
            )
        return libraries[dtype]

    monkeypatch.setattr(aot_kernels, "_library", library)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,width", [(1, 257), (4, 4096), (32, 8192)])
def test_packed_silu_matches_original_rounding(explicit_test_shaders, dtype, rows, width):
    gate = torch.linspace(-3, 3, width).to(dtype).repeat(rows, 1)
    up = torch.tensor([1.25, 1.75, 0.75, 1.5], dtype=dtype).repeat((width + 3) // 4)[:width]
    x = torch.cat((gate, up.expand_as(gate)), -1).to("mps")
    saved = x.clone()
    # Preserve the replaced MPS operation exactly; CPU SiLU differs at half ULP boundaries.
    expected = (torch.nn.functional.silu(x[..., :width]) * x[..., width:]).cpu()
    actual = aot_kernels.silu_and_mul(x)
    tol = 2e-6 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual.cpu(), expected, atol=tol, rtol=tol)
    torch.testing.assert_close(x, saved, atol=0, rtol=0)
    if rows == 1 and width == 257:
        torch.testing.assert_close(
            actual.cpu(), torch.nn.functional.silu(gate) * up, atol=tol, rtol=tol
        )


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens,dim,rot", [(1, 64, 64), (7, 64, 32)])
def test_cached_neox_rope_keeps_product_rounding_tail_and_inputs(
    explicit_test_shaders, dtype, tokens, dim, rot
):
    torch.manual_seed(53)
    x = torch.randn(tokens, 4, dim, device="mps", dtype=dtype)
    cos = torch.randn(tokens, rot // 2, dtype=dtype, device="mps")
    sin = torch.randn_like(cos)
    saved = x.clone()
    expected = x.clone()
    expected[..., :rot] = ApplyRotaryEmb.forward_static(x[..., :rot], cos, sin)
    actual = aot_kernels.cached_neox_rope(x, cos, sin, rot)
    tol = 2e-6 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    torch.testing.assert_close(x, saved, atol=0, rtol=0)
    torch.testing.assert_close(actual[..., rot:], x[..., rot:], atol=0, rtol=0)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens,q_heads,kv_heads,dim", [(1, 12, 2, 128), (19, 4, 1, 64)])
def test_packed_qk_rope_preserves_gqa_rounding_and_inputs(
    explicit_test_shaders, dtype, tokens, q_heads, kv_heads, dim
):
    torch.manual_seed(73)
    q = torch.randn(tokens, q_heads * dim, device="mps", dtype=dtype)
    k = torch.randn(tokens, kv_heads, dim, device="mps", dtype=dtype)
    cache = torch.randn(tokens, dim, device="mps", dtype=dtype)
    saved = [item.clone() for item in (q, k, cache)]
    cos, sin = cache.chunk(2, dim=-1)
    expected = [
        ApplyRotaryEmb.forward_static(x.view(tokens, -1, dim), cos, sin).view_as(x) for x in (q, k)
    ]
    actual = aot_kernels.cached_neox_rope_qk(q, k, cache, dim)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)
    for source, original in zip((q, k, cache), saved, strict=True):
        torch.testing.assert_close(source, original, atol=0, rtol=0)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1,), (3, 257), (2, 3, 4096), (1, 65537)])
def test_add_rms_norm_single_launch_double_output_and_immutable_inputs(
    explicit_test_shaders, monkeypatch, dtype, shape
):
    torch.manual_seed(71)
    x = torch.randn(shape, dtype=dtype, device="mps")
    residual = torch.randn_like(x)
    weight = torch.randn(shape[-1], dtype=dtype, device="mps")
    saved = [item.clone() for item in (x, residual, weight)]
    summed = x.cpu().float() + residual.cpu().float()
    expected = (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)).to(
        dtype
    ) * weight.cpu()
    library = aot_kernels._library(dtype)
    launches = []

    def launch(*args, **kwargs):
        launches.append(kwargs)
        return library.add_rms_norm_f16(*args, **kwargs)

    monkeypatch.setattr(aot_kernels, "_library", lambda _: SimpleNamespace(add_rms_norm_f16=launch))
    assert aot_kernels.supports_add_rms_norm(x, residual, weight)
    actual, updated_residual = aot_kernels.add_rms_norm(x, residual, weight, 1e-6)
    assert launches == [{"threads": (256, x.numel() // shape[-1]), "group_size": (256, 1)}]
    assert actual.shape == updated_residual.shape == x.shape
    assert actual.dtype == updated_residual.dtype == dtype
    assert actual.device == updated_residual.device == x.device
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(updated_residual.cpu(), summed.to(dtype), atol=0, rtol=0)
    for item, before in zip((x, residual, weight), saved, strict=True):
        torch.testing.assert_close(item, before, atol=0, rtol=0)
        assert actual.data_ptr() != item.data_ptr()
        assert updated_residual.data_ptr() != item.data_ptr()


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("key_present,is_neox", [(True, True), (False, True), (True, False)])
def test_oot_rotary_uses_upstream_cache_and_preserves_other_styles(
    explicit_test_shaders, monkeypatch, dtype, key_present, is_neox
):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    from vllm_omni_metal.ops import rotary

    config = VllmConfig()
    config.parallel_config.worker_cls = "vllm_omni_metal.native.worker.MetalARWorker"
    config.compilation_config.custom_ops = ["none", "+rotary_embedding"]
    with set_current_vllm_config(config):
        layer = RotaryEmbedding(64, 32, 128, 10000, is_neox, dtype).to("mps")
        # Deliberately non-default cache: no recomputed frequency/scaling may replace it.
        layer.cos_sin_cache.copy_(torch.randn_like(layer.cos_sin_cache))
        positions = torch.tensor([73, 5, 29], device="mps")
        q = torch.randn(3, 256, dtype=dtype, device="mps")
        k = torch.randn(3, 128, dtype=dtype, device="mps") if key_present else None
        expected = layer.forward_native(positions, q, k)
        actual = layer.forward_oot(positions, q, k)
    assert isinstance(layer, rotary.MetalRotaryEmbedding)
    tol = 2e-6 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual[0], expected[0], atol=tol, rtol=tol)
    if k is None:
        assert actual[1] is None
    else:
        torch.testing.assert_close(actual[1], expected[1], atol=tol, rtol=tol)
