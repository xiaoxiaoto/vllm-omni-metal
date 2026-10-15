# SPDX-License-Identifier: Apache-2.0
"""vpipe ports must preserve Torch rounding, cache tables and input storage."""

import hashlib
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm_omni_metal import vpipe_kernels


def test_adapted_vpipe_fragment_hash_is_frozen():
    assert hashlib.sha256(vpipe_kernels._SOURCE.read_bytes()).hexdigest() == (
        "ef51b6bbd6cf06ceae1eae032b308a7eb4697b9b250296ea7964b7b2a3df1cc4"
    )


@pytest.mark.parametrize("args", [["--rows", "0"], ["--rows", "128", "--hidden-size", "16384"]])
def test_benchmark_budget_rejects_before_allocating(monkeypatch, args):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        vpipe_kernels.psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 1024**3)
    )
    monkeypatch.setattr(sys, "argv", ["benchmark_vpipe.py", *args, "--compile-for-test"])
    monkeypatch.setattr(
        torch.mps, "compile_shader", lambda _: pytest.fail("budget gate must precede compilation")
    )
    script = Path(__file__).resolve().parents[1] / "examples/benchmark_vpipe.py"
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2


def test_missing_artifact_never_compiles_or_creates_cache(tmp_path, monkeypatch):
    path = tmp_path / "absent" / "ops.metallib"
    monkeypatch.setattr(vpipe_kernels, "_artifact_path", lambda _: path)
    monkeypatch.setattr(torch.mps, "compile_shader", lambda _: pytest.fail("worker JIT forbidden"))
    vpipe_kernels._library.cache_clear()
    try:
        assert vpipe_kernels._library(torch.float16) is None
        assert not path.parent.exists()
    finally:
        vpipe_kernels._library.cache_clear()


@pytest.mark.parametrize("case", ["symlink", "public_directory", "public_file"])
def test_unsafe_vpipe_artifact_fails_explicitly(tmp_path, monkeypatch, case):
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
    monkeypatch.setattr(vpipe_kernels, "_artifact_path", lambda _: path)
    vpipe_kernels._library.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="private|unsafe"):
            vpipe_kernels._library(torch.float16)
    finally:
        vpipe_kernels._library.cache_clear()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cpu_never_uses_vpipe_shader(dtype):
    assert not vpipe_kernels.supports_silu(torch.ones(1, 128, dtype=dtype))


def test_vpipe_build_missing_compiler_creates_nothing(tmp_path, monkeypatch):
    path = tmp_path / "absent" / "ops.metallib"
    monkeypatch.setattr(vpipe_kernels, "_artifact_path", lambda _: path)
    monkeypatch.setattr(vpipe_kernels.shutil, "which", lambda _: "/usr/bin/xcrun")
    monkeypatch.setattr(
        vpipe_kernels.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1)
    )
    with pytest.raises(RuntimeError, match="compiler unavailable"):
        vpipe_kernels.build_vpipe_kernels()
    assert not path.parent.exists()


@pytest.mark.parametrize("available,disk", [(2 * 1024**3, 1024**3), (8 * 1024**3, 128 * 1024**2)])
def test_vpipe_build_budget_stops_before_compiler(tmp_path, monkeypatch, available, disk):
    path = tmp_path / "absent" / "ops.metallib"
    monkeypatch.setattr(vpipe_kernels, "_artifact_path", lambda _: path)
    monkeypatch.setattr(
        vpipe_kernels.psutil, "virtual_memory", lambda: SimpleNamespace(available=available)
    )
    monkeypatch.setattr(vpipe_kernels.shutil, "disk_usage", lambda _: SimpleNamespace(free=disk))
    with pytest.raises(RuntimeError, match="budget"):
        vpipe_kernels.build_vpipe_kernels()
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
            scalar = vpipe_kernels._SCALARS[dtype]
            libraries[dtype] = torch.mps.compile_shader(
                f"#define VPIPE_ELT {scalar}\n" + vpipe_kernels._SOURCE.read_text()
            )
        return libraries[dtype]

    monkeypatch.setattr(vpipe_kernels, "_library", library)


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
    actual = vpipe_kernels.silu_and_mul(x)
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
    actual = vpipe_kernels.cached_neox_rope(x, cos, sin, rot)
    tol = 2e-6 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    torch.testing.assert_close(x, saved, atol=0, rtol=0)
    torch.testing.assert_close(actual[..., rot:], x[..., rot:], atol=0, rtol=0)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("key_present,is_neox", [(True, True), (False, True), (True, False)])
def test_oot_rotary_uses_upstream_cache_and_preserves_other_styles(
    explicit_test_shaders, monkeypatch, dtype, key_present, is_neox
):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    from vllm_omni_metal import rotary

    config = VllmConfig()
    config.parallel_config.worker_cls = "vllm_omni_metal.native_worker.MetalARWorker"
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
