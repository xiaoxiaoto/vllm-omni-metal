# SPDX-License-Identifier: Apache-2.0
"""Optional upstream Torch-stream launcher, without runtime compilation."""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_omni_metal.ops import mps_attention


def test_missing_artifact_does_not_compile_import_or_create_cache(tmp_path, monkeypatch):
    path = tmp_path / "not-created" / "native.so"
    monkeypatch.setattr(mps_attention, "_artifact_path", lambda: path)
    mps_attention.get_mps_ops.cache_clear()
    try:
        assert mps_attention.get_mps_ops() is None
        assert not path.parent.exists()
    finally:
        mps_attention.get_mps_ops.cache_clear()


@pytest.mark.parametrize("case", ["symlink", "public_directory", "public_binary"])
def test_native_artifact_rejects_unsafe_cache(tmp_path, monkeypatch, case):
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    path = directory / "native.so"
    path.write_bytes(b"not executable")
    if case == "symlink":
        original = directory / "original.so"
        path.rename(original)
        path.symlink_to(original)
    elif case == "public_directory":
        directory.chmod(0o755)
    else:
        path.chmod(0o666)
    monkeypatch.setattr(mps_attention, "_artifact_path", lambda: path)
    mps_attention.get_mps_ops.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="private|unsafe"):
            mps_attention.get_mps_ops()
    finally:
        mps_attention.get_mps_ops.cache_clear()


@pytest.mark.parametrize("available,disk", [(2 * 1024**3, 1024**3), (8 * 1024**3, 128 * 1024**2)])
def test_build_budget_stops_before_creating_artifacts(tmp_path, monkeypatch, available, disk):
    path = tmp_path / "absent" / "native.so"
    monkeypatch.setattr(mps_attention, "_artifact_path", lambda: path)
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=available))
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=disk))
    with pytest.raises(RuntimeError, match="budget|headroom"):
        mps_attention.build_mps_launcher()
    assert not path.parent.exists()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bridge_keeps_absent_native_provider_for_all_dtypes(monkeypatch, dtype):
    from vllm_omni_metal.native.attention import MetalAttentionMetadata
    from vllm_omni_metal.ops import metal_attention

    calls = []
    monkeypatch.setattr(mps_attention, "get_mps_ops", lambda: calls.append("lookup") or None)
    # A missing mlx proves fallback selection before any kernel or cache write.
    monkeypatch.setitem(__import__("sys").modules, "mlx.core", None)
    q = torch.empty(1, 4, 64, dtype=dtype)
    k = torch.empty(1, 2, 64, dtype=dtype)
    cache = torch.empty(2, 2, 16, 2, 64, dtype=dtype)
    meta = MetalAttentionMetadata([0, 1], [1], [[0]])
    with pytest.raises(ModuleNotFoundError):
        metal_attention.paged_attention(q, k, k, cache, meta, 0.125, None)
    assert calls == ["lookup"]


def test_unrelated_shader_and_notice_do_not_invalidate_host_artifact(tmp_path, monkeypatch):
    native = tmp_path / "native"
    shutil.copytree(mps_attention._NATIVE, native)
    monkeypatch.setattr(mps_attention, "_NATIVE", native)
    original = mps_attention._artifact_path()
    (native / "vpipe_elementwise.metal").write_text("unrelated shader change")
    (native / "VPIPE.md").write_text("unrelated notice change")
    assert mps_attention._artifact_path() == original


def test_cache_identity_changes_with_torch_version(monkeypatch):
    original = mps_attention._artifact_path()
    monkeypatch.setattr(torch, "__version__", "different-abi-version")
    assert mps_attention._artifact_path() != original


def test_strides_over_int32_fail_before_native_cache_write(monkeypatch):
    from vllm_omni_metal.native.attention import MetalAttentionMetadata
    from vllm_omni_metal.ops import metal_attention

    class ForbiddenLauncher:
        def forward(self, *args):
            raise AssertionError("bad strides must fail before KV writes")

    class MetadataOnlyCache:
        shape = (2, 1, 16, 2, 64)

        def stride(self, dim):
            return 2**31 if dim == 1 else 64

    monkeypatch.setattr(mps_attention, "get_mps_ops", lambda: ForbiddenLauncher())
    q = torch.empty(1, 4, 64, dtype=torch.float16)
    k = torch.empty(1, 2, 64, dtype=q.dtype)
    meta = MetalAttentionMetadata([0, 1], [1], [[0]])
    with pytest.raises(ValueError, match="int32"):
        metal_attention.paged_attention(q, k, k, MetadataOnlyCache(), meta, 0.125, None)


def test_vendored_launcher_only_adapts_fp32_dispatch_from_immutable_upstream():
    root = Path(mps_attention.__file__).parents[1] / "_native"
    expected = {
        "pytorch_backend/mps_ops.mm": "cf4d246d5b6fd7388c55fb0d1a2ffa70f868a6f93c5ec3fc69a25762a42cd9d5",
        "metal/metal_device.h": "c706594db1efcc2285ad7a77c070bd233aa1a02636af974325c0589df57edb4f",
        "metal/paged_attention_kernels.h": "6bfc32d2d8458744c1bc0978904c7a979e4855470a4de0428da8cf268777d662",
    }
    for name, digest in expected.items():
        source = (root / name).read_bytes()
        if name == "pytorch_backend/mps_ops.mm":
            source = (
                source.decode()
                .replace(
                    "TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16 ||\n"
                    "                q.scalar_type() == at::kFloat,\n"
                    '                "MPS attention requires fp32, fp16 or bf16");',
                    "TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,\n"
                    '                "Experimental MPS attention requires fp16 or bf16");',
                )
                .replace(
                    "    // The released tiled kernels only cover half/bfloat; float uses the same paged kernel.\n"
                    "    const bool prefill = tokens > seqs && !window && q.scalar_type() != at::kFloat;",
                    "    const bool prefill = tokens > seqs && !window;",
                )
                .replace(
                    '    const std::string dt = q.scalar_type() == at::kHalf ? "half" :\n'
                    '        q.scalar_type() == at::kBFloat16 ? "bfloat16_t" : "float";',
                    '    const std::string dt = q.scalar_type() == at::kHalf ? "half" : "bfloat16_t";',
                )
                .encode()
            )
        assert hashlib.sha256(source).hexdigest() == digest


MPS = pytest.mark.skipif(
    os.getenv("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="requires explicit MPS integration opt-in",
)


@pytest.mark.integration
@MPS
def test_matching_native_artifact_is_available():
    if mps_attention.get_mps_ops() is None:
        pytest.skip("optional launcher has not been explicitly built")


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_fresh_native_route_without_mlx_synchronize_or_runtime_build(dtype):
    if mps_attention.get_mps_ops() is None:
        pytest.skip("optional launcher has not been explicitly built")
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['mlx'] = None
sys.modules['mlx.core'] = None
import torch
from vllm_omni_metal.ops import mps_attention
from vllm_omni_metal.native.attention import MetalAttentionImpl, MetalAttentionMetadata
assert mps_attention.get_mps_ops() is not None
def forbidden(*args, **kwargs):
    raise AssertionError('native attention must not synchronize or compile')
mps_attention.build_mps_launcher = forbidden
torch.mps.synchronize = forbidden
heads, kv_heads, dim = 4, 2, 64
impl = MetalAttentionImpl(heads, dim, dim**-0.5, kv_heads)
cache = torch.empty(2, 4, 16, kv_heads, dim, dtype=torch.bfloat16, device='mps')
outputs = []
for step in range(4):
    # No CPU read between Torch producers, cache scatter, attention, and consumer.
    x = torch.ones(2, heads, dim, device='mps', dtype=torch.bfloat16) * (step + 1)
    q = x * 0.1
    k = x[:, :kv_heads].contiguous()
    v = torch.full_like(k, 3.0)
    meta = MetalAttentionMetadata([0, 1, 2], [step+1, step+1], [[1], [3]])
    actual = impl.forward(None, q, k, v, cache, meta)
    outputs.append(actual * 2)
for result in outputs:
    # FP32 paged softmax retains its existing FP64-reference tolerance, not bitwise reduction.
    tolerance = 2e-5 if result.dtype == torch.float32 else 0
    torch.testing.assert_close(result.cpu(), torch.full(result.shape, 6.0, dtype=result.dtype), atol=tolerance, rtol=tolerance)
assert sys.modules['mlx'] is None and sys.modules['mlx.core'] is None
""".replace("torch.bfloat16", f"torch.{dtype}"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
