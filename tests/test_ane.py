# SPDX-License-Identifier: Apache-2.0
"""Generic cooperative SwiGLU contracts; no model registry or real weights."""

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from vllm_omni_metal.ane import cache
from vllm_omni_metal.ane.build import publish_profile
from vllm_omni_metal.ane.runtime import DenseSwiGLU, _headroom_retry, execution_scope


@pytest.fixture
def private_cache(tmp_path, monkeypatch):
    root = tmp_path / "private" / "ane" / ("a" * 64)
    monkeypatch.setattr(cache, "cache_root", lambda: root)
    cache.ensure_cache()
    return root


@pytest.mark.parametrize("changed", ["ops/activation.py", "ane/runtime.py", "_native/ane/emit.cc"])
def test_cache_identity_tracks_nested_sources(tmp_path, monkeypatch, changed):
    package = tmp_path / "vllm_omni_metal"
    for name in ("ane/cache.py", "ops/activation.py", "ane/runtime.py", "_native/ane/emit.cc"):
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    monkeypatch.setattr(cache, "__file__", str(package / "ane/cache.py"))
    cache.cache_root.cache_clear()
    try:
        initial = cache.cache_root()
        source = package / changed
        original = source.read_text()
        source.write_text("changed")
        cache.cache_root.cache_clear()
        assert cache.cache_root() != initial
        source.write_text(original)
        cache.cache_root.cache_clear()
        assert cache.cache_root() == initial
    finally:
        cache.cache_root.cache_clear()


def test_execution_scope_uses_real_ir_dataclass_and_resource_settings(tmp_path, monkeypatch):
    from vllm.config import CacheConfig, CompilationConfig
    from vllm.config.kernel import KernelConfig

    monkeypatch.setattr("vllm_omni_metal.ops.mps_attention.get_mps_ops", lambda: None)
    monkeypatch.setattr(
        "vllm_omni_metal.ops.aot_kernels._artifact_path", lambda dtype: tmp_path / "absent"
    )
    config = SimpleNamespace(
        cache_config=CacheConfig(),
        compilation_config=CompilationConfig(),
        kernel_config=KernelConfig(),
        model_config=SimpleNamespace(dtype=torch.bfloat16, max_model_len=6144),
        attention_config=SimpleNamespace(backend="CUSTOM"),
        scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=4096),
    )
    initial = execution_scope("revision", config)
    assert initial == execution_scope("revision", config)
    config.cache_config.kv_cache_memory_bytes = 1024**3
    assert initial != execution_scope("revision", config)
    assert initial != execution_scope("other-revision", config)


def test_missing_artifacts_keep_gpu_but_invalid_artifacts_fail(private_cache, monkeypatch):
    weight = SimpleNamespace(device=SimpleNamespace(type="mps"), _version=0)
    key = "b" * 64
    monkeypatch.setattr(cache, "weight_shape", lambda *args: (1024, 1024))
    monkeypatch.setattr(cache, "eligible", lambda *args: True)
    monkeypatch.setattr(cache, "weight_key", lambda *args: key)
    monkeypatch.setattr(
        "vllm_omni_metal.ane.runtime._Bridge", lambda: SimpleNamespace(device="test")
    )
    (private_cache / "tools.json").write_text("stub")
    identity = cache.group_key([key], "dense-swiglu")
    cache.write_json(
        private_cache / f"{identity}.json",
        {
            "group": identity,
            "device": "test",
            "profiles": {"514": cache.select_profile(514, [10, 11], {2: [8, 9]})},
        },
    )

    def absent(*args, **kwargs):
        raise FileNotFoundError("missing graph")

    monkeypatch.setattr(cache, "verify_model", absent)
    with DenseSwiGLU([(weight, weight)], precision="fp16") as group:
        assert group._pool is None
    with pytest.raises(FileNotFoundError):
        DenseSwiGLU([(weight, weight)], precision="fp16", _benchmark=True)

    def invalid(*args, **kwargs):
        raise RuntimeError("checksum mismatch")

    monkeypatch.setattr(cache, "verify_model", invalid)
    with pytest.raises(RuntimeError, match="checksum"):
        DenseSwiGLU([(weight, weight)], precision="fp16")


def test_weight_and_group_identity():
    p = torch.zeros(2048, 1024, dtype=torch.bfloat16)
    d = torch.zeros(1024, 1024, dtype=p.dtype)
    assert cache.eligible(p, d)
    before = cache.weight_key(p, d)
    p[0, 0] = 1
    assert cache.weight_key(p, d) != before
    assert cache.weight_key(p.half(), d.half()) != cache.weight_key(p, d)
    assert cache.group_key([before], "a") != cache.group_key([before], "b")
    assert not cache.eligible(p.float(), d.float())
    assert not cache.eligible(torch.zeros(4096, 1536), torch.zeros(1536, 2048))
    large_p = torch.empty((32768, 4096), dtype=torch.float16, device="meta")
    large_d = torch.empty((4096, 16384), dtype=torch.float16, device="meta")
    assert not cache.eligible(large_p, large_d)
    with pytest.raises(ValueError):
        cache.weight_shape(p, d.T[:, :2])
    with pytest.raises(ValueError):
        cache.group_key([before], "")
    with pytest.raises(RuntimeError):
        cache.model_directory("../outside")


def test_private_cache_integrity_and_atomic_json(private_cache):
    path = private_cache / "test.json"
    cache.write_json(path, {"value": 1})
    assert cache.read_json(path)["value"] == 1
    with pytest.raises(ValueError):
        cache.write_json(path, {"bad": float("nan")})
    assert cache.read_json(path)["value"] == 1
    assert list(private_cache.iterdir()) == [path]
    path.chmod(0o666)
    with pytest.raises(RuntimeError, match="Unsafe"):
        cache.read_json(path)
    path.chmod(0o600)
    path.write_text('{"schema": true}')
    with pytest.raises(RuntimeError, match="schema"):
        cache.read_json(path)
    path.unlink()
    path.symlink_to(private_cache.parent / "missing")
    with pytest.raises(RuntimeError):
        cache.read_json(path)
    path.unlink()
    private_cache.chmod(0o755)
    with pytest.raises(RuntimeError, match="private"):
        cache.check_existing_cache()


def test_model_digest_and_selftest_gate(private_cache):
    key = "b" * 64
    directory = cache.model_directory(key)
    model = directory / "module.mlmodelc"
    (model / "weights").mkdir(mode=0o700, parents=True)
    directory.chmod(0o700)
    model.chmod(0o700)
    names = ["model.mil", "coremldata.bin", "weights/weight.bin"]
    for name in names:
        (model / name).write_bytes(b"test")
    record = {
        "key": key,
        "shape": [256, 1024, 1024],
        "device": "test",
        "runtime": False,
        "ane_preferred": True,
        "files": {name: cache.checksum(model / name) for name in names},
        "numerical_tests": [{"relative_l2": 0.001, "scaled_max_abs": 0.002}] * 2,
    }
    cache.write_json(directory / "manifest.json", record)
    assert cache.verify_model(key, 1024, 1024, "test") == model
    with pytest.raises(RuntimeError, match="contract"):
        cache.verify_model(key, 1024, 1024, "another-device")
    (model / "model.mil").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="checksum"):
        cache.verify_model(key, 1024, 1024, "test")
    (model / "model.mil").write_bytes(b"test")
    record["numerical_tests"][0] = {"relative_l2": 0.02, "scaled_max_abs": 0}
    cache.write_json(directory / "manifest.json", record)
    with pytest.raises(RuntimeError, match="self-test"):
        cache.verify_model(key, 1024, 1024, "test")


def test_whole_pipeline_policy_is_conservative_and_shape_bound():
    # Half splits can be slower; a fast isolated CoreML kernel is not a pipeline profile.
    record = cache.select_profile(514, [37, 38], {1: [38, 39], 2: [35, 36]})
    assert record["chunks"] == 2
    assert cache.validate_profiles({"514:566": record}) == {(514, 566): 2}
    assert cache.select_profile(514, [37, 38], {1: [36.5, 37]})["chunks"] == 0
    assert cache.select_profile(514, [30, 40], {1: [29, 32]})["chunks"] == 0
    for rows in (1, 54, 202, 256):
        with pytest.raises(ValueError):
            cache.select_profile(rows, [1, 1], {})
    for chunks in (True, -1, 3):
        with pytest.raises(ValueError):
            cache.select_profile(514, [1, 1], {chunks: [0.5, 0.5]})
    for invalid in ([1], [1, float("inf")], [True, 2], [0, 1]):
        with pytest.raises(ValueError):
            cache.samples(invalid)
    for label in ("0514", "514:-1", "514:True", "514:1:2"):
        with pytest.raises(RuntimeError):
            cache.validate_profiles({label: record})


def test_bounded_up_only_headroom():
    scales = []
    raw = [torch.tensor([float("inf")]), torch.tensor([float("inf")]), torch.tensor([1.0])]
    value, scale, retries = _headroom_retry(lambda: raw.pop(0), scales.append, 1)
    assert value.item() == 1 and scale == 16 and retries == 2 and scales == [4, 16]
    scales.clear()
    value, scale, retries = _headroom_retry(lambda: torch.tensor([float("nan")]), scales.append, 1)
    assert value is None and scale == 4096 and retries == 6
    assert scales == [4, 16, 64, 256, 1024, 4096]


def test_native_default_opt_out_and_gradient_boundary(monkeypatch):
    p, d = torch.randn(8, 4), torch.randn(4, 4)
    x = torch.randn(54, 4)
    monkeypatch.setenv("VLLM_OMNI_METAL_ANE", "off")
    with DenseSwiGLU([(p, d)]) as group:
        with pytest.raises(RuntimeError, match="inference"):
            group(0, x)
        with torch.inference_mode():
            assert torch.equal(group(0, x), DenseSwiGLU.native(x, p, d))
            with pytest.raises(ValueError):
                group(0, x.double())
            with pytest.raises(ValueError, match="context"):
                group(0, x, context=True)
        assert group._pool is None
    with torch.inference_mode(), pytest.raises(RuntimeError, match="closed"):
        group(0, x)
    monkeypatch.setenv("VLLM_OMNI_METAL_ANE", "force")
    with pytest.raises(ValueError, match="auto or off"):
        DenseSwiGLU([(p, d)])


def test_split_joins_worker_preserves_input_and_recomputes_only_lost_tail(monkeypatch):
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=10 * 1024**3))
    packed, down = torch.cat([torch.eye(2), torch.eye(2)]).half(), torch.eye(2).half()
    modes = []
    calls = []
    unloaded = []

    def predict(handle, inp, out, inputs, outputs):
        modes.append(torch.is_inference_mode_enabled())
        calls.append(handle)
        if len(calls) == 2:
            return -4
        source = torch.frombuffer(
            (ctypes.c_uint16 * inputs).from_address(inp), dtype=torch.float16
        ).reshape(256, 2)
        target = torch.frombuffer(
            (ctypes.c_uint16 * outputs).from_address(out), dtype=torch.float16
        )
        target.copy_(DenseSwiGLU.native(source, packed, down).flatten())
        return 0

    with torch.inference_mode(), DenseSwiGLU([(packed, down)]) as group:
        group._bridge = SimpleNamespace(
            lib=SimpleNamespace(ffn_predict=predict, ffn_unload=unloaded.append)
        )
        group._handles = [0]
        group._runtime = {(2, 2): 1}
        group._pool = ThreadPoolExecutor(max_workers=1)
        group._benchmark = True
        group._benchmark_chunks = 2
        x = torch.ones(514, 2, dtype=torch.float16)
        before = x.clone()
        result = group(0, x)
        assert torch.equal(x, before)
        assert torch.equal(result, DenseSwiGLU.native(x, packed, down))
        assert group.disabled and modes == [True, True]
        assert group.metrics["ane_chunks"] == 1
        assert group.metrics["gpu_recomputed_rows"] == 256
        assert sorted(unloaded) == [0, 1]
        assert torch.equal(group(0, x), result) and len(calls) == 2


def test_approval_requires_every_live_layer(private_cache):
    group = SimpleNamespace(
        _benchmark=True,
        disabled=False,
        identity="b" * 64,
        _bridge=SimpleNamespace(device="test"),
        weights=[1, 2],
        checks={},
    )
    with pytest.raises(ValueError, match="every FFN"):
        publish_profile(group, 514, [10, 11], {2: [8, 9]}, context=566)
    group.checks = {(i, 514, 2, 566): {} for i in range(2)}
    assert publish_profile(group, 514, [10, 11], {2: [8, 9]}, context=566)["chunks"] == 2
    with pytest.raises(ValueError):
        publish_profile(group, 514, [10, 11], {2: [8, 9]}, context=1000)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1", reason="explicit MPS opt-in"
)
def test_real_coreml_bridge_and_mps_source_immutability(tmp_path, monkeypatch):
    """Tiny structured FFN only; fake timing is confined to a temporary test cache."""
    root = tmp_path / "private" / "ane" / cache.cache_root().name
    monkeypatch.setattr(cache, "cache_root", lambda: root)
    from vllm_omni_metal.ane.build import prepare_group

    with torch.inference_mode():
        generator = torch.Generator().manual_seed(42)
        p = torch.zeros((12288, 2048), dtype=torch.bfloat16)
        d = torch.zeros((2048, 6144), dtype=torch.bfloat16)
        diagonal = torch.arange(2048)
        p[diagonal, diagonal] = 0.5
        p[6144 + diagonal, diagonal] = 0.5
        d[diagonal, diagonal] = 0.5
        p, d = p.to("mps"), d.to("mps")
        x = torch.randn(514, 2048, generator=generator).to(device="mps", dtype=p.dtype)
        original = x.clone()
        prepare_group([(p, d)], scope="random-ffn-test")
        with DenseSwiGLU(
            [(p, d)], scope="random-ffn-test", precision="fp16", _benchmark=True
        ) as group:
            for chunks in (1, 2):
                group._benchmark_chunks = chunks
                result = group(0, x, context=7)
                assert torch.isfinite(result).all() and not group.disabled
            assert torch.equal(x, original)
            assert group.metrics["ane_chunks"] == 3
            # Fixture policy exercises default reader/identity, NOT a performance claim.
            publish_profile(group, 514, [10, 11], {2: [8, 9]}, context=7)
        with DenseSwiGLU([(p, d)], scope="random-ffn-test", precision="fp16") as group:
            assert not group._benchmark
            assert torch.equal(group(0, x, context=7), result)
            before = group.metrics["ane_chunks"]
            group(0, x, context=8)
            assert group.metrics["ane_chunks"] == before
