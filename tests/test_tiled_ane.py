# SPDX-License-Identifier: Apache-2.0
"""Tiled FFN consumers: unchanged native routing, final gates and bounded ownership."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from vllm_omni_metal.ane import build as ane_build
from vllm_omni_metal.ane import cache
from vllm_omni_metal.ane import runtime as ane


@pytest.fixture
def private_cache(tmp_path, monkeypatch):
    root = tmp_path / "private" / "ane" / ("a" * 64)
    monkeypatch.setattr(cache, "cache_root", lambda: root)
    cache.ensure_cache()
    return root


@pytest.fixture(autouse=True)
def resource_headroom(monkeypatch):
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=64 * 1024**3))
    monkeypatch.setenv("VLLM_OMNI_METAL_ANE", "auto")


@pytest.mark.parametrize(
    ("hidden", "inner", "expected"),
    [
        (4096, 10240, ((0, 10240),)),
        (4096, 11264, ((0, 10240), (10240, 11264))),
        (4096, 12288, ((0, 10240), (10240, 12288))),
        (1024, 44032, ((0, 43008), (43008, 44032))),
    ],
)
def test_inner_partition_keeps_each_graph_within_existing_budget(hidden, inner, expected):
    assert cache.inner_tiles(hidden, inner) == expected
    assert expected[0][0] == 0 and expected[-1][1] == inner
    for start, end in expected:
        assert start % cache.TILE == end % cache.TILE == 0
        assert 6 * hidden * (end - start) <= cache.MAX_WEIGHTS
    packed = torch.empty((inner * 2, hidden), dtype=torch.float16, device="meta")
    down = torch.empty((hidden, inner), dtype=torch.float16, device="meta")
    if len(expected) > 1:
        assert not cache.eligible(packed, down)


@pytest.mark.parametrize("shape", [(0, 1024), (1024, 1025), (1536, 2048), (65536, 1024)])
def test_invalid_or_oversize_single_tile_is_rejected(shape):
    with pytest.raises(ValueError):
        cache.inner_tiles(*shape)


def test_packing_preserves_gate_up_and_corresponding_down_columns():
    gate = torch.arange(24, dtype=torch.float16).reshape(6, 4)
    up = gate + 100
    down = torch.arange(24, dtype=torch.float16).reshape(4, 6)
    plan = ((0, 4), (4, 6))
    for host in (False, True):
        parts = list(cache.tiled_weights([(gate, up, down)], [plan], host=host))
        for (packed, projected), (start, end) in zip(parts, plan, strict=True):
            g, u = packed.chunk(2)
            assert torch.equal(g, gate[start:end])
            assert torch.equal(u, up[start:end])
            assert torch.equal(projected, down[:, start:end])
            assert packed.is_contiguous() and projected.is_contiguous()
    assert cache.tiled_weight_shape(gate, up, down) == (4, 6)
    with pytest.raises(ValueError):
        cache.tiled_weight_shape(gate, up[:, :3], down)


@pytest.mark.parametrize("precision,mode", [("native", "auto"), ("fp16", "off"), ("fp16", "auto")])
@pytest.mark.parametrize("strided", [False, True])
def test_inactive_routing_calls_original_whole_bsh_ffn(monkeypatch, precision, mode, strided):
    monkeypatch.setenv("VLLM_OMNI_METAL_ANE", mode)
    gate = torch.ones(3, 2, dtype=torch.float16)
    x = torch.ones(2, 257, 2, dtype=torch.float16)
    if strided:
        x = x.transpose(0, 1)
    sentinel = torch.full_like(x, 7)
    seen = []

    def original(value):
        seen.append(value)
        return sentinel

    with ane.TiledDenseSwiGLU(
        [(gate, gate, gate.T.contiguous())], scope="pipeline", precision=precision
    ) as group:
        with torch.inference_mode():
            assert group(0, x, native=original) is sentinel
        assert seen == [x]
        assert not group.tile_metrics
        assert group.metrics["native_calls"] == 1
    with torch.inference_mode(), pytest.raises(RuntimeError, match="closed"):
        group(0, x, native=original)


def active_group(values, *, benchmark=True):
    """Inject deterministic tile predictions without loading hardware or compiling."""
    gate = torch.ones(len(values), 1, dtype=torch.float16)
    group = ane.TiledDenseSwiGLU([(gate, gate, gate.T.contiguous())], scope="pipeline")
    group.plans = [tuple((i, i + 1) for i in range(len(values)))]
    group._benchmark = benchmark
    group._benchmark_chunks = 1
    group.identity = cache.tiled_group_key(["weights"], group.shapes, group.plans, group.scope)
    child = SimpleNamespace(
        metrics=Counter(), disabled=False, closed=0, _bridge=SimpleNamespace(device="test")
    )

    def close():
        child.closed += 1

    def split(index, x, chunks):
        child.metrics.update(calls=1, ane_chunks=chunks, ane_rows=chunks * cache.CHUNK)
        value = values[index]
        if isinstance(value, Exception):
            raise value
        return torch.full_like(x, value)

    child.close = close
    child._split = split
    group._group = child
    return group, child


def test_fp32_final_reduction_and_live_error_evidence(private_cache):
    group, child = active_group([2048, 1, -2048])
    x = torch.ones(1, 257, 1, dtype=torch.float16)
    with group, torch.inference_mode():
        assert torch.equal(group(0, x, native=lambda value: value.clone(), context=42), x)
        check = group.checks[0, 257, 1, 42]
        assert check == {
            "relative_l2": 0,
            "scaled_max_abs": 0,
            "tiles": 3,
            "shape": [1, 257, 1],
        }
        assert all(group.tile_metrics[0, i]["ane_chunks"] == 1 for i in range(3))
        assert (
            ane_build.publish_tiled_profile(group, 257, [10, 11], {1: [8, 9]}, context=42)["chunks"]
            == 1
        )
        policy = cache.read_json(private_cache / f"{group.identity}.json")
        assert cache.validate_profiles(policy["profiles"]) == {(257, 42): 1}
        with pytest.raises(ValueError, match="final live-input"):
            ane_build.publish_tiled_profile(group, 257, [10, 11], {1: [8, 9]}, context=43)
        group._benchmark = False
        group.profiles, group._input_shapes = cache.validate_tiled_profiles(
            policy["profiles"], group.shapes, group.plans
        )
        marker = torch.full_like(x, 13)
        assert group(0, x, native=lambda value: marker, context=43) is marker
        assert torch.equal(group(0, x, native=lambda value: marker, context=42), x)
        flat = x.reshape(257, 1)
        assert group(0, flat, native=lambda value: marker, context=42) is marker
    assert child.closed == 1


def test_timed_path_does_not_repeat_gpu_audit_and_rejects_unaudited_shape():
    group, _ = active_group([1])
    x = torch.ones(257, 1, dtype=torch.float16)
    with group, torch.inference_mode():
        with pytest.raises(RuntimeError, match="every FFN"):
            group.finish_audit()
        torch.testing.assert_close(group(0, x, native=lambda value: value.clone(), context=7), x)
        group.finish_audit()

        def audit_would_distort_timing(value):
            raise AssertionError("formal timing repeated the whole GPU reference")

        torch.testing.assert_close(group(0, x, native=audit_would_distort_timing, context=7), x)
        with pytest.raises(RuntimeError, match="completed audit"):
            group(0, x, native=audit_would_distort_timing, context=8)


@pytest.mark.parametrize(
    "values", [[1, 0.1], [1, float("nan")], [1, RuntimeError("prediction failed")]]
)
def test_final_error_or_tile_failure_rejects_approval_and_restores_whole_native(
    private_cache, values
):
    group, child = active_group(values)
    x = torch.ones(257, 1, dtype=torch.float16)
    sentinel = x.clone()
    with torch.inference_mode():
        assert group(0, x, native=lambda value: sentinel) is sentinel
        assert group.disabled and child.closed == 1
        assert group.metrics["failures"] == 1
        assert group(0, x, native=lambda value: sentinel) is sentinel
    with pytest.raises(ValueError):
        ane_build.publish_tiled_profile(group, 257, [10, 11], {1: [8, 9]})
    assert not (private_cache / f"{group.identity}.json").exists()
    group.close()
    assert child.closed == 1


def test_weight_mutation_and_memory_guard_restore_original(monkeypatch):
    for mutate in (True, False):
        group, child = active_group([1])
        if mutate:
            group.weights[0][0].add_(1)
        else:
            monkeypatch.setattr(
                "psutil.virtual_memory", lambda: SimpleNamespace(available=cache.RESERVE)
            )
        x = torch.ones(257, 1, dtype=torch.float16)
        with torch.inference_mode():
            assert group(0, x, native=lambda value: value) is x
        assert child.closed == 1 and not child.metrics


def test_profile_isolation_covers_partition_order_shape_and_pipeline():
    keys = ["first", "second"]
    shapes = [(4096, 12288)]
    plans = [((0, 10240), (10240, 12288))]
    original = cache.tiled_group_key(keys, shapes, plans, "full-pipeline-a")
    assert original != cache.group_key(keys, "full-pipeline-a")
    assert original != cache.tiled_group_key(keys, shapes, plans, "full-pipeline-b")
    assert original != cache.tiled_group_key(keys[::-1], shapes, plans, "full-pipeline-a")
    assert original != cache.tiled_group_key(keys, [(2048, 12288)], plans, "full-pipeline-a")
    assert original != cache.tiled_group_key(
        keys, shapes, [((0, 6144), (6144, 12288))], "full-pipeline-a"
    )


def test_dense_constructor_unloads_all_handles_on_unexpected_failure(private_cache, monkeypatch):
    (private_cache / "tools.json").touch(mode=0o600)
    weight = SimpleNamespace(
        device=SimpleNamespace(type="mps"), _version=0, numel=lambda: 1, element_size=lambda: 2
    )
    monkeypatch.setattr(cache, "weight_shape", lambda *args: (1024, 1024))
    monkeypatch.setattr(cache, "eligible", lambda *args: True)
    monkeypatch.setattr(cache, "weight_key", lambda *args: "b" * 64)
    monkeypatch.setattr(cache, "verify_model", lambda *args, **kwargs: private_cache)
    unloaded = []
    attempts = []

    def load(*args):
        attempts.append(args)
        if len(attempts) == 3:
            raise ValueError("load interrupted")
        return len(attempts)

    monkeypatch.setattr(
        ane,
        "_Bridge",
        lambda: SimpleNamespace(
            load=load, device="test", lib=SimpleNamespace(ffn_unload=unloaded.append)
        ),
    )
    with pytest.raises(ValueError, match="interrupted"):
        ane.DenseSwiGLU([(weight, weight), (weight, weight)], precision="fp16", _benchmark=True)
    assert sorted(unloaded) == [1, 2]


def test_prepare_is_sequential_and_never_publishes_approval_on_failure(private_cache, monkeypatch):
    gate = torch.arange(24, dtype=torch.float16).reshape(6, 4)
    weights = [(gate, gate + 100, gate.T.contiguous())]
    monkeypatch.setattr(cache, "inner_tiles", lambda *args: ((0, 4), (4, 6)))
    monkeypatch.setattr(ane_build, "build_tools", lambda: private_cache)
    monkeypatch.setattr(ane_build, "_Bridge", lambda: SimpleNamespace(device="test"))
    monkeypatch.setattr(ane_build, "_budget", lambda *args, **kwargs: None)
    prepared = []

    def emit(bridge, key, hidden, inner, packed=None, down=None):
        if packed is not None:
            prepared.append((packed.clone(), down.clone()))
            if len(prepared) == 2:
                raise RuntimeError("second tile self-test failed")

    monkeypatch.setattr(ane_build, "_emit_model", emit)
    with pytest.raises(RuntimeError, match="self-test"):
        ane_build.prepare_tiled_group(weights, scope="pipeline")
    assert [p.shape for p, _ in prepared] == [torch.Size([8, 4]), torch.Size([4, 4])]
    assert list(private_cache.glob("*.json")) == []


def test_resource_guard_runs_before_any_packed_model_copy(private_cache, monkeypatch):
    (private_cache / "tools.json").touch(mode=0o600)
    weight = SimpleNamespace(
        device=SimpleNamespace(type="mps"),
        dtype=torch.float16,
        _version=0,
        numel=lambda: 4096 * 12288,
        element_size=lambda: 2,
    )
    monkeypatch.setattr(cache, "tiled_weight_shape", lambda *args: (4096, 12288))
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=8 * 1024**3))

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("resource guard must precede every packed copy")

    monkeypatch.setattr(cache, "tiled_weights", unexpected_copy)
    with ane.TiledDenseSwiGLU(
        [(weight, weight, weight)] * 32, scope="pipeline", precision="fp16"
    ) as group:
        assert group._group is None and not group.keys
    with pytest.raises(RuntimeError, match="headroom"):
        ane.TiledDenseSwiGLU(
            [(weight, weight, weight)] * 32,
            scope="pipeline",
            precision="fp16",
            _benchmark=True,
        )


def test_tiled_binding_failure_closes_constructed_child(private_cache, monkeypatch):
    (private_cache / "tools.json").touch(mode=0o600)
    weight = SimpleNamespace(
        device=SimpleNamespace(type="mps"),
        dtype=torch.float16,
        _version=0,
        numel=lambda: 1,
        element_size=lambda: 2,
    )
    monkeypatch.setattr(cache, "tiled_weight_shape", lambda *args: (1024, 1024))
    monkeypatch.setattr(cache, "tiled_weights", lambda *args, **kwargs: [(weight, weight)])
    monkeypatch.setattr(cache, "weight_key", lambda *args: "b" * 64)
    closed = []
    original = ane.DenseSwiGLU

    class Child:
        _version = staticmethod(original._version)
        disabled = False
        _pool = object()
        keys = ["different-weight-identity"]

        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr(ane, "DenseSwiGLU", Child)
    with pytest.raises(RuntimeError, match="changed during"):
        ane.TiledDenseSwiGLU(
            [(weight, weight, weight)], scope="pipeline", precision="fp16", _benchmark=True
        )
    assert closed == [True]


def test_unexpected_prediction_exception_still_closes_transaction():
    group, child = active_group([TypeError("bad bridge response")])
    x = torch.ones(257, 1, dtype=torch.float16)
    with torch.inference_mode(), pytest.raises(TypeError, match="bridge response"):
        group(0, x, native=lambda value: value)
    assert group.disabled and child.closed == 1


def test_calibration_cannot_mix_bsh_shapes_with_equal_row_count(private_cache):
    group, child = active_group([1])
    x = torch.ones(1, 257, 1, dtype=torch.float16)
    with torch.inference_mode():
        group(0, x, native=lambda value: value.clone())
        flat = x.reshape(257, 1)
        assert group(0, flat, native=lambda value: value) is flat
    assert group.disabled and child.closed == 1
    with pytest.raises(ValueError, match="healthy"):
        ane_build.publish_tiled_profile(group, 257, [10, 11], {1: [8, 9]})


def test_tiled_profile_requires_persisted_final_error_and_shape_evidence():
    shapes, plans = [(1, 1)], [((0, 1),)]
    record = cache.select_profile(257, [10, 11], {1: [8, 9]})
    with pytest.raises(RuntimeError, match="evidence"):
        cache.validate_tiled_profiles({"257": record}, shapes, plans)
    evidence = {
        "shape": [1, 257, 1],
        "tiles": 1,
        "relative_l2": 0.001,
        "scaled_max_abs": 0.002,
    }
    record["numerical_checks"] = [evidence]
    profiles, inputs = cache.validate_tiled_profiles({"257": record}, shapes, plans)
    assert profiles == {(257, None): 1}
    assert inputs == {((257, None), 0): (1, 257, 1)}
    for field, value in (
        ("shape", [1, 258, 1]),
        ("tiles", 2),
        ("relative_l2", 0.011),
        ("scaled_max_abs", float("nan")),
    ):
        record["numerical_checks"] = [{**evidence, field: value}]
        with pytest.raises(RuntimeError, match="numerical evidence"):
            cache.validate_tiled_profiles({"257": record}, shapes, plans)
