# SPDX-License-Identifier: Apache-2.0
"""Private deployment contracts; small tensors do not prove whole-image quality."""

import copy
import gc
import importlib.util
import json
import os
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni_metal.quantization.engine import DynamicEngine
from vllm_omni_metal.quantization.linear import QuantizedLinear, build_profile

DEPLOY = Path(__file__).resolve().parents[2] / "docker-compose/qwen-image-2.1-turbo"
pytestmark = pytest.mark.skipif(not DEPLOY.is_dir(), reason="独立Qwen部署不在此checkout旁")


@pytest.fixture
def adapter_module(monkeypatch):
    monkeypatch.setenv("QWEN_IMAGE_METAL_QUANTIZATION", "off")
    monkeypatch.setenv("VLLM_OMNI_METAL_ANE", "off")
    spec = importlib.util.spec_from_file_location(
        "qwen_dynamic_contract", DEPLOY / "quantization_adapter.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def single_rank_parallel(tmp_path):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    assert not torch.distributed.is_initialized()
    with set_current_vllm_config(VllmConfig()):
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{tmp_path / 'gloo-init'}",
                backend="gloo",
            )
            initialize_model_parallel(backend="gloo")
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def owner(*, count=1, width=64, device="cpu", dtype=torch.float16):
    root = torch.nn.Module()
    blocks = []
    for _ in range(count):
        block = torch.nn.Module()
        block.img_mlp = torch.nn.Module()
        block.img_mlp.out = torch.nn.Linear(width, 64, bias=False, dtype=dtype, device=device)
        block.img_mlp.out.requires_grad_(False)
        blocks.append(block)
    root.transformer_blocks = torch.nn.ModuleList(blocks)
    return SimpleNamespace(
        pipeline=SimpleNamespace(transformer=root),
        layers=[],
        group=None,
        idle=lambda: None,
        scope="test-qwen-contract",
    )


def calibration(quantized_modules=1):
    return {
        "policy": "qwen-whole-image-dynamic-v1",
        "approved": True,
        "lossy": True,
        "threshold": 0.01,
        "quantized_modules": quantized_modules,
        "samples": [
            {
                "quantized": q,
                "calls": 8 * quantized_modules if q else 0,
                "sha256": ("a" if q and quantized_modules else "b") * 64,
            }
            for q in (False, True, True, False)
        ],
        "errors": [{"relative_l2": 0.001, "scaled_max_abs": 0.002}] * 2,
    }


def test_default_off_and_explicit_off_override_global(adapter_module, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION", "dynamic")
    model = owner()
    original = model.pipeline.transformer.transformer_blocks[0].img_mlp.out
    adapter = adapter_module.QwenDynamicAdapter(model)
    assert adapter.snapshot()["status"] == "off"
    assert model.pipeline.transformer.transformer_blocks[0].img_mlp.out is original
    assert model.pipeline._metal_quantization_metadata == {
        "managed": True,
        "mode": "off",
        "approved": False,
    }
    monkeypatch.delenv("QWEN_IMAGE_METAL_QUANTIZATION")
    monkeypatch.delenv("VLLM_OMNI_METAL_QUANTIZATION")
    assert adapter_module.QwenDynamicAdapter(owner()).status == "off"


def test_global_dynamic_requires_profile(adapter_module, monkeypatch):
    monkeypatch.delenv("QWEN_IMAGE_METAL_QUANTIZATION")
    monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION", "dynamic")
    monkeypatch.delenv("QWEN_IMAGE_METAL_QUANTIZATION_PROFILE", raising=False)
    monkeypatch.delenv("VLLM_OMNI_METAL_QUANTIZATION_PROFILE", raising=False)
    with pytest.raises(ValueError, match="profile"):
        adapter_module.QwenDynamicAdapter(owner())
    monkeypatch.setenv("QWEN_IMAGE_METAL_QUANTIZATION", "auto")
    with pytest.raises(ValueError, match="off或dynamic"):
        adapter_module.QwenDynamicAdapter(owner())


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(approved=False),
        lambda c: c.update(threshold=0.02),
        lambda c: c["samples"][1].update(calls=0),
        lambda c: c["samples"][3].update(sha256="c" * 64),
        lambda c: c["errors"][0].update(relative_l2=0.010001),
        lambda c: c["errors"][1].update(scaled_max_abs=float("nan")),
    ],
)
def test_whole_image_gate_rejects_failed_evidence(adapter_module, change):
    value = copy.deepcopy(calibration())
    adapter_module.approval(value)
    change(value)
    with pytest.raises(ValueError):
        adapter_module.approval(value)


def test_all_retained_has_no_fake_quantized_calls(adapter_module):
    value = calibration(0)
    adapter_module.approval(value)
    value["samples"][1]["calls"] = 1
    with pytest.raises(ValueError):
        adapter_module.approval(value)


def test_pixel_errors_include_alpha(adapter_module):
    reference = np.full((4, 4, 4), 100, dtype=np.uint8)
    candidate = reference.copy()
    candidate[0, 0, 3] = 110
    assert adapter_module.pixel_errors(reference, candidate)["scaled_max_abs"] == 0.1
    with pytest.raises(ValueError, match="dtype/shape"):
        adapter_module.pixel_errors(reference, candidate[..., :3])


@pytest.mark.parametrize("alias", ["module", "parameter", "buffer", "ane"])
def test_shared_and_ane_leave_tree_unchanged(adapter_module, alias):
    model = owner()
    root = model.pipeline.transformer
    original = root.transformer_blocks[0].img_mlp.out
    if alias == "module":
        root.alias = original
    elif alias == "parameter":
        root.register_parameter("alias", original.weight)
    elif alias == "buffer":
        root.register_buffer("alias", original.weight.detach())
    else:
        model.group = object()
    adapter = adapter_module.QwenDynamicAdapter(model)
    with pytest.raises(ValueError, match="shared|tied|互斥"):
        adapter._sources(["transformer_blocks.0.img_mlp.out"])
    assert root.transformer_blocks[0].img_mlp.out is original


def test_capture_four_real_steps_without_cast(adapter_module, monkeypatch):
    monkeypatch.setattr(adapter_module.QwenDynamicAdapter, "_scope", lambda self: "scope")
    model = owner(count=2)
    adapter = adapter_module.QwenDynamicAdapter(model)
    adapter.prepare([0, 1])
    tensors = []
    with torch.inference_mode():
        for step in range(8):
            tensor = torch.full((2, 64), step, dtype=torch.float16)
            tensors.append(tensor)
            for block in model.pipeline.transformer.transformer_blocks:
                block.img_mlp.out(tensor)
    for captures in adapter.local.values():
        assert [step for step, _ in captures] == [0, 2, 4, 6]
        for step, tensor in captures:
            assert tensor.device.type == "cpu" and tensor.dtype == torch.float16
            assert tensor.data_ptr() != tensors[step].data_ptr()
            assert torch.equal(tensor, tensors[step])
    adapter.close()
    assert not adapter.hooks and not adapter.local


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="显式MPS/MLX验收",
)
def test_capture_batches_merge_all_candidates_after_full_generations(adapter_module, monkeypatch):
    monkeypatch.setattr(adapter_module.QwenDynamicAdapter, "_scope", lambda self: "scope")
    model = owner(count=2, device="mps")
    adapter = adapter_module.QwenDynamicAdapter(model)
    names = [f"transformer_blocks.{i}.img_mlp.out" for i in range(2)]
    adapter.prepare([0, 1])
    try:
        for batch in range(2):
            with torch.inference_mode():
                for step in range(8):
                    tensor = torch.full((2, 64), step / 8, dtype=torch.float16, device="mps")
                    for block in model.pipeline.transformer.transformer_blocks:
                        block.img_mlp.out(tensor)
            result = adapter.finish_capture()
            assert result["complete"] == (batch == 1)
            if batch == 0:
                assert not adapter.candidates
                assert list(adapter.local) == [names[1]]
        assert set(result["precisions"]) == set(names)
        captures = result["profile"]["calibration"]["captures"]
        assert set(captures) == set(names)
        assert all(
            [record["step"] for record in items] == [0, 2, 4, 6] for items in captures.values()
        )
        assert not adapter.local and not adapter.hooks
    finally:
        adapter.close()


def test_incomplete_capture_is_not_search(adapter_module, monkeypatch):
    monkeypatch.setattr(adapter_module.QwenDynamicAdapter, "_scope", lambda self: "scope")
    adapter = adapter_module.QwenDynamicAdapter(owner())
    adapter.prepare([0])
    with pytest.raises(ValueError, match="0/2/4/6"):
        adapter.finish_capture()
    assert adapter.candidates == {}


def test_global_profile_uses_complete_scope_before_loading(adapter_module, monkeypatch):
    monkeypatch.delenv("QWEN_IMAGE_METAL_QUANTIZATION")
    monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION", "dynamic")
    monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION_PROFILE", "/approved.json")
    monkeypatch.setattr(adapter_module.QwenDynamicAdapter, "_scope", lambda self: "complete-sha")

    def reject(path, *, scope):
        assert path == "/approved.json" and scope == "complete-sha"
        raise ValueError("scope rejected")

    monkeypatch.setattr(adapter_module, "load_profile", reject)
    model = owner()
    source = model.pipeline.transformer.transformer_blocks[0].img_mlp.out
    with pytest.raises(ValueError, match="scope rejected"):
        adapter_module.QwenDynamicAdapter(model)
    assert model.pipeline.transformer.transformer_blocks[0].img_mlp.out is source


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="显式MPS/MLX验收",
)
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("return_bias", [False, True])
@pytest.mark.parametrize("skip_bias_add", [False, True])
def test_vllm_projection_contract(
    adapter_module, single_rank_parallel, bits, return_bias, skip_bias_add
):
    from vllm.model_executor.layers.linear import RowParallelLinear

    source = RowParallelLinear(
        64,
        64,
        bias=False,
        input_is_parallel=True,
        params_dtype=torch.float16,
        disable_tp=True,
        return_bias=return_bias,
        skip_bias_add=skip_bias_add,
    ).to("mps")
    source.requires_grad_(False)
    with torch.inference_mode():
        source.weight.copy_(torch.eye(64, dtype=torch.float16, device="mps"))
        packed = QuantizedLinear.from_linear(adapter_module.dense_view(source), bits=bits)
        projection = adapter_module.DynamicProjection(source, packed)
        inputs = torch.ones(2, 64, dtype=torch.float16, device="mps")
        result = projection(inputs)
    if return_bias:
        assert isinstance(result, tuple) and result[1] is None
        result = result[0]
    else:
        assert isinstance(result, torch.Tensor)
    assert projection.skip_bias_add is skip_bias_add
    assert projection.quantized.bits == bits
    torch.testing.assert_close(result, inputs, atol=0.01, rtol=0.01)
    assert not hasattr(projection, "weight")


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="显式MPS/MLX验收",
)
def test_mixed_profile_constructs_each_bits_and_releases_only_quantized(
    adapter_module, monkeypatch
):
    monkeypatch.setattr(adapter_module.QwenDynamicAdapter, "_scope", lambda self: "scope")
    model = owner(count=3, device="mps")
    adapter = adapter_module.QwenDynamicAdapter(model)
    names = [f"transformer_blocks.{i}.img_mlp.out" for i in range(3)]
    adapter._sources(names)
    adapter.scope = "scope"
    precisions = dict(zip(names, ["w4a16", "w8a16", "fp16"], strict=True))
    # Real engine evidence is generated separately, not forged into a production approval.
    captures = {
        name: [(i, torch.ones(2, 64, dtype=torch.float16)) for i in range(4)] for name in names
    }
    evidence = DynamicEngine.search(adapter._views(), captures, scope="scope")["profile"][
        "calibration"
    ]
    profile = build_profile(
        adapter._views(), names, scope="scope", calibration=evidence, precisions=precisions
    )
    refs = [
        (weakref.ref(source), weakref.ref(source.weight)) for source in adapter.originals.values()
    ]
    retained = adapter.originals[names[2]]
    adapter._install(profile, require_approved=False)
    adapter.set_candidate(True)
    assert [adapter.candidates[n].quantized.bits for n in names[:2]] == [4, 8]
    assert adapter.candidates[names[2]] is retained
    adapter.release()
    gc.collect()
    assert all(module() is None and weight() is None for module, weight in refs[:2])
    assert refs[2][0]() is retained
    with pytest.raises(ValueError, match="释放"):
        adapter.set_candidate(False)


def test_atomic_profile_publish_no_overwrite(adapter_module, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    path = tmp_path / "approved.json"
    barrier = Barrier(2)

    def publish(identity):
        barrier.wait()
        try:
            adapter_module._publish_profile(path, {"winner": identity})
        except FileExistsError:
            return False, identity
        return True, identity

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, (1, 2)))
    winners = [identity for succeeded, identity in results if succeeded]
    assert len(winners) == 1
    assert json.loads(path.read_text()) == {"winner": winners[0]}
    assert path.stat().st_mode & 0o777 == 0o600
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        adapter_module._publish_profile(path, {"winner": 3})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def joint_calibration():
    value = calibration()
    value.update(
        policy="qwen-whole-image-joint-v1",
        ane_baseline_seconds=[10.0, 10.1],
        ane_candidate_seconds=[9.5, 9.6],
        ane_profiles=[{"chunks": 1}],
        lifecycle=dict(
            released_repeatable=True,
            adapter_closed=True,
            engine_closed=True,
            supervisor_success=True,
        ),
    )
    for sample in value["samples"]:
        sample["snapshot"] = {"ane_metrics": {"ane_chunks": 8 if sample["quantized"] else 0}}
    return value


@pytest.mark.parametrize("failure", ["old", "hits", "speed", "lifecycle", "repeat", "error"])
def test_joint_approval_rejects_independent_or_incomplete_evidence(adapter_module, failure):
    value = joint_calibration()
    adapter_module.joint_approval(value)
    if failure == "old":
        value["policy"] = "qwen-whole-image-dynamic-v1"
    elif failure == "hits":
        value["samples"][1]["snapshot"]["ane_metrics"]["ane_chunks"] = 0
    elif failure == "speed":
        value["ane_candidate_seconds"] = [9.7, 9.71]
    elif failure == "lifecycle":
        value["lifecycle"]["supervisor_success"] = False
    elif failure == "repeat":
        value["samples"][2]["sha256"] = "c" * 64
    else:
        value["errors"][0]["scaled_max_abs"] = 0.01001
    with pytest.raises(ValueError):
        adapter_module.joint_approval(value)


def test_joint_production_consumes_complete_scope_and_rejects_before_install(
    adapter_module, monkeypatch
):
    monkeypatch.setenv("QWEN_IMAGE_METAL_QUANTIZATION", "joint")
    monkeypatch.setenv("QWEN_IMAGE_METAL_QUANTIZATION_PROFILE", "/joint.json")
    monkeypatch.setattr(adapter_module.QwenDynamicAdapter, "_scope", lambda self: "joint-scope")
    installed = []
    monkeypatch.setattr(
        adapter_module.QwenDynamicAdapter, "_install", lambda *a, **k: installed.append(True)
    )
    value = joint_calibration()
    value["lifecycle"]["engine_closed"] = False

    def load(path, *, scope):
        assert path == "/joint.json" and scope == "joint-scope"
        return {"calibration": value}

    monkeypatch.setattr(adapter_module, "load_profile", load)
    model = owner()
    original = model.pipeline.transformer.transformer_blocks[0].img_mlp.out
    with pytest.raises(ValueError, match="关闭"):
        adapter_module.QwenDynamicAdapter(model)
    assert not installed
    assert model.pipeline.transformer.transformer_blocks[0].img_mlp.out is original


@pytest.fixture
def joint_module():
    spec = importlib.util.spec_from_file_location(
        "qwen_joint_contract", DEPLOY / "joint_adapter.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_executes_native_preserves_bias_and_rejects_fallback(joint_module, monkeypatch):
    class Projection:
        disabled = _closed = False
        _bridge = object()

        def __init__(self, weight, bias, **kwargs):
            self.metrics = {}
            self.closed = False

        def __call__(self, hidden, *, native, context):
            return native(hidden)

        def close(self):
            self.closed = True

    monkeypatch.setattr(joint_module, "prepare_projection", lambda *a, **k: None)
    monkeypatch.setattr(joint_module, "DenseProjection", Projection)
    source = torch.nn.Linear(4, 3)
    hidden = torch.randn(2, 5, 4)
    gate = joint_module.GateProjection(source, scope="scope", prepare=True)
    torch.testing.assert_close(gate(hidden), source(hidden), rtol=0, atol=0)
    assert gate.shapes == {(10, None)}
    gate.chunks = 1
    with pytest.raises(RuntimeError, match="silent GPU fallback"):
        gate(hidden)
    gate.chunks = 0
    gate.return_bias = True
    gate.skip_bias_add = True
    source.forward = lambda x: (torch.nn.functional.linear(x, source.weight), source.bias)
    output, bias = gate(hidden)
    torch.testing.assert_close(output, torch.nn.functional.linear(hidden, source.weight))
    assert bias is source.bias


def test_gate_close_on_unhealthy_projection(joint_module, monkeypatch):
    closed = []

    class Projection:
        disabled = True
        _closed = False
        _bridge = None

        def __init__(self, *a, **k):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr(joint_module, "DenseProjection", Projection)
    with pytest.raises(RuntimeError, match="healthy"):
        joint_module.GateProjection(torch.nn.Linear(4, 3), scope="scope", prepare=False)
    assert closed == [True]


def test_joint_gate_policy_consumption_rejects_shape_scope_and_chunk_drift(joint_module):
    projection = SimpleNamespace(identity="graph-id", profiles={(512, 7): 1})
    gate = SimpleNamespace(scope="scope:gate:0", projection=projection)
    group = joint_module.ProjectionGroup.__new__(joint_module.ProjectionGroup)
    group.gates = [gate]
    record = dict(gate_scope=gate.scope, identity="graph-id", rows=512, context=7, chunks=1)
    group.validate_records([record])
    for key, wrong in (
        ("gate_scope", "other"),
        ("identity", "other"),
        ("rows", 768),
        ("context", 8),
        ("chunks", 2),
    ):
        with pytest.raises(ValueError, match="policy"):
            group.validate_records([{**record, key: wrong}])
    with pytest.raises(ValueError, match="every"):
        group.validate_records([])


def test_gate_group_preparation_failure_rolls_back_and_closes(joint_module, monkeypatch):
    model = owner(count=2)
    blocks = model.pipeline.transformer.transformer_blocks
    sources = [torch.nn.Linear(4, 3) for _ in blocks]
    for block, source in zip(blocks, sources, strict=True):
        block.img_mlp.gate_layer = source
    projections = []

    class Projection:
        disabled = _closed = False
        _bridge = object()

        def __init__(self, *a, **k):
            self.closed = False
            projections.append(self)
            if len(projections) == 2:
                raise RuntimeError("load failed")

        def close(self):
            self.closed = True

    monkeypatch.setattr(joint_module, "DenseProjection", Projection)
    with pytest.raises(RuntimeError, match="load failed"):
        joint_module.ProjectionGroup(blocks, [0, 1], scope="scope", prepare=False)
    assert all(
        block.img_mlp.gate_layer is source for block, source in zip(blocks, sources, strict=True)
    )
    assert projections[0].closed


def test_joint_profiles_are_staged_and_replayed_without_publication(joint_module, monkeypatch):
    source = torch.nn.Linear(4, 3)
    old = SimpleNamespace(
        identity="graph", _bridge=SimpleNamespace(device="device"), close=lambda: None
    )
    gate = SimpleNamespace(
        scope="scope:gate:0", source=source, skip_bias_add=False, projection=old, shapes={(512, 7)}
    )
    group = joint_module.ProjectionGroup.__new__(joint_module.ProjectionGroup)
    group.gates = [gate]
    calls = []

    def stage(projection, rows, baseline, candidates, *, context, publish):
        assert projection is old and rows == 512 and context == 7
        assert publish is False
        calls.append("staged")
        return {"chunks": 1}

    class Replay:
        disabled = _closed = False
        _bridge = object()

        def __init__(self, weight, bias, *, scope, precision, _staged_policy):
            assert weight is source.weight and bias is source.bias
            assert scope == gate.scope and precision == "fp16"
            assert _staged_policy["group"] == "graph"
            assert list(_staged_policy["profiles"].values()) == [{"chunks": 1}]
            calls.append("replayed")

        def close(self):
            pass

    monkeypatch.setattr(joint_module, "publish_projection_profile", stage)
    monkeypatch.setattr(joint_module, "DenseProjection", Replay)
    records = group.approve([10, 10], {1: [9, 9]})
    assert records[0]["staged_policy"]["group"] == "graph"
    group.production()
    assert calls == ["staged", "replayed"]
    assert group._benchmark is False and gate.chunks is None


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="requires explicit real MPS execution",
)
def test_private_attention_uses_selected_metal_backend(single_rank_parallel):
    pytest.importorskip("vllm_omni.diffusion.models.qwen_image_21.metal_adapter")
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
        QwenImage21Attention,
    )

    generator = torch.Generator().manual_seed(1729)
    with torch.inference_mode():
        layer = QwenImage21Attention(dim=64, heads=1, dim_head=64)
        q, k, v = [torch.randn(1, 3, 1, 64, generator=generator).to("mps") for _ in range(3)]
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=0.125,
        ).transpose(1, 2)
        actual = layer.attn(q, k, v)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
