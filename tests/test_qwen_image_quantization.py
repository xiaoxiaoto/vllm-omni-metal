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
