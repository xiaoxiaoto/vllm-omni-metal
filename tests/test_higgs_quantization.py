# SPDX-License-Identifier: Apache-2.0
"""Higgs adapter boundary tests, never real TTS quality/approval evidence."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

DEPLOY = Path(__file__).resolve().parents[2] / "docker-compose/higgs-tts-3"
pytestmark = pytest.mark.skipif(
    not (DEPLOY / "higgs_metal_quantization.py").is_file(), reason="独立Higgs部署未检出"
)


@pytest.fixture
def adapter():
    spec = importlib.util.spec_from_file_location(
        "higgs_test_adapter", DEPLOY / "higgs_metal_quantization.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def codec(device="cpu", dtype=torch.float16):
    model = torch.nn.Module()
    model.fc2 = torch.nn.Linear(64, 32, device=device, dtype=dtype).eval()
    return model


def approval(adapter, bits=8, dtype="torch.float16"):
    # Deliberately incomplete profile: only tests rejection gates, never published.
    return {
        "selected": ["fc2"],
        "modules": {
            "fc2": {
                "bits": bits,
                "dtype": dtype,
                "group_size": 64 if bits in (4, 8) else None,
                "mode": "affine" if bits in (4, 8) else "identity",
            }
        },
        "calibration": {
            "engine": "DynamicEngine",
            "local_pass": True,
            "approved": True,
            "policy": copy.deepcopy(adapter.POLICY),
            "samples": [
                {
                    "text": text,
                    "seed_repeatable": True,
                    "natural_stop": True,
                    "released_matches": True,
                    "candidate_calls": 3,
                    "candidate_rows": 12,
                    "same_shape": True,
                    "same_dtype": True,
                    "relative_l2": 0.005,
                    "scaled_max_abs": 0.005,
                    "baseline_sha256": "a" * 64,
                    "candidate_sha256": "b" * 64,
                }
                for text in adapter.TEXTS
            ],
        },
    }


def test_default_off_and_explicit_override_global(adapter, monkeypatch):
    monkeypatch.delenv("HIGGS_METAL_QUANTIZATION", raising=False)
    monkeypatch.delenv("VLLM_OMNI_METAL_QUANTIZATION", raising=False)
    model = codec()
    original = model.fc2
    adapter.finish_load(model)
    assert model.fc2 is original
    assert model._metal_quantization_metadata == {"managed": True, "mode": "off", "approved": False}
    monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION", "dynamic")
    monkeypatch.setenv("HIGGS_METAL_QUANTIZATION", "off")
    adapter.finish_load(model)
    assert model.fc2 is original


@pytest.mark.parametrize("mode", ["auto", "on", "w8a16"])
def test_legacy_modes_rejected(adapter, monkeypatch, mode):
    monkeypatch.setenv("HIGGS_METAL_QUANTIZATION", mode)
    with pytest.raises(ValueError, match="off/dynamic"):
        adapter.finish_load(codec())


def test_global_dynamic_uses_scoped_shared_loader(adapter, monkeypatch, tmp_path):
    from vllm_omni_metal.quantization import runtime

    path = str(tmp_path / "missing.json")
    monkeypatch.delenv("HIGGS_METAL_QUANTIZATION", raising=False)
    monkeypatch.delenv("HIGGS_METAL_QUANTIZATION_PROFILE", raising=False)
    monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION", "dynamic")
    monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION_PROFILE", path)
    monkeypatch.setattr(adapter, "scope", lambda model: "actual-execution-sha")
    calls = []

    def missing(selected, *, scope):
        calls.append((selected, scope))
        raise FileNotFoundError(selected)

    monkeypatch.setattr(runtime, "load_profile", missing)
    model = codec()
    original = model.fc2
    with pytest.raises(FileNotFoundError):
        adapter.finish_load(model)
    assert calls == [(path, "actual-execution-sha")]
    assert model.fc2 is original
    assert model._metal_quantization_metadata["approved"] is False


@pytest.mark.parametrize(
    "damage", ["gate", "calls", "stop", "release", "policy", "selection", "engine"]
)
def test_incomplete_generation_is_not_approved(adapter, damage):
    value = approval(adapter)
    sample = value["calibration"]["samples"][0]
    if damage == "gate":
        sample["relative_l2"] = 0.010001
    elif damage == "calls":
        sample["candidate_calls"] = 0
    elif damage == "stop":
        sample["natural_stop"] = False
    elif damage == "release":
        del sample["released_matches"]
    elif damage == "policy":
        value["calibration"]["policy"]["seed"] = 7
    elif damage == "engine":
        del value["calibration"]["engine"]
    else:
        value["selected"].append("other")
    with pytest.raises(ValueError):
        adapter.approved(value)


@pytest.mark.parametrize(
    "bits,dtype",
    [
        (4, "torch.float16"),
        (8, "torch.bfloat16"),
        (16, "torch.float16"),
        (16, "torch.bfloat16"),
        (32, "torch.float32"),
    ],
)
def test_dynamic_descriptor_accepts_source_precision_contract(adapter, bits, dtype):
    adapter.require_dynamic_profile(approval(adapter, bits, dtype))
    adapter.approved(approval(adapter, bits, dtype))


@pytest.mark.parametrize(
    "bits,dtype",
    [(4, "torch.float32"), (8, "torch.float32"), (16, "torch.float32"), (32, "torch.float16")],
)
def test_fp32_cannot_masquerade_as_quantized(adapter, bits, dtype):
    with pytest.raises(ValueError):
        adapter.require_dynamic_profile(approval(adapter, bits, dtype))


def test_capture_across_real_module_calls_preserves_fp32(adapter, monkeypatch):
    monkeypatch.setenv("HIGGS_METAL_QUANTIZATION", "off")
    monkeypatch.setattr(adapter, "scope", lambda model: "capture-scope")
    model = codec(dtype=torch.float32)
    worker = adapter.QuantizationPreparation()
    worker.get_model = lambda: model
    worker.model_runner = SimpleNamespace(requests={})
    worker.higgs_quant_prepare()
    for _ in range(4):
        model.fc2(torch.randn(1, 2, 64))
    captures = model._higgs_benchmark["captures"]
    assert [step for step, _ in captures] == [0, 1, 2, 3]
    assert all(t.device.type == "cpu" and t.dtype == torch.float32 for _, t in captures)
    model._higgs_benchmark["capture_hook"].remove()


def test_identity_projection_keeps_dtype_device_layout_and_counts(adapter):
    source = codec(dtype=torch.float32).fc2
    wrapped = adapter.CodecProjection(source)
    value = torch.randn(2, 64, 7).transpose(1, 2)
    assert not value.is_contiguous()
    with torch.inference_mode():
        assert torch.equal(wrapped(value), source(value))
    assert wrapped.dtype == torch.float32 and wrapped.device == source.weight.device
    assert wrapped.metrics == {"calls": 1, "rows": 14}


def test_malformed_profile_is_transactional(adapter):
    model = codec()
    original = model.fc2
    with pytest.raises(ValueError):
        adapter.install(model, {"selected": ["fc2"]}, "scope", require_approved=False)
    assert model.fc2 is original


def preparation():
    spec = importlib.util.spec_from_file_location(
        "higgs_test_preparation", DEPLOY / "prepare_quantization.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_approved_publication_never_overwrites_and_is_private(tmp_path):
    module = preparation()
    target = tmp_path / "approved-profile.json"
    module.publish(target, {"boundary_test": "not quality evidence"})
    before = target.read_bytes()
    assert target.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        module.publish(target, {})
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]


def test_full_output_comparison_preserves_raw_bits():
    reference = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    candidate = reference.copy()
    candidate[0] = -0.0
    result = preparation().comparison(reference, candidate)
    assert result["pcm16_identical"] and not result["bitwise_identical"]
    assert result["relative_l2"] == 0
    assert not preparation().comparison(reference, candidate[:-1])["same_shape"]


@pytest.mark.integration
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="需要真实MPS")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_real_dynamic_search_and_projection(adapter, dtype):
    from vllm_omni_metal.quantization.engine import DynamicEngine

    model = codec("mps", dtype)
    captures = {"fc2": [(i, torch.randn(1, 3, 64, dtype=dtype)) for i in range(4)]}
    with torch.inference_mode():
        result = DynamicEngine.search({"fc2": model.fc2}, captures, scope="higgs-test")
        profile = result["profile"]
        assert profile["calibration"]["approved"] is False
        if dtype == torch.float32:
            assert result["precisions"]["fc2"] == "fp32"
        original = model.fc2
        with pytest.raises(ValueError, match="scope"):
            adapter.install(model, profile, "different", require_approved=False)
        assert model.fc2 is original
        model.alias = original
        with pytest.raises(ValueError, match="shared/tied"):
            adapter.install(model, profile, "higgs-test", require_approved=False)
        assert model.fc2 is original and model.alias is original
        del model.alias
        wrapped = adapter.install(model, profile, "higgs-test", require_approved=False)
        value = torch.randn(2, 64, 7, device="mps", dtype=dtype).transpose(1, 2)
        output = wrapped(value)
        assert output.shape == (2, 7, 32) and output.dtype == dtype
        assert torch.isfinite(output).all().item()
        assert wrapped.metrics == {"calls": 1, "rows": 14}


@pytest.mark.parametrize("change", ["appear", "modify", "disappear"])
def test_generation_config_state_remains_bound(adapter, tmp_path, change):
    path = tmp_path / "generation_config.json"
    if change != "appear":
        path.write_text('{"eos_token_id":7}')
    before = adapter.generation_config_revision(tmp_path)
    if change == "disappear":
        path.unlink()
    else:
        path.write_text('{"eos_token_id":8}')
    assert adapter.generation_config_revision(tmp_path) != before


def test_release_requires_actual_baseline_weakref_expiry(adapter, monkeypatch):
    import weakref

    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    model = codec(dtype=torch.float32)
    original = model.fc2
    state = {
        "float": original,
        "dynamic": adapter.CodecProjection(copy.deepcopy(original)),
        "baseline_ref": weakref.ref(original),
        "baseline_weight_ref": weakref.ref(original.weight),
    }
    model._higgs_benchmark = state
    worker = adapter.QuantizationPreparation()
    worker.get_model = lambda: model
    worker.model_runner = SimpleNamespace(requests={})
    with pytest.raises(RuntimeError, match="仍被引用"):
        worker.higgs_quant_release()
    assert state.get("released") is not True
    del original
    # Restore only the bookkeeping key needed by the release RPC; no baseline
    # module/weight reference survives.
    state["float"] = None
    assert worker.higgs_quant_release()["result"]["float_baseline_released"] is True


@pytest.fixture
def joint(adapter, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "higgs_metal_quantization", adapter)
    spec = importlib.util.spec_from_file_location("higgs_test_joint", DEPLOY / "higgs_joint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_joint_off_does_not_touch_talker(joint, monkeypatch):
    monkeypatch.delenv("HIGGS_METAL_JOINT", raising=False)
    model = torch.nn.Module()
    joint.finish_load(model)
    assert model._metal_quantization_metadata == {
        "managed": True,
        "mode": "off",
        "approved": False,
    }
    assert not hasattr(model, "_higgs_joint_gates")


def test_joint_rejects_independent_codec_approval(joint, adapter):
    with pytest.raises(ValueError, match="complete Higgs joint"):
        joint.approved(approval(adapter))


def test_joint_rejects_incomplete_talker(joint):
    with pytest.raises(ValueError, match="36-layer"):
        joint.projections(SimpleNamespace(model=SimpleNamespace(layers=[])))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_joint_preserves_actual_codec_source_precision(adapter, monkeypatch, dtype):
    monkeypatch.setenv("HIGGS_METAL_JOINT", "joint")
    monkeypatch.setenv("HIGGS_METAL_QUANTIZATION", "off")
    model = codec(dtype=dtype)
    original, weight = model.fc2, model.fc2.weight
    value = torch.randn(3, 64, dtype=dtype)
    expected = original(value)
    adapter.finish_load(model)
    assert model.fc2 is original and model.fc2.weight is weight
    assert model.fc2.weight.dtype == dtype
    assert torch.equal(model.fc2(value), expected)
    monkeypatch.setenv("HIGGS_METAL_QUANTIZATION", "dynamic")
    with pytest.raises(ValueError, match="source identity"):
        adapter.finish_load(model)
    assert model.fc2 is original


def test_joint_codec_detects_equal_value_parameter_replacement(adapter, joint, monkeypatch):
    monkeypatch.setenv("HIGGS_METAL_JOINT_CALIBRATION", "1")
    monkeypatch.setenv("HIGGS_METAL_QUANTIZATION", "off")
    model = codec(dtype=torch.bfloat16)
    adapter.finish_load(model)
    worker = adapter.QuantizationPreparation()
    worker.get_model = lambda: model
    worker.model_runner = SimpleNamespace(requests={})
    before = worker.higgs_quant_snapshot()["result"]["source_identity"]
    joint.validate_codec_identity(before)
    model.fc2.weight = torch.nn.Parameter(model.fc2.weight.detach().clone())
    after = worker.higgs_quant_snapshot()["result"]["source_identity"]
    assert after["identity"] is False
    with pytest.raises(ValueError, match="native identity"):
        joint.validate_codec_identity(after)


def test_joint_count_preserves_native_tuple_and_strides(joint):
    class Native(torch.nn.Module):
        def forward(self, x):
            assert not x.is_contiguous()
            return x, None

    wrapped = joint.CountProjection(Native())
    value = torch.randn(2, 64, 7).transpose(1, 2)
    result = wrapped(value)
    assert result[0] is value and result[1] is None
    assert wrapped.metrics == {"calls": 1, "rows": 14}


@pytest.mark.parametrize(
    "field,value",
    [
        ("ane_chunks", 0),
        ("quantized_calls", 0),
        ("engine_shutdown", False),
        ("ane_relative_l2", 0.02),
        ("ane_scaled_max_abs", float("nan")),
    ],
)
def test_joint_rejects_inactive_unclosed_or_inaccurate_execution(joint, field, value):
    # Synthetic rejection fixture only; never a publishable source-bound profile.
    numerical = {
        "relative_l2": 0.001,
        "scaled_max_abs": 0.001,
        "shape": [257, 2560],
        "ane_chunks": 1,
    }
    record = {
        "chunks": 1,
        "baseline": [10.0, 10.1],
        "candidate": [9.0, 9.1],
        "numerical_check": numerical,
    }
    sample = {
        "seed_repeatable": True,
        "natural_stop": True,
        "released_matches": True,
        "same_shape": True,
        "same_dtype": True,
        "relative_l2": 0.001,
        "scaled_max_abs": 0.001,
        "ane_relative_l2": 0.001,
        "ane_scaled_max_abs": 0.001,
        "candidate_calls": 4,
        "candidate_rows": 257,
        "ane_chunks": 1,
        "quantized_calls": 4,
        "baseline_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
        "ane_reference_sha256": "c" * 64,
    }
    evidence = {
        "policy": copy.deepcopy(joint.POLICY),
        "float_released": True,
        "ane_closed": True,
        "codec_identity": {"dtype": "torch.bfloat16", "identity": True},
        "engine_shutdown": True,
        "supervisor_success": True,
        "samples": [{**sample, "text": text} for text in joint.TEXTS],
        "projections": {
            str(i): {"identity": "a" * 64, "profiles": {"257": copy.deepcopy(record)}}
            for i in joint.LAYERS
        },
    }
    if field == "engine_shutdown":
        evidence[field] = value
    else:
        evidence["samples"][0][field] = value
    profile = {
        "selected": list(joint.SELECTED),
        "modules": {name: {"bits": 8} for name in joint.SELECTED},
        "calibration": {
            "engine": "DynamicEngine",
            "approved": True,
            "local_pass": True,
            "joint": evidence,
        },
    }
    import hashlib

    from vllm_omni_metal.quantization.linear import _encoded

    candidate = copy.deepcopy(profile)
    del candidate["calibration"]["joint"]
    candidate["calibration"]["approved"] = False
    evidence["quantization_candidate_sha256"] = hashlib.sha256(_encoded(candidate)).hexdigest()
    with pytest.raises(ValueError, match="shutdown|hit/repeat/release"):
        joint.approved(profile)
