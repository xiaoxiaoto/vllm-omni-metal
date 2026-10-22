# SPDX-License-Identifier: Apache-2.0
"""Independent YuE2 calibration behavior only; no GPU/model loading or fake audio evidence."""

import asyncio
import importlib.util
import json
import sys
import types
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

DEPLOY = Path(__file__).resolve().parents[2] / "docker-compose" / "yue2-3b"
pytestmark = pytest.mark.skipif(
    not (DEPLOY / "prepare_quantization.py").is_file(),
    reason="independent YuE2 deployment checkout is not present",
)


@pytest.fixture
def preparation(monkeypatch):
    monkeypatch.syspath_prepend(str(DEPLOY))
    spec = importlib.util.spec_from_file_location(
        "yue2_quantization_preparation", DEPLOY / "prepare_quantization.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("VLLM_OMNI_METAL_ANE", "off")
    return module


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = torch.nn.Linear(4, 2, bias=False)
        self.nar_layers = torch.nn.ModuleList([layer])
        self.packed = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 20
        self.nar_fused = [
            (None, None, None, None, None, None, self.packed, layer.mlp.down_proj.weight)
        ]
        self._states = {}
        self._pending = None

    def nar_weights(self):
        return self.nar_fused


class CpuProjection(torch.nn.Module):
    """CPU lifecycle fixture, not a quantization/precision/performance substitute."""

    def __init__(self, source):
        super().__init__()
        self.register_buffer("fixture_weight", source.weight.detach().clone())
        self.metrics = {"calls": 0, "rows": 0}
        self.storage_bytes = self.source_bytes = self.fixture_weight.numel() * 4

    def forward(self, x):
        self.metrics["calls"] += 1
        self.metrics["rows"] += x.numel() // x.shape[-1]
        return functional.linear(x, self.fixture_weight)


@pytest.fixture
def extension(preparation):
    model = TinyModel()
    name = "nar_layers.0.mlp.down_proj"
    model._metal_ffn = preparation.NarDownAdapter(model, [name], "cpu-behavior-only")
    worker = preparation.QuantizationPreparation()
    worker.get_model = lambda: model
    worker.model_runner = SimpleNamespace(requests={})
    return worker, model, name


def test_candidates_preserve_inseparable_fused_weights(preparation):
    assert preparation.candidate_names(28, "auto") == [
        f"nar_layers.{layer}.mlp.down_proj" for layer in (0, 9, 18, 27)
    ]
    for value in ("0,0", "28", "-1", "0,1,2,3,4"):
        with pytest.raises(ValueError):
            preparation.candidate_names(28, value)


def test_audio_bits_are_stricter_than_pcm_and_json_has_no_infinity(preparation):
    reference = np.array([[0.0, 0.5]], dtype=np.float32)
    candidate = reference.copy()
    candidate[0, 0] = -0.0
    result = preparation.compare_arrays(reference, candidate)
    assert not result["bitwise_identical"]
    assert result["pcm16_identical"]
    assert result["relative_l2"] == 0
    assert result["snr_infinite"] and result["snr_db"] is None
    json.dumps(result, allow_nan=False)
    assert preparation.compare_arrays(reference, reference.copy())["bitwise_identical"]
    shorter = preparation.compare_arrays(reference, reference[:, :1])
    assert not shorter["same_shape"] and "relative_l2" not in shorter


def test_ranking_requires_both_error_bounds_and_complete_capture(preparation):
    audits = [
        dict(name="high-peak", captured_calls=4, relative_l2=0.001, scaled_max_abs=0.02),
        dict(name="good", captured_calls=4, relative_l2=0.002, scaled_max_abs=0.003),
        dict(name="incomplete", captured_calls=3, relative_l2=0, scaled_max_abs=0),
    ]
    ranked, selected = preparation.select_ranked(audits, 2)
    assert [entry["name"] for entry in ranked] == ["incomplete", "good", "high-peak"]
    assert selected == ["good"]


def test_explicit_adapter_hits_down_module_without_global_patching(preparation, extension):
    _, model, name = extension
    adapter = model._metal_ffn
    source = model.get_submodule(name)
    hits = []
    handle = source.register_forward_hook(lambda *values: hits.append(True))
    x = torch.tensor([[0.25, -0.75], [0.5, 1.0]])
    gate, up = functional.linear(x, model.packed).chunk(2, dim=-1)
    with torch.inference_mode():
        expected = functional.linear(functional.silu(gate) * up, source.weight)
        for _ in range(25):
            torch.testing.assert_close(adapter(0, x, context=512), expected, rtol=0, atol=0)
    handle.remove()
    assert len(hits) == 25
    assert [step for step, _ in adapter.captures[name]] == list(preparation.CAPTURE_STEPS)
    replacement = CpuProjection(source)
    model.nar_layers[0].mlp.down_proj = replacement
    adapter.mode = "w8a16"
    with torch.inference_mode():
        torch.testing.assert_close(adapter(0, x), expected, rtol=0, atol=0)
    assert replacement.metrics == {"calls": 1, "rows": 2}
    adapter.close()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="关闭"):
        adapter(0, x)


def test_apply_failure_preserves_float_group_and_fused_reference(monkeypatch, extension):
    from vllm_omni_metal.quantization import linear as quantization

    worker, model, name = extension
    original = model.get_submodule(name)
    monkeypatch.setattr(quantization, "validate_profile", lambda *a, **k: [name])

    def fail(*args, **kwargs):
        assert torch.is_inference_mode_enabled()
        raise ValueError("candidate conversion rejected")

    monkeypatch.setattr(quantization, "apply_profile", fail)
    with pytest.raises(ValueError, match="conversion rejected"):
        worker.quant_apply(
            {"selected": [name], "modules": {name: {"bits": 8, "group_size": 64, "mode": "affine"}}}
        )
    assert model.get_submodule(name) is original
    assert model.nar_fused[0][7] is original.weight
    assert model._metal_ffn.baselines == {}
    assert model._metal_ffn.quantized == {}


def test_release_removes_all_selected_float_references(monkeypatch, preparation, extension):
    from vllm_omni_metal.quantization import linear as quantization

    worker, model, name = extension
    old = weakref.ref(model.get_submodule(name))
    monkeypatch.setattr(quantization, "validate_profile", lambda *a, **k: [name])

    def replace(root, profile, *, scope):
        assert torch.is_inference_mode_enabled()
        candidate = CpuProjection(root.get_submodule(name))
        root.nar_layers[0].mlp.down_proj = candidate
        return {name: candidate}

    monkeypatch.setattr(quantization, "apply_profile", replace)
    worker.quant_apply(
        {"selected": [name], "modules": {name: {"bits": 8, "group_size": 64, "mode": "affine"}}}
    )
    assert model.nar_fused[0][7] is None
    assert old() is model._metal_ffn.baselines[name]
    worker.quant_mode("w8a16")
    assert model.get_submodule(name).metrics == {"calls": 0, "rows": 0}
    monkeypatch.setattr(preparation, "memory_snapshot", lambda: {"fixture": True})
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    mlx = types.ModuleType("mlx")
    mlx.core = SimpleNamespace(clear_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx.core)
    result = worker.quant_release_baseline()["result"]
    assert old() is None
    assert result["benchmark_held_float_bytes"] == 0
    assert not result["independent_process"]
    with pytest.raises(RuntimeError, match="浮点已释放"):
        worker.quant_mode("float")
    with torch.inference_mode():
        model._metal_ffn(0, torch.ones(2, 2))
    assert model.get_submodule(name).metrics == {"calls": 1, "rows": 2}


def test_approved_metadata_is_rebuilt_and_validated_before_float_release(monkeypatch, extension):
    from vllm_omni_metal.quantization import linear as quantization

    worker, model, name = extension
    source = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    adapter = model._metal_ffn
    adapter.baselines = {name: source}
    adapter.quantized = {name: object()}
    monkeypatch.setattr(quantization, "_environment", lambda: {"fixture": "CPU profile binding"})
    with torch.inference_mode():
        profile = quantization.build_profile(
            adapter.baselines,
            [name],
            scope=adapter.scope,
            calibration={"approved": False, "local_pass": True},
        )
    completion = {"approved": True, "lossless": True, "same_workload": True}
    approved = worker.quant_approve_profile(profile, completion)["result"]["profile"]
    assert not profile["calibration"]["approved"]
    assert approved["calibration"]["approved"]
    assert approved["calibration"]["full_pipeline"] == completion
    assert approved["sha256"] != profile["sha256"]
    with torch.inference_mode():
        assert quantization.validate_profile(approved, adapter.baselines, scope=adapter.scope) == [
            name
        ]
    adapter.baselines.clear()
    with pytest.raises(RuntimeError, match="浮点已释放"):
        worker.quant_approve_profile(profile, completion)


@pytest.mark.parametrize("precision", ["w4a16", "bf16"])
def test_w8_apply_and_approval_reject_other_precisions(monkeypatch, extension, precision):
    from vllm_omni_metal.quantization import linear as quantization

    worker, model, name = extension
    adapter = model._metal_ffn
    original = model.get_submodule(name)
    fused = model.nar_fused[0][7]
    source = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    monkeypatch.setattr(quantization, "_environment", lambda: {})
    with torch.inference_mode():
        profile = quantization.build_profile(
            {name: source},
            [name],
            scope=adapter.scope,
            calibration={"approved": False, "local_pass": True},
            precisions={name: precision},
        )
    with pytest.raises(ValueError, match="仅支持 W8A16"):
        worker.quant_apply(profile)
    assert model.get_submodule(name) is original and model.nar_fused[0][7] is fused
    assert not adapter.baselines and not adapter.quantized
    adapter.baselines = {name: source}
    adapter.quantized = {name: object()}
    with pytest.raises(ValueError, match="仅支持 W8A16"):
        worker.quant_approve_profile(profile, {"approved": True})
    assert not profile["calibration"]["approved"]


def test_busy_worker_refuses_switch_before_mutation(extension):
    worker, model, _ = extension
    worker.model_runner.requests["active"] = object()
    with pytest.raises(RuntimeError, match="请求尚未清理"):
        worker.quant_mode("w8a16")
    assert model._metal_ffn.mode == "float"


def test_failed_quantization_rpc_is_not_a_success(preparation, tmp_path):
    class Engine:
        async def collective_rpc(self, method, **kwargs):
            assert isinstance(method, str) and kwargs["stage_ids"] == [0]
            return [{"supported": False, "error": "audit rejected"}]

    with pytest.raises(RuntimeError, match="quant_audit failed"):
        asyncio.run(preparation.rpc(Engine(), tmp_path, "quant_audit", 1))
    assert json.loads((tmp_path / "rpc-error.json").read_text())["method"] == "quant_audit"


@pytest.mark.parametrize("arguments", [["--frames", "256"], []])
def test_cli_refuses_unsafe_or_shortened_requests_before_weights(
    preparation, monkeypatch, arguments
):
    monkeypatch.setattr(sys, "argv", ["prepare_quantization.py", *arguments])
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1" if not arguments else "0")
    monkeypatch.setattr(preparation, "validate_model", lambda *args: pytest.fail("loaded weights"))
    with pytest.raises(SystemExit):
        preparation.main()


def test_supervisor_reuses_limits_and_propagates_child_failure(preparation, monkeypatch, tmp_path):
    helper = sys.modules["prepare_ane"]
    commands = []
    monkeypatch.setattr(
        helper.psutil, "virtual_memory", lambda: SimpleNamespace(available=20 * 1024**3)
    )

    def spawn(command, **kwargs):
        commands.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=7, poll=lambda: 7)

    monkeypatch.setattr(helper.subprocess, "Popen", spawn)
    args = SimpleNamespace(
        gpu_only=True,
        attention_ab=False,
        model="local-model",
        vae="local-vae",
        frames=512,
        deploy_config="profile.yaml",
    )
    with pytest.raises(RuntimeError, match="量化 失败"):
        helper.supervise(
            args,
            script=Path("prepare_quantization.py"),
            directory=tmp_path,
            extra_args=("--candidate-layers", "0"),
            label="量化",
        )
    command, environment = commands[0]
    assert command[1] == "prepare_quantization.py"
    assert command[-2:] == ["--candidate-layers", "0"]
    assert environment["VLLM_OMNI_METAL_ANE"] == "off"
    assert environment["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert json.loads((tmp_path / "supervisor.json").read_text())["returncode"] == 7
    assert not (tmp_path / "completed.json").exists()


def test_weight_scope_hashes_actual_existing_files(preparation, tmp_path):
    model, vae = tmp_path / "model", tmp_path / "vae"
    model.mkdir()
    vae.mkdir()
    (model / "model.safetensors").write_bytes(b"weight-a")
    (vae / "model.safetensors").write_bytes(b"vae-a")
    first = preparation.weight_fingerprint(model, vae)
    (model / "model.safetensors").write_bytes(b"weight-b")
    second = preparation.weight_fingerprint(model, vae)
    assert first["model"][0]["sha256"] != second["model"][0]["sha256"]
    assert first["vae"] == second["vae"]
