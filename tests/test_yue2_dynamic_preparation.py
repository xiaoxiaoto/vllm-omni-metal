# SPDX-License-Identifier: Apache-2.0
"""Offline control-plane tests only; no model execution or benchmark approval claims."""

import copy
import gc
import importlib.util
import json
import os
import sys
import types
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

DEPLOY = Path(__file__).resolve().parents[2] / "docker-compose/yue2-3b"
pytestmark = pytest.mark.skipif(
    not (DEPLOY / "prepare_dynamic_quantization.py").is_file(),
    reason="independent deployment checkout absent",
)
NAME = "nar_layers.0.mlp.down_proj"


@pytest.fixture
def preparation(monkeypatch):
    monkeypatch.syspath_prepend(str(DEPLOY))
    spec = importlib.util.spec_from_file_location(
        "prepare_dynamic_quantization", DEPLOY / "prepare_dynamic_quantization.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setenv("VLLM_OMNI_METAL_ANE", "off")
    return module


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = torch.nn.Linear(64, 32, bias=False, dtype=torch.bfloat16)
        self.nar_layers = torch.nn.ModuleList([layer])
        self.nar_fused = [
            (None,) * 6 + (torch.randn(128, 32, dtype=torch.bfloat16), layer.mlp.down_proj.weight)
        ]
        self._states = {}
        self._pending = None

    def nar_weights(self):
        return self.nar_fused


@pytest.fixture
def extension(preparation):
    model = TinyModel()
    adapter = preparation.NarDownAdapter(model, [NAME], "native-scope")
    adapter.revision = {"source_weights": {}, "input_sha256": "a" * 64}
    adapter.retained = {}
    model._metal_ffn = adapter
    worker = preparation.DynamicPreparation()
    worker.get_model = lambda: model
    worker.model_runner = SimpleNamespace(requests={})
    return worker, model, adapter


def test_uniform_or_explicit_candidates_do_not_force_precision_distribution(preparation):
    assert preparation.candidate_names(28, "auto") == [
        f"nar_layers.{n}.mlp.down_proj" for n in (0, 9, 18, 27)
    ]
    assert preparation.candidate_names(28, "7") == ["nar_layers.7.mlp.down_proj"]
    for value in ("0,0", "-1", "28", "0,1,2,3,4"):
        with pytest.raises(ValueError):
            preparation.candidate_names(28, value)


def test_real_hook_captures_four_full_cpu_activations(preparation, extension):
    _, _, adapter = extension
    x = torch.ones(3, 32, dtype=torch.bfloat16)
    with torch.inference_mode():
        for _ in range(25):
            adapter(0, x)
    samples = adapter.captures[NAME]
    assert [step for step, _ in samples] == [0, 8, 16, 24]
    assert all(
        tensor.shape == (3, 64) and tensor.device.type == "cpu" and tensor.dtype == torch.bfloat16
        for _, tensor in samples
    )
    assert adapter.capture_bytes == 4 * 3 * 64 * 2
    assert adapter.seen[NAME] == 25


def test_calibration_cli_discards_inherited_dynamic_before_native_capture(
    preparation, monkeypatch, tmp_path
):
    from yue2_dynamic_quantization import install

    from vllm_omni_metal.ane.runtime import DenseSwiGLU

    monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION", "dynamic")
    monkeypatch.setenv("YUE2_METAL_QUANTIZATION", "dynamic")
    for key in ("VLLM_OMNI_METAL_QUANTIZATION_PROFILE", "YUE2_METAL_QUANTIZATION_PROFILE"):
        monkeypatch.setenv(key, str(tmp_path / "absent-production-profile.json"))
    monkeypatch.setattr(sys, "argv", ["prepare_dynamic_quantization.py", "--worker"])
    monkeypatch.setattr(preparation, "validate_model", lambda path, kind: Path(path))
    models = []

    async def cpu_worker(args):
        model = TinyModel()
        original = model.nar_layers[0].mlp.down_proj
        native = DenseSwiGLU([(w[6], w[7]) for w in model.nar_fused], scope="native-scope")
        model._metal_ffn = native
        assert install(model, scope="native-scope") is None
        assert model._metal_ffn is native
        assert model._metal_quantization_metadata["mode"] == "off"
        capture = preparation.NarDownAdapter(model, [NAME], "native-scope")
        with torch.inference_mode():
            for _ in range(25):
                capture(0, torch.ones(3, 32, dtype=torch.bfloat16))
        assert model.nar_layers[0].mlp.down_proj is original
        assert [step for step, _ in capture.captures[NAME]] == [0, 8, 16, 24]
        assert os.environ["VLLM_OMNI_METAL_QUANTIZATION"] == "off"
        models.append(model)

    monkeypatch.setattr(preparation, "worker", cpu_worker)
    preparation.main()
    assert len(models) == 1


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="explicit MPS integration opt-in required",
)
@torch.inference_mode()
def test_real_mps_capture_search_apply_and_release(extension):
    worker, model, adapter = extension
    model.to("mps")
    model.nar_fused[0] = (
        *model.nar_fused[0][:6],
        model.nar_fused[0][6].to("mps"),
        model.get_submodule(NAME).weight,
    )
    original_weight = weakref.ref(model.get_submodule(NAME).weight)
    x = torch.ones(3, 32, device="mps", dtype=torch.bfloat16)
    before = adapter(0, x).cpu()
    for _ in range(24):
        adapter(0, x)
    result = worker.dynamic_search()["result"]
    profile = result["profile"]
    worker.dynamic_apply(profile)
    worker.quant_mode("float")
    torch.testing.assert_close(adapter(0, x).cpu(), before, atol=0, rtol=0)
    worker.quant_mode("dynamic")
    actual = adapter(0, x).cpu()
    delta = actual.double() - before.double()
    assert torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(before.double()) < 0.01
    bits = profile["modules"][NAME]["bits"]
    if bits == 16:
        assert model.get_submodule(NAME).weight is original_weight()
    else:
        assert model.get_submodule(NAME).bits == bits
        assert worker.quant_snapshot()["result"]["quantized"][NAME]["metrics"]["calls"] == 1
    released = worker.quant_release_baseline()["result"]
    assert released["benchmark_held_float_bytes"] == 0
    assert released["float_weakrefs_released"] is True
    if bits != 16:
        assert original_weight() is None
    torch.testing.assert_close(adapter(0, x).cpu(), actual, atol=0, rtol=0)


@pytest.mark.parametrize(
    "profile",
    [
        {"selected": ["nar_layers.1.mlp.down_proj"], "modules": {}},
        {"selected": [NAME], "modules": {NAME: {"bits": 8}}},
    ],
)
def test_invalid_apply_preserves_fused_original_and_benchmark_state(extension, profile):
    worker, model, adapter = extension
    original = model.get_submodule(NAME)
    with pytest.raises(ValueError):
        worker.dynamic_apply(profile)
    assert model.get_submodule(NAME) is original
    assert model.nar_fused[0][7] is original.weight
    assert adapter.baselines == {} and adapter.quantized == {}


def test_release_removes_benchmark_refs_and_float_route_cannot_resume(
    preparation, extension, monkeypatch
):
    worker, model, adapter = extension
    original = model.get_submodule(NAME)
    reference = weakref.ref(original)
    adapter.baselines[NAME] = original
    adapter.float_references = [weakref.ref(original), weakref.ref(original.weight)]
    adapter.quantized[NAME] = SimpleNamespace(
        bits=8, metrics={"calls": 0, "rows": 0}, source_bytes=4096, storage_bytes=2048
    )
    # This fixture tests ownership, not conversion or numerical equivalence.
    model.nar_layers[0].mlp.down_proj = torch.nn.Identity()
    model.nar_fused[0] = (*model.nar_fused[0][:7], None)
    del original
    monkeypatch.setattr(preparation, "memory_snapshot", lambda: {"fixture": True})
    monkeypatch.setitem(
        preparation.QuantizationPreparation.quant_snapshot.__globals__,
        "memory_snapshot",
        lambda: {"fixture": True},
    )
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    package = types.ModuleType("mlx")
    package.__path__ = []
    core = types.ModuleType("mlx.core")
    core.clear_cache = lambda: None
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    worker.quant_mode("dynamic")
    result = worker.quant_release_baseline()["result"]
    gc.collect()
    assert reference() is None
    assert result["benchmark_held_float_bytes"] == 0
    assert result["released_benchmark_float_bytes"] == 64 * 32 * 2
    assert result["independent_process"] is False
    with pytest.raises(RuntimeError, match="released"):
        worker.quant_mode("float")


def test_preflight_records_refusal_without_model_or_service_start(
    preparation, tmp_path, monkeypatch
):
    model, vae = tmp_path / "model", tmp_path / "vae"
    model.mkdir()
    vae.mkdir()
    output = tmp_path / "evidence"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_dynamic_quantization.py",
            "--model",
            str(model),
            "--vae",
            str(vae),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(preparation, "validate_model", lambda path, kind: Path(path))
    monkeypatch.setattr(
        preparation.psutil, "virtual_memory", lambda: SimpleNamespace(available=15 * 1024**3)
    )
    monkeypatch.setattr(
        preparation.shutil, "disk_usage", lambda path: SimpleNamespace(free=3 * 1024**3)
    )
    monkeypatch.setattr(
        preparation, "supervise", lambda *args, **kwargs: pytest.fail("must not spawn")
    )
    with pytest.raises(MemoryError, match="preflight"):
        preparation.main()
    record = json.loads((output / "preflight.json").read_text())
    assert record["startup_gate_pass"] is False
    assert record["model_process_started"] is False
    assert record["required_available_bytes"] == 16 * 1024**3


def test_disk_preflight_also_records_refusal(preparation, tmp_path, monkeypatch):
    output = tmp_path / "evidence"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_dynamic_quantization.py",
            "--model",
            str(tmp_path),
            "--vae",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(preparation, "validate_model", lambda path, kind: Path(path))
    monkeypatch.setattr(
        preparation.psutil, "virtual_memory", lambda: SimpleNamespace(available=20 * 1024**3)
    )
    monkeypatch.setattr(
        preparation.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024**3)
    )
    monkeypatch.setattr(
        preparation, "supervise", lambda *args, **kwargs: pytest.fail("must not spawn")
    )
    with pytest.raises(MemoryError):
        preparation.main()
    assert json.loads((output / "preflight.json").read_text())["startup_gate_pass"] is False


def synthetic_records():
    """Control-plane fixture, not audio/timing measurements."""
    result = []
    order = [
        ("native", True, False, 10),
        ("float", True, True, 10),
        ("dynamic", True, False, 8),
        ("float", False, False, 10),
        ("dynamic", False, False, 8),
        ("dynamic", False, False, 8),
        ("float", False, False, 10),
    ]
    for mode, warmup, capture, seconds in order:
        result.append(
            {
                "mode": mode,
                "warmup": warmup,
                "capture": capture,
                "seconds": seconds,
                "pending": 0,
                "frames": 512,
                "seed": 42,
                "sample_rate": 48000,
                "shape": [240000, 2],
                "dtype": "float32",
                "audio_sha256": "b" * 64,
                "quantized": {NAME: {"metrics": {"calls": 1, "rows": 512}}}
                if mode == "dynamic"
                else {},
            }
        )
    return result


def test_summary_keeps_all_bf16_valid_but_never_approved(preparation):
    records = synthetic_records()
    for record in records:
        record["quantized"] = {}
    completion = preparation.summarize(records, [], True)
    assert completion["approved"] is False
    assert completion["local_pass"] is True
    assert completion["rejection_reason"] == "no_quantized_candidate"


def test_summary_distinguishes_assertion_from_measured_release(preparation):
    completion = preparation.summarize(synthetic_records(), [], True)
    assert completion["approved"] is True
    assert completion["release"]["kind"] == "expected-post-release-assertion"
    assert "release_observation" not in completion
    assert "engine_closed" not in completion
    assert completion["publication_requirements"]["supervisor_success"] is True


def test_approval_rebuild_preserves_search_fingerprints(preparation, extension, monkeypatch):
    from vllm_omni_metal.quantization import linear

    worker, model, adapter = extension
    original = model.get_submodule(NAME)
    monkeypatch.setattr(linear, "_environment", lambda: {"fixture": "CPU profile metadata"})
    calibration = {
        "engine": "DynamicEngine",
        "approved": False,
        "local_pass": True,
        "precisions": {NAME: "w8a16"},
        "captures": {NAME: [{"sha256": "c" * 64}]},
        "audits": {NAME: {"fixture": True}},
        "implementation_sha256": "a" * 64,
    }
    with torch.inference_mode():
        profile = linear.build_profile(
            {NAME: original},
            [NAME],
            scope=adapter.scope,
            calibration=calibration,
            precisions=calibration["precisions"],
        )
    adapter.baselines = {NAME: original}
    adapter.quantized = {NAME: object()}
    completion = preparation.summarize(synthetic_records(), [], True)
    approved = worker.dynamic_approval_candidate(profile, completion)["result"]["profile"]
    assert approved["modules"] == profile["modules"]
    assert approved["sha256"] != profile["sha256"]
    for key in ("precisions", "captures", "audits", "implementation_sha256"):
        assert approved["calibration"][key] == calibration[key]
    assert profile["calibration"]["approved"] is False
    assert approved["calibration"]["approved"] is True
    with torch.inference_mode():
        linear.validate_profile(approved, {NAME: original}, scope=adapter.scope)


@pytest.mark.parametrize("failure", ["supervisor", "shutdown", "release", None])
def test_publication_waits_for_actual_release_shutdown_and_supervisor(
    preparation, tmp_path, failure
):
    measured = preparation.summarize(synthetic_records(), [], True)
    completed = copy.deepcopy(measured)
    completed.update(
        engine_closed=True,
        supervisor_finished=False,
        release_observation={**measured["release"], "kind": "actual-post-release-observation"},
    )
    supervisor = {"returncode": 0, "stop_reason": None}
    if failure == "supervisor":
        supervisor["returncode"] = 1
    elif failure == "shutdown":
        completed["engine_closed"] = False
    elif failure == "release":
        completed["release_observation"]["float_weakrefs_released"] = False
    preparation.write_json(tmp_path / "completed.json", completed)
    preparation.write_json(tmp_path / "supervisor.json", supervisor)
    pending = tmp_path / "pending-approved-profile.json"
    preparation.write_json(
        pending,
        {
            "selected": [NAME],
            "modules": {NAME: {"bits": 8}},
            "calibration": {"full_pipeline": measured},
        },
    )
    if failure is not None:
        with pytest.raises((RuntimeError, ValueError)):
            preparation.publish_after_supervision(tmp_path)
        assert not (tmp_path / "approved-profile.json").exists()
        assert pending.exists()
    else:
        preparation.publish_after_supervision(tmp_path)
        assert not pending.exists()
        published = tmp_path / "approved-profile.json"
        assert published.stat().st_mode & 0o777 == 0o600
        assert json.loads((tmp_path / "completed.json").read_text())["supervisor_finished"] is True


def test_native_snapshot_reports_pending_but_mutating_rpcs_reject_it(
    preparation, extension, monkeypatch
):
    worker, model, adapter = extension
    monkeypatch.setattr(preparation, "memory_snapshot", lambda: {"fixture": True})
    model._pending = object()
    assert worker.dynamic_native_snapshot()["result"]["pending"] == 1
    with pytest.raises(RuntimeError, match="pending"):
        worker.quant_mode("dynamic")
    assert adapter.mode == "float"


def test_unexpected_float_alias_refuses_release(preparation, extension, monkeypatch):
    worker, model, adapter = extension
    original = model.get_submodule(NAME)
    alias = original.weight
    adapter.baselines = {NAME: original}
    adapter.float_references = [weakref.ref(original), weakref.ref(original.weight)]
    adapter.mode = "dynamic"
    model.nar_layers[0].mlp.down_proj = torch.nn.Identity()
    model.nar_fused[0] = (*model.nar_fused[0][:7], None)
    del original
    package = types.ModuleType("mlx")
    package.__path__ = []
    core = types.ModuleType("mlx.core")
    core.clear_cache = lambda: None
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    with pytest.raises(RuntimeError, match="unexpected aliases"):
        worker.quant_release_baseline()
    assert alias.numel() == 64 * 32
