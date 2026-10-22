# SPDX-License-Identifier: Apache-2.0
"""Sidecar boundary tests; tiny fixtures are not real YuE2 quality/performance evidence."""

import hashlib
import importlib.util
import json
import os
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

DEPLOY = Path(__file__).resolve().parents[2] / "docker-compose" / "yue2-3b"
pytestmark = pytest.mark.skipif(
    not (DEPLOY / "mixed_quantization.py").is_file(),
    reason="independent YuE2 mixed-precision sidecar is not present",
)
PLAN = {
    "nar_layers.0.mlp.down_proj": "w4a16",
    "nar_layers.1.mlp.down_proj": "w8a16",
    "nar_layers.2.mlp.down_proj": "bf16",
}


@pytest.fixture
def mixed(monkeypatch):
    monkeypatch.syspath_prepend(str(DEPLOY))
    spec = importlib.util.spec_from_file_location(
        "yue2_mixed_quantization", DEPLOY / "mixed_quantization.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("VLLM_OMNI_METAL_ANE", "off")
    return module


class TinyModel(torch.nn.Module):
    """Three independent BF16 down projections and their native packed gate/up."""

    def __init__(self, device="cpu"):
        super().__init__()
        generator = torch.Generator().manual_seed(42)
        self.nar_layers = torch.nn.ModuleList()
        self.nar_fused = []
        for _ in range(3):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.down_proj = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
            with torch.no_grad():
                layer.mlp.down_proj.weight.copy_(torch.randn(64, 64, generator=generator) / 16)
            layer.to(device)
            self.nar_layers.append(layer)
            packed = (torch.randn(128, 64, generator=generator) / 16).to(
                device=device, dtype=torch.bfloat16
            )
            self.nar_fused.append(
                (None, None, None, None, None, None, packed, layer.mlp.down_proj.weight)
            )
        self._states = {}
        self._pending = None
        self.vllm_config = SimpleNamespace()
        self._metal_ffn = SimpleNamespace(_closed=False, disabled=False, _benchmark=False)
        self._metal_ffn.close = lambda: setattr(self._metal_ffn_previous, "_closed", True)
        self._metal_ffn_previous = self._metal_ffn

    def nar_weights(self):
        return self.nar_fused


@pytest.fixture
def worker_factory(mixed, monkeypatch):
    from vllm_omni_metal.ane import runtime

    # No runtime/compiler probing: scope still binds the actual sidecar revision and plan.
    monkeypatch.setattr(
        runtime,
        "execution_scope",
        lambda revision, config: hashlib.sha256(revision.encode()).hexdigest(),
    )

    def create(device="cpu", install=True):
        model = TinyModel(device)
        worker = mixed.MixedPreparation()
        worker.get_model = lambda: model
        worker.model_runner = SimpleNamespace(requests={})
        if install:
            worker.mixed_install(dict(PLAN), {"seed": 42, "frames": 512})
        return worker, model

    return create


@pytest.fixture
def cpu_worker(worker_factory, monkeypatch):
    from vllm_omni_metal.quantization import linear as quantization

    # Metadata only; conversion, profile validation and failure remain the real API.
    monkeypatch.setattr(quantization, "_environment", lambda: {"fixture": "CPU boundary"})
    return worker_factory()


@pytest.mark.parametrize(
    "plan",
    [
        None,
        [],
        {},
        {"nar_layers.0.mlp.down_proj": "w4a16"},
        {**PLAN, "nar_layers.3.mlp.down_proj": "w8a16"},
        {**PLAN, "nar_layers.00.mlp.down_proj": "w8a16"},
        {**PLAN, "nar_layers.-1.mlp.down_proj": "w8a16"},
        {**PLAN, "nar_layers.0.mlp.gate_proj": "w8a16"},
        {**PLAN, "nar_layers.0.mlp.down_proj.weight": "w8a16"},
        {**PLAN, "nar_layers.0.mlp.down_proj": "W4A16"},
        {**PLAN, "nar_layers.0.mlp.down_proj": 4},
        {**PLAN, "nar_layers.0.mlp.down_proj": "w8a16"},
        {**PLAN, "nar_layers.1.mlp.down_proj": "bf16"},
        {**PLAN, "nar_layers.2.mlp.down_proj": "w4a16"},
        {
            f"nar_layers.{i}.mlp.down_proj": value
            for i, value in enumerate(("w4a16", "w8a16", "bf16", "bf16", "bf16"))
        },
    ],
)
def test_invalid_plans_are_rejected_before_install(mixed, worker_factory, plan):
    worker, model = worker_factory(install=False)
    previous = model._metal_ffn
    originals = list(model.nar_fused)
    with pytest.raises((ValueError, TypeError)):
        worker.mixed_install(plan, {})
    assert model._metal_ffn is previous and not previous._closed
    assert all(
        current is original for current, original in zip(model.nar_fused, originals, strict=True)
    )


def test_explicit_plan_disables_capture_and_binds_scope(worker_factory):
    worker, model = worker_factory(install=False)
    result = worker.mixed_install(dict(PLAN), {"seed": 42})["result"]
    adapter = model._metal_ffn
    assert adapter.names == list(PLAN)
    assert not adapter.capture and adapter.capture_bytes == 0
    assert all(not captures for captures in adapter.captures.values())
    assert model._metal_ffn_previous._closed
    assert result["precisions"] == PLAN
    first_scope = adapter.scope
    other_worker, other_model = worker_factory(install=False)
    changed = dict(PLAN)
    changed.update({list(PLAN)[0]: "w8a16", list(PLAN)[1]: "w4a16"})
    other_worker.mixed_install(changed, {"seed": 42})
    assert other_model._metal_ffn.scope != first_scope
    # Caller mutation must not change the installed fixed plan.
    changed.clear()
    assert other_model._metal_ffn.names == list(PLAN)


def test_cpu_conversion_failure_preserves_all_modules_and_fused_aliases(cpu_worker):
    worker, model = cpu_worker
    originals = {name: model.get_submodule(name) for name in PLAN}
    fused = list(model.nar_fused)
    weights = {name: module.weight.detach().clone() for name, module in originals.items()}
    for _ in range(2):
        with pytest.raises(ValueError, match="MPS weights"):
            worker.mixed_prepare()
        assert model._metal_ffn.baselines == {}
        assert model._metal_ffn.quantized == {}
        for name, module in originals.items():
            assert model.get_submodule(name) is module
            torch.testing.assert_close(module.weight, weights[name], rtol=0, atol=0)
        assert all(now is old for now, old in zip(model.nar_fused, fused, strict=True))
        assert all(model.nar_fused[i][7] is originals[name].weight for i, name in enumerate(PLAN))


@pytest.mark.parametrize("alias", ["module", "weight"])
def test_shared_quantized_sources_are_not_accepted(cpu_worker, alias):
    worker, model = cpu_worker
    first = model.nar_layers[0].mlp.down_proj
    if alias == "module":
        model.nar_layers[1].mlp.down_proj = first
    else:
        model.nar_layers[1].mlp.down_proj.weight = first.weight
    weights = model.nar_fused[1]
    model.nar_fused[1] = (*weights[:7], first.weight)
    modules = [model.get_submodule(name) for name in PLAN]
    fused = list(model.nar_fused)
    with pytest.raises(ValueError, match="shared|tied"):
        worker.mixed_prepare()
    assert all(
        model.get_submodule(name) is module for name, module in zip(PLAN, modules, strict=True)
    )
    assert all(now is old for now, old in zip(model.nar_fused, fused, strict=True))
    assert not model._metal_ffn.quantized and not model._metal_ffn.baselines


def test_valid_plan_sizes_and_order_are_explicit(mixed):
    mixed.validate_precisions(dict(PLAN), 3)
    fourth = {**PLAN, "nar_layers.27.mlp.down_proj": "bf16"}
    mixed.validate_precisions(fourth, 28)
    for invalid in (
        {**fourth, "nar_layers.26.mlp.down_proj": "bf16"},
        {**PLAN, "0": "bf16"},
        {**PLAN, "nar_layers.28.mlp.down_proj": "bf16"},
    ):
        with pytest.raises(ValueError):
            mixed.validate_precisions(invalid, 28)


@torch.inference_mode()
def test_installed_float_hook_uses_original_bf16_down(cpu_worker):
    worker, model = cpu_worker
    x = torch.linspace(-0.5, 0.5, 128).reshape(2, 64).to(torch.bfloat16)
    for index, name in enumerate(PLAN):
        gate, up = torch.nn.functional.linear(x, model.nar_fused[index][6]).chunk(2, dim=-1)
        expected = torch.nn.functional.linear(
            torch.nn.functional.silu(gate) * up, model.get_submodule(name).weight
        )
        actual = model._metal_ffn(index, x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.dtype == torch.bfloat16 and torch.isfinite(actual).all()
    assert model._metal_ffn.metrics["calls"] == 3
    assert all(not values for values in model._metal_ffn.captures.values())


@pytest.mark.parametrize("mode", ["float", "mixed", "w8a16", "w4a16", "", None])
def test_mode_requires_prepared_candidates(cpu_worker, mode):
    worker, model = cpu_worker
    with pytest.raises((ValueError, RuntimeError)):
        worker.quant_mode(mode)
    assert model._metal_ffn.mode == "float"
    with pytest.raises(RuntimeError):
        worker.quant_release_baseline()


@pytest.mark.parametrize("pending", ["requests", "states", "pending"])
@pytest.mark.parametrize("operation", ["install", "prepare", "mode", "release"])
def test_pending_requests_block_mutation(cpu_worker, pending, operation):
    worker, model = cpu_worker
    adapter = model._metal_ffn
    fused = list(model.nar_fused)
    if pending == "requests":
        worker.model_runner.requests["active"] = object()
    elif pending == "states":
        model._states["active"] = object()
    else:
        model._pending = object()
    actions = {
        "install": lambda: worker.mixed_install(dict(PLAN), {}),
        "prepare": worker.mixed_prepare,
        "mode": lambda: worker.quant_mode("mixed"),
        "release": worker.quant_release_baseline,
    }
    with pytest.raises(RuntimeError):
        actions[operation]()
    assert model._metal_ffn is adapter and adapter.mode == "float"
    assert not adapter.quantized and not adapter.baselines
    assert all(now is old for now, old in zip(model.nar_fused, fused, strict=True))


def test_native_reference_rejects_installed_adapter_and_pending(mixed, worker_factory, monkeypatch):
    monkeypatch.setattr(mixed, "memory_snapshot", lambda: {})
    worker, model = worker_factory(install=False)
    assert worker.mixed_native_snapshot()["result"]["mode"] == "native"
    worker.model_runner.requests["active"] = object()
    with pytest.raises(RuntimeError):
        worker.mixed_native_snapshot()
    worker.model_runner.requests.clear()
    worker.mixed_install(dict(PLAN), {})
    with pytest.raises(RuntimeError):
        worker.mixed_native_snapshot()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="explicit MPS integration opt-in required",
)
@torch.inference_mode()
def test_real_tiny_mixed_forward_identity_and_release(worker_factory):
    worker, model = worker_factory(device="mps")
    names = list(PLAN)
    original_refs = {name: weakref.ref(model.get_submodule(name)) for name in names[:2]}
    weight_refs = {name: weakref.ref(model.get_submodule(name).weight) for name in names[:2]}
    retained = model.get_submodule(names[2])
    retained_weight = retained.weight
    x = torch.linspace(-0.5, 0.5, 128).reshape(2, 64).to("mps", torch.bfloat16)
    baseline = [model._metal_ffn(i, x).cpu().clone() for i in range(3)]
    result = worker.mixed_prepare()["result"]
    assert result["profile"]["schema"] == 2
    assert result["profile"]["calibration"]["approved"] is False
    assert result["approved"] is False and result["local_pass"] is None
    assert result["no_sensitivity_search"] is True
    assert result["precisions"] == PLAN
    assert set(model._metal_ffn.quantized) == set(names[:2])
    assert set(model._metal_ffn.baselines) == set(names[:2])
    assert all(model.nar_fused[i][7] is None for i in range(2))
    assert model.get_submodule(names[2]) is retained
    assert model.nar_fused[2][7] is retained_weight
    with pytest.raises(RuntimeError):
        worker.mixed_prepare()
    with pytest.raises((ValueError, RuntimeError)):
        worker.quant_mode("w8a16")
    worker.quant_mode("float")
    for i in range(3):
        torch.testing.assert_close(model._metal_ffn(i, x).cpu(), baseline[i], rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        worker.quant_release_baseline()
    worker.quant_mode("mixed")
    hits = []
    handles = [
        model.get_submodule(name).register_forward_hook(
            lambda module, args, output, name=name: hits.append(name)
        )
        for name in names
    ]
    try:
        outputs = [model._metal_ffn(i, x).cpu().clone() for i in range(3)]
    finally:
        for handle in handles:
            handle.remove()
    assert hits == names
    for i, name in enumerate(names[:2]):
        projection = model.get_submodule(name)
        assert projection.metrics == {"calls": 1, "rows": 2}
        assert projection.bits == (4 if i == 0 else 8)
        assert torch.isfinite(outputs[i]).all()
        relative = torch.linalg.vector_norm(outputs[i].float() - baseline[i].float())
        relative /= torch.linalg.vector_norm(baseline[i].float()).clamp_min(1e-12)
        assert relative.item() < (0.25 if i == 0 else 0.03)
    torch.testing.assert_close(outputs[2], baseline[2], rtol=0, atol=0)
    snapshot = worker.quant_snapshot()["result"]
    assert snapshot["retained_identity"] is True
    assert snapshot["retained_modules"] == {names[2]: True}
    released = worker.quant_release_baseline()["result"]
    assert released["released_benchmark_float_bytes"] == 2 * 64 * 64 * 2
    assert released["benchmark_held_float_bytes"] == 0
    assert released["all_quantized_sources_released"] is True
    assert all(ref() is None for ref in (*original_refs.values(), *weight_refs.values()))
    assert not released["independent_process"]
    assert model.get_submodule(names[2]) is retained and retained.weight is retained_weight
    assert model.nar_fused[2][7] is retained_weight
    with pytest.raises(RuntimeError):
        worker.quant_mode("float")
    for i in range(3):
        torch.testing.assert_close(model._metal_ffn(i, x).cpu(), outputs[i], rtol=0, atol=0)
    json.dumps(released, allow_nan=False)
