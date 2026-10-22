# SPDX-License-Identifier: Apache-2.0
"""Profile admission and complete-state identity; fixtures are not model approval."""

import copy
import dataclasses
import json
import os
from types import SimpleNamespace

import pytest
import torch

from vllm_omni_metal.quantization import linear, runtime
from vllm_omni_metal.quantization.adapter import DenseModelAdapter
from vllm_omni_metal.quantization.engine import _implementation_sha256


class ComponentPipeline:
    def __init__(self, component):
        self.components = {"transformer": component}


def model():
    root = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Linear(32, 16)).eval()
    root.register_buffer("offset", torch.tensor([1.0]))
    return root.requires_grad_(False)


def envelope(root, scope, *, approved=False):
    """Unit-only identity envelope, not an offline quality approval."""
    return linear.build_profile(
        {"0": root[0]},
        ["0"],
        scope=scope,
        precisions={"0": "fp32"},
        calibration={
            "engine": "DynamicEngine",
            "implementation_sha256": _implementation_sha256(),
            "approved": approved,
            "projections": {"0": {"kind": "torch"}},
        },
    )


def save(tmp_path, data):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))
    path.chmod(0o600)
    return path


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv(runtime.MODE_ENV, raising=False)
    monkeypatch.delenv(runtime.PROFILE_ENV, raising=False)


def test_bundle_selects_exact_stage_and_rejects_modified_content(tmp_path):
    root = model()
    first, second = envelope(root, "first"), envelope(root, "second")
    path = save(tmp_path, {"profiles": {"first": first, "second": second}})
    assert runtime.load_profile(path, scope="second") == second
    with pytest.raises(ValueError, match="no profile"):
        runtime.load_profile(path, scope="missing")
    first["calibration"]["approved"] = True
    save(tmp_path, {"profiles": {"first": first, "second": second}})
    with pytest.raises(ValueError, match="content hash"):
        runtime.load_profile(path, scope="second")


def test_profile_file_admission(tmp_path):
    path = save(tmp_path, envelope(model(), "test"))
    path.chmod(0o644)
    with pytest.raises(ValueError, match="mode0600"):
        runtime.load_profile(path, scope="test")
    path.chmod(0o600)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(OSError):
        runtime.load_profile(link, scope="test")
    with pytest.raises(ValueError, match="absolute"):
        runtime.load_profile("relative.json", scope="test")
    path.write_bytes(b" " * (runtime.PROFILE_LIMIT + 1))
    with pytest.raises(ValueError, match="bounded regular"):
        runtime.load_profile(path, scope="test")
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo, 0o600)
    with pytest.raises(ValueError, match="bounded regular"):
        runtime.load_profile(fifo, scope="test")


@pytest.mark.parametrize("payload", ['{"profiles":{},"profiles":{}}', '{"value":NaN}'])
def test_profile_rejects_ambiguous_json(tmp_path, payload):
    path = save(tmp_path, {})
    path.write_text(payload)
    with pytest.raises(ValueError, match="duplicate|nonfinite"):
        runtime.load_profile(path, scope="test")


@pytest.mark.parametrize(
    "part", ["selected", "unselected", "buffer", "config", "training", "child"]
)
def test_scope_binds_complete_state_and_execution(part):
    root = model()
    config = SimpleNamespace(dtype=torch.float32, options={"steps": 4})
    original = runtime.model_scope(root, config)
    assert runtime.model_scope(copy.deepcopy(root), copy.deepcopy(config)) == original
    if part == "selected":
        root[0].weight.add_(1)
    elif part == "unselected":
        root[1].weight.add_(1)
    elif part == "buffer":
        root.offset.add_(1)
    elif part == "config":
        config.options["steps"] = 5
    elif part == "training":
        root.train()
    else:
        root[1].train()
    assert runtime.model_scope(root, config) != original


def test_scope_binds_unregistered_diffusers_component():
    root = model()
    component = torch.nn.Linear(64, 64).requires_grad_(False)
    root._pipeline = ComponentPipeline(component)
    original = runtime.model_scope(root)
    component.weight.add_(1)
    assert runtime.model_scope(root) != original


def test_dataclass_configuration_and_invalid_values():
    @dataclasses.dataclass
    class Config:
        dtype: object
        dimensions: tuple

    root = model()
    assert runtime.model_scope(root, Config(torch.float16, (64, 32))) != runtime.model_scope(
        root, Config(torch.bfloat16, (64, 32))
    )
    with pytest.raises(ValueError, match="finite"):
        runtime.model_scope(root, {"threshold": float("inf")})
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="cyclic"):
        runtime.model_scope(root, cycle)
    with pytest.raises(TypeError, match="explicit scope"):
        runtime.model_scope(root, object())


def test_default_off_preserves_model_without_scope_requirements():
    opaque = object()
    assert runtime.configure_model_quantization(opaque)["mode"] == "off"


def test_dynamic_requires_approved_matching_profile(tmp_path, monkeypatch):
    root = model()
    original = root[0]
    scope = runtime.model_scope(root)
    monkeypatch.setenv(runtime.MODE_ENV, "dynamic")
    with pytest.raises(ValueError, match="PROFILE.*required"):
        runtime.configure_model_quantization(root)
    path = save(tmp_path, envelope(root, scope))
    monkeypatch.setenv(runtime.PROFILE_ENV, str(path))
    with pytest.raises(ValueError, match="approved"):
        runtime.configure_model_quantization(root)
    assert root[0] is original
    save(tmp_path, envelope(root, scope, approved=True))
    root[1].weight.add_(1)
    with pytest.raises(ValueError, match="scope mismatch"):
        runtime.configure_model_quantization(root)
    assert root[0] is original


def test_identity_install_and_private_optout(tmp_path, monkeypatch):
    root = model()
    inputs = torch.arange(128, dtype=torch.float32).reshape(2, 64) / 128
    expected = root(inputs)
    original = root[0]
    scope = runtime.model_scope(root)
    monkeypatch.setenv(runtime.MODE_ENV, "dynamic")
    monkeypatch.setenv(
        runtime.PROFILE_ENV, str(save(tmp_path, envelope(root, scope, approved=True)))
    )
    metadata = runtime.configure_model_quantization(root)
    assert metadata["approved"] is True
    assert root[0] is original
    torch.testing.assert_close(root(inputs), expected, rtol=0, atol=0)
    private = model()
    private._metal_quantization_metadata = {"managed": True, "mode": "off", "approved": False}
    assert runtime.configure_model_quantization(private)["mode"] == "off"
    assert not hasattr(private[0], "bits")


def test_self_managed_runtime_requires_explicit_identity(monkeypatch):
    monkeypatch.setenv(runtime.MODE_ENV, "dynamic")
    monkeypatch.setenv(runtime.PROFILE_ENV, "/unreachable/profile.json")
    with pytest.raises(NotImplementedError, match="metal_quantization_scope"):
        runtime.configure_model_quantization(object(), runtime="mlx")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_real_mps_capture_search_profile_loading_and_forward(tmp_path, monkeypatch):
    root = torch.nn.Sequential(
        torch.nn.Linear(64, 64, bias=False, device="mps", dtype=torch.float16)
    )
    root.eval().requires_grad_(False)
    root[0].weight.fill_(0.25)
    adapter = DenseModelAdapter(root)
    inputs = [
        torch.full((2, 64), (step + 1) / 8, device="mps", dtype=torch.float16) for step in range(4)
    ]
    scope = runtime.model_scope(root, {"steps": 4})
    with adapter.capture(["0"]) as session:
        expected = [root(value).cpu() for value in inputs]
    result = adapter.search(session.captures, scope=scope)
    profile = result["profile"]
    adapter.apply(profile, scope=scope, require_approved=False)
    for value, reference in zip(inputs, expected, strict=True):
        torch.testing.assert_close(root(value).cpu(), reference, rtol=0, atol=0)
    # The complete tiny pipeline matched on all four actual executions. Reload the
    # unchanged original to exercise production admission, not a module forwarding spy.
    fresh = torch.nn.Sequential(
        torch.nn.Linear(64, 64, bias=False, device="mps", dtype=torch.float16)
    )
    fresh.eval().requires_grad_(False)
    fresh[0].weight.fill_(0.25)
    calibration = {**profile["calibration"], "approved": True}
    approved = linear.build_profile(
        {"0": fresh[0]},
        ["0"],
        scope=scope,
        precisions={"0": result["precisions"]["0"]},
        calibration=calibration,
    )
    monkeypatch.setenv(runtime.MODE_ENV, "dynamic")
    monkeypatch.setenv(runtime.PROFILE_ENV, str(save(tmp_path, approved)))
    metadata = runtime.configure_model_quantization(fresh, {"steps": 4})
    assert metadata["approved"] is True
    assert approved["modules"]["0"]["bits"] == 4
    for value, reference in zip(inputs, expected, strict=True):
        torch.testing.assert_close(fresh(value).cpu(), reference, rtol=0, atol=0)
