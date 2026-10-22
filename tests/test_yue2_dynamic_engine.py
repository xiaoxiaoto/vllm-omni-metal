# SPDX-License-Identifier: Apache-2.0
"""Private adapter contracts; CPU fixtures are not real-model approval evidence."""

import gc
import importlib.util
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

DEPLOY = Path(__file__).resolve().parents[2] / "docker-compose/yue2-3b"
pytestmark = pytest.mark.skipif(
    not (DEPLOY / "yue2_dynamic_quantization.py").is_file(),
    reason="independent deployment checkout absent",
)
NAME = "nar_layers.0.mlp.down_proj"


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(DEPLOY))
    spec = importlib.util.spec_from_file_location(
        "yue2_dynamic_quantization", DEPLOY / "yue2_dynamic_quantization.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setenv("VLLM_OMNI_METAL_ANE", "off")
    for key in (
        "YUE2_METAL_QUANTIZATION",
        "YUE2_METAL_QUANTIZATION_PROFILE",
        "VLLM_OMNI_METAL_QUANTIZATION",
        "VLLM_OMNI_METAL_QUANTIZATION_PROFILE",
    ):
        monkeypatch.delenv(key, raising=False)
    return module


def records():
    result = []
    for mode, warmup, capture, seconds in [
        ("native", True, False, 10),
        ("float", True, True, 10),
        ("dynamic", True, False, 8),
        ("float", False, False, 10),
        ("dynamic", False, False, 8),
        ("dynamic", False, False, 8.1),
        ("float", False, False, 10.1),
    ]:
        result.append(
            {
                "mode": mode,
                "warmup": warmup,
                "capture": capture,
                "seconds": seconds,
                "pending": 0,
                "seed": 42,
                "frames": 512,
                "sample_rate": 48000,
                "shape": [240000, 2],
                "dtype": "float32",
                "audio_sha256": "a" * 64,
                "quantized": {
                    NAME: {
                        "metrics": {
                            "calls": int(mode == "dynamic"),
                            "rows": int(mode == "dynamic") * 512,
                        }
                    }
                }
                if mode not in ("native",) and not capture
                else {},
            }
        )
    return result


def completion():
    return {
        "approved": True,
        "local_pass": True,
        "records": records(),
        "publication_requirements": {
            "engine_shutdown": True,
            "supervisor_success": True,
            "release_assertion_verified": True,
        },
        "release": {
            "kind": "expected-post-release-assertion",
            "audio_sha256": "a" * 64,
            "pending": 0,
            "float_weakrefs_released": True,
            "benchmark_held_float_bytes": 0,
            "reproduced": True,
        },
    }


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nar_layers = torch.nn.ModuleList()
        self.nar_fused = []
        for _ in range(2):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.down_proj = torch.nn.Linear(64, 32, bias=False, dtype=torch.bfloat16)
            self.nar_layers.append(layer)
            packed = torch.randn(128, 32, dtype=torch.bfloat16) / 20
            self.nar_fused.append((None,) * 6 + (packed, layer.mlp.down_proj.weight))
        self._states = {}
        self._pending = None

    def nar_weights(self):
        return self.nar_fused


class PackedCpuFixture(torch.nn.Module):
    """Actual integer state for lifetime testing, not an MLX quality substitute."""

    def __init__(self, linear):
        super().__init__()
        scales = linear.weight.detach().float().abs().amax(1, keepdim=True).clamp_min(1e-6) / 127
        self.register_buffer(
            "codes", (linear.weight.detach().float() / scales).round().to(torch.int8)
        )
        self.register_buffer("scales", scales)
        self.metrics = {"calls": 0, "rows": 0}
        self.source_bytes = linear.weight.numel() * linear.weight.element_size()
        self.storage_bytes = sum(t.numel() * t.element_size() for t in self.buffers())

    def forward(self, x):
        self.metrics["calls"] += 1
        self.metrics["rows"] += x.numel() // x.shape[-1]
        return F.linear(x, (self.codes.float() * self.scales).to(x.dtype))


def test_off_is_identity_and_old_mode_is_rejected(adapter, monkeypatch):
    model = SimpleNamespace(_metal_ffn=object())
    previous = model._metal_ffn
    monkeypatch.delenv("YUE2_METAL_QUANTIZATION", raising=False)
    assert adapter.install(model, scope="unused") is None
    assert model._metal_ffn is previous
    assert model._metal_quantization_metadata == {
        "managed": True,
        "mode": "off",
        "approved": False,
    }
    monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION", "dynamic")
    monkeypatch.setenv("YUE2_METAL_QUANTIZATION", "off")
    monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION_PROFILE", "/missing/production.json")
    assert adapter.install(model, scope="unused") is None
    assert model._metal_ffn is previous
    monkeypatch.setenv("YUE2_METAL_QUANTIZATION", "manual-w8a16")
    with pytest.raises(ValueError, match="dynamic or off"):
        adapter.install(model, scope="unused")


def test_adapter_executes_root_and_preserves_packed_gate_up(adapter):
    model = TinyModel()
    bridge = adapter.DynamicNarAdapter(model, "native-scope")
    x = torch.randn(3, 32, dtype=torch.bfloat16)
    packed = model.nar_fused[0][6]
    hits = []
    hook = model.nar_layers[0].mlp.down_proj.register_forward_hook(lambda *args: hits.append(True))
    with torch.inference_mode():
        gate, up = F.linear(x, packed).chunk(2, -1)
        expected = F.linear(F.silu(gate) * up, model.nar_fused[0][7])
        assert torch.equal(bridge(0, x), expected)
    hook.remove()
    assert hits == [True]
    assert model.nar_fused[0][6] is packed
    original = model.nar_layers[0].mlp.down_proj
    packed_projection = PackedCpuFixture(original)
    model.nar_layers[0].mlp.down_proj = packed_projection
    model.nar_fused[0] = (*model.nar_fused[0][:7], None)
    del original
    with torch.inference_mode():
        bridge(0, x)
    assert packed_projection.metrics == {"calls": 1, "rows": 3}
    with pytest.raises(RuntimeError, match="gradients"):
        bridge(0, x)
    bridge.close()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="closed"):
        bridge(0, x)


def test_adapter_only_weakly_owns_model(adapter):
    model = TinyModel()
    bridge = adapter.DynamicNarAdapter(model, "scope")
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None
    with torch.inference_mode(), pytest.raises(RuntimeError, match="no longer exists"):
        bridge(0, torch.ones(1, 32))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["records"].pop(),
        lambda c: c["records"][0].update(audio_sha256="b" * 64),
        lambda c: c["records"][4].update(audio_sha256="b" * 64),
        lambda c: c["records"][4].update(seconds=10),
        lambda c: c["records"][4].update(pending=1),
        lambda c: c["records"][4]["quantized"][NAME]["metrics"].update(calls=0),
        lambda c: c["release"].update(reproduced=False),
        lambda c: c["release"].update(benchmark_held_float_bytes=4096),
        lambda c: c["publication_requirements"].update(engine_shutdown=False),
        lambda c: c.update(local_pass=False),
    ],
)
def test_production_recomputes_full_flow_approval(adapter, mutation):
    evidence = completion()
    assert adapter.validate_completion(evidence, selected=[NAME]) is evidence
    mutation(evidence)
    with pytest.raises(ValueError):
        adapter.validate_completion(evidence, selected=[NAME])


def test_all_bf16_is_not_quantization_approval(adapter):
    with pytest.raises(ValueError, match="quantized"):
        adapter.validate_completion(completion(), selected=[])


def test_profile_reader_rejects_ambiguous_nonfinite_and_public_files(tmp_path):
    from vllm_omni_metal.quantization.runtime import load_profile

    path = tmp_path / "profile.json"
    path.write_text('{"a":1,"a":2}')
    path.chmod(0o600)
    with pytest.raises(ValueError, match="duplicate"):
        load_profile(path, scope="native-scope")
    path.write_text('{"a":NaN}')
    with pytest.raises(ValueError):
        load_profile(path, scope="native-scope")
    path.write_text("{}")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        load_profile(path, scope="native-scope")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="schema2"):
        load_profile(path, scope="native-scope")
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(OSError):
        load_profile(link, scope="native-scope")


def test_source_checkpoint_set_and_content_are_bound(adapter, tmp_path, monkeypatch):
    model_dir, vae_dir = tmp_path / "model", tmp_path / "vae"
    sources = {}
    for label, directory in (("model", model_dir), ("vae", vae_dir)):
        directory.mkdir()
        path = directory / "model.safetensors"
        path.write_bytes(b"synthetic checkpoint identity only")
        sources[label] = [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": adapter.file_sha256(path),
            }
        ]
    monkeypatch.setenv("YUE2_VAE", str(vae_dir))
    model = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(model=str(model_dir)))
    )
    adapter.verify_sources(model, sources)
    (model_dir / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="content"):
        adapter.verify_sources(model, sources)


def test_unknown_floating_alias_is_rejected(adapter):
    model = TinyModel()
    weight = model.nar_layers[0].mlp.down_proj.weight
    adapter.reject_float_aliases(model, weight, 0)
    model.hidden_alias = {"weight": weight}
    with pytest.raises(ValueError, match="unregistered"):
        adapter.reject_float_aliases(model, weight, 0)


def test_idle_health_rejects_active_and_resource_owning_groups(adapter, monkeypatch):
    from vllm_omni_metal.ane.runtime import DenseSwiGLU

    model = TinyModel()
    group = DenseSwiGLU([(w[6], w[7]) for w in model.nar_fused], scope="scope")
    model._metal_ffn = group
    assert adapter.healthy_group(model) is group
    model._pending = object()
    with pytest.raises(RuntimeError, match="idle"):
        adapter.healthy_group(model)
    model._pending = None
    group._bridge = object()
    with pytest.raises(RuntimeError, match="resource-free"):
        adapter.healthy_group(model)


def local_calibration(adapter, model, *, quantized=True):
    """Synthetic metadata for rejection/cutover tests; never benchmark evidence."""
    from vllm_omni_metal.quantization.engine import _implementation_sha256

    names = [f"nar_layers.{i}.mlp.down_proj" for i in range(2)]
    precisions = {names[0]: "w8a16" if quantized else "bf16", names[1]: "bf16"}
    audits, captures = {}, {}
    for name in names:
        candidates = {}
        for precision, storage in (("bf16", 4096), ("w4a16", 1200), ("w8a16", 2304)):
            error = 0 if precision == "bf16" else (0.001 if precision == precisions[name] else 0.1)
            candidates[precision] = {
                "storage_bytes": storage,
                "relative_l2": error,
                "scaled_max_abs": error,
                "passed": error <= 0.01,
                "steps": [
                    {"step": step, "relative_l2": error, "scaled_max_abs": error}
                    for step in (0, 8, 16, 24)
                ],
            }
        audits[name] = {
            "source_bytes": 4096,
            "candidates": candidates,
            "selected": precisions[name],
        }
        captures[name] = [
            {"step": step, "shape": [512, 64], "dtype": "torch.bfloat16", "sha256": "b" * 64}
            for step in (0, 8, 16, 24)
        ]
    return {
        "engine": "DynamicEngine",
        "implementation_sha256": _implementation_sha256(),
        "approved": True,
        "local_pass": True,
        "thresholds": {"relative_l2": 0.01, "scaled_max_abs": 0.01},
        "audits": audits,
        "captures": captures,
        "precisions": precisions,
        "adapter_sha256": adapter.file_sha256(DEPLOY / "yue2_dynamic_quantization.py"),
        "preparation_sha256": adapter.file_sha256(DEPLOY / "prepare_dynamic_quantization.py"),
        "model_source_sha256": adapter.file_sha256(__file__),
        "source_weights": {},
        "full_pipeline": completion(),
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["calibration"]["captures"][NAME].pop(),
        lambda p: p["calibration"]["audits"][NAME]["candidates"]["w8a16"].update(relative_l2=0.5),
        lambda p: p["calibration"]["precisions"].update({NAME: "w4a16"}),
        lambda p: p["calibration"]["thresholds"].update(relative_l2=0.02),
    ],
)
def test_local_evidence_cannot_lose_steps_or_relax_quality(adapter, mutate):
    model = TinyModel()
    profile = {
        "selected": [f"nar_layers.{i}.mlp.down_proj" for i in range(2)],
        "modules": {
            NAME: {"bits": 8, "shape": [32, 64]},
            "nar_layers.1.mlp.down_proj": {"bits": 16, "shape": [32, 64]},
        },
        "calibration": local_calibration(adapter, model),
    }
    adapter.validate_search(profile)
    mutate(profile)
    with pytest.raises(ValueError):
        adapter.validate_search(profile)


@pytest.mark.parametrize("source", ["private", "global", "private-over-global"])
def test_production_cutover_clears_float_owners_and_preserves_bf16(
    adapter, monkeypatch, tmp_path, source
):
    import inspect
    import json

    from vllm_omni_metal.ane.runtime import DenseSwiGLU
    from vllm_omni_metal.quantization import linear

    monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION", "off" if source == "private" else "dynamic")
    if source != "global":
        monkeypatch.setenv("YUE2_METAL_QUANTIZATION", "dynamic")
    monkeypatch.setattr(linear, "_environment", lambda: {"fixture": "CPU ownership only"})
    monkeypatch.setattr(
        linear.QuantizedLinear,
        "from_linear",
        classmethod(lambda cls, module, **kwargs: PackedCpuFixture(module)),
    )
    monkeypatch.setattr(adapter, "verify_sources", lambda *args: None)
    model = TinyModel()
    previous = DenseSwiGLU([(w[6], w[7]) for w in model.nar_fused], scope="native-scope")
    model._metal_ffn = previous
    original = model.get_submodule(NAME)
    old = weakref.ref(original)
    retained = model.nar_layers[1].mlp.down_proj
    retained_weight = retained.weight
    packed = model.nar_fused[0][6]
    calibration = local_calibration(adapter, model)
    calibration["model_source_sha256"] = adapter.file_sha256(inspect.getfile(type(model)))
    modules = {n: model.get_submodule(n) for n in calibration["precisions"]}
    with torch.inference_mode():
        profile = linear.build_profile(
            modules,
            list(modules),
            scope="native-scope",
            calibration=calibration,
            precisions=calibration["precisions"],
        )
    path = tmp_path / "profile.json"
    other = dict(profile, scope="another-execution")
    other["sha256"] = adapter.digest({k: v for k, v in other.items() if k != "sha256"})
    path.write_text(
        json.dumps(
            {
                "profiles": {"another-execution": other, "native-scope": profile},
            }
        )
    )
    path.chmod(0o600)
    if source == "global":
        monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION_PROFILE", str(path))
    else:
        monkeypatch.setenv("YUE2_METAL_QUANTIZATION_PROFILE", str(path))
        monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION_PROFILE", "/wrong-global.json")
    del original, modules
    installed = adapter.install(model, scope="native-scope")
    gc.collect()
    assert old() is None
    assert previous._closed and previous.weights == ()
    assert model.nar_fused[0][7] is None
    assert model.nar_fused[0][6] is packed
    assert model.nar_layers[1].mlp.down_proj is retained
    assert retained.weight is retained_weight and model.nar_fused[1][7] is retained_weight
    assert installed is model._metal_ffn
    assert model._metal_quantization_metadata["managed"] is True
    assert model._metal_quantization_metadata["approved"] is True
    assert model._metal_quantization_metadata["profile"] == profile
    assert not hasattr(installed, "baselines") and not hasattr(installed, "weights")
    with torch.inference_mode():
        installed(0, torch.ones(2, 32, dtype=torch.bfloat16))
    assert model.get_submodule(NAME).metrics == {"calls": 1, "rows": 2}


def test_unapproved_profile_does_not_touch_native_group(adapter, monkeypatch, tmp_path):
    import json

    from vllm_omni_metal.ane.runtime import DenseSwiGLU

    model = TinyModel()
    previous = DenseSwiGLU([(w[6], w[7]) for w in model.nar_fused], scope="native-scope")
    model._metal_ffn = previous
    original = model.get_submodule(NAME)
    calibration = local_calibration(adapter, model)
    calibration["approved"] = False
    path = tmp_path / "profile.json"
    profile = {
        "schema": 2,
        "scope": "native-scope",
        "environment": {},
        "calibration": calibration,
        "selected": [NAME],
        "modules": {NAME: {"bits": 8}},
    }
    profile["sha256"] = adapter.digest(profile)
    path.write_text(json.dumps(profile))
    path.chmod(0o600)
    monkeypatch.setenv("YUE2_METAL_QUANTIZATION", "dynamic")
    monkeypatch.setenv("YUE2_METAL_QUANTIZATION_PROFILE", str(path))
    with pytest.raises(ValueError, match="freshly approved"):
        adapter.install(model, scope="native-scope")
    assert model._metal_ffn is previous and not previous._closed
    assert model.get_submodule(NAME) is original and model.nar_fused[0][7] is original.weight
