# SPDX-License-Identifier: Apache-2.0
"""独立 SeedVR2 消费者边界/事务；小模块不是 7B 质量或批准证据。"""

import copy
import importlib.util
import os
import sys
import types
import weakref
from pathlib import Path

import pytest
import torch

from vllm_omni_metal.quantization import linear as generic
from vllm_omni_metal.quantization.engine import DynamicEngine, _implementation_sha256

DEPLOY = Path(__file__).resolve().parents[2] / "docker-compose/seedvr2"
pytestmark = pytest.mark.skipif(
    not (DEPLOY / "quantization.py").is_file(), reason="独立 SeedVR2 部署 checkout 不存在"
)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.delenv("SEEDVR2_METAL_QUANTIZATION", raising=False)
    monkeypatch.delenv("SEEDVR2_METAL_QUANTIZATION_PROFILE", raising=False)
    monkeypatch.delenv("VLLM_OMNI_METAL_QUANTIZATION", raising=False)
    monkeypatch.delenv("VLLM_OMNI_METAL_QUANTIZATION_PROFILE", raising=False)
    package = types.ModuleType("_seedvr2_boundary_tests")
    package.__path__ = [str(DEPLOY)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    result = None
    for source, name in (("prepare.py", "weight_manifest"), ("quantization.py", "quantization")):
        full_name = package.__name__ + "." + name
        spec = importlib.util.spec_from_file_location(full_name, DEPLOY / source)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
        if name == "quantization":
            result = module
    return result


class BiasedGelu(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.proj_in = torch.nn.Linear(64, 128, dtype=torch.float16, device=device)
        self.act = torch.nn.GELU(approximate="tanh")
        self.proj_out = torch.nn.Linear(128, 64, dtype=torch.float16, device=device)

    def forward(self, x):
        return self.proj_out(self.act(self.proj_in(x)))


class SmallGraph(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.variant = "7b-comfy-fp16"
        self.transformer = torch.nn.Module()
        self.transformer.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(36)])
        for index in (0, 12, 24, 35):
            block = self.transformer.blocks[index]
            block.mlp = torch.nn.Module()
            block.mlp.vid = BiasedGelu(device)

    def forward(self, x):
        for index in (0, 12, 24, 35):
            x = self.transformer.blocks[index].mlp.vid(x)
        return x


def boundary_metadata(adapter):
    """只用于校验消费者证据 schema，不能作为真实校准产物发布。"""
    selected = adapter.CANDIDATES[0]
    samples = []
    for kind in ("image", "video"):
        for seed in adapter.SEEDS:
            samples.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "frames": 1 if kind == "image" else 5,
                    "relative_l2": 0.001,
                    "scaled_max_abs": 0.001,
                    "repeat_identical": True,
                    "released_baseline_identical": True,
                    "calls": {selected: 1},
                    "released_calls": {selected: 1},
                    "input_sha256": "1" * 64,
                    "baseline_sha256": "2" * 64,
                    "candidate_sha256": "3" * 64,
                }
            )
    return {
        "adapter": adapter.ADAPTER_VERSION,
        "status": "approved",
        "engine": "DynamicEngine",
        "approved": True,
        "implementation_sha256": _implementation_sha256(),
        "thresholds": {"relative_l2": 0.01, "scaled_max_abs": 0.01},
        "precisions": {selected: "w8a16"},
        "audits": {selected: {"selected": "w8a16", "candidates": {"w8a16": {"passed": True}}}},
        "captures": {
            selected: [
                {"step": step, "dtype": "torch.float16", "shape": [2, 128], "sha256": "4" * 64}
                for step in range(4)
            ]
        },
        "lossy": True,
        "samples": samples,
        "audited_calls": 4,
        "local": [
            {
                "relative_l2": 0.001,
                "scaled_max_abs": 0.001,
                "kind": sample["kind"],
                "seed": sample["seed"],
                "calls": 1,
                "shape": [2, 128],
                "output_shape": [2, 64],
            }
            for sample in samples
        ],
    }


def profile_for(adapter, graph):
    with torch.inference_mode():
        return generic.build_profile(
            adapter.modules(graph),
            [adapter.CANDIDATES[0]],
            scope="behavior-only",
            calibration=boundary_metadata(adapter),
        )


def test_default_and_explicit_off_never_read_profile(adapter, monkeypatch, tmp_path):
    monkeypatch.setenv("SEEDVR2_METAL_QUANTIZATION_PROFILE", str(tmp_path / "absent.json"))
    graph = SmallGraph()
    before = graph.get_submodule(adapter.CANDIDATES[0])
    adapter.consume_profile(graph)
    assert graph.seedvr2_quantization == {
        "enabled": False,
        "reason": "explicit-off",
        "selected": [],
    }
    assert graph.get_submodule(adapter.CANDIDATES[0]) is before
    assert graph._metal_quantization_metadata == {
        "managed": True,
        "mode": "off",
        "approved": False,
    }
    monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION", "dynamic")
    monkeypatch.setenv("SEEDVR2_METAL_QUANTIZATION", "off")
    adapter.consume_profile(graph)
    assert graph._metal_quantization_metadata["mode"] == "off"


def test_dynamic_missing_or_unknown_mode_fails(adapter, monkeypatch):
    graph = SmallGraph()
    monkeypatch.setenv("SEEDVR2_METAL_QUANTIZATION", "dynamic")
    with pytest.raises(ValueError, match="缺少批准"):
        adapter.consume_profile(graph)
    for mode in ("auto", "on", "w8a16"):
        monkeypatch.setenv("SEEDVR2_METAL_QUANTIZATION", mode)
        with pytest.raises(ValueError, match="off/dynamic"):
            adapter.consume_profile(graph)


def test_global_fallback_uses_scoped_loader(adapter, monkeypatch, tmp_path):
    from vllm_omni_metal.quantization import runtime

    path = str(tmp_path / "bundle.json")
    graph = SmallGraph()
    monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION", "dynamic")
    monkeypatch.setenv("VLLM_OMNI_METAL_QUANTIZATION_PROFILE", path)
    monkeypatch.setattr(adapter, "scope", lambda graph: "behavior-only")
    calls = []

    def load(file, *, scope):
        calls.append((file, scope))
        return {"selected": [], "calibration": {}}

    monkeypatch.setattr(runtime, "load_profile", load)
    with pytest.raises(ValueError, match="批准"):
        adapter.consume_profile(graph)
    assert calls == [(path, "behavior-only")]
    assert graph._metal_quantization_metadata["approved"] is False


def test_local_engine_search_is_not_model_approval(adapter, monkeypatch):
    monkeypatch.setattr(generic, "_environment", lambda: {})
    graph = SmallGraph()
    profile = profile_for(adapter, graph)
    profile["calibration"]["approved"] = False
    with pytest.raises(ValueError, match="批准"):
        adapter.validate_approval(profile)


@pytest.mark.parametrize(
    "change",
    [
        "video-missing",
        "zero-calls",
        "release-missing",
        "repeat-failed",
        "over-one-percent",
        "local-only",
        "local-coverage",
        "local-zero-calls",
    ],
)
def test_approval_requires_actual_complete_outputs(adapter, change):
    name = adapter.CANDIDATES[0]
    profile = {
        "selected": [name],
        "modules": {
            name: {"bits": 8, "group_size": 64, "mode": "affine", "dtype": "torch.float16"}
        },
        "calibration": boundary_metadata(adapter),
    }
    data = profile["calibration"]
    if change == "video-missing":
        data["samples"] = data["samples"][:2]
    elif change == "zero-calls":
        data["samples"][0]["calls"][adapter.CANDIDATES[0]] = 0
    elif change == "release-missing":
        data["samples"][0].pop("released_baseline_identical")
    elif change == "repeat-failed":
        data["samples"][0]["repeat_identical"] = False
    elif change == "over-one-percent":
        data["samples"][0]["relative_l2"] = 0.010001
    elif change == "local-coverage":
        for entry in data["local"]:
            entry["kind"] = "image"
    elif change == "local-zero-calls":
        data["local"][0]["calls"] = 0
    else:
        data.pop("samples")
    with pytest.raises(ValueError):
        adapter.validate_approval(profile)


@pytest.mark.parametrize("precision", ["w4a16", "w8a16", "fp16"])
def test_dynamic_consumer_accepts_quantized_or_source_identity(adapter, monkeypatch, precision):
    monkeypatch.setattr(generic, "_environment", lambda: {})
    graph = SmallGraph()
    name = adapter.CANDIDATES[0]
    metadata = boundary_metadata(adapter)
    metadata["precisions"][name] = precision
    metadata["audits"][name] = {"selected": precision, "candidates": {precision: {"passed": True}}}
    with torch.inference_mode():
        profile = generic.build_profile(
            {name: graph.get_submodule(name)},
            [name],
            scope="behavior-only",
            calibration=metadata,
            precisions={name: precision},
        )
    adapter.validate_approval(profile)


def test_dynamic_consumer_rejects_bf16_conversion(adapter):
    name = adapter.CANDIDATES[0]
    profile = {
        "selected": [name],
        "calibration": boundary_metadata(adapter),
        "modules": {
            name: {"bits": 16, "mode": "identity", "group_size": None, "dtype": "torch.bfloat16"}
        },
    }
    with pytest.raises(ValueError, match="源 FP16"):
        adapter.validate_approval(profile)


def test_rejects_swiglu_unbiased_or_tuple_linears(adapter):
    graph = SmallGraph()
    parent = graph.get_submodule(adapter.CANDIDATES[0].rpartition(".")[0])
    parent.act = torch.nn.SiLU()
    with pytest.raises(ValueError, match="GELU"):
        adapter.modules(graph)
    parent.act = torch.nn.GELU(approximate="tanh")
    parent.proj_out = torch.nn.Linear(128, 64, bias=False, dtype=torch.float16)
    with pytest.raises(ValueError, match="bias"):
        adapter.modules(graph)

    class TupleLinear(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    parent.proj_out = TupleLinear(128, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="投影合同"):
        adapter.modules(graph)


@pytest.mark.parametrize("alias", ["module", "weight", "buffer"])
def test_alias_outside_candidates_rejected_before_replacement(adapter, monkeypatch, alias):
    monkeypatch.setattr(generic, "_environment", lambda: {})
    graph = SmallGraph()
    selected = graph.get_submodule(adapter.CANDIDATES[0])
    profile = profile_for(adapter, graph)
    if alias == "module":
        graph.alias = selected
    elif alias == "weight":
        graph.extra_weight = selected.weight
    else:
        graph.register_buffer("extra_buffer", selected.weight.detach())
    with torch.inference_mode(), pytest.raises(ValueError, match="shared/tied"):
        adapter.apply(graph, profile, bound_scope="behavior-only")
    assert graph.get_submodule(adapter.CANDIDATES[0]) is selected


def test_failed_construction_and_stale_profile_are_transactional(adapter, monkeypatch):
    monkeypatch.setattr(generic, "_environment", lambda: {})
    graph = SmallGraph()
    selected = graph.get_submodule(adapter.CANDIDATES[0])
    profile = profile_for(adapter, graph)
    changed = copy.deepcopy(profile)
    changed["sha256"] = "0" * 64
    with torch.inference_mode(), pytest.raises(ValueError, match="hash"):
        adapter.apply(graph, changed, bound_scope="behavior-only")

    def fail(cls, linear, **kwargs):
        raise MemoryError("construction-failure")

    monkeypatch.setattr(generic.QuantizedLinear, "from_linear", classmethod(fail))
    with torch.inference_mode(), pytest.raises(MemoryError, match="construction-failure"):
        adapter.apply(graph, profile, bound_scope="behavior-only")
    assert graph.get_submodule(adapter.CANDIDATES[0]) is selected


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="需要显式真实 MPS 集成测试",
)
def test_real_quantized_gelu_forward_counts_and_input_boundaries(adapter):
    graph = SmallGraph("mps").eval()
    with torch.inference_mode():
        profile = profile_for(adapter, graph)
        original = weakref.ref(graph.get_submodule(adapter.CANDIDATES[0]))
        replacements = adapter.apply(graph, profile, bound_scope="behavior-only")
        projection = replacements[adapter.CANDIDATES[0]]
        output = graph(torch.ones(2, 64, dtype=torch.float16, device="mps"))
        assert output.shape == (2, 64) and torch.isfinite(output).all()
        assert projection.metrics["calls"] == 1
        assert original() is None
        for x in (
            torch.ones(2, 128, device="mps"),
            torch.ones(2, 256, dtype=torch.float16, device="mps")[:, ::2],
            torch.ones(2, 128, dtype=torch.float16),
        ):
            with pytest.raises(ValueError, match="activation"):
                projection(x)
        assert projection.metrics["calls"] == 1


def test_missing_real_model_rejected_without_output_or_profile(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "seedvr2_offline_boundary", DEPLOY / "prepare_quantization.py"
    )
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    output, profile = tmp_path / "evidence", tmp_path / "approved.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_quantization.py",
            "--model",
            str(tmp_path / "missing-model"),
            "--image",
            str(tmp_path / "real.png"),
            "--video",
            str(tmp_path / "real.mp4"),
            "--output",
            str(output),
            "--profile",
            str(profile),
        ],
    )
    with pytest.raises(ValueError, match="model_index"):
        preparation.main()
    assert not output.exists() and not profile.exists()


@pytest.fixture
def calibration():
    spec = importlib.util.spec_from_file_location(
        "seedvr2_stream_boundary", DEPLOY / "prepare_quantization.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_streaming_local_error_covers_tail_and_preserves_budget(calibration):
    reference = torch.arange(40, dtype=torch.float32) / 40 + 1
    actual = reference.clone()
    actual[-1] += 1
    # 小预算强制多个归约块；尾部差异必须计入，不是抽样或前缀比较。
    error = calibration.projection_error(reference, actual, budget=actual.nbytes + 16 * 7)
    expected = torch.linalg.vector_norm((actual - reference).double()) / torch.linalg.vector_norm(
        reference.double()
    )
    assert error["relative_l2"] == pytest.approx(float(expected), rel=1e-6)
    assert error["scaled_max_abs"] > 0
    with pytest.raises(MemoryError, match="128MiB"):
        calibration.projection_error(reference, actual, budget=actual.nbytes)


def test_atomic_approval_publication_never_replaces_existing(calibration, tmp_path):
    target = tmp_path / "approved.json"
    calibration.publish_json(target, {"approved": "first"})
    before = target.read_bytes()
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        calibration.publish_json(target, {"approved": "second"})
    assert target.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["approved.json"]
    link = tmp_path / "linked.json"
    link.symlink_to(tmp_path / "missing-target.json")
    with pytest.raises(FileExistsError):
        calibration.publish_json(link, {"approved": "third"})
    assert link.is_symlink() and not (tmp_path / "missing-target.json").exists()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="需要显式真实 MPS 集成测试",
)
def test_disk_candidate_four_workloads_release_complete_inputs(calibration, tmp_path):
    module = (
        torch.nn.Linear(128, 64, dtype=torch.float16, device="mps").eval().requires_grad_(False)
    )
    workloads = [(kind, seed) for kind in ("image", "video") for seed in (7723, 7724)]
    baseline = {}
    references = []
    calls = []

    def run(workload, label):
        # A new full input per workload, including a second projection call that
        # must not be captured. Weakrefs detect accidental retention by hooks/store.
        assert all(reference() is None for reference in references)
        activation = torch.full(
            (2, 128), float(workloads.index(workload) + 1), dtype=torch.float16, device="mps"
        )
        references.append(weakref.ref(activation))
        with torch.inference_mode():
            output = module(activation).cpu().numpy()
            module(activation + 1)
        calls.append(workload)
        return output, {}

    for workload in workloads:
        baseline[workload] = run(workload, "baseline")[0]
    calls.clear()
    result = calibration.capture_candidate(
        "projection",
        module,
        workloads,
        run,
        baseline,
        tmp_path / "capture-0",
        label="capture",
        scope="behavior-only",
    )
    assert calls == workloads
    assert all(reference() is None for reference in references)
    assert not module._forward_pre_hooks
    assert not (tmp_path / "capture-0").exists()
    assert (tmp_path / "capture-0.json").is_file()
    assert [
        entry["step"] for entry in result["profile"]["calibration"]["captures"]["projection"]
    ] == list(range(4))
    assert module.weight.dtype == torch.float16


@pytest.mark.parametrize("failure", ["record", "search"])
def test_disk_candidate_failures_close_hook_and_store(calibration, tmp_path, monkeypatch, failure):
    from vllm_omni_metal.quantization.capture import DiskCaptures

    module = torch.nn.Linear(128, 64, dtype=torch.float16).eval().requires_grad_(False)
    workloads = [("image", 7723), ("image", 7724), ("video", 7723), ("video", 7724)]
    baseline = {}

    def run(workload, label):
        with torch.inference_mode():
            output = module(torch.ones(2, 128, dtype=torch.float16)).numpy()
        return output, {}

    for workload in workloads:
        baseline[workload] = run(workload, "baseline")[0]
    if failure == "record":
        original_record = DiskCaptures.record

        def fail_record(self, name, step, tensor):
            original_record(self, name, step, tensor)
            if step == 1:
                raise RuntimeError("record failure")

        monkeypatch.setattr(DiskCaptures, "record", fail_record)
    else:

        def fail_search(modules, captures, *, scope):
            assert captures.steps("projection") == [0, 1, 2, 3]
            assert not module._forward_pre_hooks
            raise RuntimeError("search failure")

        monkeypatch.setattr(DynamicEngine, "search", fail_search)
    with pytest.raises(RuntimeError, match=f"{failure} failure"):
        calibration.capture_candidate(
            "projection",
            module,
            workloads,
            run,
            baseline,
            tmp_path / "capture-0",
            label="capture",
            scope="behavior-only",
        )
    assert not module._forward_pre_hooks
    assert not (tmp_path / "capture-0").exists()
    assert (tmp_path / "capture-0.json").is_file()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="需要显式真实 MPS 集成测试",
)
def test_real_dynamic_engine_search_preserves_seedvr2_source_fp16(adapter):
    graph = SmallGraph("mps").eval()
    name = adapter.CANDIDATES[0]
    module = graph.get_submodule(name)
    generator = torch.Generator().manual_seed(7723)
    captures = {
        name: [
            (step, torch.randn(2, 128, generator=generator, dtype=torch.float16))
            for step in range(4)
        ]
    }
    result = DynamicEngine.search({name: module}, captures, scope="behavior-only")
    assert result["precisions"][name] in ("w4a16", "w8a16", "fp16")
    assert result["profile"]["modules"][name]["dtype"] == "torch.float16"
    assert module.weight.dtype == torch.float16
    assert result["profile"]["calibration"]["approved"] is False
    with pytest.raises(ValueError, match="批准"):
        adapter.apply(graph, result["profile"], bound_scope="behavior-only")


def load_joint(adapter):
    return __import__(adapter.__package__ + ".joint", fromlist=["joint"])


def test_joint_zero_hit_is_fail_closed_with_real_cpu_projection(adapter):
    joint = load_joint(adapter)
    linear = torch.nn.Linear(64, 128, bias=True, dtype=torch.float16)
    original = linear.weight.detach().clone()
    wrapper = joint.JointProjection(linear, scope="behavior-only")
    try:
        with torch.inference_mode(), pytest.raises(RuntimeError, match="nonzero ANE hit"):
            wrapper(torch.ones(1, 300, 64, dtype=torch.float16))
        assert torch.equal(linear.weight, original)
        assert wrapper.projection.metrics["ane_chunks"] == 0
    finally:
        wrapper.close()
    assert wrapper.projection._closed


def test_joint_requires_original_biased_fp16_input_projection(adapter):
    joint = load_joint(adapter)
    for linear in (
        torch.nn.Linear(64, 128, bias=False, dtype=torch.float16),
        torch.nn.Linear(64, 128, bias=True, dtype=torch.float32),
    ):
        with pytest.raises(ValueError, match="original FP16 biased"):
            joint.JointProjection(linear, scope="behavior-only")


def test_joint_mode_requires_explicit_approval_before_model_access(adapter, monkeypatch):
    monkeypatch.setenv("SEEDVR2_METAL_QUANTIZATION", "joint")
    graph = SmallGraph()
    with pytest.raises(ValueError, match="explicit approved profile"):
        adapter.consume_profile(graph)
    assert graph._metal_quantization_metadata["approved"] is False


def test_joint_calibration_requires_prequantization_scope(adapter):
    module = __import__(adapter.__package__ + ".joint_calibration", fromlist=["calibrate"])
    with pytest.raises(TypeError, match="bound_scope"):
        module.calibrate(None, None, None, None, None, None, None)


def test_joint_scope_binds_prequantization_identity_without_reopening_model(adapter):
    joint = load_joint(adapter)
    # This object has no packed.weight or model configuration to inspect.
    pipeline = object()
    first = joint.execution_scope(pipeline, bound_scope="original-full-model-a")
    assert first == joint.execution_scope(pipeline, bound_scope="original-full-model-a")
    assert first != joint.execution_scope(pipeline, bound_scope="original-full-model-b")
