# SPDX-License-Identifier: Apache-2.0
"""CPU policy regressions; opt-in MPS tests exercise actual packed MLX kernels."""

import gc
import json
import os
import subprocess
import sys
import weakref

import pytest
import torch

from vllm_omni_metal.quantization import engine, linear
from vllm_omni_metal.quantization.engine import DynamicEngine

SCOPE = "dynamic-test/source-and-configuration"
MPS = pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="explicit MPS integration opt-in required",
)


def source(width=64, outputs=64, device="cpu"):
    module = torch.nn.Linear(width, outputs, dtype=torch.bfloat16, bias=False)
    module.requires_grad_(False)
    generator = torch.Generator().manual_seed(42)
    module.weight.copy_(torch.randn(outputs, width, generator=generator).to(torch.bfloat16) * 0.02)
    return module.to(device)


def captures(width=64):
    generator = torch.Generator().manual_seed(79)
    return [
        (step, torch.randn(2, 3, width, generator=generator).to(torch.bfloat16))
        for step in (0, 8, 16, 24)
    ]


@pytest.fixture
def environment(monkeypatch):
    monkeypatch.setattr(linear, "_environment", lambda: {"test_environment": True})


@pytest.fixture
def cpu_search(monkeypatch, environment):
    # Only policy tests bypass the MPS admission gate. Their deterministic fake
    # kernels do not claim evidence about MLX math or accelerator execution.
    validate = linear._linear

    def allow_cpu(module, *, mps=False, bits=8):
        return validate(module, bits=bits)

    monkeypatch.setattr(linear, "_linear", allow_cpu)
    calls = []
    outputs = {4: 0.0, 8: 0.0}
    sizes = {4: 100, 8: 200}

    class Candidate:
        def __init__(self, module, bits):
            self.module = module
            self.bits = bits
            self.storage_bytes = sizes[bits]

        def __call__(self, tensor):
            calls.append((self.bits, tuple(tensor.shape)))
            return self.module(tensor) + outputs[self.bits]

    monkeypatch.setattr(
        linear.QuantizedLinear,
        "from_linear",
        classmethod(lambda cls, module, *, bits=8, **kwargs: Candidate(module, bits)),
    )
    return calls, outputs, sizes


def test_namespace_import_is_runtime_lazy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vllm_omni_metal.quantization; "
            "assert 'mlx.core' not in sys.modules; "
            "assert 'vllm_omni_metal.quantization.linear' not in sys.modules; "
            "assert 'vllm_omni_metal.quantization.engine' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_metrics_keep_real_bf16_error_and_zero_reference():
    reference = torch.tensor([1, 0], dtype=torch.bfloat16)
    changed = torch.tensor([1, 0.01], dtype=torch.bfloat16)
    error = engine._errors(reference, changed)
    assert error["relative_l2"] == pytest.approx(float(changed[1]))
    assert error["scaled_max_abs"] == pytest.approx(float(changed[1]))
    assert engine._errors(reference * 0, reference * 0) == {"relative_l2": 0, "scaled_max_abs": 0}
    assert engine._errors(reference * 0, reference)["relative_l2"] > 1


@pytest.mark.parametrize("candidate", [torch.tensor([float("nan")]), torch.ones(2)])
def test_metrics_reject_nonfinite_or_changed_shape(candidate):
    with pytest.raises(ValueError):
        engine._errors(torch.ones(1), candidate)


def test_both_bounds_inclusive_and_independent():
    audit = engine._audit(10, [{"step": 0, "relative_l2": 0.01, "scaled_max_abs": 0.01}])
    assert audit["passed"]
    for field in ("relative_l2", "scaled_max_abs"):
        error = {"step": 0, "relative_l2": 0, "scaled_max_abs": 0, field: 0.010000001}
        assert not engine._audit(10, [error])["passed"]


def test_search_independent_trials_all_steps_and_smallest_actual_storage(cpu_search):
    calls, _, sizes = cpu_search
    sizes.update({4: 300, 8: 100})  # Do not select using bit width alone.
    module = source()
    result = DynamicEngine.search({"proj": module}, {"proj": captures()}, scope=SCOPE)
    assert result["precisions"] == {"proj": "w8a16"}
    assert [bits for bits, _ in calls] == [4] * 4 + [8] * 4
    calibration = result["profile"]["calibration"]
    assert calibration["engine"] == "DynamicEngine" and calibration["approved"] is False
    assert calibration["precisions"] == result["precisions"]
    assert calibration["audits"] == result["audits"]
    assert calibration["captures"]["proj"][0]["shape"] == [2, 3, 64]
    assert calibration["captures"]["proj"][0]["dtype"] == "torch.bfloat16"
    assert len(calibration["captures"]["proj"][0]["sha256"]) == 64
    assert linear.validate_profile(result["profile"], {"proj": module}, scope=SCOPE) == ["proj"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "four,eight,expected", [(10, 0, "w8a16"), (10, 10, "bf16"), (0, 10, "w4a16")]
)
def test_search_errors_select_or_retain(cpu_search, four, eight, expected):
    _, errors, _ = cpu_search
    errors.update({4: four, 8: eight})
    result = DynamicEngine.search({"proj": source()}, {"proj": captures()}, scope=SCOPE)
    assert result["precisions"]["proj"] == expected
    assert result["local_pass"] is True  # BF16 is legal, not evidence of a benefit.


def test_storage_ties_and_unaligned_layers_retain_bf16(cpu_search):
    _, _, sizes = cpu_search
    module = source()
    sizes.update({4: module.weight.numel() * 2, 8: module.weight.numel() * 2})
    assert DynamicEngine.search({"proj": module}, {"proj": captures()}, scope=SCOPE)[
        "precisions"
    ] == {"proj": "bf16"}
    assert DynamicEngine.search({"proj": source(63)}, {"proj": captures(63)}, scope=SCOPE)[
        "precisions"
    ] == {"proj": "bf16"}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.clear(),
        lambda data: data.update(proj=data["proj"][:3]),
        lambda data: data["proj"].__setitem__(0, (-1, data["proj"][0][1])),
        lambda data: data["proj"].__setitem__(0, (True, data["proj"][0][1])),
        lambda data: data["proj"].__setitem__(0, data["proj"][1]),
        lambda data: data["proj"].__setitem__(0, (0, torch.zeros(1, 63, dtype=torch.bfloat16))),
        lambda data: data["proj"].__setitem__(0, (0, torch.zeros(1, 64))),
        lambda data: data["proj"][0][1].fill_(float("nan")),
        lambda data: data["proj"].__setitem__(0, (0, torch.zeros(0, 64, dtype=torch.bfloat16))),
    ],
)
def test_capture_admission_before_quantization(cpu_search, mutation):
    calls, _, _ = cpu_search
    data = {"proj": captures()}
    mutation(data)
    with pytest.raises(ValueError):
        DynamicEngine.search({"proj": source()}, data, scope=SCOPE)
    assert calls == []


def test_candidate_steps_must_match_but_order_does_not_matter(cpu_search):
    modules = {"first": source(), "second": source()}
    data = {"first": captures(), "second": list(reversed(captures()))}
    result = DynamicEngine.search(modules, data, scope=SCOPE)
    assert (
        result["profile"]["calibration"]["captures"]["first"]
        == result["profile"]["calibration"]["captures"]["second"]
    )
    data["second"][0] = (25, data["second"][0][1])
    with pytest.raises(ValueError, match="same captured steps"):
        DynamicEngine.search(modules, data, scope=SCOPE)


def test_capture_fingerprint_binds_bits_shape_and_steps(cpu_search):
    module = source()
    data = {"proj": [(step, torch.zeros(2, 64, dtype=torch.bfloat16)) for step in range(4)]}
    before = DynamicEngine.search({"proj": module}, data, scope=SCOPE)["profile"]["calibration"][
        "captures"
    ]
    data["proj"][0][1][0, 0] = -0.0
    after = DynamicEngine.search({"proj": module}, data, scope=SCOPE)["profile"]["calibration"][
        "captures"
    ]
    assert before != after


def test_cpu_source_never_falls_back(environment):
    with pytest.raises(ValueError, match="requires MPS"):
        DynamicEngine.search({"proj": source()}, {"proj": captures()}, scope=SCOPE)


def test_aliases_rejected_before_trials(cpu_search):
    module = source()
    with pytest.raises(ValueError, match="shared/tied"):
        DynamicEngine.search(
            {"a": module, "b": module}, {"a": captures(), "b": captures()}, scope=SCOPE
        )
    other = source()
    other.weight = module.weight
    with pytest.raises(ValueError, match="shared/tied"):
        DynamicEngine.search(
            {"a": module, "b": other}, {"a": captures(), "b": captures()}, scope=SCOPE
        )


def test_search_mixed_source_dtypes_and_fp32_is_never_quantized(cpu_search):
    calls, errors, _ = cpu_search
    errors.update({4: 10, 8: 10})
    modules = {
        "half": source().to(torch.float16),
        "brain": source(),
        "single": source().to(torch.float32),
    }
    data = {
        name: [(step, tensor.to(module.weight.dtype)) for step, tensor in captures()]
        for name, module in modules.items()
    }
    result = DynamicEngine.search(modules, data, scope=SCOPE)
    assert result["precisions"] == {"half": "fp16", "brain": "bf16", "single": "fp32"}
    assert len(calls) == 16  # FP32 has reference execution only, no packed trials.
    assert result["audits"]["single"]["candidates"]["w4a16"]["reason"] == (
        "FP32 source is retention-only"
    )
    for name, module in modules.items():
        assert result["profile"]["calibration"]["captures"][name][0]["dtype"] == str(
            module.weight.dtype
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_search_capture_dtype_must_match_each_source(cpu_search, dtype):
    calls, _, _ = cpu_search
    with pytest.raises(ValueError, match="matching source dtype"):
        DynamicEngine.search({"proj": source().to(dtype)}, {"proj": captures()}, scope=SCOPE)
    assert not calls


@MPS
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_real_mps_source_dtype_search_preserves_numerical_contract(dtype):
    module = source(device="mps").to(dtype)
    data = {"proj": [(step, tensor.to(dtype)) for step, tensor in captures()]}
    result = DynamicEngine.search({"proj": module}, data, scope=SCOPE)
    assert result["profile"]["modules"]["proj"]["dtype"] == str(dtype)
    if dtype == torch.float32:
        assert result["precisions"] == {"proj": "fp32"}
        assert result["profile"]["modules"]["proj"]["bits"] == 32
    else:
        # Each trial really executed on four inputs; no local result is approval.
        for precision in ("w4a16", "w8a16"):
            audit = result["audits"]["proj"]["candidates"][precision]
            assert len(audit["steps"]) == 4
            assert audit["relative_l2"] >= 0 and audit["scaled_max_abs"] >= 0
    assert result["profile"]["calibration"]["approved"] is False


def test_apply_requires_explicit_approval_and_keeps_bf16_identity(environment):
    root = torch.nn.Sequential(source())
    original = root[0]
    profile = linear.build_profile(
        {"0": original},
        ["0"],
        scope=SCOPE,
        calibration={
            "engine": "DynamicEngine",
            "approved": False,
            "implementation_sha256": engine._implementation_sha256(),
        },
        precisions={"0": "bf16"},
    )
    with pytest.raises(ValueError, match="not been approved"):
        DynamicEngine.apply(root, profile, scope=SCOPE)
    assert DynamicEngine.apply(root, profile, scope=SCOPE, require_approved=False)["0"] is original
    with pytest.raises(ValueError, match="scope"):
        DynamicEngine.apply(root, profile, scope="changed", require_approved=False)
    assert root[0] is original


def test_apply_never_inherits_another_engine_approval(environment):
    root = torch.nn.Sequential(source())
    profile = linear.build_profile(
        {"0": root[0]},
        ["0"],
        scope=SCOPE,
        calibration={"engine": "old-engine", "approved": True},
        precisions={"0": "bf16"},
    )
    with pytest.raises(ValueError, match="not calibrated"):
        DynamicEngine.apply(root, profile, scope=SCOPE)


@MPS
def test_real_mps_search_calls_w4_w8_and_persistent_state(monkeypatch):
    import mlx.core as mx

    root = torch.nn.Sequential(source(device="mps"))
    data = {"0": captures()}
    original = root[0]
    reference = weakref.ref(original)
    bits_called = []
    actual = mx.quantized_matmul

    def counted(*args, **kwargs):
        bits_called.append(kwargs["bits"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(mx, "quantized_matmul", counted)
    result = DynamicEngine.search({"0": original}, data, scope=SCOPE)
    assert bits_called.count(4) == bits_called.count(8) == 4
    assert root[0] is original
    # Independently apply both real storage representations and prove neither
    # retains a Tensor/Linear source or a floating-weight fallback.
    for precision, bits in (("w4a16", 4), ("w8a16", 8)):
        profile = linear.build_profile(
            {"0": original},
            ["0"],
            scope=SCOPE,
            calibration={
                "engine": "DynamicEngine",
                "approved": False,
                "implementation_sha256": engine._implementation_sha256(),
            },
            precisions={"0": precision},
        )
        applied = DynamicEngine.apply(root, profile, scope=SCOPE, require_approved=False)
        quantized = applied["0"]
        assert quantized.bits == bits
        assert not list(quantized.parameters()) and not list(quantized.buffers())
        assert not any(
            isinstance(value, (torch.Tensor, torch.nn.Linear)) for value in vars(quantized).values()
        )
        assert quantized.storage_bytes < quantized.source_bytes
        output = root(data["0"][0][1].to("mps"))
        assert output.dtype == torch.bfloat16 and output.device.type == "mps"
        root[0] = original
    del original
    # A quantized replacement owns no reference to its former BF16 module.
    root[0] = quantized
    gc.collect()
    assert reference() is None
    assert result["profile"]["calibration"]["approved"] is False


@MPS
def test_real_mps_nonzero_quant_error_on_zero_bf16_output_retains_identity():
    # Derive a null-space activation from the actual weight. Exact zero BF16
    # reference with nonzero quantized output exceeds both strict relative bounds.
    module = source(outputs=1, device="mps")
    with torch.no_grad():
        module.weight.zero_()
        module.weight[0, 0] = 1
        module.weight[0, 1] = 0.125
        module.weight[0, 2] = -1
    activation = torch.zeros(1, 64, dtype=torch.bfloat16)
    activation[0, 0] = -0.125
    activation[0, 1] = 1
    data = {"0": [(step, activation.clone()) for step in range(4)]}
    root = torch.nn.Sequential(module)
    result = DynamicEngine.search({"0": module}, data, scope=SCOPE)
    assert result["precisions"] == {"0": "bf16"}
    assert (
        DynamicEngine.apply(root, result["profile"], scope=SCOPE, require_approved=False)["0"]
        is module
    )


def test_search_rejects_nonfinite_candidate_outputs_instead_of_silent_fallback(cpu_search):
    _, errors, _ = cpu_search
    errors[4] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        DynamicEngine.search({"proj": source()}, {"proj": captures()}, scope=SCOPE)


@pytest.mark.parametrize("kind", ["empty", "dtype_mismatch", "subclass", "hook"])
def test_search_rejects_ineligible_modules(cpu_search, kind):
    module = source()
    if kind == "empty":
        modules, data = {}, {}
    else:
        if kind == "dtype_mismatch":
            module = module.to(torch.float16)
        elif kind == "subclass":

            class SpecializedLinear(torch.nn.Linear):
                pass

            module = SpecializedLinear(64, 64, dtype=torch.bfloat16).requires_grad_(False)
        elif kind == "hook":
            module.register_forward_hook(lambda *args: None)
        modules, data = {"proj": module}, {"proj": captures()}
    with pytest.raises(ValueError):
        DynamicEngine.search(modules, data, scope=SCOPE)


def test_apply_checks_current_engine_source_and_approved_boolean(environment):
    root = torch.nn.Sequential(source())
    base = {
        "engine": "DynamicEngine",
        "approved": True,
        "implementation_sha256": engine._implementation_sha256(),
    }
    profile = linear.build_profile(
        {"0": root[0]}, ["0"], scope=SCOPE, calibration=base, precisions={"0": "bf16"}
    )
    assert DynamicEngine.apply(root, profile, scope=SCOPE)["0"] is root[0]
    for field, value, match in [
        ("implementation_sha256", "0" * 64, "implementation mismatch"),
        ("approved", 1, "explicit boolean"),
    ]:
        calibration = {**base, field: value}
        profile = linear.build_profile(
            {"0": root[0]}, ["0"], scope=SCOPE, calibration=calibration, precisions={"0": "bf16"}
        )
        with pytest.raises(ValueError, match=match):
            DynamicEngine.apply(root, profile, scope=SCOPE, require_approved=False)


def test_apply_stale_source_and_nonselected_alias_leave_tree_unchanged(environment):
    root = torch.nn.Sequential(source(), source())
    original = root[0]
    calibration = {
        "engine": "DynamicEngine",
        "approved": True,
        "implementation_sha256": engine._implementation_sha256(),
    }
    profile = linear.build_profile(
        {"0": original}, ["0"], scope=SCOPE, calibration=calibration, precisions={"0": "bf16"}
    )
    original.weight[0, 0] += 1
    with pytest.raises(ValueError, match="weight, shape, dtype or group"):
        DynamicEngine.apply(root, profile, scope=SCOPE)
    assert root[0] is original
    profile = linear.build_profile(
        {"0": original}, ["0"], scope=SCOPE, calibration=calibration, precisions={"0": "bf16"}
    )
    root.register_buffer("hidden_alias", original.weight)
    with pytest.raises(ValueError, match="shared/tied"):
        DynamicEngine.apply(root, profile, scope=SCOPE)
    assert root[0] is original
