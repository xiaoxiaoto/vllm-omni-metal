# SPDX-License-Identifier: Apache-2.0
"""Content/transaction CPU regressions and opt-in real MLX/MPS mixed-precision math."""

import copy
import gc
import hashlib
import json
import os
import weakref

import pytest
import torch

from vllm_omni_metal.quantization.linear import (
    QuantizedLinear,
    apply_profile,
    build_profile,
    validate_profile,
)

MPS = pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="explicit MPS integration opt-in required",
)
SCOPE = "test-pipeline-revision/seed42/complete-configuration"


def test_bridge_path_tracks_vllm_metal_package_layout(monkeypatch, tmp_path):
    from vllm_omni_metal.quantization import linear

    package = tmp_path / "vllm_metal"
    package.mkdir()
    bridge = package / "tensor_bridge.py"
    bridge.write_text("# 0.32 package layout\n")

    class Distribution:
        def locate_file(self, relative):
            return tmp_path / relative

    monkeypatch.setattr(linear.importlib.metadata, "distribution", lambda name: Distribution())

    assert linear._bridge_path() == bridge


def linear(*, dtype=torch.float16, device="cpu", bias=True, width=128, outputs=32):
    # CPU-generated source data keeps the reference independent of MPS RNG.
    generator = torch.Generator().manual_seed(872)
    result = torch.nn.Linear(width, outputs, bias=bias, dtype=dtype)
    result.requires_grad_(False)
    result.weight.copy_(torch.randn(outputs, width, generator=generator).to(dtype) * 0.05)
    if bias:
        result.bias.copy_(torch.linspace(-0.1, 0.1, outputs, dtype=dtype))
    return result.to(device)


def profile(modules, selected=None, calibration=None, *, precisions=None):
    return build_profile(
        modules,
        list(modules) if selected is None else selected,
        scope=SCOPE,
        calibration={"seed": 42, "audio_bitwise_equal": False}
        if calibration is None
        else calibration,
        precisions=precisions,
    )


def resign(value):
    payload = {key: item for key, item in value.items() if key != "sha256"}
    value["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return value


def test_cpu_profile_is_plain_json_deterministic_and_copies_metadata():
    modules = {"decoder.0.proj": linear(), "untouched": linear()}
    calibration = {"seed": 42, "samples": [1.5, None, True], "audio_bitwise_equal": False}
    first = profile(modules, ["decoder.0.proj"], calibration)
    assert first == profile(modules, ["decoder.0.proj"], calibration)
    calibration["samples"].append("not-in-profile")
    restored = json.loads(json.dumps(first, allow_nan=False))
    assert validate_profile(restored, modules, scope=SCOPE) == ["decoder.0.proj"]
    assert list(restored["modules"]) == ["decoder.0.proj"]
    assert restored["calibration"]["samples"] == [1.5, None, True]
    assert restored["schema"] == 2
    assert restored["modules"]["decoder.0.proj"]["bits"] == 8
    modules["untouched"].weight.add_(1)
    # Only selected weights are bound; a complete pipeline identity belongs in scope.
    assert validate_profile(restored, modules, scope=SCOPE) == ["decoder.0.proj"]


@pytest.mark.parametrize("precision", ["w4a16", "w8a16", "bf16"])
@pytest.mark.parametrize("field", ["weight", "bias"])
def test_cpu_profile_rejects_changed_selected_tensor_bits(precision, field):
    module = linear(dtype=torch.bfloat16)
    modules = {"proj": module}
    saved = profile(modules, precisions={"proj": precision})
    tensor = getattr(module, field)
    tensor.view(-1)[0] += 1
    with pytest.raises(ValueError, match="weight, shape, dtype or group"):
        validate_profile(saved, modules, scope=SCOPE)


def test_cpu_profile_hash_preserves_signed_zero():
    module = linear()
    module.weight.zero_()
    saved = profile({"proj": module})
    module.weight[0, 0] = -0.0
    with pytest.raises(ValueError, match="weight, shape, dtype or group"):
        validate_profile(saved, {"proj": module}, scope=SCOPE)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(schema=True),
        lambda p: p.update(schema=999),
        lambda p: p.update(scope="other-pipeline"),
        lambda p: p["environment"].update(implementation_sha256="0" * 64),
        lambda p: p["environment"].update(bridge_sha256="0" * 64),
        lambda p: p["environment"].update(torch_path="/other/python/torch"),
        lambda p: p["modules"]["proj"].update(shape=[32, 64]),
        lambda p: p["modules"]["proj"].update(dtype="torch.float32"),
        lambda p: p["modules"]["proj"].update(group_size=32),
        lambda p: p["modules"]["proj"].update(bits=3),
        lambda p: p["modules"]["proj"].update(mode="mxfp8"),
        lambda p: p.update(selected=["missing"]),
        lambda p: p.update(selected=["proj", "proj"]),
        lambda p: p.update(extra="unexpected"),
    ],
)
def test_cpu_profile_strict_validation_even_with_recomputed_digest(change):
    modules = {"proj": linear()}
    saved = profile(modules)
    change(saved)
    resign(saved)
    with pytest.raises(ValueError):
        validate_profile(saved, modules, scope=SCOPE)


def test_cpu_default_precision_remains_explicit_w8a16():
    modules = {"first": linear(), "second": linear(dtype=torch.bfloat16)}
    assert profile(modules) == profile(modules, precisions={"first": "w8a16", "second": "w8a16"})


def test_cpu_mixed_profile_roundtrips_exact_precision_descriptors():
    modules = {
        "four": linear(),
        "eight": linear(dtype=torch.bfloat16),
        "keep": linear(dtype=torch.bfloat16, width=63, outputs=17),
        "unselected": linear(),
    }
    selected = ["keep", "four", "eight"]
    saved = profile(
        modules,
        selected,
        precisions={"eight": "w8a16", "keep": "bf16", "four": "w4a16"},
    )
    restored = json.loads(json.dumps(saved, allow_nan=False))
    assert restored["schema"] == 2
    assert validate_profile(restored, modules, scope=SCOPE) == selected
    assert list(restored["modules"]) == selected
    for name, bits, mode, group in (
        ("four", 4, "affine", 64),
        ("eight", 8, "affine", 64),
        ("keep", 16, "identity", None),
    ):
        descriptor = restored["modules"][name]
        assert (descriptor["bits"], descriptor["mode"], descriptor["group_size"]) == (
            bits,
            mode,
            group,
        )


@pytest.mark.parametrize(
    "precisions",
    [
        {},
        {"first": "w4a16"},
        {"first": "w4a16", "second": "w8a16", "unselected": "bf16"},
        {"first": "w4a16", "missing": "w8a16"},
        [("first", "w4a16"), ("second", "w8a16")],
        "w4a16",
    ],
)
def test_cpu_precision_mapping_requires_exact_selected_keys(precisions):
    modules = {"first": linear(), "second": linear(), "unselected": linear()}
    with pytest.raises(ValueError):
        profile(modules, ["first", "second"], precisions=precisions)


@pytest.mark.parametrize("precision", ["W4A16", "w4", "float16", "", 4, 8, 16, True, None, []])
def test_cpu_precision_mapping_rejects_invalid_values(precision):
    with pytest.raises(ValueError):
        profile({"proj": linear()}, precisions={"proj": precision})


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cpu_bf16_selection_rejects_casting_source(dtype):
    source = linear(dtype=dtype, width=63)
    weight, bias = source.weight, source.bias
    before = copy.deepcopy(source.state_dict())
    with pytest.raises(ValueError):
        profile({"proj": source}, precisions={"proj": "bf16"})
    assert source.weight is weight and source.bias is bias
    assert source.weight.dtype == dtype and source.bias.dtype == dtype
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)


@pytest.mark.parametrize("bias", [False, True])
def test_cpu_bf16_apply_preserves_module_parameters_and_non_group64_dimensions(bias):
    root = torch.nn.Sequential(linear(dtype=torch.bfloat16, width=63, outputs=17, bias=bias))
    source = root[0]
    weight, original_bias = source.weight, source.bias
    weight_before = weight.clone()
    bias_before = None if original_bias is None else original_bias.clone()
    x = torch.ones(2, 63, dtype=torch.bfloat16)
    before = root(x)
    saved = profile({"0": source}, precisions={"0": "bf16"})
    applied = apply_profile(root, saved, scope=SCOPE)
    assert list(applied) == ["0"]
    assert applied["0"] is source and root[0] is source
    assert source.weight is weight and source.bias is original_bias
    assert source.weight.dtype == torch.bfloat16
    torch.testing.assert_close(source.weight, weight_before, atol=0, rtol=0)
    if original_bias is not None:
        torch.testing.assert_close(source.bias, bias_before, atol=0, rtol=0)
    torch.testing.assert_close(root(x), before, atol=0, rtol=0)
    assert validate_profile(saved, {"0": source}, scope=SCOPE) == ["0"]


@pytest.mark.parametrize(
    "dtype,precision,bits",
    [(torch.float16, "fp16", 16), (torch.bfloat16, "bf16", 16), (torch.float32, "fp32", 32)],
)
def test_cpu_source_identity_profiles_preserve_dtype_and_objects(dtype, precision, bits):
    root = torch.nn.Sequential(linear(dtype=dtype, width=63, outputs=17))
    original = root[0]
    weight, bias = original.weight, original.bias
    saved = profile({"0": original}, precisions={"0": precision})
    descriptor = saved["modules"]["0"]
    assert (
        descriptor["bits"],
        descriptor["dtype"],
        descriptor["mode"],
        descriptor["group_size"],
    ) == (bits, str(dtype), "identity", None)
    assert validate_profile(saved, {"0": original}, scope=SCOPE) == ["0"]
    assert apply_profile(root, saved, scope=SCOPE)["0"] is original
    assert original.weight is weight and original.bias is bias
    assert original.weight.dtype == dtype


@pytest.mark.parametrize("precision", ["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cpu_identity_precision_never_casts(dtype, precision):
    expected = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype == expected[precision]:
        return
    module = linear(dtype=dtype)
    with pytest.raises(ValueError):
        profile({"proj": module}, precisions={"proj": precision})
    assert module.weight.dtype == dtype


@pytest.mark.parametrize(
    "precision,dtype,change",
    [
        ("fp16", torch.float16, {"dtype": "torch.bfloat16"}),
        ("fp32", torch.float32, {"dtype": "torch.float16"}),
        ("fp32", torch.float32, {"bits": 16}),
        ("fp32", torch.float32, {"mode": "affine"}),
        ("fp32", torch.float32, {"group_size": 64}),
    ],
)
def test_cpu_identity_descriptor_tampering_is_rejected(precision, dtype, change):
    module = linear(dtype=dtype)
    saved = profile({"proj": module}, precisions={"proj": precision})
    saved["modules"]["proj"].update(change)
    resign(saved)
    with pytest.raises(ValueError):
        validate_profile(saved, {"proj": module}, scope=SCOPE)


def test_cpu_old_schema_is_rejected_even_with_fresh_outer_digest():
    modules = {"proj": linear()}
    saved = profile(modules)
    saved["schema"] = 1
    resign(saved)
    with pytest.raises(ValueError, match="schema"):
        validate_profile(saved, modules, scope=SCOPE)


@pytest.mark.parametrize(
    ("precision", "bits"),
    [
        ("w4a16", 8),
        ("w8a16", 4),
        ("bf16", 8),
        ("w8a16", 16),
        ("w4a16", True),
        ("w8a16", 8.0),
        ("bf16", 16.0),
    ],
)
def test_cpu_precision_metadata_tampering_rejected_with_fresh_outer_digest(precision, bits):
    modules = {"proj": linear(dtype=torch.bfloat16)}
    saved = profile(modules, precisions={"proj": precision})
    saved["modules"]["proj"]["bits"] = bits
    resign(saved)
    with pytest.raises(ValueError):
        validate_profile(saved, modules, scope=SCOPE)


@pytest.mark.parametrize(
    ("precision", "field", "value"),
    [
        ("bf16", "mode", "affine"),
        ("bf16", "group_size", 64),
        ("w4a16", "mode", "identity"),
        ("w8a16", "group_size", None),
    ],
)
def test_cpu_precision_mode_and_group_tampering_rejected(precision, field, value):
    modules = {"proj": linear(dtype=torch.bfloat16)}
    saved = profile(modules, precisions={"proj": precision})
    saved["modules"]["proj"][field] = value
    resign(saved)
    with pytest.raises(ValueError):
        validate_profile(saved, modules, scope=SCOPE)


def test_cpu_calibration_tampering_requires_new_content_hash():
    modules = {"proj": linear()}
    saved = profile(modules)
    saved["calibration"]["audio_bitwise_equal"] = True
    with pytest.raises(ValueError, match="content hash"):
        validate_profile(saved, modules, scope=SCOPE)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), torch.tensor(1), (1, 2), {1: 2}])
def test_cpu_profile_rejects_non_plain_calibration(bad):
    with pytest.raises(ValueError, match="JSON"):
        profile({"proj": linear()}, calibration={"bad": bad})


@pytest.mark.parametrize("selected", [["missing"], ["proj", "proj"], [""], ["proj..child"], [1]])
def test_cpu_selection_rejects_invalid_names(selected):
    with pytest.raises(ValueError):
        profile({"proj": linear()}, selected)


def test_cpu_scope_mismatch_and_empty_scope_are_rejected():
    modules = {"proj": linear()}
    saved = profile(modules)
    with pytest.raises(ValueError, match="scope"):
        validate_profile(saved, modules, scope="different-request-config")
    with pytest.raises(ValueError, match="scope"):
        build_profile(modules, ["proj"], scope="", calibration={})


@pytest.mark.parametrize("precision", ["w4a16", "w8a16", "bf16"])
def test_cpu_grad_parameters_require_explicit_inference_mode(precision):
    module = linear(dtype=torch.bfloat16).requires_grad_(True)
    with pytest.raises(ValueError, match="without gradients"):
        profile({"proj": module}, precisions={"proj": precision})
    with torch.inference_mode():
        saved = profile({"proj": module}, precisions={"proj": precision})
        assert validate_profile(saved, {"proj": module}, scope=SCOPE) == ["proj"]
    assert module.weight.requires_grad and module.bias.requires_grad


@pytest.mark.parametrize("precision", ["w4a16", "w8a16", "bf16"])
@pytest.mark.parametrize("which", ["weight", "bias"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_cpu_rejects_nonfinite_source_without_mutation(precision, which, value):
    module = linear(dtype=torch.bfloat16)
    getattr(module, which).view(-1)[0] = value
    before = copy.deepcopy(module.state_dict())
    with pytest.raises(ValueError, match="finite"):
        profile({"proj": module}, precisions={"proj": precision})
    for name, tensor in module.state_dict().items():
        torch.testing.assert_close(tensor, before[name], equal_nan=True, atol=0, rtol=0)


@pytest.mark.parametrize("precision", ["w4a16", "w8a16", "bf16"])
@pytest.mark.parametrize("which", ["weight", "bias"])
def test_cpu_rejects_noncontiguous_source(precision, which):
    module = linear(dtype=torch.bfloat16)
    source = getattr(module, which)
    backing = torch.empty((*source.shape, 2), dtype=source.dtype)
    view = backing[..., 0]
    view.copy_(source)
    setattr(module, which, torch.nn.Parameter(view, requires_grad=False))
    with pytest.raises(ValueError, match="contiguous"):
        profile({"proj": module}, precisions={"proj": precision})


def test_cpu_rejects_wrong_dimension_dtype_bias_and_subclass():
    with pytest.raises(ValueError, match="dimensions"):
        profile({"proj": linear(width=63)})
    with pytest.raises(ValueError, match="FP16 or BF16"):
        profile({"proj": linear(dtype=torch.float32)})
    bad_bias = linear()
    bad_bias.bias = torch.nn.Parameter(torch.zeros(31, dtype=torch.float16), requires_grad=False)
    with pytest.raises(ValueError, match="bias"):
        profile({"proj": bad_bias})

    class SpecializedLinear(torch.nn.Linear):
        pass

    specialized = SpecializedLinear(128, 32, dtype=torch.float16).requires_grad_(False)
    with pytest.raises(ValueError, match="ordinary"):
        profile({"proj": specialized})


@pytest.mark.parametrize("precision", ["w4a16", "w8a16", "bf16"])
def test_cpu_rejects_hooks_instead_of_silently_discarding_them(precision):
    module = linear(dtype=torch.bfloat16)
    handle = module.register_forward_hook(lambda _module, _args, output: output + 1)
    with pytest.raises(ValueError, match="hooks"):
        profile({"proj": module}, precisions={"proj": precision})
    handle.remove()
    saved = profile({"proj": module}, precisions={"proj": precision})
    assert validate_profile(saved, {"proj": module}, scope=SCOPE) == ["proj"]


@pytest.mark.parametrize("precision", ["w4a16", "w8a16", "bf16"])
@pytest.mark.parametrize("sharing", ["module", "weight", "bias", "view"])
def test_cpu_rejects_selected_shared_or_tied_storage(precision, sharing):
    first, second = linear(dtype=torch.bfloat16), linear(dtype=torch.bfloat16)
    if sharing == "module":
        second = first
    elif sharing == "view":
        second.weight = torch.nn.Parameter(first.weight.view_as(first.weight), requires_grad=False)
    else:
        setattr(second, sharing, getattr(first, sharing))
    with pytest.raises(ValueError, match="shared/tied"):
        profile({"first": first, "second": second}, ["first"], precisions={"first": precision})


def test_cpu_apply_rejects_external_embedding_tie_before_conversion():
    root = torch.nn.Module()
    root.proj = linear()
    saved = profile({"proj": root.proj})
    root.embedding = torch.nn.Embedding(32, 128, dtype=torch.float16)
    root.embedding.weight = root.proj.weight
    original = root.proj
    with pytest.raises(ValueError, match="shared/tied"):
        apply_profile(root, saved, scope=SCOPE)
    assert root.proj is original and root.embedding.weight is original.weight


def test_cpu_apply_rejects_root_buffer_alias_before_conversion():
    root = torch.nn.Module()
    root.proj = linear()
    saved = profile({"proj": root.proj})
    root.register_buffer("shared", root.proj.weight.detach())
    original = root.proj
    with pytest.raises(ValueError, match="shared/tied"):
        apply_profile(root, saved, scope=SCOPE)
    assert root.proj is original


def test_cpu_apply_failure_leaves_entire_tree_callable_and_unchanged():
    root = torch.nn.Sequential(
        linear(dtype=torch.bfloat16, outputs=64),
        linear(dtype=torch.bfloat16, width=64, outputs=64),
        linear(dtype=torch.bfloat16, width=64),
    )
    modules = dict(root.named_children())
    saved = profile(modules, precisions={"0": "bf16", "1": "w4a16", "2": "w8a16"})
    parameters = list(root.parameters())
    x = torch.ones(2, 128, dtype=torch.bfloat16)
    before = root(x)
    with pytest.raises(ValueError, match="MPS weights"):
        apply_profile(root, saved, scope=SCOPE)
    assert list(root.children()) == list(modules.values())
    assert all(
        current is original for current, original in zip(root.parameters(), parameters, strict=True)
    )
    torch.testing.assert_close(root(x), before, atol=0, rtol=0)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group", [32, 128, 64.0, True, None])
def test_cpu_conversion_rejects_non_group64_without_touching_source(bits, group):
    module = linear()
    original = module.weight.clone()
    with pytest.raises(ValueError, match="group_size=64"):
        QuantizedLinear.from_linear(module, bits=bits, group_size=group)
    torch.testing.assert_close(module.weight, original, atol=0, rtol=0)


@pytest.mark.parametrize("bits", [0, 3, 16, True, False, 4.0, 8.0, "4", None])
def test_cpu_conversion_requires_strict_integer_4_or_8_bits(bits):
    source = linear()
    weight, bias = source.weight, source.bias
    before = copy.deepcopy(source.state_dict())
    with pytest.raises(ValueError, match="bits"):
        QuantizedLinear.from_linear(source, bits=bits)
    assert source.weight is weight and source.bias is bias
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)


def independent_weight(module):
    """Decode documented uint32 little-bit affine packing, without MLX dequantize."""
    packed = torch.tensor(module._packed.tolist(), dtype=torch.int64)
    shifts = torch.arange(32 // module.bits, dtype=torch.int64) * module.bits
    integers = ((packed[..., None] >> shifts) & ((1 << module.bits) - 1)).reshape(
        module.out_features, -1
    )
    scales = torch.tensor(module._scales.tolist(), dtype=torch.float64).repeat_interleave(64, -1)
    offsets = torch.tensor(module._offsets.tolist(), dtype=torch.float64).repeat_interleave(64, -1)
    return integers.double() * scales + offsets


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("shape", [(128,), (1, 128), (2, 3, 128), (0, 128)])
def test_real_quantized_matches_independent_affine_reference(bits, dtype, bias, shape):
    source = linear(dtype=dtype, device="mps", bias=bias)
    original_weight = source.weight.cpu().clone()
    original_bias = None if source.bias is None else source.bias.cpu().clone()
    quantized = QuantizedLinear.from_linear(source, bits=bits)
    generator = torch.Generator().manual_seed(53)
    host = torch.randn(shape, generator=generator).to(dtype)
    x = host.to("mps")
    result = quantized(x)
    reconstructed = independent_weight(quantized)
    assert not torch.equal(reconstructed, original_weight.double())
    expected = (host.double() @ reconstructed.T).to(dtype)
    if original_bias is not None:
        expected = expected + original_bias
    tolerance = 0.003 if dtype == torch.float16 else 0.025
    torch.testing.assert_close(result.cpu(), expected, atol=tolerance, rtol=tolerance)
    original = (host.double() @ original_weight.double().T).to(dtype)
    if original_bias is not None:
        original = original + original_bias
    if result.numel():
        relative = torch.linalg.vector_norm(result.cpu().double() - original.double()) / max(
            torch.linalg.vector_norm(original.double()).item(), 1e-12
        )
        # W4 has a deliberately larger loss budget; neither precision is lossless.
        assert 0 < relative < (0.20 if bits == 4 else 0.035)
    assert result.shape == (*shape[:-1], 32)
    assert result.dtype == dtype and result.device.type == "mps"
    torch.testing.assert_close(x.cpu(), host, atol=0, rtol=0)
    assert quantized.metrics == {"calls": 1, "rows": host.numel() // 128}
    assert quantized.bits == bits
    assert quantized._packed.shape == (32, 128 * bits // 32)
    arrays = [quantized._packed, quantized._scales, quantized._offsets]
    if bias:
        arrays.append(quantized._bias)
    assert quantized.storage_bytes == sum(array.nbytes for array in arrays)
    assert quantized.source_bytes == (32 * 128 + (32 if bias else 0)) * 2
    assert quantized.storage_bytes < quantized.source_bytes
    assert not list(quantized.parameters()) and not list(quantized.buffers())


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("bits", [4, 8])
def test_real_conversion_owns_bias_and_releases_float_weight_lifetime(bits):
    source = linear(device="mps")
    weight_ref, bias_ref, source_ref = (
        weakref.ref(source.weight),
        weakref.ref(source.bias),
        weakref.ref(source),
    )
    quantized = QuantizedLinear.from_linear(source, bits=bits)
    x = torch.ones(1, 128, dtype=torch.float16, device="mps")
    first = quantized(x).cpu().clone()
    source.weight.zero_()
    source.bias.fill_(100)
    torch.testing.assert_close(quantized(x).cpu(), first, atol=0, rtol=0)
    del source
    gc.collect()
    assert source_ref() is None and weight_ref() is None and bias_ref() is None
    # Torch producer -> MLX -> Torch consumer; outputs outlive the quantized module.
    x.add_(0.5)
    second = quantized(x)
    retained = second.cpu().clone()
    del quantized
    gc.collect()
    torch.testing.assert_close((second + 1).cpu(), retained + 1, atol=0, rtol=0)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize(
    "invalid", ["dtype", "device", "shape", "strides", "grad", "nan", "inf", "overflow"]
)
def test_real_forward_rejects_illegal_activations_without_counting(bits, invalid):
    quantized = QuantizedLinear.from_linear(linear(device="mps"), bits=bits)
    x = torch.ones(2, 128, dtype=torch.float16, device="mps")
    if invalid == "dtype":
        x = x.float()
    elif invalid == "device":
        x = x.cpu()
    elif invalid == "shape":
        x = x[:, :64].contiguous()
    elif invalid == "strides":
        x = torch.ones(2, 256, dtype=x.dtype, device=x.device)[:, ::2]
    elif invalid == "grad":
        x.requires_grad_(True)
    elif invalid in ("nan", "inf"):
        x[0, 0] = float(invalid)
    else:
        source = linear(device="mps")
        source.weight.fill_(1)
        quantized = QuantizedLinear.from_linear(source, bits=bits)
        x.fill_(torch.finfo(x.dtype).max)
    with pytest.raises(ValueError):
        quantized(x)
    assert quantized.metrics == {"calls": 0, "rows": 0}


@pytest.mark.integration
@MPS
def test_real_apply_construct_failure_after_first_candidate_is_transactional():
    root = torch.nn.Module()
    root.first = linear(dtype=torch.bfloat16, device="mps")
    root.keep = linear(dtype=torch.bfloat16, device="mps", width=63)
    root.second = linear(dtype=torch.bfloat16)  # Fingerprintable, not executable.
    modules = {"first": root.first, "keep": root.keep, "second": root.second}
    saved = profile(modules, precisions={"first": "w4a16", "keep": "bf16", "second": "w8a16"})
    parameters = list(root.parameters())
    x = torch.ones(1, 128, dtype=torch.bfloat16, device="mps")
    before = root.first(x).cpu()
    with pytest.raises(ValueError, match="MPS weights"):
        apply_profile(root, saved, scope=SCOPE)
    assert all(getattr(root, name) is module for name, module in modules.items())
    assert all(
        current is original for current, original in zip(root.parameters(), parameters, strict=True)
    )
    torch.testing.assert_close(root.first(x).cpu(), before, atol=0, rtol=0)


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("first_bits", [4, 8])
def test_real_mixed_pipeline_generates_tensors_preserves_bf16_and_releases_sources(first_bits):
    root = torch.nn.Sequential(
        linear(dtype=torch.bfloat16, device="mps", outputs=64),
        linear(dtype=torch.bfloat16, device="mps", width=64, outputs=63),
        linear(dtype=torch.bfloat16, device="mps", width=63, outputs=17),
    )
    preserved = root[2]
    preserved_weight, preserved_bias = preserved.weight, preserved.bias
    weight_before, bias_before = preserved_weight.cpu().clone(), preserved_bias.cpu().clone()
    source_refs = [
        (weakref.ref(root[index]), weakref.ref(root[index].weight), weakref.ref(root[index].bias))
        for index in (0, 1)
    ]
    biases = [root[index].bias.cpu().clone() for index in (0, 1)]
    modules = dict(root.named_children())
    saved = profile(
        modules,
        precisions={"0": f"w{first_bits}a16", "1": f"w{12 - first_bits}a16", "2": "bf16"},
    )
    generator = torch.Generator().manual_seed(953)
    host = torch.randn(2, 3, 128, generator=generator).to(torch.bfloat16)
    x = host.to("mps")
    original = root(x).cpu().clone()
    applied = apply_profile(root, saved, scope=SCOPE)
    assert list(applied) == ["0", "1", "2"]
    assert root[0] is applied["0"] and root[1] is applied["1"]
    assert isinstance(root[0], QuantizedLinear) and isinstance(root[1], QuantizedLinear)
    assert (root[0].bits, root[1].bits) == (first_bits, 12 - first_bits)
    assert root[2] is applied["2"] is preserved
    assert preserved.weight is preserved_weight and preserved.bias is preserved_bias
    del modules
    gc.collect()
    assert all(reference() is None for references in source_refs for reference in references)
    torch.testing.assert_close(preserved.weight.cpu(), weight_before, atol=0, rtol=0)
    torch.testing.assert_close(preserved.bias.cpu(), bias_before, atol=0, rtol=0)
    expected = host
    for index in (0, 1):
        expected = (expected.double() @ independent_weight(root[index]).T).to(torch.bfloat16)
        expected = expected + biases[index]
    expected = torch.nn.functional.linear(expected, weight_before, bias_before)
    result = root(x)
    assert result.shape == (2, 3, 17) and result.dtype == torch.bfloat16
    assert result.device.type == "mps" and torch.isfinite(result).all().item()
    torch.testing.assert_close(result.cpu(), expected, atol=0.025, rtol=0.025)
    relative = torch.linalg.vector_norm(result.cpu().double() - original.double()) / max(
        torch.linalg.vector_norm(original.double()).item(), 1e-12
    )
    assert relative < 0.25  # The complete W4/W8/BF16 pipeline is intentionally lossy.
    torch.testing.assert_close(root(x).cpu(), result.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(x.cpu(), host, atol=0, rtol=0)
    assert root[0].metrics == root[1].metrics == {"calls": 2, "rows": 12}


@pytest.mark.integration
@MPS
def test_real_apply_only_explicit_modules_and_rejects_second_application():
    root = torch.nn.Module()
    root.selected = torch.nn.Sequential(linear(device="mps"))
    root.untouched = linear(device="mps")
    untouched = root.untouched
    source = root.selected[0]
    weight_ref = weakref.ref(source.weight)
    saved = profile({"selected.0": source, "untouched": untouched}, ["selected.0"])
    replacements = apply_profile(root, saved, scope=SCOPE)
    assert set(replacements) == {"selected.0"}
    assert root.selected[0] is replacements["selected.0"] and root.untouched is untouched
    del source
    gc.collect()
    assert weight_ref() is None
    with pytest.raises(ValueError, match="ordinary"):
        apply_profile(root, saved, scope=SCOPE)
    result = root.selected(torch.ones(1, 128, dtype=torch.float16, device="mps"))
    assert torch.isfinite(result).all().item()
    assert replacements["selected.0"].metrics == {"calls": 1, "rows": 1}


@pytest.mark.integration
@MPS
@pytest.mark.parametrize("bits", [4, 8])
def test_real_inference_mode_detaches_trainable_parameters_and_odd_output_width(bits):
    source = linear(device="mps", dtype=torch.bfloat16, bias=False, outputs=17)
    source.requires_grad_(True)
    with torch.inference_mode():
        quantized = QuantizedLinear.from_linear(source, bits=bits)
        x = torch.ones(2, 128, dtype=torch.bfloat16, device="mps")
        output = quantized(x)
    expected = (x.cpu().double() @ independent_weight(quantized).T).to(torch.bfloat16)
    torch.testing.assert_close(output.cpu(), expected, atol=0.025, rtol=0.025)
    assert output.dtype == torch.bfloat16 and output.shape == (2, 17)
    assert source.weight.requires_grad
    with pytest.raises(ValueError, match="moved/cast"):
        quantized.to("cpu")
