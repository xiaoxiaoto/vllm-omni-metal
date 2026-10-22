# SPDX-License-Identifier: Apache-2.0
"""Small real-projection contracts, not whole-model numerical approval."""

import copy
import gc
import os
import weakref

import pytest
import torch

from vllm_omni_metal.quantization import linear
from vllm_omni_metal.quantization.adapter import DenseModelAdapter, _semantics
from vllm_omni_metal.quantization.engine import _implementation_sha256


def root_with_linears(*, count=2, dtype=torch.float16, device="cpu"):
    root = torch.nn.Module()
    root.layers = torch.nn.ModuleList(
        [
            torch.nn.Linear(64, 64, dtype=dtype, device=device).requires_grad_(False)
            for _ in range(count)
        ]
    )
    return root


def profile_for(adapter, precisions):
    """Unit-only calibrated envelope; never a production approval or quality claim."""
    selected = list(precisions)
    root = adapter._live_root()
    return linear.build_profile(
        adapter.views(selected),
        selected,
        scope="test",
        precisions=precisions,
        calibration={
            "engine": "DynamicEngine",
            "implementation_sha256": _implementation_sha256(),
            "approved": False,
            "projections": {name: _semantics(root.get_submodule(name)) for name in selected},
        },
    )


@pytest.mark.parametrize(
    "dtype,precision",
    [
        (torch.float16, "fp16"),
        (torch.bfloat16, "bf16"),
        (torch.float32, "fp32"),
    ],
)
def test_identity_preserves_original_and_unselected(dtype, precision):
    root = root_with_linears(dtype=dtype)
    adapter = DenseModelAdapter(root)
    originals = list(root.layers)
    profile = profile_for(adapter, {"layers.0": precision})
    result = adapter.apply(profile, scope="test", require_approved=False)
    assert result == {"layers.0": originals[0]}
    assert list(root.layers) == originals


def test_ordinary_path_does_not_import_vllm(monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith("vllm.")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    root = root_with_linears()
    adapter = DenseModelAdapter(root)
    assert set(adapter.candidates()) == {"layers.0", "layers.1"}
    profile = profile_for(adapter, {"layers.0": "fp16"})
    adapter.apply(profile, scope="test", require_approved=False)


@pytest.mark.parametrize("alias", ["module", "parameter", "buffer", "view"])
def test_whole_root_alias_rejected_without_mutation(alias):
    root = root_with_linears()
    adapter = DenseModelAdapter(root)
    profile = profile_for(adapter, {"layers.0": "fp16"})
    source = root.layers[0]
    if alias == "module":
        root.alias = source
    elif alias == "parameter":
        root.register_parameter("alias", source.weight)
    elif alias == "buffer":
        root.register_buffer("alias", source.weight.detach())
    else:
        root.register_buffer("alias", source.weight.detach().view(-1)[1:])
    with pytest.raises(ValueError, match="shared|tied"):
        adapter.apply(profile, scope="test", require_approved=False)
    assert root.layers[0] is source


def test_hooks_not_discovered_or_discarded():
    root = root_with_linears()
    handle = root.layers[0].register_forward_hook(lambda *args: None)
    adapter = DenseModelAdapter(root)
    assert set(adapter.candidates()) == {"layers.1"}
    with pytest.raises(ValueError, match="hooks"):
        adapter.views(["layers.0"])
    handle.remove()


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda p: p["calibration"].update(engine="other"), "DynamicEngine"),
        (lambda p: p["calibration"].update(implementation_sha256="stale"), "implementation"),
        (lambda p: p["calibration"].update(approved=1), "boolean"),
    ],
)
def test_engine_gate_is_required_even_offline(change, match):
    root = root_with_linears()
    adapter = DenseModelAdapter(root)
    profile = profile_for(adapter, {"layers.0": "fp16"})
    change(profile)
    with pytest.raises(ValueError, match=match):
        adapter.apply(profile, scope="test", require_approved=False)


def test_unapproved_default_and_source_mismatch():
    root = root_with_linears()
    adapter = DenseModelAdapter(root)
    profile = profile_for(adapter, {"layers.0": "fp16"})
    with pytest.raises(ValueError, match="approved"):
        adapter.apply(profile, scope="test")
    with torch.inference_mode():
        root.layers[0].weight.add_(1)
    with pytest.raises(ValueError, match="weight"):
        adapter.apply(profile, scope="test", require_approved=False)


def test_constructor_failure_is_transactional(monkeypatch):
    root = root_with_linears()
    adapter = DenseModelAdapter(root)
    profile = profile_for(adapter, {"layers.0": "w4a16", "layers.1": "w8a16"})
    originals = list(root.layers)
    bits_seen = []

    def construct(view, *, bits):
        bits_seen.append(bits)
        if bits == 8:
            raise RuntimeError("second conversion failed")
        return torch.nn.Identity()

    monkeypatch.setattr(linear.QuantizedLinear, "from_linear", construct)
    with pytest.raises(RuntimeError, match="second"):
        adapter.apply(profile, scope="test", require_approved=False)
    assert bits_seen == [4, 8]
    assert list(root.layers) == originals


def test_capture_search_does_not_keep_sources(monkeypatch):
    root = root_with_linears(count=1)
    adapter = DenseModelAdapter(root)
    with adapter.capture(["layers.0"]) as session:
        for _ in range(4):
            root.layers[0](torch.ones(2, 64, dtype=torch.float16))
    assert len(session.captures["layers.0"]) == 4
    # CPU test exercises adapter dispatch and profile augmentation, not PTQ math.
    from vllm_omni_metal.quantization.engine import DynamicEngine

    def search(views, captures, *, scope):
        assert views["layers.0"] is root.layers[0]
        return {
            "profile": profile_for(adapter, {"layers.0": "fp16"}),
            "precisions": {"layers.0": "fp16"},
            "audits": {},
            "local_pass": True,
        }

    monkeypatch.setattr(DynamicEngine, "search", search)
    result = adapter.search(session.captures, scope="test")
    assert result["profile"]["calibration"]["projections"] == {"layers.0": {"kind": "torch"}}
    source_ref, weight_ref = weakref.ref(root.layers[0]), weakref.ref(root.layers[0].weight)
    root.layers[0] = torch.nn.Identity()
    gc.collect()
    assert source_ref() is None and weight_ref() is None


@pytest.fixture(scope="module")
def single_rank_parallel(tmp_path_factory):
    pytest.importorskip("vllm")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    assert not torch.distributed.is_initialized()
    directory = tmp_path_factory.mktemp("quantization-gloo")
    with set_current_vllm_config(VllmConfig()):
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{directory / 'init'}",
                backend="gloo",
            )
            initialize_model_parallel(backend="gloo")
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def projection(kind, *, return_bias=True, skip_bias_add=False, device="cpu", dtype=torch.float16):
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
    )

    options = dict(
        bias=True,
        return_bias=return_bias,
        skip_bias_add=skip_bias_add,
        disable_tp=True,
        params_dtype=dtype,
    )
    if kind == "row":
        module = RowParallelLinear(64, 64, **options)
    elif kind == "column":
        module = ColumnParallelLinear(64, 64, **options)
    elif kind == "replicated":
        module = ReplicatedLinear(64, 64, **options)
    elif kind == "merged":
        module = MergedColumnParallelLinear(64, [64, 64], **options)
    else:
        module = QKVParallelLinear(64, 16, 4, **options)
    module = module.to(device).requires_grad_(False)
    with torch.inference_mode():
        # Constant groups retain distinct rows without BF16 affine rounding of
        # zero/one mixtures; forced W4 here checks execution, not sensitivity.
        rows = module.weight.shape[0]
        values = torch.arange(1, rows + 1, dtype=torch.float32, device=device) / 256
        module.weight.copy_(values.to(dtype)[:, None].expand_as(module.weight))
        module.bias.fill_(0.25)
    root = torch.nn.Module()
    root.proj = module
    return root


@pytest.mark.parametrize("kind", ["row", "column", "replicated", "merged", "qkv"])
def test_complete_vllm_view_and_skip_bias_binding(single_rank_parallel, kind):
    root = projection(kind, skip_bias_add=True)
    adapter = DenseModelAdapter(root)
    view = adapter.views(["proj"])["proj"]
    assert view.weight is root.proj.weight and view.bias is None
    assert view.out_features == root.proj.weight.shape[0]
    profile = profile_for(adapter, {"proj": "fp16"})
    source = root.proj
    source.skip_bias_add = False
    with pytest.raises(ValueError):
        adapter.apply(profile, scope="test", require_approved=False)
    source.skip_bias_add = True
    source.return_bias = False
    with pytest.raises(ValueError, match="semantics"):
        adapter.apply(profile, scope="test", require_approved=False)
    source.return_bias = True
    with torch.inference_mode():
        source.bias.add_(1)
    with pytest.raises(ValueError, match="bias source"):
        adapter.apply(profile, scope="test", require_approved=False)
    assert root.proj is source


mps_integration = pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="explicit real MPS/MLX integration",
)


@pytest.mark.integration
@mps_integration
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["row", "column", "replicated", "merged", "qkv"])
@pytest.mark.parametrize("return_bias", [False, True])
@pytest.mark.parametrize("skip_bias_add", [False, True])
def test_real_vllm_forward_contract_and_release(
    single_rank_parallel,
    bits,
    dtype,
    kind,
    return_bias,
    skip_bias_add,
):
    root = projection(
        kind, return_bias=return_bias, skip_bias_add=skip_bias_add, device="mps", dtype=dtype
    )
    adapter = DenseModelAdapter(root)
    # Use the actual original vLLM forward, not a dense-view baseline.
    inputs = torch.ones(3, 128, dtype=dtype, device="mps")[:, ::2]
    assert not inputs.is_contiguous()
    keyword = "x" if kind == "replicated" else "input_"
    with torch.inference_mode():
        expected = root.proj(**{keyword: inputs})
        with adapter.capture(["proj"]) as session:
            for _ in range(4):
                root.proj(**{keyword: inputs})
        evidence = adapter.search(session.captures, scope="test")
        # Force both real implementations for a contract check; still unapproved.
        profile = linear.build_profile(
            adapter.views(["proj"]),
            ["proj"],
            scope="test",
            calibration=copy.deepcopy(evidence["profile"]["calibration"]),
            precisions={"proj": f"w{bits}a16"},
        )
        source_ref = weakref.ref(root.proj)
        weight_ref = weakref.ref(root.proj.weight)
        replacements = adapter.apply(profile, scope="test", require_approved=False)
        result = root.proj(**{keyword: inputs})
    assert replacements["proj"] is root.proj
    assert root.proj.quantized.bits == bits
    assert root.proj.skip_bias_add is skip_bias_add
    if return_bias:
        assert isinstance(result, tuple)
        expected_output, expected_bias = expected
        output, bias = result
        if skip_bias_add:
            torch.testing.assert_close(bias, expected_bias, atol=0, rtol=0)
            assert bias is not expected_bias
        else:
            assert bias is expected_bias is None
    else:
        assert isinstance(result, torch.Tensor)
        output, expected_output = result, expected
    torch.testing.assert_close(output, expected_output, atol=0.01, rtol=0.01)
    del expected
    if return_bias:
        del expected_bias
    gc.collect()
    assert source_ref() is None and weight_ref() is None


class ComponentPipeline:
    def __init__(self, **components):
        self.__dict__.update(components)

    @property
    def components(self):
        return {name: value for name, value in self.__dict__.items()}


@pytest.mark.parametrize("nested_owner", [False, True])
@pytest.mark.parametrize("component_root", [False, True])
def test_unregistered_components_commit_and_identity(monkeypatch, nested_owner, component_root):
    root = torch.nn.Module()
    source = torch.nn.Linear(64, 64, dtype=torch.float16).requires_grad_(False)
    component = source if component_root else torch.nn.Sequential(source)
    if nested_owner:
        root._pipeline = ComponentPipeline(denoiser=component, tokenizer=object())
        owner = root._pipeline
    else:
        root.components = {"denoiser": component, "tokenizer": object()}
        owner = root
    # Also exercise an ordinary direct child while a synthetic root is active.
    root.other = torch.nn.Linear(64, 64, dtype=torch.float16).requires_grad_(False)
    adapter = DenseModelAdapter(root)
    name = "components.denoiser" + ("" if component_root else ".0")
    assert set(adapter.candidates()) == {name, "other"}
    identity = profile_for(adapter, {name: "fp16", "other": "fp16"})
    assert adapter.apply(identity, scope="test", require_approved=False)[name] is source
    source_ref, weight_ref = weakref.ref(source), weakref.ref(source.weight)
    profile = profile_for(adapter, {name: "w4a16", "other": "fp16"})
    # Component routing unit check only; real packed execution is tested above.
    monkeypatch.setattr(
        linear.QuantizedLinear, "from_linear", lambda view, *, bits: torch.nn.Identity()
    )
    replacement = adapter.apply(profile, scope="test", require_approved=False)[name]
    actual = owner.components["denoiser"]
    assert (actual if component_root else actual[0]) is replacement
    if nested_owner and component_root:
        assert owner.denoiser is replacement
    assert replacement is not source
    del source, component, identity
    gc.collect()
    assert source_ref() is None and weight_ref() is None


def test_components_share_storage_rejected():
    root = torch.nn.Module()
    first = torch.nn.Linear(64, 64, dtype=torch.float16).requires_grad_(False)
    second = torch.nn.Linear(64, 64, dtype=torch.float16).requires_grad_(False)
    second.weight = first.weight
    root._pipeline = ComponentPipeline(first=first, second=second)
    adapter = DenseModelAdapter(root)
    with pytest.raises(ValueError, match="shared|tied"):
        adapter.views(["components.first"])


def test_readonly_component_root_requires_hook():
    root = torch.nn.Module()
    module = torch.nn.Linear(64, 64, dtype=torch.float16).requires_grad_(False)

    class ReadOnlyPipeline:
        @property
        def components(self):
            return {"projection": module}

    root._pipeline = ReadOnlyPipeline()
    adapter = DenseModelAdapter(root)
    profile = profile_for(adapter, {"components.projection": "fp16"})
    with pytest.raises(ValueError, match="custom hook"):
        adapter.apply(profile, scope="test", require_approved=False)
