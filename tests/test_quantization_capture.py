# SPDX-License-Identifier: Apache-2.0
"""Real CPU forwards cover bounded capture and source ownership boundaries."""

import gc
import weakref

import pytest
import torch

from vllm_omni_metal.quantization.capture import CaptureSession
from vllm_omni_metal.quantization.engine import _captures


def root(dtype=torch.float16):
    return torch.nn.Sequential(torch.nn.Linear(5, 3, dtype=dtype)).requires_grad_(False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_real_capture_counts_calls_supports_kwargs_and_owns_cpu_contiguous_copy(dtype):
    model = root(dtype)
    x = torch.arange(20, dtype=dtype).reshape(5, 4).T
    assert not x.is_contiguous()
    with CaptureSession(model, ["0"], steps=(1, 3)) as session:
        for _ in range(5):
            model[0](input=x)
    samples = session.captures["0"]
    assert [step for step, _ in samples] == [1, 3]
    assert session.bytes_used == 2 * x.numel() * x.element_size()
    for _, tensor in samples:
        assert tensor.device.type == "cpu" and tensor.dtype == dtype
        assert tensor.is_contiguous() and not tensor.requires_grad
        assert tensor.data_ptr() != x.data_ptr()
        torch.testing.assert_close(tensor, x, rtol=0, atol=0)
    x.zero_()
    assert bool(samples[0][1].any())
    assert not model[0]._forward_pre_hooks


def test_missing_calls_stay_missing_and_search_admission_rejects():
    model = root()
    with CaptureSession(model, ["0"]) as session:
        model(torch.ones(1, 5, dtype=torch.float16))
    assert len(session.captures["0"]) == 1
    with pytest.raises(ValueError, match="four distinct"):
        _captures({"0": model[0]}, session.captures)


@pytest.mark.parametrize("steps", [(), [], (0, 0), (-1,), (True,), (1.0,), "0123", None])
def test_invalid_steps_are_rejected_without_hooks(steps):
    model = root()
    with pytest.raises(ValueError, match="steps"):
        CaptureSession(model, ["0"], steps=steps)
    assert not model[0]._forward_pre_hooks


@pytest.mark.parametrize("selected", [[], ["0", "0"], ["missing"], [""], "0"])
def test_invalid_selection_is_rejected_without_hooks(selected):
    model = root()
    with pytest.raises(ValueError):
        CaptureSession(model, selected)
    assert not model[0]._forward_pre_hooks


def test_module_alias_and_tied_storage_are_rejected_even_if_one_alias_selected():
    model = root()
    model.add_module("alias", model[0])
    with pytest.raises(ValueError, match="shared/tied"):
        CaptureSession(model, ["0"])
    model = root()
    other = torch.nn.Linear(5, 3, dtype=torch.float16).requires_grad_(False)
    other.weight = model[0].weight
    model.add_module("other", other)
    with pytest.raises(ValueError, match="shared/tied"):
        CaptureSession(model, ["0"])


def test_only_ordinary_linears_are_admitted():
    class Specialized(torch.nn.Linear):
        pass

    model = torch.nn.Sequential(Specialized(5, 3)).requires_grad_(False)
    with pytest.raises(ValueError, match="ordinary unquantized"):
        CaptureSession(model, ["0"])


@pytest.mark.parametrize("budget", [0, -1, True, 1.0])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError, match="max_bytes"):
        CaptureSession(root(), ["0"], max_bytes=budget)


def test_budget_overflow_keeps_real_samples_and_always_removes_hooks():
    model = root()
    x = torch.ones(2, 5, dtype=torch.float16)
    with pytest.raises(ValueError, match="byte budget exceeded"):
        with CaptureSession(model, ["0"], max_bytes=20) as session:
            model(x)  # Exactly the budget is legal.
            model(x)
    assert session.bytes_used == 20
    assert len(session.captures["0"]) == 1
    assert not model[0]._forward_pre_hooks


@pytest.mark.parametrize("invalid", ["dtype", "grad", "nan", "empty", "width"])
def test_bad_real_inputs_remove_hooks_without_recording(invalid):
    model = root()
    x = torch.ones(2, 5, dtype=torch.float16)
    if invalid == "dtype":
        x = x.float()
    elif invalid == "grad":
        x.requires_grad_(True)
    elif invalid == "nan":
        x[0, 0] = float("nan")
    elif invalid == "empty":
        x = x[:0]
    else:
        x = x[:, :4]
    with pytest.raises(ValueError):
        with CaptureSession(model, ["0"]) as session:
            model(x)
    assert session.captures["0"] == []
    assert not model[0]._forward_pre_hooks


def test_exception_cleanup_and_single_use():
    model = root()
    with pytest.raises(RuntimeError, match="forward failed"):
        with CaptureSession(model, ["0"]) as session:
            raise RuntimeError("forward failed")
    assert not model[0]._forward_pre_hooks
    session.close()
    with pytest.raises(ValueError, match="single-use"):
        with session:
            pass


def test_source_changed_is_reported_at_close_and_hooks_removed():
    model = root()
    with pytest.raises(ValueError, match="source_changed"):
        with CaptureSession(model, ["0"]) as session:
            model(torch.ones(1, 5, dtype=torch.float16))
            model[0].weight.add_(1)
    assert session.source_changed and not model[0]._forward_pre_hooks


def test_source_changed_after_close_is_rejected_by_search_validation():
    model = root()
    with CaptureSession(model, ["0"]) as session:
        for _ in range(4):
            model(torch.ones(1, 5, dtype=torch.float16))
    model[0].weight.add_(1)
    with pytest.raises(ValueError, match="source_changed"):
        _captures({"0": model[0]}, session.captures)
    assert session.source_changed


def test_replaced_module_is_reported_and_old_hooks_removed():
    model = root()
    original = model[0]
    with pytest.raises(ValueError, match="source_changed"):
        with CaptureSession(model, ["0"]):
            model[0] = root()[0]
    assert not original._forward_pre_hooks


def test_capture_objects_do_not_retain_source_parameters_or_modules():
    model = root()
    module_ref = weakref.ref(model[0])
    weight_ref = weakref.ref(model[0].weight)
    with CaptureSession(model, ["0"]) as session:
        model(torch.ones(1, 5, dtype=torch.float16))
    samples = session.captures
    del model
    gc.collect()
    assert module_ref() is None and weight_ref() is None
    assert len(samples["0"]) == 1


def test_alias_introduced_during_capture_marks_source_changed_and_cleans_hooks():
    model = root()
    with pytest.raises(ValueError, match="source_changed"):
        with CaptureSession(model, ["0"]) as session:
            model.add_module("alias", model[0])
    assert session.source_changed
    assert not model[0]._forward_pre_hooks


class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.skip_bias_add = True
        self.weight = torch.nn.Parameter(torch.ones(3, 5, dtype=torch.float16), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.ones(3, dtype=torch.float16), requires_grad=False)

    def forward(self, input_):
        return torch.nn.functional.linear(input_, self.weight), self.bias


def projection_view(module, *, bias=False):
    view = torch.nn.Linear(5, 3, dtype=torch.float16, bias=bias).requires_grad_(False)
    view.weight = module.weight
    if bias:
        view.bias = module.bias
    return view


@pytest.mark.parametrize("bias", [False, True])
def test_explicit_projection_views_capture_actual_calls_without_retaining_views(bias):
    model = torch.nn.Sequential(Projection())
    view = projection_view(model[0], bias=bias)
    view_ref = weakref.ref(view)
    session = CaptureSession(model, ["0"], linear_views={"0": view})
    with session:
        del view
        gc.collect()
        assert view_ref() is None
        for _ in range(4):
            model[0](input_=torch.ones(2, 5, dtype=torch.float16))
    views = {"0": projection_view(model[0], bias=bias)}
    ordered, _ = _captures(views, session.captures)
    assert len(ordered["0"]) == 4
    assert not model[0]._forward_pre_hooks


def test_projection_requires_explicit_exact_weight_and_bias_views():
    model = torch.nn.Sequential(Projection())
    with pytest.raises(ValueError, match="ordinary unquantized"):
        CaptureSession(model, ["0"])
    with pytest.raises(ValueError, match="exactly"):
        CaptureSession(model, ["0"], linear_views={})
    wrong = projection_view(model[0])
    wrong.weight = torch.nn.Parameter(wrong.weight.clone(), requires_grad=False)
    with pytest.raises(ValueError, match="exact source tensors"):
        CaptureSession(model, ["0"], linear_views={"0": wrong})
    wrong = projection_view(model[0], bias=True)
    wrong.bias = torch.nn.Parameter(wrong.bias.clone(), requires_grad=False)
    with pytest.raises(ValueError, match="exact source tensors"):
        CaptureSession(model, ["0"], linear_views={"0": wrong})


def test_projection_input_must_be_unique_not_an_arbitrary_tensor_kwarg():
    model = torch.nn.Sequential(Projection())
    with pytest.raises(ValueError, match="one Linear input"):
        with CaptureSession(model, ["0"], linear_views={"0": projection_view(model[0])}):
            model[0](other=torch.ones(1, 5, dtype=torch.float16))
    assert not model[0]._forward_pre_hooks


def test_projection_bias_cannot_be_omitted_without_explicit_skip_bias_add():
    model = torch.nn.Sequential(Projection())
    model[0].skip_bias_add = False
    with pytest.raises(ValueError, match="exact source tensors"):
        CaptureSession(model, ["0"], linear_views={"0": projection_view(model[0])})


def test_explicit_projection_x_keyword_is_supported_without_guessing():
    class XProjection(Projection):
        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight), self.bias

    model = torch.nn.Sequential(XProjection())
    with CaptureSession(model, ["0"], linear_views={"0": projection_view(model[0])}) as session:
        for _ in range(4):
            model[0](x=torch.ones(1, 5, dtype=torch.float16))
    assert len(session.captures["0"]) == 4
    assert not model[0]._forward_pre_hooks
