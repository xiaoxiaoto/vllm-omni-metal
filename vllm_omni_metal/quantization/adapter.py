# SPDX-License-Identifier: Apache-2.0
"""Explicit dense projection discovery, offline search and atomic installation.

``calibration.projections`` binds the vLLM return/skip-bias contract and the
original bias bytes, including biases intentionally absent from a search view.
Views are transient shared-parameter objects, never installed in the model tree.
"""

import hashlib
import weakref
from collections.abc import Mapping

import torch

from . import linear
from .engine import DynamicEngine, _implementation_sha256


def _component_owner(root):
    pipeline = getattr(root, "_pipeline", None)
    if isinstance(getattr(pipeline, "components", None), Mapping):
        return pipeline
    return root if isinstance(getattr(root, "components", None), Mapping) else None


class _ComponentMap(dict):
    """A short-lived module table whose writes reach the actual pipeline."""

    def __init__(self, owner):
        self.owner = owner
        super().__init__(
            (name, value)
            for name, value in owner.components.items()
            if isinstance(value, torch.nn.Module)
        )

    def writable(self, name):
        source = self[name]
        concrete = self.owner.__dict__
        mapping = concrete.get("components")
        if concrete.get(name) is not source and not (
            type(mapping) is dict and mapping.get(name) is source
        ):
            raise ValueError("component root needs a writable pipeline attribute or custom hook")

    def __setitem__(self, name, value):
        self.writable(name)
        source = self[name]
        concrete = self.owner.__dict__
        mapping = concrete.get("components")
        if concrete.get(name) is source:
            concrete[name] = value
        if type(mapping) is dict and mapping.get(name) is source:
            mapping[name] = value
        super().__setitem__(name, value)


class _ComponentRoot(torch.nn.Module):
    """Dynamic facade: only the real owner, never a cached float tree, is weakly held."""

    def __init__(self, root):
        super().__init__()
        self._owner = weakref.ref(root)

    def _live(self):
        root = self._owner()
        if root is None:
            raise ValueError("adapter root was released")
        return root

    def _table(self):
        table = torch.nn.Module()
        table._modules = _ComponentMap(_component_owner(self._live()))
        return table

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        root = self._live()
        if "components" in root._modules:
            raise ValueError("registered components name conflicts with pipeline components")
        snapshot = torch.nn.Module()
        snapshot._parameters = root._parameters
        snapshot._buffers = root._buffers
        snapshot._modules = {**root._modules, "components": self._table()}
        return snapshot.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)

    def get_submodule(self, target):
        if not target:
            return self
        if target == "components":
            return self._table()
        if target.startswith("components."):
            return self._table().get_submodule(target[len("components.") :])
        return self._live().get_submodule(target)


def _vllm(module):
    # Do not import vLLM (and its registry) for ordinary Torch models.
    if not any(base.__module__.startswith("vllm.") for base in type(module).__mro__):
        return False
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        LinearBase,
        ReplicatedLinear,
        RowParallelLinear,
        UnquantizedLinearMethod,
    )

    if not isinstance(module, LinearBase):
        return False
    if (
        module.tp_size != 1
        or module.tp_rank != 0
        or type(module.quant_method) is not UnquantizedLinearMethod
        or getattr(module.forward, "__func__", None)
        not in (ColumnParallelLinear.forward, ReplicatedLinear.forward, RowParallelLinear.forward)
        or type(module.return_bias) is not bool
        or type(module.skip_bias_add) is not bool
    ):
        raise ValueError("requires a supported TP1 unquantized vLLM dense forward contract")
    return True


def _view(module):
    if type(module) is torch.nn.Linear:
        view = module
    elif _vllm(module):
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
            raise ValueError("cannot discard projection hooks")
        if not isinstance(module.weight, torch.Tensor) or module.weight.ndim != 2:
            raise ValueError("requires a complete dense projection weight")
        if tuple(module.weight.shape) != (module.output_size, module.input_size):
            raise ValueError("requires a full TP1 projection, not a sharded or packed-weight view")
        view = torch.nn.Linear.__new__(torch.nn.Linear)
        torch.nn.Module.__init__(view)
        view.in_features = module.weight.shape[1]
        view.out_features = module.weight.shape[0]
        view.weight = module.weight
        view.bias = None if module.skip_bias_add else module.bias
        view.training = module.training
        # Validate the otherwise skipped bias as well as the actual GEMM view.
        if module.skip_bias_add and module.bias is not None:
            view.bias = module.bias
            linear._linear(view, bits=_identity_bits(view))
            view.bias = None
    else:
        raise ValueError("requires ordinary torch Linear or supported vLLM LinearBase")
    linear._linear(view, bits=_identity_bits(view))
    return view


def _identity_bits(view):
    return 32 if view.weight.dtype == torch.float32 else 16


def _semantics(module):
    if type(module) is torch.nn.Linear:
        return {"kind": "torch"}
    bias = module.bias
    descriptor = None
    if bias is not None:
        descriptor = {
            "shape": list(bias.shape),
            "dtype": str(bias.dtype),
            "device": str(bias.device),
        }
        host = bias.detach().to(device="cpu", copy=True).contiguous()
        descriptor["sha256"] = hashlib.sha256(
            memoryview(host.view(torch.uint8).numpy()).cast("B")
        ).hexdigest()
    return {
        "kind": "vllm",
        "return_bias": module.return_bias,
        "skip_bias_add": module.skip_bias_add,
        "bias": descriptor,
    }


class _Projection(torch.nn.Module):
    """Own only packed weights and, when returned separately, a copied bias."""

    def __init__(self, source, quantized):
        super().__init__()
        self.quantized = quantized
        self.return_bias = source.return_bias
        self.skip_bias_add = source.skip_bias_add
        self.training = source.training
        self.register_buffer(
            "bias",
            source.bias.detach().clone()
            if self.skip_bias_add and self.return_bias and source.bias is not None
            else None,
        )

    def forward(self, *args, **kwargs):
        if len(args) == 1 and not kwargs:
            x = args[0]
        elif not args and len(kwargs) == 1 and next(iter(kwargs)) in ("x", "input_"):
            x = next(iter(kwargs.values()))
        else:
            raise TypeError("projection requires one positional or x=/input_= activation")
        # vLLM callers may pass strided activations; never cast or move them.
        if isinstance(x, torch.Tensor) and x.device.type == "mps" and not x.is_contiguous():
            x = x.contiguous()
        result = self.quantized(x)
        return (result, self.bias) if self.return_bias else result


class DenseModelAdapter:
    """Load-time adapter with no persistent floating source references.

    Use only with an exclusively owned, quiescent module tree. Search returns an
    unapproved profile; model-level approval remains the caller's responsibility.
    Explicit identity selections leave the original module object untouched.
    """

    def __init__(self, root):
        if not isinstance(root, torch.nn.Module):
            raise ValueError("root must be a torch.nn.Module")
        self._root = weakref.ref(root)
        self._component_root = _ComponentRoot(root)

    def _live_root(self):
        root = self._root()
        if root is None:
            raise ValueError("adapter root was released")
        return self._component_root if _component_owner(root) is not None else root

    @torch.inference_mode()
    def candidates(self):
        result = {}
        for name, module in self._live_root().named_modules(remove_duplicate=False):
            if not name:
                continue
            try:
                _view(module)
            except (ValueError, AttributeError):
                continue
            result[name] = module
        return result

    @torch.inference_mode()
    def views(self, selected):
        if type(selected) is not list or not selected:
            raise ValueError("selected must be a nonempty list of explicit module names")
        for name in selected:
            linear._name(name)
        modules = dict(self._live_root().named_modules(remove_duplicate=False))
        if len(set(selected)) != len(selected) or any(name not in modules for name in selected):
            raise ValueError("selected contains duplicate or unknown module names")
        chosen = {name: modules[name] for name in selected}
        linear._untied(chosen, selected, root=self._live_root())
        return {name: _view(module) for name, module in chosen.items()}

    def capture(self, selected, **opts):
        from .capture import CaptureSession

        return CaptureSession(
            self._live_root(), selected, linear_views=self.views(selected), **opts
        )

    @torch.inference_mode()
    def search(self, captures, *, scope):
        selected = list(captures)
        views = self.views(selected)
        result = DynamicEngine.search(views, captures, scope=scope)
        root = self._live_root()
        calibration = dict(result["profile"]["calibration"])
        calibration["projections"] = {
            name: _semantics(root.get_submodule(name)) for name in selected
        }
        result["profile"] = linear.build_profile(
            views, selected, scope=scope, calibration=calibration, precisions=result["precisions"]
        )
        return result

    @torch.inference_mode()
    def apply(self, profile, *, scope, require_approved=True):
        if type(require_approved) is not bool:
            raise ValueError("require_approved must be a boolean")
        if type(profile) is not dict or type(profile.get("calibration")) is not dict:
            raise ValueError("DynamicEngine requires a calibrated schema2 profile")
        calibration = profile["calibration"]
        if calibration.get("engine") != "DynamicEngine":
            raise ValueError("profile was not calibrated with DynamicEngine")
        if calibration.get("implementation_sha256") != _implementation_sha256():
            raise ValueError("DynamicEngine implementation mismatch; recalibration is required")
        if type(calibration.get("approved")) is not bool:
            raise ValueError("profile approval must be an explicit boolean")
        if require_approved and not calibration["approved"]:
            raise ValueError("DynamicEngine profile has not been approved by its model adapter")
        selected = profile.get("selected")
        views = self.views(selected)
        linear.validate_profile(profile, views, scope=scope)
        root = self._live_root()
        sources = {name: root.get_submodule(name) for name in selected}
        expected = {name: _semantics(module) for name, module in sources.items()}
        if linear._encoded(calibration.get("projections")) != linear._encoded(expected):
            raise ValueError("projection return/skip-bias semantics or bias source mismatch")
        linear._untied(sources, selected, root=root)
        locations = {}
        for name, source in sources.items():
            parent_name, _, child = name.rpartition(".")
            parent = root.get_submodule(parent_name) if parent_name else root
            if parent is root and isinstance(root, _ComponentRoot):
                parent = root._live()
            if parent._modules.get(child) is not source:
                raise ValueError("selected module tree changed during profile validation")
            if isinstance(parent._modules, _ComponentMap):
                parent._modules.writable(child)
            locations[name] = (parent, child)
        replacements = {}
        for name, source in sources.items():
            bits = profile["modules"][name]["bits"]
            if bits in (16, 32):
                replacement = source
            else:
                packed = linear.QuantizedLinear.from_linear(views[name], bits=bits)
                replacement = (
                    packed if type(source) is torch.nn.Linear else _Projection(source, packed)
                )
            replacements[name] = replacement
        # Constructors must not alter the source tree or introduce new aliases.
        linear._untied(sources, selected, root=root)
        if any(
            parent._modules.get(child) is not sources[name]
            for name, (parent, child) in locations.items()
        ):
            raise ValueError("selected module tree changed during candidate construction")
        for name, (parent, child) in locations.items():
            parent._modules[child] = replacements[name]
        return replacements
