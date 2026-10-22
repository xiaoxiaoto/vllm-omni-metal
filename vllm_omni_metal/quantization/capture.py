# SPDX-License-Identifier: Apache-2.0
"""Bounded real-forward activation capture for offline DynamicEngine search.

Use ``with CaptureSession(root, ["projection"]) as session: ...`` and pass
``session.captures`` to search after exit. Missing calls remain missing; this
utility neither generates inputs nor approves a model's quality contract.
"""

import weakref

import torch

from . import linear


class Captures(dict):
    """CPU samples plus source fingerprints, never source tensors or parameters."""

    def __init__(self, descriptors, source_descriptors=None):
        super().__init__((name, []) for name in descriptors)
        self._descriptors = descriptors
        self._source_descriptors = source_descriptors or descriptors
        self.source_changed = False

    def validate_source(self, modules, *, original=False):
        descriptors = self._source_descriptors if original else self._descriptors
        if set(modules) != set(descriptors):
            self.source_changed = True
        else:
            for name, descriptor in descriptors.items():
                if linear._descriptor(modules[name], descriptor["bits"]) != descriptor:
                    self.source_changed = True
                    break
        if self.source_changed:
            raise ValueError("source_changed: candidate weights changed after capture began")


class CaptureSession:
    """Record selected ordinary Linear inputs at zero-based actual call indices.

    Captures own detached contiguous CPU copies. The byte budget covers those
    copies across all selected modules and calls; overflow raises before copying.
    Sessions are single-use and hooks are removed on every exit, including errors.
    The source must remain quiescent except for inference during the session.
    Explicit ``linear_views`` admit adapter-validated projection modules: each
    ordinary Linear view must share the actual module's exact weight and bias
    (bias may be omitted only for skip_bias_add). Hooks observe the actual module.
    """

    def __init__(
        self,
        root,
        selected,
        *,
        steps=(0, 1, 2, 3),
        max_bytes=128 * 1024**2,
        linear_views=None,
    ):
        if not isinstance(root, torch.nn.Module):
            raise ValueError("root must be a torch.nn.Module")
        if type(selected) is not list or not selected:
            raise ValueError("selected must be a nonempty list of explicit module names")
        if (
            type(steps) not in (tuple, list)
            or not steps
            or any(type(step) is not int or step < 0 for step in steps)
            or len(set(steps)) != len(steps)
        ):
            raise ValueError("steps must be distinct nonnegative integer call indices")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        modules = dict(root.named_modules(remove_duplicate=False))
        for name in selected:
            linear._name(name)
        if len(set(selected)) != len(selected) or any(name not in modules for name in selected):
            raise ValueError("selected contains duplicate or unknown module names")
        chosen = {name: modules[name] for name in selected}
        if linear_views is None:
            views = chosen
        else:
            if type(linear_views) is not dict or set(linear_views) != set(selected):
                raise ValueError("linear_views must map exactly the selected module names")
            views = linear_views
            for name, view in views.items():
                if (
                    type(view) is not torch.nn.Linear
                    or view.weight is not getattr(chosen[name], "weight", None)
                    or (
                        view.bias is not None
                        and view.bias is not getattr(chosen[name], "bias", None)
                    )
                    or (
                        view.bias is None
                        and getattr(chosen[name], "bias", None) is not None
                        and getattr(chosen[name], "skip_bias_add", False) is not True
                    )
                ):
                    raise ValueError(
                        "linear_views require ordinary Linear views of exact source tensors"
                    )
        self._uses_views = linear_views is not None
        precisions = {
            name: {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}.get(
                getattr(getattr(module, "weight", None), "dtype", None), "bf16"
            )
            for name, module in views.items()
        }
        bits = linear._modules(views, selected, precisions)
        linear._untied(chosen, selected, root=root)
        self.captures = Captures(
            {name: linear._descriptor(module, bits[name]) for name, module in views.items()},
            {name: linear._descriptor(module, bits[name]) for name, module in chosen.items()},
        )
        self._root = weakref.ref(root)
        self._modules = {name: weakref.ref(module) for name, module in chosen.items()}
        self._steps = frozenset(steps)
        self._counts = dict.fromkeys(selected, 0)
        self._max_bytes = max_bytes
        self.bytes_used = 0
        self._handles = []
        self._entered = False
        self._closed = False

    @property
    def source_changed(self):
        return self.captures.source_changed

    def _live_modules(self):
        root = self._root()
        if root is None:
            self.captures.source_changed = True
            raise ValueError("source_changed: capture root was released")
        current = dict(root.named_modules(remove_duplicate=False))
        chosen = {}
        for name, reference in self._modules.items():
            module = reference()
            if module is None or current.get(name) is not module:
                self.captures.source_changed = True
                raise ValueError("source_changed: selected module tree changed")
            chosen[name] = module
        try:
            linear._untied(chosen, list(chosen), root=root)
        except ValueError as error:
            self.captures.source_changed = True
            raise ValueError("source_changed: selected source gained aliases") from error
        return chosen

    def _hook(self, name, module, args, kwargs):
        step = self._counts[name]
        self._counts[name] += 1
        if step not in self._steps:
            return
        if len(args) == 1 and not kwargs:
            tensor = args[0]
        elif not args and (
            set(kwargs) in ({"input"}, {"input_"}) or (self._uses_views and set(kwargs) == {"x"})
        ):
            tensor = next(iter(kwargs.values()))
        else:
            raise ValueError(
                "capture requires one Linear input, positional or input=/input_= keyword"
            )
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.requires_grad
            or tensor.layout != torch.strided
            or tensor.dtype != module.weight.dtype
            or tensor.device != module.weight.device
            or tensor.ndim < 1
            or tensor.numel() == 0
            or tensor.shape[-1] != self.captures._descriptors[name]["shape"][1]
        ):
            raise ValueError("capture requires nonempty matching source input without gradients")
        size = tensor.numel() * tensor.element_size()
        if size > self._max_bytes - self.bytes_used:
            raise ValueError("capture byte budget exceeded")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("capture requires finite input")
        host = tensor.detach().to(device="cpu", copy=True).contiguous()
        self.captures[name].append((step, host))
        self.bytes_used += size

    def __enter__(self):
        if self._entered or self._closed:
            raise ValueError("capture sessions are single-use")
        self._entered = True
        try:
            modules = self._live_modules()
            self.captures.validate_source(modules, original=True)
            for name, module in modules.items():
                # Recheck hooks/weights before attaching anything; all handles are
                # removed if any registration or validation fails.
                if self._uses_views:
                    if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
                        raise ValueError("capture cannot discard projection hooks")
                else:
                    linear._linear(module, bits=self.captures._descriptors[name]["bits"])
                self._handles.append(
                    module.register_forward_pre_hook(
                        lambda mod, args, kwargs, name=name: self._hook(name, mod, args, kwargs),
                        with_kwargs=True,
                    )
                )
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        if self._closed:
            return
        self._closed = True
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.captures.validate_source(self._live_modules(), original=True)

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
