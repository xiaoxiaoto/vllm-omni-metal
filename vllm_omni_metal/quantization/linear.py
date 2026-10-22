# SPDX-License-Identifier: Apache-2.0
"""Explicit MLX W4/W8 linears, source precision retention, and JSON profiles.

Construction is an offline/load-time operation, never a forward-time conversion.
Profiles contain metadata only: no pickle, executable objects, or weight payloads.
"""

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path

import torch

SCHEMA = 2
GROUP_SIZE = 64
_DTYPES = (torch.float16, torch.bfloat16)
_PRECISIONS = {"w4a16": 4, "w8a16": 8, "bf16": 16, "fp16": 16, "fp32": 32}


def _plain(value, depth=0):
    if depth > 32:
        raise ValueError("profile metadata is too deeply nested")
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _plain(item, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _plain(item, depth + 1)
        return
    raise ValueError("profile metadata must contain only finite plain JSON values")


def _encoded(value):
    _plain(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _scope(scope):
    if type(scope) is not str or not scope or len(scope) > 1024:
        raise ValueError("a bounded nonempty pipeline revision scope is required")


def _name(name):
    if (
        type(name) is not str
        or not name
        or len(name) > 1024
        or any(not part for part in name.split("."))
    ):
        raise ValueError("selected module names must be nonempty dotted paths")


def _group(group_size):
    if type(group_size) is not int or group_size != GROUP_SIZE:
        raise ValueError("W4A16/W8A16 requires group_size=64")


def _bits(bits, *, retained=False):
    if type(bits) is not int or bits not in ((4, 8, 16, 32) if retained else (4, 8)):
        raise ValueError("precision bits must be 4/8 (or 16/32 for retained source)")


def _linear(linear, *, mps=False, bits=8):
    if type(linear) is not torch.nn.Linear:
        raise ValueError("quantization requires an ordinary unquantized torch.nn.Linear")
    weight = linear.weight
    if (
        not isinstance(weight, torch.Tensor)
        or weight.layout != torch.strided
        or weight.ndim != 2
        or type(linear.in_features) is not int
        or type(linear.out_features) is not int
        or min(linear.in_features, linear.out_features) <= 0
        or tuple(weight.shape) != (linear.out_features, linear.in_features)
        or (bits not in (16, 32) and linear.in_features % GROUP_SIZE)
    ):
        raise ValueError("Linear dimensions must match a positive group64-aligned weight")
    dtypes = (torch.float32,) if bits == 32 else _DTYPES
    if weight.dtype not in dtypes or weight.device.type not in ("cpu", "mps"):
        raise ValueError("Linear weights must be CPU/MPS FP16 or BF16 (FP32 retention only)")
    if mps and weight.device.type != "mps":
        raise ValueError("W4A16/W8A16 execution requires MPS weights")
    tensors = [weight]
    if linear.bias is not None:
        bias = linear.bias
        if (
            not isinstance(bias, torch.Tensor)
            or bias.layout != torch.strided
            or tuple(bias.shape) != (linear.out_features,)
            or bias.dtype != weight.dtype
            or bias.device != weight.device
        ):
            raise ValueError("Linear bias must match weight shape, dtype and device")
        tensors.append(bias)
    for tensor in tensors:
        if (
            tensor.requires_grad and not torch.is_inference_mode_enabled()
        ) or not tensor.is_contiguous():
            raise ValueError(
                "quantization requires contiguous weights/bias without gradients "
                "(or explicit torch.inference_mode for detached offline conversion)"
            )
        # Offline validation may synchronize; forward never reads whole activations back.
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("quantization requires finite weights and bias")
    if len(tensors) == 2 and _storage(tensors[0]) == _storage(tensors[1]):
        raise ValueError("selected shared/tied weight or bias storage is unsupported")
    if linear._forward_hooks or linear._forward_pre_hooks or linear._backward_hooks:
        raise ValueError("quantization cannot discard Linear hooks")
    return tensors


def _storage(tensor):
    return (str(tensor.device), tensor.untyped_storage().data_ptr())


def _untied(modules, selected, *, root=None):
    chosen = {id(modules[name]) for name in selected}
    counts = {}
    for module in modules.values():
        counts[id(module)] = counts.get(id(module), 0) + 1
    if any(counts[identity] != 1 for identity in chosen):
        raise ValueError("selected shared/tied modules are unsupported")
    owners = {}
    if root is None:
        tensors = (
            (id(module) in chosen, tensor)
            for module in modules.values()
            for tensor in module.parameters(recurse=False)
        )
    else:
        aliases = {}
        for _, module in root.named_modules(remove_duplicate=False):
            aliases[id(module)] = aliases.get(id(module), 0) + 1
        if any(aliases[identity] != 1 for identity in chosen):
            raise ValueError("selected shared/tied modules are unsupported")
        tensors = (
            (id(module) in chosen, tensor)
            for _, module in root.named_modules(remove_duplicate=False)
            for tensor in (
                *module.parameters(recurse=False),
                *module.buffers(recurse=False),
            )
        )
    for is_selected, tensor in tensors:
        key = _storage(tensor)
        if tensor.numel() == 0:
            continue
        previous = owners.get(key)
        if previous is not None and (previous or is_selected):
            raise ValueError("selected shared/tied weight or bias storage is unsupported")
        owners[key] = is_selected


def _modules(modules, selected, precisions=None):
    if type(modules) is not dict or type(selected) is not list:
        raise ValueError("modules must be a dict and selected must be a list")
    for name in modules:
        _name(name)
    if any(not isinstance(module, torch.nn.Module) for module in modules.values()):
        raise ValueError("modules must map names to torch modules")
    for name in selected:
        _name(name)
    if len(set(selected)) != len(selected) or any(name not in modules for name in selected):
        raise ValueError("selected contains duplicate or unknown module names")
    if precisions is None:
        precisions = dict.fromkeys(selected, "w8a16")
    if type(precisions) is not dict or set(precisions) != set(selected):
        raise ValueError("precisions must map exactly the selected module names")
    if any(type(value) is not str or value not in _PRECISIONS for value in precisions.values()):
        raise ValueError("precisions must be w4a16, w8a16, bf16, fp16 or fp32")
    bits = {name: _PRECISIONS[precisions[name]] for name in selected}
    for name in selected:
        _linear(modules[name], bits=bits[name])
        precision = precisions[name]
        retained_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        if precision in retained_dtype and modules[name].weight.dtype != retained_dtype[precision]:
            raise ValueError(
                f"{precision} retention requires existing {precision.upper()} weights; no implicit cast"
            )
    _untied(modules, selected)
    return bits


def _descriptor(linear, bits):
    tensors = [linear.weight] + ([] if linear.bias is None else [linear.bias])
    descriptor = {
        "shape": list(linear.weight.shape),
        "dtype": str(linear.weight.dtype),
        "device": str(linear.weight.device),
        "bias": linear.bias is not None,
        "group_size": None if bits in (16, 32) else GROUP_SIZE,
        "bits": bits,
        "mode": "identity" if bits in (16, 32) else "affine",
    }
    digest = hashlib.sha256(_encoded(descriptor))
    for tensor in tensors:
        # Preserve BF16 and signed-zero bits; never cast weights for fingerprinting.
        host = tensor.detach().to(device="cpu", copy=True)
        digest.update(memoryview(host.view(torch.uint8).numpy()).cast("B"))
    return {**descriptor, "sha256": digest.hexdigest()}


def _environment():
    packages = {}
    for name in ("mlx", "vllm-metal", "vllm-omni-metal"):
        distribution = importlib.metadata.distribution(name)
        packages[name] = {
            "version": str(distribution.version),
            "path": str(Path(distribution.locate_file("")).resolve()),
        }
    bridge = Path(
        importlib.metadata.distribution("vllm-metal").locate_file(
            "vllm_metal/pytorch_backend/tensor_bridge.py"
        )
    )
    device = {}
    if torch.backends.mps.is_available():
        import mlx.core as mx

        device = dict(mx.device_info(mx.gpu))
    result = {
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bridge_sha256": hashlib.sha256(bridge.read_bytes()).hexdigest(),
        "packages": packages,
        "torch": str(torch.__version__),
        "torch_git": str(torch.version.git_version),
        "torch_path": str(Path(torch.__file__).resolve()),
        "python": str(Path(sys.executable).resolve()),
        "python_abi": str(sys.implementation.cache_tag),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "kernel": platform.version(),
        "device": device,
    }
    _plain(result)
    return result


def build_profile(modules, selected, *, scope, calibration, precisions=None):
    """Bind an explicit selection to exact source tensors and calibration metadata.

    Only selected weights are hashed; adapters must bind the complete pipeline
    revision/configuration/input policy in scope. CPU source tensors can be
    fingerprinted offline, but applying cannot silently move them to MPS.
    Callers own calibration policy; metadata is not proof of losslessness.
    An explicit precisions map covers exactly selected: w4a16/w8a16/bf16/fp16/fp32.
    Identity entries bind existing source modules without converting or replacing them.
    """
    _scope(scope)
    bits = _modules(modules, selected, precisions)
    if type(calibration) is not dict or len(_encoded(calibration)) > 1024**2:
        raise ValueError("calibration must be a bounded plain JSON dict")
    profile = {
        "schema": SCHEMA,
        "scope": scope,
        "environment": _environment(),
        "selected": list(selected),
        "modules": {name: _descriptor(modules[name], bits[name]) for name in selected},
        "calibration": json.loads(_encoded(calibration)),
    }
    profile["sha256"] = hashlib.sha256(_encoded(profile)).hexdigest()
    return profile


def validate_profile(profile, modules, *, scope):
    """Return explicit selected names, rejecting any stale or malformed profile."""
    _scope(scope)
    if type(profile) is not dict:
        raise ValueError("quantization profile must be a plain JSON dict")
    _plain(profile)
    if set(profile) != {
        "schema",
        "scope",
        "environment",
        "selected",
        "modules",
        "calibration",
        "sha256",
    }:
        raise ValueError("invalid quantization profile fields")
    if type(profile["schema"]) is not int or profile["schema"] != SCHEMA:
        raise ValueError("unsupported quantization profile schema")
    if profile["scope"] != scope or _encoded(profile["environment"]) != _encoded(_environment()):
        raise ValueError("quantization profile scope or source environment mismatch")
    payload = {key: value for key, value in profile.items() if key != "sha256"}
    if profile["sha256"] != hashlib.sha256(_encoded(payload)).hexdigest():
        raise ValueError("quantization profile content hash mismatch")
    selected = profile["selected"]
    descriptors = profile["modules"]
    if (
        type(descriptors) is not dict
        or type(selected) is not list
        or any(type(name) is not str for name in selected)
        or set(descriptors) != set(selected)
    ):
        raise ValueError("invalid quantization module descriptors")
    precisions = {}
    for name in selected:
        descriptor = descriptors[name]
        if type(descriptor) is not dict:
            raise ValueError("invalid quantization module descriptor")
        bits = descriptor.get("bits")
        _bits(bits, retained=True)
        if bits == 16:
            if type(descriptor.get("dtype")) is not str:
                raise ValueError("16-bit identity descriptor requires FP16 or BF16 dtype")
            precision = {"torch.float16": "fp16", "torch.bfloat16": "bf16"}.get(
                descriptor.get("dtype")
            )
            if precision is None:
                raise ValueError("16-bit identity descriptor requires FP16 or BF16 dtype")
        elif bits == 32:
            if descriptor.get("dtype") != "torch.float32":
                raise ValueError("32-bit identity descriptor requires FP32 dtype")
            precision = "fp32"
        else:
            precision = {4: "w4a16", 8: "w8a16"}[bits]
        precisions[name] = precision
    bits = _modules(modules, selected, precisions)
    if type(profile["calibration"]) is not dict or len(_encoded(profile["calibration"])) > 1024**2:
        raise ValueError("invalid quantization calibration metadata")
    expected = {name: _descriptor(modules[name], bits[name]) for name in selected}
    if _encoded(profile["modules"]) != _encoded(expected):
        raise ValueError("quantization profile weight, shape, dtype or group mismatch")
    return list(selected)


class QuantizedLinear(torch.nn.Module):
    """Inference-only MPS linear; persistent state is MLX packed W4/W8 + affine data.

    Supports contiguous FP16/BF16 activations of matching dtype/device and any
    rank >= 1. Nonfinite input/output values are checked with a GPU reduction and one
    scalar read, not an activation-sized CPU transfer. No compilation, weight
    conversion, full dequantization, or float-weight fallback occurs in forward.
    Use from_linear explicitly while the model is quiescent, before requests.
    """

    def __init__(self):
        super().__init__()
        raise TypeError("use QuantizedLinear.from_linear for explicit load-time construction")

    @classmethod
    def from_linear(cls, linear, *, group_size=GROUP_SIZE, bits=8):
        _group(group_size)
        _bits(bits)
        tensors = _linear(linear, mps=True, bits=bits)
        import mlx.core as mx
        from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx

        module = cls.__new__(cls)
        torch.nn.Module.__init__(module)
        module.in_features = linear.in_features
        module.out_features = linear.out_features
        module.dtype = linear.weight.dtype
        module.device = linear.weight.device
        module.group_size = group_size
        module.bits = bits
        module.training = linear.training
        module.metrics = {"calls": 0, "rows": 0}
        module.source_bytes = int(sum(t.numel() * t.element_size() for t in tensors))
        with mx.stream(mx.gpu):
            shared_weight = torch_to_mlx(linear.weight)
            packed, scales, offsets = mx.quantize(
                shared_weight, group_size=group_size, bits=bits, mode="affine"
            )
            # Bias must own its storage: no original Linear/float-weight references survive.
            bias = None if linear.bias is None else torch_to_mlx(linear.bias, copy=True)
            arrays = [packed, scales, offsets] + ([] if bias is None else [bias])
            finite = mx.all(mx.isfinite(scales)) & mx.all(mx.isfinite(offsets))
            mx.eval(*arrays, finite)
            if not bool(finite.item()):
                raise ValueError("quantized affine data overflowed; original Linear is unchanged")
        module._packed = packed
        module._scales = scales
        module._offsets = offsets
        module._bias = bias
        module.storage_bytes = int(sum(array.nbytes for array in arrays))
        return module

    def _apply(self, fn, recurse=True):
        raise ValueError("quantized MLX storage cannot be moved/cast through torch.Module.to")

    def forward(self, x):
        if (
            not isinstance(x, torch.Tensor)
            or x.device != self.device
            or x.dtype != self.dtype
            or x.layout != torch.strided
            or x.ndim < 1
            or x.shape[-1] != self.in_features
            or not x.is_contiguous()
            or x.requires_grad
        ):
            raise ValueError(
                "activation must be contiguous matching MPS FP16/BF16 without gradients"
            )
        rows = x.numel() // self.in_features
        if rows == 0:
            result = x.new_empty((*x.shape[:-1], self.out_features))
        else:
            import mlx.core as mx
            from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch, torch_to_mlx

            with mx.stream(mx.gpu):
                shared = torch_to_mlx(x)
                matrix = shared.reshape(rows, self.in_features)
                finite = mx.all(mx.isfinite(shared))
                output = mx.quantized_matmul(
                    matrix,
                    self._packed,
                    self._scales,
                    self._offsets,
                    transpose=True,
                    group_size=self.group_size,
                    bits=self.bits,
                    mode="affine",
                )
                if self._bias is not None:
                    output = output + self._bias
                finite = finite & mx.all(mx.isfinite(output))
                # Borrowed input and all persistent arrays remain alive through evaluation.
                mx.eval(output, finite)
                if not bool(finite.item()):
                    raise ValueError("activation and output must contain only finite values")
                result = mlx_to_torch(output, self.device).reshape(*x.shape[:-1], self.out_features)
        self.metrics["calls"] += 1
        self.metrics["rows"] += int(rows)
        return result


def apply_profile(root, profile, *, scope):
    """Construct every candidate before replacing any child; reject all sharing.

    Call only at load time with exclusive access to the quiescent module tree.
    Quantized entries retain no original float modules. Identity entries stay identical;
    the returned map includes every selected name, including retained modules.
    """
    if not isinstance(root, torch.nn.Module):
        raise ValueError("root must be a torch.nn.Module")
    modules = {
        name: module
        for name, module in root.named_modules(remove_duplicate=False)
        if name and isinstance(module, (torch.nn.Linear, QuantizedLinear))
    }
    selected = validate_profile(profile, modules, scope=scope)
    _untied(modules, selected, root=root)
    locations = {}
    for name in selected:
        parent_name, _, child = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        # Write the standard module table directly; custom __setattr__ cannot half-commit.
        if parent._modules.get(child) is not modules[name]:
            raise ValueError("selected module tree changed during profile validation")
        locations[name] = (parent, child)
    replacements = {
        name: (
            modules[name]
            if profile["modules"][name]["bits"] in (16, 32)
            else QuantizedLinear.from_linear(
                modules[name], group_size=GROUP_SIZE, bits=profile["modules"][name]["bits"]
            )
        )
        for name in selected
    }
    for name in selected:
        parent, child = locations[name]
        parent._modules[child] = replacements[name]
    return replacements
