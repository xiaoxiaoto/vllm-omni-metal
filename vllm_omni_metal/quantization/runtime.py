# SPDX-License-Identifier: Apache-2.0
"""Model-independent dynamic-quantization loading and complete execution identity.

Private adapters own fused execution and task-quality approval. Ordinary Torch
models use the shared dense adapter; self-managed facades provide explicit hooks.
"""

import dataclasses
import enum
import hashlib
import inspect
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

PROFILE_LIMIT = 4 * 1024**2
MODE_ENV = "VLLM_OMNI_METAL_QUANTIZATION"
PROFILE_ENV = "VLLM_OMNI_METAL_QUANTIZATION_PROFILE"


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate quantization JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite quantization JSON value: {value}")


def _profile_header(profile, scope):
    if type(profile) is not dict or set(profile) != {
        "schema",
        "scope",
        "environment",
        "selected",
        "modules",
        "calibration",
        "sha256",
    }:
        raise ValueError("expected a plain schema2 quantization profile")
    if type(profile["schema"]) is not int or profile["schema"] != 2:
        raise ValueError("unsupported quantization profile schema")
    if profile["scope"] != scope:
        raise ValueError("quantization profile execution scope mismatch")
    payload = {key: value for key, value in profile.items() if key != "sha256"}
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if profile["sha256"] != expected:
        raise ValueError("quantization profile content hash mismatch")
    return profile


def load_profile(path, *, scope):
    """Read owned0600 JSON; select this exact execution from a multi-stage bundle."""
    if type(scope) is not str or not scope or len(scope) > 1024:
        raise ValueError("a bounded nonempty quantization scope is required")
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ValueError("quantization profile path must be absolute")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > PROFILE_LIMIT:
            raise ValueError("quantization profile must be a bounded regular file")
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("quantization profile must be user-owned mode0600")
        raw = stream.read(PROFILE_LIMIT + 1)
    if len(raw) > PROFILE_LIMIT:
        raise ValueError("quantization profile exceeds the size limit")
    data = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    if type(data) is dict and set(data) == {"profiles"}:
        profiles = data["profiles"]
        if type(profiles) is not dict or not profiles:
            raise ValueError("quantization profile bundle must be nonempty")
        for key, profile in profiles.items():
            _profile_header(profile, key)
        if scope not in profiles:
            raise ValueError("quantization bundle has no profile for this execution scope")
        data = profiles[scope]
    return _profile_header(data, scope)


def _type_name(value):
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _code_identity(value):
    filename = inspect.getsourcefile(value)
    if not filename:
        raise TypeError("a native quantization adapter must bind opaque implementation identity")
    return {
        "name": f"{value.__module__}.{value.__qualname__}",
        "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
    }


def _config_identity(value, seen=None):
    """Canonicalize supported config values without repr addresses or executable data."""
    import torch

    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("quantization configuration must be finite")
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return {"type": _type_name(value), "value": str(value)}
    if isinstance(value, Path):
        return {"path": str(value)}
    if isinstance(value, enum.Enum):
        return {"enum": _type_name(value), "value": _config_identity(value.value)}
    if isinstance(value, type) or inspect.isfunction(value):
        return {"implementation": _code_identity(value)}
    seen = set() if seen is None else seen
    if id(value) in seen:
        raise ValueError("cyclic quantization configuration")
    seen.add(id(value))
    try:
        compute_hash = getattr(value, "compute_hash", None)
        if callable(compute_hash):
            hashed = compute_hash()
            if type(hashed) is not str or not hashed:
                raise TypeError("configuration compute_hash must return a nonempty string")
            return {"type": _type_name(value), "hash": hashed}
        if isinstance(value, Mapping):
            if any(type(key) is not str for key in value):
                raise TypeError("quantization config dictionary keys must be strings")
            return {key: _config_identity(item, seen) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [_config_identity(item, seen) for item in value]
        if dataclasses.is_dataclass(value):
            return {
                "type": _type_name(value),
                "fields": {
                    field.name: _config_identity(getattr(value, field.name), seen)
                    for field in dataclasses.fields(value)
                },
            }
        if isinstance(value, SimpleNamespace):
            return {key: _config_identity(item, seen) for key, item in vars(value).items()}
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return {"type": _type_name(value), "fields": _config_identity(model_dump(), seen)}
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return {"type": _type_name(value), "fields": _config_identity(to_dict(), seen)}
        raise TypeError(f"configuration {_type_name(value)} requires an explicit scope adapter")
    finally:
        seen.remove(id(value))


def _components(model):
    """Expose documented pipeline components, never arbitrary object attributes."""
    import torch

    pipeline = getattr(model, "_pipeline", model)
    components = getattr(pipeline, "components", None)
    if not isinstance(components, Mapping):
        return {}
    return {
        name: module for name, module in components.items() if isinstance(module, torch.nn.Module)
    }


def _tensor_bytes(digest, tensor):
    """Stream exact source bits to CPU with a one-MiB staging bound."""
    import torch

    tensor = tensor.detach()
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        raise ValueError("quantization identity requires loaded strided state")
    elements = max(1, 1024**2 // tensor.element_size())
    if tensor.is_contiguous():
        flat = tensor.view(-1)
        for start in range(0, flat.numel(), elements):
            host = flat[start : start + elements].cpu().contiguous()
            digest.update(memoryview(host.view(torch.uint8).numpy()).cast("B"))
    elif tensor.numel() <= elements:
        host = tensor.cpu().contiguous()
        digest.update(memoryview(host.view(torch.uint8).numpy()).cast("B"))
    else:
        for part in tensor.unbind(0):
            _tensor_bytes(digest, part)


def model_scope(model, config=None, *, runtime="torch"):
    """Bind all model/component state and execution config, not only selected layers."""
    explicit = getattr(model, "metal_quantization_scope", None)
    if callable(explicit):
        scope = explicit(config)
        if type(scope) is not str or not scope or len(scope) > 1024:
            raise ValueError("model quantization scope must be a bounded nonempty string")
        return scope
    if runtime != "torch":
        raise NotImplementedError(
            "self-managed runtimes must provide metal_quantization_scope(config)"
        )
    import torch

    if not isinstance(model, torch.nn.Module):
        raise NotImplementedError(
            "non-Torch pipelines require an explicit quantization scope adapter"
        )
    implementations = {}

    def implementation(value):
        cls = type(value)
        if cls not in implementations:
            implementations[cls] = _code_identity(cls)
        return implementations[cls]

    descriptor = {
        "model": implementation(model),
        "runtime": runtime,
        "config": _config_identity(config),
        "training": model.training,
        "pipeline": (
            {
                "implementation": implementation(model._pipeline),
                "config": _config_identity(getattr(model._pipeline, "config", None)),
            }
            if hasattr(model, "_pipeline")
            else None
        ),
    }
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True, allow_nan=False).encode())
    roots = {
        "model": model,
        **{f"components.{key}": value for key, value in _components(model).items()},
    }
    for prefix, root in sorted(roots.items()):
        for name, module in root.named_modules(remove_duplicate=False):
            digest.update(
                json.dumps(
                    {
                        "root": prefix,
                        "module": name,
                        "implementation": implementation(module),
                        "training": module.training,
                    },
                    sort_keys=True,
                ).encode()
            )
        tensors = [
            ("parameter", name, value)
            for name, value in root.named_parameters(remove_duplicate=False)
        ]
        tensors += [
            ("buffer", name, value) for name, value in root.named_buffers(remove_duplicate=False)
        ]
        for kind, name, tensor in tensors:
            identity = {
                "kind": kind,
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "stride": list(tensor.stride()),
            }
            digest.update(json.dumps(identity, sort_keys=True).encode())
            _tensor_bytes(digest, tensor)
    return digest.hexdigest()


def _approved(profile):
    from .engine import _implementation_sha256

    calibration = profile.get("calibration") if type(profile) is dict else None
    if (
        type(calibration) is not dict
        or calibration.get("engine") != "DynamicEngine"
        or calibration.get("approved") is not True
        or calibration.get("implementation_sha256") != _implementation_sha256()
    ):
        raise ValueError("loading requires a matching approved DynamicEngine profile")


def configure_model_quantization(model, config=None, *, runtime="torch"):
    """Run once after loading; no discovery-time activation, search, or fallback."""
    mode = os.environ.get(MODE_ENV, "off")
    if mode not in ("off", "dynamic"):
        raise ValueError(f"{MODE_ENV} must be off or dynamic")
    managed = getattr(model, "_metal_quantization_metadata", None)
    if type(managed) is dict and managed.get("managed") is True:
        if managed.get("mode") == "dynamic":
            _approved(managed.get("profile"))
            if managed.get("approved") is not True:
                raise ValueError("private quantization adapter did not approve the installed state")
        elif managed.get("mode") != "off":
            raise ValueError("private quantization adapter reported an unsupported mode")
        return managed
    if mode == "off":
        return {"managed": False, "mode": "off", "approved": False}
    path = os.environ.get(PROFILE_ENV)
    if not path:
        raise ValueError(f"{PROFILE_ENV} is required when dynamic quantization is requested")
    scope = model_scope(model, config, runtime=runtime)
    profile = load_profile(path, scope=scope)
    _approved(profile)
    hook = getattr(model, "apply_metal_quantization_profile", None)
    if callable(hook):
        applied = hook(profile, scope=scope)
    elif runtime == "torch":
        from .adapter import DenseModelAdapter

        applied = DenseModelAdapter(model).apply(profile, scope=scope)
    else:
        raise NotImplementedError(
            "self-managed runtimes must provide apply_metal_quantization_profile(profile, scope=...)"
        )
    if type(applied) is not dict or set(applied) != set(profile["selected"]):
        raise RuntimeError("quantization adapter must return every applied projection")
    metadata = {
        "managed": True,
        "mode": "dynamic",
        "approved": True,
        "scope": scope,
        "profile": profile,
        "profile_path": str(path),
        "selected": list(applied),
        "runtime": runtime,
    }
    model._metal_quantization_metadata = metadata
    return metadata
