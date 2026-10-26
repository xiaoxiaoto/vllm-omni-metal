# SPDX-License-Identifier: Apache-2.0
"""Offline activation-sensitive W4/W8/source selection, never request-time switching.

Local numerical acceptance is not model-level approval. An adapter owns complete
pipeline identity, end-to-end evidence, and approval before production install.
"""

import hashlib
import math
from pathlib import Path

import torch

from . import linear

_THRESHOLD = 0.01
_EPSILON = 1e-12


def _implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _errors(reference, candidate):
    """Measure real outputs in FP64 on CPU, without rounding away BF16 errors."""
    if (
        not isinstance(reference, torch.Tensor)
        or not isinstance(candidate, torch.Tensor)
        or reference.shape != candidate.shape
        or reference.dtype != candidate.dtype
        or reference.numel() == 0
    ):
        raise ValueError("calibration output shape/dtype must match nonempty source reference")
    a = reference.detach().cpu().to(dtype=torch.float64)
    b = candidate.detach().cpu().to(dtype=torch.float64)
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError("calibration outputs must be finite")
    delta = a - b
    result = {
        "relative_l2": float(torch.linalg.vector_norm(delta))
        / max(float(torch.linalg.vector_norm(a)), _EPSILON),
        "scaled_max_abs": float(delta.abs().max()) / max(float(a.abs().max()), _EPSILON),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("calibration errors must be finite")
    return result


def _captures(modules, captures):
    from .capture import Captures, DiskCaptures

    if isinstance(captures, DiskCaptures):
        captures.validate_source(modules)
        if set(captures.names) != set(modules):
            raise ValueError("captures must cover exactly every candidate module")
        ordered = {name: captures.steps(name) for name in modules}
        common_steps = None
        for steps in ordered.values():
            if len(steps) < 4:
                raise ValueError("each candidate requires at least four distinct captured steps")
            if any(type(step) is not int or step < 0 for step in steps):
                raise ValueError("captured steps must be distinct nonnegative integers")
            if common_steps is not None and set(steps) != common_steps:
                raise ValueError("every candidate must have the same captured steps")
            common_steps = set(steps)
        metadata = captures.summary()["captures"]
        metadata = {
            name: [
                {key: sample[key] for key in ("step", "shape", "dtype", "sha256")}
                for sample in samples
            ]
            for name, samples in metadata.items()
        }
        return ordered, metadata

    if isinstance(captures, Captures):
        captures.validate_source(modules)
    if type(captures) not in (dict, Captures) or set(captures) != set(modules):
        raise ValueError("captures must cover exactly every candidate module")
    metadata = {}
    ordered = {}
    common_steps = None
    for name, module in modules.items():
        samples = captures[name]
        if type(samples) is not list or len(samples) < 4:
            raise ValueError("each candidate requires at least four distinct captured steps")
        steps = {}
        for sample in samples:
            if type(sample) not in (tuple, list) or len(sample) != 2:
                raise ValueError("captures must contain (step, CPU tensor) pairs")
            step, tensor = sample
            if type(step) is not int or step < 0 or step in steps:
                raise ValueError("captured steps must be distinct nonnegative integers")
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.layout != torch.strided
                or tensor.dtype != module.weight.dtype
                or tensor.requires_grad
                or tensor.ndim < 1
                or tensor.numel() == 0
                or tensor.shape[-1] != module.in_features
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(
                    "captures require finite nonempty CPU activations of matching source dtype/width"
                )
            steps[step] = tensor
        if common_steps is not None and set(steps) != common_steps:
            raise ValueError("every candidate must have the same captured steps")
        common_steps = set(steps)
        ordered[name] = sorted(steps.items())
        metadata[name] = []
        for step, tensor in ordered[name]:
            descriptor = {"step": step, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            digest = hashlib.sha256(linear._encoded(descriptor))
            digest.update(
                memoryview(tensor.detach().contiguous().view(torch.uint8).numpy()).cast("B")
            )
            metadata[name].append({**descriptor, "sha256": digest.hexdigest()})
    return ordered, metadata


def _audit(storage_bytes, errors):
    result = {
        "storage_bytes": storage_bytes,
        "relative_l2": max(error["relative_l2"] for error in errors),
        "scaled_max_abs": max(error["scaled_max_abs"] for error in errors),
        "steps": errors,
    }
    result["passed"] = (
        result["relative_l2"] <= _THRESHOLD and result["scaled_max_abs"] <= _THRESHOLD
    )
    return result


def _select(candidates):
    # Source precision wins ties: no storage benefit means no quantization.
    return min(
        (name for name, audit in candidates.items() if audit["passed"]),
        key=lambda name: (
            candidates[name]["storage_bytes"],
            {"bf16": 0, "fp16": 0, "fp32": 0, "w4a16": 1, "w8a16": 2}[name],
        ),
    )


def _trial_reason(module):
    # Source dtype and shape are never changed to make a candidate eligible.
    if module.weight.dtype == torch.float32:
        return "FP32 source is retention-only"
    if module.in_features % linear.GROUP_SIZE:
        return "group64 alignment required"
    return None


def _streamed_candidates(module, captures, name, steps, source_precision, source_bytes):
    """Reuse one full sample/reference for both packed trials, then release it."""
    reason = _trial_reason(module)
    quantized = {}
    errors = {source_precision: []}
    candidates = {}
    try:
        for precision, bits in (("w4a16", 4), ("w8a16", 8)):
            if reason is not None:
                candidates[precision] = {
                    "passed": False,
                    "reason": reason,
                    "storage_bytes": None,
                }
            else:
                quantized[precision] = linear.QuantizedLinear.from_linear(module, bits=bits)
                errors[precision] = []
        for step in steps:
            host = captures.read(name, step)
            if (
                host.device.type != "cpu"
                or host.layout != torch.strided
                or host.dtype != module.weight.dtype
                or host.requires_grad
                or host.ndim < 1
                or host.numel() == 0
                or host.shape[-1] != module.in_features
                or not bool(torch.isfinite(host).all())
            ):
                raise ValueError(
                    "captures require finite nonempty CPU activations of matching source dtype/width"
                )
            tensor = host.to(device=module.weight.device).contiguous()
            del host
            reference = module(tensor).detach().cpu()
            errors[source_precision].append({"step": step, **_errors(reference, reference)})
            for precision, candidate in quantized.items():
                errors[precision].append({"step": step, **_errors(reference, candidate(tensor))})
            del tensor, reference
        candidates[source_precision] = _audit(source_bytes, errors[source_precision])
        for precision, candidate in quantized.items():
            candidates[precision] = _audit(candidate.storage_bytes, errors[precision])
        return candidates
    finally:
        quantized.clear()


class DynamicEngine:
    """Stateless offline PTQ search and explicit, transactional profile application.

    Search runs each W4/W8 candidate against original source outputs on every
    captured step. DiskCaptures streams one full activation/reference at a time,
    sharing it between the two packed candidates without accumulating samples.
    The existing in-memory capture API keeps its independent trial order.
    Neither path mutates the source tree or retains floating weights. FP32 is
    retained only; all-source results are allowed without claiming a benefit.
    """

    @staticmethod
    @torch.inference_mode()
    def search(modules, captures, *, scope):
        linear._scope(scope)
        if type(modules) is not dict or not modules:
            raise ValueError("search requires a nonempty candidate module dict")
        selected = list(modules)
        source_precisions = {
            name: {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}.get(
                getattr(getattr(module, "weight", None), "dtype", None), "bf16"
            )
            for name, module in modules.items()
        }
        bits = linear._modules(modules, selected, source_precisions)
        for name, module in modules.items():
            linear._linear(module, mps=True, bits=bits[name])
        ordered, fingerprints = _captures(modules, captures)
        from .capture import DiskCaptures

        streamed = isinstance(captures, DiskCaptures)
        audits = {}
        precisions = {}
        for name, module in modules.items():
            source_bytes = int(
                sum(
                    t.numel() * t.element_size()
                    for t in (module.weight, module.bias)
                    if t is not None
                )
            )
            if streamed:
                candidates = _streamed_candidates(
                    module, captures, name, ordered[name], source_precisions[name], source_bytes
                )
            else:
                # Preserve the established in-memory API and independent trial order.
                inputs = [
                    (step, tensor.to(device=module.weight.device).contiguous())
                    for step, tensor in ordered[name]
                ]
                references = [(step, module(tensor).detach().cpu()) for step, tensor in inputs]
                zero_errors = [
                    {"step": step, **_errors(reference, reference)}
                    for step, reference in references
                ]
                candidates = {source_precisions[name]: _audit(source_bytes, zero_errors)}
                for precision, trial_bits in (("w4a16", 4), ("w8a16", 8)):
                    reason = _trial_reason(module)
                    if reason is not None:
                        candidates[precision] = {
                            "passed": False,
                            "reason": reason,
                            "storage_bytes": None,
                        }
                        continue
                    quantized = linear.QuantizedLinear.from_linear(module, bits=trial_bits)
                    errors = [
                        {"step": step, **_errors(reference, quantized(tensor))}
                        for (step, tensor), (_, reference) in zip(inputs, references, strict=True)
                    ]
                    candidates[precision] = _audit(quantized.storage_bytes, errors)
                    del quantized
                del inputs, references
            chosen = _select(candidates)
            precisions[name] = chosen
            audits[name] = {
                "source_bytes": source_bytes,
                "candidates": candidates,
                "selected": chosen,
            }
        if streamed:
            captures.validate_source(modules)
        calibration = {
            "engine": "DynamicEngine",
            "implementation_sha256": _implementation_sha256(),
            "approved": False,
            "thresholds": {"relative_l2": _THRESHOLD, "scaled_max_abs": _THRESHOLD},
            "audits": audits,
            "captures": fingerprints,
            "precisions": precisions,
            "local_pass": True,
        }
        profile = linear.build_profile(
            modules, selected, scope=scope, calibration=calibration, precisions=precisions
        )
        return {"profile": profile, "audits": audits, "precisions": precisions, "local_pass": True}

    @staticmethod
    @torch.inference_mode()
    def apply(root, profile, *, scope, require_approved=True):
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
        # apply_profile validates schema, source, environment and aliases before
        # constructing all replacements and atomically committing the module map.
        return linear.apply_profile(root, profile, scope=scope)
