# SPDX-License-Identifier: Apache-2.0
"""Explicit FP16 linear ANE graphs; offline preparation, load-only production.

Build tools: ``python -m vllm_omni_metal.ane.build --bridge``.
Real NE numerical smoke: ``python -m vllm_omni_metal.ane.projection
--inputs 1024 --outputs 1024 --rows 513 --bias --dtype bfloat16``.
This smoke prepares a random graph and audits it, but never approves a pipeline.
Only adapters measuring identical full-pipeline work may publish profiles.
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F

from vllm_omni_metal.ane import cache as artifacts
from vllm_omni_metal.ane.allocation import RowAllocator
from vllm_omni_metal.ane.build import _budget, _lock, _seal
from vllm_omni_metal.ane.runtime import DenseSwiGLU, _Bridge, _errors


def _shape(weight, bias):
    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 2
        or min(weight.shape) <= 0
        or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not weight.is_contiguous()
    ):
        raise ValueError("a contiguous FP16/BF16/FP32 [out,in] source weight is required")
    if bias is not None and (
        not isinstance(bias, torch.Tensor)
        or bias.shape != (weight.shape[0],)
        or bias.dtype != weight.dtype
        or bias.device != weight.device
        or not bias.is_contiguous()
    ):
        raise ValueError("bias must be contiguous [out] with the source dtype/device")
    size = (weight.numel() + (0 if bias is None else bias.numel())) * 2
    if size > artifacts.MAX_WEIGHTS:
        raise ValueError("projection exceeds the 256 MiB FP16 graph weight limit")
    return weight.shape[1], weight.shape[0]


def _host(value):
    # MPS view readback and conversion must be separate to preserve source bytes.
    return value.detach().to(device="cpu", copy=True)


def _key(weight, bias, scope):
    inputs, outputs = _shape(weight, bias)
    digest = hashlib.sha256(
        repr(
            (
                "linear-bias-v1",
                artifacts.CHUNK,
                inputs,
                outputs,
                str(weight.dtype),
                bias is not None,
            )
        ).encode()
    )
    for value in (weight, bias):
        if value is not None:
            digest.update(memoryview(_host(value).view(torch.uint8).numpy()).cast("B"))
    return artifacts.group_key([digest.hexdigest()], scope)


def _passing(check):
    return isinstance(check, dict) and all(
        type(check.get(name)) in (int, float)
        and math.isfinite(check[name])
        and 0 <= check[name] <= 0.01
        for name in ("relative_l2", "scaled_max_abs")
    )


def _verify(key, inputs, outputs, bias, device):
    directory = artifacts.model_directory(key)
    artifacts.private_directory(directory)
    data = artifacts.read_json(directory / "manifest.json")
    if (
        data.get("key") != key
        or data.get("kind") != "linear-bias-v1"
        or data.get("shape") != [artifacts.CHUNK, inputs, outputs]
        or data.get("bias") is not bias
        or data.get("device") != device
        or data.get("ane_preferred") is not True
        or not isinstance(data.get("numerical_tests"), list)
        or len(data["numerical_tests"]) != 2
        or not all(_passing(check) for check in data["numerical_tests"])
    ):
        raise RuntimeError("projection artifact does not match its execution contract")
    path = directory / "module.mlmodelc"
    artifacts.verify_files(
        path, data.get("files"), ["model.mil", "coremldata.bin", "weights/weight.bin"]
    )
    return path


def _load(bridge, path, inputs, outputs):
    function = bridge.lib.projection_load
    function.argtypes = [ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    function.restype = ctypes.c_int32
    handle = function(str(path).encode(), artifacts.CHUNK, inputs, outputs)
    if handle < 0:
        raise RuntimeError("CoreML could not load the prepared linear projection")
    return handle


def _predict(bridge, handle, incoming, output):
    if (
        bridge.lib.ffn_predict(
            handle, incoming.data_ptr(), output.data_ptr(), incoming.numel(), output.numel()
        )
        != 0
    ):
        raise RuntimeError("CoreML linear projection prediction failed")
    if not torch.isfinite(output).all():
        raise RuntimeError("non-finite FP16 linear output; no bias-unsafe rescaling")


@torch.inference_mode()
def prepare_projection(weight, bias=None, *, scope):
    """Prepare and numerically validate a real, NE-preferred constant linear graph.

    Tools must already have been explicitly built. No source tensor is cast in
    place; no other projection (in particular a quantized down) is retained.
    """
    inputs, outputs = _shape(weight, bias)
    artifacts.group_key([], scope)
    if weight.device.type != "mps":
        raise ValueError("offline projection preparation requires an MPS source")
    _budget(
        4 * artifacts.MAX_WEIGHTS, artifacts.RESERVE + 3 * (weight.numel() * weight.element_size())
    )
    root = artifacts.ensure_cache()
    bridge = _Bridge()
    key = _key(weight, bias, scope)
    destination = artifacts.model_directory(key)
    with _lock(root):
        if destination.exists() or destination.is_symlink():
            _verify(key, inputs, outputs, bias is not None, bridge.device)
            return key
        host = _host(weight)
        host_bias = None if bias is None else _host(bias)
        with tempfile.TemporaryDirectory(prefix="projection-", dir=root) as temporary:
            temporary = Path(temporary)
            model = temporary / "module.mlmodelc"
            payload = temporary / "slabs.bin"
            bk = artifacts.TILE if inputs % artifacts.TILE == 0 else inputs
            bn = artifacts.TILE if outputs % artifacts.TILE == 0 else outputs
            with payload.open("wb") as file:
                for col in range(0, outputs, bn):
                    for reduction in range(0, inputs, bk):
                        tile = (
                            host[col : col + bn, reduction : reduction + bk].half().T.contiguous()
                        )
                        if not torch.isfinite(tile).all():
                            raise ValueError("source weight is outside finite FP16 range")
                        tile.numpy().tofile(file)
                if host_bias is not None:
                    value = host_bias.half()
                    if not torch.isfinite(value).all():
                        raise ValueError("source bias is outside finite FP16 range")
                    value.numpy().tofile(file)
            subprocess.run(
                [
                    str(root / "emit"),
                    str(artifacts.CHUNK),
                    str(inputs),
                    str(outputs),
                    str(payload),
                    str(model),
                    "projection",
                    "bias" if bias is not None else "none",
                ],
                check=True,
                timeout=60,
            )
            payload.unlink()
            _seal(model)
            if bridge.lib.ffn_plan(str(model).encode()) != 1:
                raise RuntimeError("CoreML linear matmuls do not all prefer Neural Engine")
            handle = _load(bridge, model, inputs, outputs)
            checks = []
            try:
                generator = torch.Generator(device="cpu").manual_seed(1729)
                for scale in (0.125, 1.0):
                    x = torch.randn((artifacts.CHUNK, inputs), generator=generator).mul_(scale)
                    incoming = x.half().contiguous()
                    result = torch.empty((artifacts.CHUNK, outputs), dtype=torch.float16)
                    _predict(bridge, handle, incoming, result)
                    expected = F.linear(
                        incoming.float(),
                        host.float(),
                        None if host_bias is None else host_bias.float(),
                    )
                    rel, peak = _errors(result, expected)
                    check = {"relative_l2": rel, "scaled_max_abs": peak}
                    if not _passing(check):
                        raise RuntimeError(
                            f"linear ANE graph failed unchanged 1% self-test: {check}"
                        )
                    checks.append(check)
            finally:
                bridge.lib.ffn_unload(handle)
            names = ["model.mil", "coremldata.bin", "weights/weight.bin"]
            artifacts.write_json(
                temporary / "manifest.json",
                {
                    "key": key,
                    "kind": "linear-bias-v1",
                    "shape": [artifacts.CHUNK, inputs, outputs],
                    "source_dtype": str(weight.dtype),
                    "bias": bias is not None,
                    "device": bridge.device,
                    "ane_preferred": True,
                    "numerical_tests": checks,
                    "files": {name: artifacts.checksum(model / name) for name in names},
                },
            )
            os.rename(temporary, destination)
    return key


def _policy(path):
    artifacts.checked_file(path, 1024**2)
    if stat.S_IMODE(path.lstat().st_mode) != 0o600:
        raise RuntimeError("projection policy must be owned mode 0600")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate policy key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number

    try:
        data = json.loads(
            path.read_text(),
            object_pairs_hook=pairs,
            parse_constant=invalid,
            parse_float=finite_float,
        )
    except (ValueError, UnicodeError) as error:
        raise RuntimeError("invalid strict projection policy") from error
    _policy_header(data)
    return data


def _policy_header(data):
    if (
        type(data) is not dict
        or set(data) != {"schema", "group", "device", "kind", "profiles"}
        or type(data["schema"]) is not int
        or data["schema"] != artifacts.SCHEMA
        or type(data["group"]) is not str
        or artifacts.KEY.fullmatch(data["group"]) is None
        or type(data["device"]) is not str
        or not data["device"]
        or data["kind"] != "linear-bias-v1"
        or type(data["profiles"]) is not dict
    ):
        raise RuntimeError("invalid projection policy header")
    try:
        json.dumps(data, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise RuntimeError("projection policy requires finite plain JSON") from error


def _profiles(records, inputs):
    profiles = artifacts.validate_profiles(records)
    shapes = {}
    for key, chunks in profiles.items():
        if not chunks:
            continue
        check = records[artifacts.profile_label(*key)].get("numerical_check")
        if not _passing(check):
            raise RuntimeError("projection policy lacks full-output dual 1% evidence")
        shape = check.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) < 2
            or any(type(v) is not int or v <= 0 for v in shape)
            or shape[-1] != inputs
            or math.prod(shape[:-1]) != key[0]
            or check.get("ane_chunks") != chunks
        ):
            raise RuntimeError("projection policy lacks shape-bound actual ANE hits")
        shapes[key] = tuple(shape)
    return profiles, shapes


class DenseProjection:
    """One immutable, unquantized source projection and one persistent NE worker.

    ``native`` is the original Tensor-returning linear callable, never a whole
    FFN or a bound quantized down. It receives the original rank on both routes.
    Production failures on an approved split raise; missing approval stays GPU.
    """

    def __init__(
        self,
        weight,
        bias=None,
        *,
        scope,
        precision="fp16",
        _benchmark=False,
        _staged_policy=None,
    ):
        self.inputs, self.outputs = _shape(weight, bias)
        artifacts.group_key([], scope)
        if precision != "fp16":
            raise ValueError("DenseProjection requires explicit precision='fp16' approximation")
        mode = os.environ.get("VLLM_OMNI_METAL_ANE", "auto")
        if mode not in ("auto", "off"):
            raise ValueError("VLLM_OMNI_METAL_ANE must be auto or off")
        if _staged_policy is not None and (
            _benchmark or mode == "off" or weight.device.type != "mps"
        ):
            raise ValueError("staged replay requires immutable MPS ANE execution")
        self.weight, self.bias = weight, bias
        self.scope, self.precision = scope, precision
        self._versions = DenseSwiGLU._version(tuple(v for v in (weight, bias) if v is not None))
        self.metrics, self.checks = Counter(), {}
        self.disabled = self._closed = False
        self._benchmark, self._benchmark_chunks, self._audit = _benchmark, 0, True
        self._auto_allocation = False
        self._allocators, self._allocation_chunks, self._allocation_used = {}, {}, {}
        self.allocation_records = []
        self._bridge = self._pool = None
        self._handle = None
        self._identity = self.identity = None
        self.profiles, self._input_shapes = {}, {}
        if mode == "off" or weight.device.type != "mps":
            return
        artifacts.check_existing_cache()
        root = artifacts.cache_root()
        if not (root / "tools.json").exists() and not (root / "tools.json").is_symlink():
            if _benchmark or _staged_policy is not None:
                raise RuntimeError("projection tools must be explicitly built offline")
            return
        required = 3 * (weight.numel() * weight.element_size())
        if psutil.virtual_memory().available < artifacts.RESERVE + required:
            raise RuntimeError("projection loading exceeds 4 GiB memory headroom")
        self._identity = self.identity = _key(weight, bias, scope)
        if not _benchmark:
            if _staged_policy is None:
                path = root / f"{self.identity}.json"
                if not path.exists() and not path.is_symlink():
                    return
                policy = _policy(path)
            else:
                policy = _staged_policy
                _policy_header(policy)
            if policy["group"] != self.identity:
                raise RuntimeError("projection policy identity mismatch")
            self.profiles, self._input_shapes = _profiles(policy["profiles"], self.inputs)
            if not any(self.profiles.values()):
                if _staged_policy is not None:
                    raise ValueError("staged replay requires a nonzero audited ANE split")
                return
        bridge = _Bridge()
        if not _benchmark and policy.get("device") != bridge.device:
            raise RuntimeError("projection policy belongs to another device")
        path = _verify(self.identity, self.inputs, self.outputs, bias is not None, bridge.device)
        self._bridge = bridge
        self._handle = _load(bridge, path, self.inputs, self.outputs)
        try:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omni-projection")
        except BaseException:
            self.close()
            raise

    def set_benchmark_chunks(self, chunks):
        if not self._benchmark:
            raise RuntimeError("production projection splits are immutable")
        if type(chunks) is str and chunks == "auto":
            if not self._audit:
                raise RuntimeError("adaptive allocation requires an open offline audit")
            self._auto_allocation = True
            return
        if type(chunks) is not int or chunks < 0:
            raise ValueError("benchmark chunks must be a nonnegative integer")
        self._auto_allocation = False
        self._benchmark_chunks = chunks

    def finish_audit(self):
        if (
            not self._benchmark
            or self.disabled
            or self._closed
            or self._pool is None
            or not self.checks
            or (self._auto_allocation and not self._allocation_used)
            or (
                not self._auto_allocation
                and not any(key[1] == self._benchmark_chunks for key in self.checks)
            )
        ):
            raise RuntimeError("a healthy audited offline projection is required")
        if self._auto_allocation:
            self._allocation_chunks = dict(self._allocation_used)
        self._audit = False

    def __call__(self, x, *, native, context=None):
        if self._closed:
            raise RuntimeError("DenseProjection is closed")
        if torch.is_grad_enabled():
            raise RuntimeError("DenseProjection requires inference/no-grad")
        if context is not None and (type(context) is not int or not 0 <= context < 2**31):
            raise ValueError("pipeline context must be a bounded nonnegative integer")
        if not isinstance(x, torch.Tensor) or x.ndim < 1:
            raise ValueError("projection activations must be a non-scalar Tensor")
        self.metrics["calls"] += 1
        rows = math.prod(x.shape[:-1])
        chunks = (
            self._benchmark_chunks if self._benchmark else self.profiles.get((rows, context), 0)
        )
        allocation_key = rows, context
        if self._benchmark and self._auto_allocation and rows > artifacts.CHUNK:
            if not self._audit and allocation_key not in self._allocation_chunks:
                raise RuntimeError("timed allocation requires an audited shape/context")
            chunks = self._allocation_chunks.get(
                allocation_key,
                min(
                    (rows - 1) // artifacts.CHUNK,
                    max(1, int(rows / (2 * artifacts.CHUNK) + 0.5)),
                ),
            )
        if self.disabled or self._pool is None or chunks == 0 or rows <= artifacts.CHUNK:
            self.metrics.update(native_calls=1, gpu_rows=rows)
            return native(x)
        if (
            x.ndim < 2
            or x.shape[-1] != self.inputs
            or x.dtype != self.weight.dtype
            or x.device != self.weight.device
            or not x.is_contiguous()
        ):
            raise ValueError("approved projection requires matching contiguous source activations")
        if type(chunks) is not int or not 1 <= chunks <= (rows - 1) // artifacts.CHUNK:
            raise ValueError("ANE must take full 256-row chunks and retain a GPU row")
        if not self._benchmark and self._input_shapes.get((rows, context)) != tuple(x.shape):
            raise RuntimeError("activation shape differs from the approved projection")
        key = rows, chunks, context
        if (
            self._benchmark
            and not self._audit
            and (key not in self.checks or self.checks[key]["shape"] != list(x.shape))
        ):
            raise RuntimeError("timed execution requires a completed audit for this shape")
        values = tuple(v for v in (self.weight, self.bias) if v is not None)
        if DenseSwiGLU._version(values) != self._versions:
            self._fail("source projection changed after binding")
        if (
            psutil.virtual_memory().available
            < artifacts.RESERVE + x.numel() * 12 + rows * self.outputs * 8
        ):
            self._fail("projection execution exceeds 4 GiB memory headroom")
        try:
            result = self._split(x, chunks, native)
            if self._benchmark and self._audit:
                expected = native(x)
                if not isinstance(expected, torch.Tensor) or expected.shape != result.shape:
                    raise ValueError("native projection must return the original linear Tensor")
                rel, peak = _errors(result, expected)
                old = self.checks.get(key, {})
                if old and old["shape"] != list(x.shape):
                    raise RuntimeError("shape changed within a projection calibration key")
                check = {
                    "relative_l2": max(rel, old.get("relative_l2", 0)),
                    "scaled_max_abs": max(peak, old.get("scaled_max_abs", 0)),
                    "shape": list(x.shape),
                    "ane_chunks": chunks,
                }
                self.checks[key] = check
                if not _passing(check):
                    raise RuntimeError(f"full projection failed dual 1% gate: {check}")
            if self._benchmark and self._auto_allocation and self._audit:
                allocator = self._allocators.setdefault(allocation_key, RowAllocator())
                sample = self._allocation_sample
                next_chunks = allocator.observe(
                    total_rows=rows,
                    ane_chunks=chunks,
                    gpu_seconds=sample["gpu_seconds"],
                    ane_seconds=sample["ane_seconds"],
                )
                self._allocation_used[allocation_key] = chunks
                self._allocation_chunks[allocation_key] = next_chunks
                self.allocation_records.append(
                    {
                        "rows": rows,
                        "context": context,
                        "chunks": chunks,
                        "next_chunks": next_chunks,
                        **sample,
                        "gpu_seconds_per_row": allocator.gpu_seconds_per_row,
                        "ane_seconds_per_chunk": allocator.ane_seconds_per_chunk,
                        "predicted_seconds": allocator.prediction(rows, chunks),
                        "next_predicted_seconds": allocator.prediction(rows, next_chunks),
                    }
                )
            return result
        except BaseException:
            self.disabled = True
            self.metrics["failures"] += 1
            self.close()
            raise

    def _fail(self, reason):
        self.disabled = True
        self.metrics["failures"] += 1
        self.close()
        raise RuntimeError(reason)

    def _split(self, x, chunks, native):
        flat = x.view(-1, self.inputs)
        ane_rows = chunks * artifacts.CHUNK
        gpu_rows = flat.shape[0] - ane_rows
        tuning = self._benchmark and self._auto_allocation and self._audit
        if tuning:
            torch.mps.synchronize()
            transfer_start = time.perf_counter()
        incoming = _host(flat[gpu_rows:]).half().contiguous()
        if not torch.isfinite(incoming).all():
            raise RuntimeError("activation outside finite FP16 range")
        answer = torch.empty((ane_rows, self.outputs), dtype=torch.float16)
        incoming_seconds = time.perf_counter() - transfer_start if tuning else 0.0

        @torch.inference_mode()
        def predict():
            started = time.perf_counter() if tuning else 0.0
            for start in range(0, ane_rows, artifacts.CHUNK):
                _predict(
                    self._bridge,
                    self._handle,
                    incoming[start : start + artifacts.CHUNK],
                    answer[start : start + artifacts.CHUNK],
                )
            return time.perf_counter() - started if tuning else 0.0

        future = self._pool.submit(predict)
        result = torch.empty((flat.shape[0], self.outputs), dtype=x.dtype, device=x.device)
        # Preserve rank without padding/repeating workload: use one rank-preserving
        # slice per original leading-index block crossing the GPU/ANE boundary.
        try:
            gpu_start = time.perf_counter() if tuning else 0.0
            block_rows = math.prod(x.shape[1:-1])
            complete, remainder = divmod(gpu_rows, block_rows)
            if complete:
                value = native(x[:complete])
                result[: complete * block_rows].copy_(value.reshape(-1, self.outputs))
            if remainder:
                # A single flattened partial block is represented with singleton
                # leading axes, so linear/packed adapters retain their rank ABI.
                partial = flat[complete * block_rows : gpu_rows].reshape(
                    *((1,) * (x.ndim - 2)), remainder, self.inputs
                )
                value = native(partial)
                result[complete * block_rows : gpu_rows].copy_(value.reshape(-1, self.outputs))
            if tuning:
                torch.mps.synchronize()
                gpu_seconds = time.perf_counter() - gpu_start
        finally:
            ane_seconds = future.result()
        if tuning:
            transfer_start = time.perf_counter()
        result[gpu_rows:].copy_(answer.to(device=x.device, dtype=x.dtype))
        if tuning:
            torch.mps.synchronize()
            self._allocation_sample = {
                "gpu_seconds": gpu_seconds,
                "ane_seconds": incoming_seconds
                + ane_seconds
                + time.perf_counter()
                - transfer_start,
            }
        self.metrics.update(ane_chunks=chunks, ane_rows=ane_rows, gpu_rows=gpu_rows)
        return result.view(*x.shape[:-1], self.outputs)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._handle is not None:
            self._bridge.lib.ffn_unload(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def publish_projection_profile(
    projection, rows, baseline, candidates, *, context=None, publish=True
):
    """Publish shape-bound full-output audit plus identical full-pipeline timings.

    The caller supplies real full-pipeline samples, as for publish_tiled_profile;
    projection timings alone are not a valid whole-pipeline calibration.
    Zero selections are GPU records, never joint-execution approvals.
    """
    p = projection
    if type(publish) is not bool:
        raise ValueError("publish must be an explicit boolean")
    if (
        not isinstance(p, DenseProjection)
        or not p._benchmark
        or p.disabled
        or p._closed
        or p._bridge is None
        or p.identity is None
    ):
        raise ValueError("a healthy offline projection is required")
    record = artifacts.select_profile(rows, baseline, candidates)
    chosen = record["chunks"]
    if chosen:
        check = p.checks.get((rows, chosen, context))
        if (
            not _passing(check)
            or check.get("ane_chunks") != chosen
            or p.metrics["ane_chunks"] < chosen
        ):
            raise ValueError("full projection dual 1% audit and actual ANE hits are required")
        record["numerical_check"] = dict(check)
    else:
        record["numerical_check"] = None
    if publish:
        publish_staged_projection_policy(
            {
                "schema": artifacts.SCHEMA,
                "group": p.identity,
                "device": p._bridge.device,
                "kind": "linear-bias-v1",
                "profiles": {artifacts.profile_label(rows, context): record},
            }
        )
    return record


def publish_staged_projection_policy(policy):
    """Commit verified subordinate records only after the adapter supervisor succeeds."""
    _policy_header(policy)
    root = artifacts.cache_root()
    key = policy["group"]
    with _lock(root):
        manifest = artifacts.read_json(artifacts.model_directory(key) / "manifest.json")
        shape = manifest.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or shape[0] != artifacts.CHUNK
            or any(type(value) is not int or value <= 0 for value in shape)
            or type(manifest.get("bias")) is not bool
        ):
            raise RuntimeError("prepared projection shape is invalid")
        bridge = _Bridge()
        if policy["device"] != bridge.device:
            raise RuntimeError("projection policy belongs to another device")
        _verify(key, shape[1], shape[2], manifest["bias"], bridge.device)
        _profiles(policy["profiles"], shape[1])
        path = root / f"{key}.json"
        records = {}
        if path.exists() or path.is_symlink():
            previous = _policy(path)
            if previous["group"] != key or previous["device"] != bridge.device:
                raise RuntimeError("existing projection policy identity mismatch")
            _profiles(previous["profiles"], shape[1])
            records.update(previous["profiles"])
        records.update(policy["profiles"])
        _profiles(records, shape[1])
        artifacts.write_json(path, {**policy, "profiles": records})


def main():
    parser = argparse.ArgumentParser(
        description="Explicit random real-NE linear numerical smoke (no approval)"
    )
    parser.add_argument("--inputs", type=int, default=1024)
    parser.add_argument("--outputs", type=int, default=1024)
    parser.add_argument("--rows", type=int, default=513)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--allocation", choices=("fixed", "predictive"), default="fixed")
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()
    if args.rows <= artifacts.CHUNK:
        parser.error("--rows must retain at least one GPU row")
    if args.warmup < 2:
        parser.error("--warmup must be at least two")
    dtype = getattr(torch, args.dtype)
    with torch.inference_mode():
        generator = torch.Generator().manual_seed(1729)
        weight = (
            torch.randn(args.outputs, args.inputs, generator=generator) / math.sqrt(args.inputs)
        ).to(device="mps", dtype=dtype)
        bias = (
            torch.randn(args.outputs, generator=generator).to(device="mps", dtype=dtype)
            if args.bias
            else None
        )
        x = torch.randn(args.rows, args.inputs, generator=generator).to(device="mps", dtype=dtype)
        identity = prepare_projection(weight, bias, scope="explicit-linear-numerical-smoke-v1")
        with DenseProjection(
            weight, bias, scope="explicit-linear-numerical-smoke-v1", _benchmark=True
        ) as p:
            p.set_benchmark_chunks("auto" if args.allocation == "predictive" else 1)
            for _ in range(args.warmup if args.allocation == "predictive" else 1):
                p(x, native=lambda value: F.linear(value, weight, bias))
            if args.allocation == "predictive":
                p.finish_audit()
                for _ in range(2):
                    p(x, native=lambda value: F.linear(value, weight, bias))
            torch.mps.synchronize()
            print(
                json.dumps(
                    {
                        "identity": identity,
                        "metrics": dict(p.metrics),
                        "checks": list(p.checks.values()),
                        "allocation": args.allocation,
                        "allocation_records": p.allocation_records,
                        "approved": False,
                    },
                    allow_nan=False,
                )
            )


if __name__ == "__main__":
    main()
