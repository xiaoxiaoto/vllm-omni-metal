# SPDX-License-Identifier: Apache-2.0
"""Generic dense SwiGLU with opt-in FP16 math and offline-profiled GPU/ANE splits.

Default auto consumes approved full-pipeline profiles only. No model registry,
model-name routing, graph emission, compilation or online performance probing.
"""

import ctypes
import hashlib
import json
import logging
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import psutil
import torch
import torch.nn.functional as F

from vllm_omni_metal.ane import cache as artifacts

logger = logging.getLogger(__name__)


def execution_scope(revision, config):
    """Bind pipeline calibration to execution settings, not a model name."""
    from vllm_omni_metal.ops.mps_attention import get_mps_ops
    from vllm_omni_metal.ops.vpipe_kernels import _artifact_path

    cache = config.cache_config
    descriptor = {
        "revision": revision,
        "dtype": str(config.model_config.dtype),
        "context": config.model_config.max_model_len,
        "max_num_seqs": config.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        "kv_bytes": cache.kv_cache_memory_bytes,
        "kv_fraction": cache.gpu_memory_utilization,
        "block_size": cache.block_size,
        "attention": str(config.attention_config.backend),
        "torch_stream_attention": get_mps_ops() is not None,
        "aot_rope": _artifact_path(config.model_config.dtype).is_file(),
        "custom_ops": config.compilation_config.custom_ops,
        "ir_priorities": asdict(config.kernel_config.ir_op_priority),
    }
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()


class _Bridge:
    def __init__(self):
        root = artifacts.cache_root()
        artifacts.private_directory(root)
        data = artifacts.read_json(root / "tools.json")
        if data.get("identity") != root.name:
            raise RuntimeError("ANE tools do not match this backend/OS/Torch")
        artifacts.verify_files(root, data.get("files"), ["bridge.dylib", "emit"])
        self.lib = ctypes.CDLL(str(root / "bridge.dylib"))
        declarations = {
            "ffn_device": ([], ctypes.c_char_p),
            "ffn_plan": ([ctypes.c_char_p], ctypes.c_int32),
            "ffn_load": ([ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64], ctypes.c_int32),
            "ffn_unload": ([ctypes.c_int32], None),
            "ffn_stage": (
                [ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int64],
                ctypes.c_int32,
            ),
            "ffn_predict": (
                [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64],
                ctypes.c_int32,
            ),
        }
        for name, (args, result) in declarations.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        self.device = self.lib.ffn_device().decode()
        if self.device != data.get("device"):
            raise RuntimeError("ANE tools/calibration belong to another device")

    def load(self, path, hidden):
        handle = self.lib.ffn_load(str(path).encode(), artifacts.CHUNK, hidden)
        if handle < 0:
            raise RuntimeError("CoreML could not load the prepared ANE graph")
        return handle


def _headroom_retry(predict, rescale, scale):
    if scale not in (1, 4, 16, 64, 256, 1024, 4096):
        raise ValueError("invalid ANE headroom scale")
    retries = 0
    while True:
        raw = predict()
        if torch.isfinite(raw).all():
            return raw, scale, retries
        if scale >= 4096:
            return None, scale, retries
        scale *= 4
        rescale(scale)
        retries += 1


def _errors(actual, expected):
    # Audit only: CPU FP64 avoids overflow/NaN accepting an invalid numerical gate.
    a = actual.detach().to(device="cpu", copy=True).double()
    e = expected.detach().to(device="cpu", copy=True).double()
    if not torch.isfinite(a).all() or not torch.isfinite(e).all():
        return math.inf, math.inf
    delta = a - e
    if not e.norm():
        return (0.0, 0.0) if not delta.count_nonzero() else (math.inf, math.inf)
    return float(delta.norm() / e.norm()), float(delta.abs().max() / e.abs().max())


class DenseSwiGLU:
    """One immutable dense-FFN weight group and one persistent ANE worker.

    A registered model/independent adapter explicitly supplies packed gate/up,
    down matrices and a pipeline revision scope. precision='fp16' is required:
    FP16 ANE is approximate and is not a transparent BF16 implementation.
    Call close() with the owning model lifecycle. Unprofiled shapes stay native.
    """

    def __init__(self, weights, *, scope="dense-swiglu", precision="native", _benchmark=False):
        self.weights = tuple(weights)
        self.scope, self.precision = scope, precision
        if precision not in ("native", "fp16"):
            raise ValueError("ANE precision must be native or explicitly fp16")
        mode = os.environ.get("VLLM_OMNI_METAL_ANE", "auto")
        if mode not in ("auto", "off"):
            raise ValueError("VLLM_OMNI_METAL_ANE must be auto or off")
        self.shapes = [artifacts.weight_shape(*pair) for pair in self.weights]
        self.metrics = Counter()
        self.checks = {}
        self.disabled = False
        self._closed = False
        self._benchmark = _benchmark
        self._benchmark_chunks = 0
        self._bridge = None
        self._pool = None
        self._handles = []
        self._runtime = {}
        self._scales = [1] * len(self.weights)
        self.profiles = {}
        self.keys = []
        self.identity = None
        self._versions = [self._version(pair) for pair in self.weights]
        root = artifacts.cache_root()
        if (
            mode == "off"
            or precision != "fp16"
            or not self.weights
            or not all(p.device.type == "mps" and artifacts.eligible(p, d) for p, d in self.weights)
        ):
            return
        artifacts.check_existing_cache()
        if not (root / "tools.json").exists() and not (root / "tools.json").is_symlink():
            return
        self.keys = [artifacts.weight_key(*pair) for pair in self.weights]
        self.identity = artifacts.group_key(self.keys, scope)
        policy_path = root / f"{self.identity}.json"
        if not _benchmark:
            if not policy_path.exists() and not policy_path.is_symlink():
                return
            policy = artifacts.read_json(policy_path)
            if policy.get("group") != self.identity:
                raise RuntimeError("ANE policy does not match the immutable FFN group")
            self.profiles = artifacts.validate_profiles(policy.get("profiles"))
            if not any(self.profiles.values()):
                return
        prepared = []
        try:
            bridge = _Bridge()
            if not _benchmark and policy.get("device") != bridge.device:
                raise RuntimeError("ANE policy belongs to another device")
            for key, (hidden, inner) in zip(self.keys, self.shapes, strict=True):
                prepared.append(artifacts.verify_model(key, hidden, inner, bridge.device))
                artifacts.verify_model(
                    artifacts.runtime_key(hidden, inner), hidden, inner, bridge.device, runtime=True
                )
        except FileNotFoundError:
            if _benchmark:
                raise
            logger.info("Prepared ANE artifacts are missing; retaining native GPU")
            return
        required = 2 * sum(
            p.numel() * p.element_size() + d.numel() * d.element_size() for p, d in self.weights
        )
        if psutil.virtual_memory().available < artifacts.RESERVE + required:
            logger.warning("ANE group exceeds system memory headroom; retaining native GPU")
            return
        self._bridge = bridge
        try:
            for path, (hidden, inner) in zip(prepared, self.shapes, strict=True):
                self._handles.append(bridge.load(path, hidden))
                if (hidden, inner) not in self._runtime:
                    path = (
                        artifacts.model_directory(artifacts.runtime_key(hidden, inner))
                        / "module.mlmodelc"
                    )
                    self._runtime[hidden, inner] = bridge.load(path, hidden)
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omni-ane")
        except BaseException as error:
            self.close()
            if _benchmark or not isinstance(error, RuntimeError):
                raise
            self._closed = False
            logger.exception("ANE graph loading failed; retaining native GPU")
            self.disabled = True
        logger.info(
            "ANE auto group: %s layers, profiles=%s, precision=%s",
            len(self.weights),
            self.profiles,
            precision,
        )

    @staticmethod
    def _version(pair):
        versions = []
        for weight in pair:
            try:
                versions.append(weight._version)
            except RuntimeError:
                # Inference tensors have no counter; immutable ownership is the caller contract.
                versions.append(None)
        return tuple(versions)

    @staticmethod
    def native(x, packed, down):
        gate, up = F.linear(x, packed).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, down)

    def __call__(self, index, x, *, context=None):
        if torch.is_grad_enabled():
            raise RuntimeError("DenseSwiGLU requires an inference/no-grad context")
        if type(index) is not int or not 0 <= index < len(self.weights):
            raise ValueError("invalid dense FFN index")
        if self._closed:
            raise RuntimeError("DenseSwiGLU group is closed")
        packed, down = self.weights[index]
        hidden, _ = self.shapes[index]
        if (
            x.ndim != 2
            or x.shape[1] != hidden
            or x.dtype != packed.dtype
            or x.device != packed.device
            or not x.is_contiguous()
        ):
            raise ValueError("matching contiguous dense SwiGLU activations are required")
        if context is not None and (type(context) is not int or not 0 <= context < 2**31):
            raise ValueError("pipeline context must be a bounded nonnegative shape qualifier")
        if x.shape[0] > artifacts.CHUNK:
            artifacts.profile_label(x.shape[0], context)
        chunks = (
            self._benchmark_chunks
            if self._benchmark
            else self.profiles.get((x.shape[0], context), 0)
        )
        if self.disabled or self._pool is None or chunks == 0 or x.shape[0] <= artifacts.CHUNK:
            self.metrics["native_calls"] += 1
            return self.native(x, packed, down)
        if type(chunks) is not int or not 1 <= chunks <= (x.shape[0] - 1) // artifacts.CHUNK:
            raise ValueError("ANE must take complete chunks and GPU must retain a row")
        if self._version((packed, down)) != self._versions[index]:
            self._disable("weights were modified after ANE binding")
            return self.native(x, packed, down)
        if psutil.virtual_memory().available < artifacts.RESERVE:
            self._disable("system memory headroom exhausted")
            return self.native(x, packed, down)
        result = self._split(index, x, chunks)
        key = index, x.shape[0], chunks, context
        if self._benchmark and key not in self.checks and not self.disabled:
            expected = self.native(x, packed, down)
            rel, scaled = _errors(result, expected)
            self.checks[key] = {"relative_l2": rel, "scaled_max_abs": scaled}
            if rel > 0.01 or scaled > 0.01:
                self.metrics["numerical_rejections"] += 1
                self._disable(f"FP16 numerical gate failed: rel={rel}, max={scaled}")
                return expected
        return result

    def _disable(self, reason):
        self.disabled = True
        self.close()
        self._closed = False
        self.metrics["failures"] += 1
        logger.warning("ANE disabled for this group: %s; retaining native GPU", reason)

    def _split(self, index, x, chunks):
        packed, down = self.weights[index]
        hidden, inner = self.shapes[index]
        ane_rows = chunks * artifacts.CHUNK
        gpu_rows = x.shape[0] - ane_rows
        # shortcut: Torch 2.13 combined MPS view readback/cast mutates its source;
        # remove this separate copy after the upstream source-mutation regression passes.
        tail = x[gpu_rows:].to(device="cpu", copy=True)
        answer = torch.empty(tail.shape, dtype=x.dtype, device="cpu")

        @torch.inference_mode()
        def predict():
            valid = retries = 0
            reason = None
            scale = self._scales[index]
            active = self._handles[index]
            incoming = tail.half().contiguous()
            raw = torch.empty((artifacts.CHUNK, hidden), dtype=torch.float16, device="cpu")

            def stage(new_scale):
                nonlocal active
                active = self._runtime[hidden, inner]
                for part, weight in enumerate((packed[:inner], packed[inner:], down)):
                    value = weight.detach().to(device="cpu", copy=True).float()
                    if part == 1:
                        value = value / new_scale
                    value = value.half().contiguous()
                    if (
                        not torch.isfinite(value).all()
                        or self._bridge.lib.ffn_stage(active, part, value.data_ptr(), value.numel())
                        != 0
                    ):
                        raise RuntimeError("ANE headroom weight staging failed")

            try:
                if not torch.isfinite(incoming).all():
                    return valid, retries, "activation outside FP16 range"
                if scale > 1:
                    stage(scale)
                for chunk in range(chunks):
                    offset = chunk * artifacts.CHUNK * hidden * 2

                    def run(offset=offset):
                        if (
                            self._bridge.lib.ffn_predict(
                                active,
                                incoming.data_ptr() + offset,
                                raw.data_ptr(),
                                raw.numel(),
                                raw.numel(),
                            )
                            != 0
                        ):
                            raise RuntimeError("CoreML prediction failed")
                        return raw

                    finite, scale, repeated = _headroom_retry(run, stage, scale)
                    retries += repeated
                    self._scales[index] = scale
                    if finite is None:
                        reason = "non-finite after bounded FP16 retries"
                        break
                    restored = (finite.float() * scale).to(x.dtype)
                    if not torch.isfinite(restored).all():
                        reason = "non-finite restored output"
                        break
                    answer[valid : valid + artifacts.CHUNK].copy_(restored)
                    valid += artifacts.CHUNK
            except (RuntimeError, ValueError) as error:
                reason = str(error)
            return valid, retries, reason

        future = self._pool.submit(predict)
        result = torch.empty_like(x)
        try:
            result[:gpu_rows].copy_(self.native(x[:gpu_rows], packed, down))
        finally:
            valid, retries, reason = future.result()
        if valid:
            result[gpu_rows : gpu_rows + valid].copy_(answer[:valid].to(device=x.device))
        if reason is not None or valid != ane_rows:
            first_lost = gpu_rows + valid
            result[first_lost:].copy_(self.native(x[first_lost:], packed, down))
            self.metrics["gpu_recomputed_rows"] += x.shape[0] - first_lost
            self._disable(reason or "incomplete ANE output")
        self.metrics.update(
            calls=1,
            ane_chunks=valid // artifacts.CHUNK,
            ane_rows=valid,
            gpu_rows=gpu_rows,
            headroom_retries=retries,
        )
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        if self._bridge is not None:
            for handle in [*self._handles, *self._runtime.values()]:
                self._bridge.lib.ffn_unload(handle)
        self._handles.clear()
        self._runtime.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class TiledDenseSwiGLU:
    """A bounded group of separate gate/up/down projections, tiled along inner.

    The owning adapter supplies native(x), its original *whole* FFN, on every
    call. Off, native precision, missing approval and any failed tile use that
    callable unchanged. Explicit offline benchmarking gates the final FP32 sum,
    not just individual projections. Bind only the layers permitted by the
    adapter's resource policy; no model-wide packed weight copy is required.
    """

    def __init__(self, weights, *, scope, precision="native", _benchmark=False):
        self.weights = tuple(tuple(layer) for layer in weights)
        self.shapes = [artifacts.tiled_weight_shape(*layer) for layer in self.weights]
        self.scope, self.precision = scope, precision
        if precision not in ("native", "fp16"):
            raise ValueError("ANE precision must be native or explicitly fp16")
        mode = os.environ.get("VLLM_OMNI_METAL_ANE", "auto")
        if mode not in ("auto", "off"):
            raise ValueError("VLLM_OMNI_METAL_ANE must be auto or off")
        # Validate scope even for an inactive group; never share anonymous approval.
        artifacts.group_key([], scope)
        self.metrics = Counter()
        self._tile_metrics = {}
        self.checks = {}
        self.profiles = {}
        self._input_shapes = {}
        self.plans = []
        self.keys = []
        self.identity = None
        self.disabled = False
        self._closed = False
        self._benchmark = _benchmark
        self._benchmark_chunks = 0
        self._audit = True
        self._group = None
        self._versions = [DenseSwiGLU._version(layer) for layer in self.weights]
        if mode == "off" or precision != "fp16" or not self.weights:
            return
        if not all(layer[0].dtype in (torch.float16, torch.bfloat16) for layer in self.weights):
            return
        if not all(layer[0].device.type == "mps" for layer in self.weights):
            return
        try:
            self.plans = [artifacts.inner_tiles(*shape) for shape in self.shapes]
        except ValueError:
            return
        artifacts.check_existing_cache()
        root = artifacts.cache_root()
        if not (root / "tools.json").exists() and not (root / "tools.json").is_symlink():
            return
        # Includes owned tiled copies plus the existing Dense/CoreML allowance.
        required = 3 * sum(
            value.numel() * value.element_size() for layer in self.weights for value in layer
        )
        if psutil.virtual_memory().available < artifacts.RESERVE + required:
            if _benchmark:
                raise RuntimeError("tiled ANE group exceeds system memory headroom")
            return
        # Hash sequential CPU tiles, not a second packed copy of every model layer.
        self.keys = [
            artifacts.weight_key(*pair)
            for pair in artifacts.tiled_weights(self.weights, self.plans, host=True)
        ]
        self.identity = artifacts.tiled_group_key(self.keys, self.shapes, self.plans, scope)
        if not _benchmark:
            path = root / f"{self.identity}.json"
            if not path.exists() and not path.is_symlink():
                return
            policy = artifacts.read_json(path)
            if policy.get("group") != self.identity:
                raise RuntimeError("ANE policy does not match the immutable tiled FFN group")
            self.profiles, self._input_shapes = artifacts.validate_tiled_profiles(
                policy.get("profiles"), self.shapes, self.plans
            )
            if not any(self.profiles.values()):
                return
        group = None
        try:
            group = DenseSwiGLU(
                artifacts.tiled_weights(self.weights, self.plans),
                scope=scope,
                precision="fp16",
                _benchmark=True,
            )
            if group._pool is None or group.disabled:
                raise RuntimeError("tiled ANE group is not fully prepared")
            if group.keys != self.keys:
                raise RuntimeError("SwiGLU weights changed during tiled ANE binding")
            if not _benchmark and policy.get("device") != group._bridge.device:
                raise RuntimeError("ANE policy belongs to another device")
            self._group = group
        except BaseException as error:
            if group is not None:
                group.close()
            if _benchmark or not isinstance(error, FileNotFoundError):
                raise
            logger.info("Prepared tiled ANE artifacts are missing; retaining native GPU")

    @property
    def _bridge(self):
        return None if self._group is None else self._group._bridge

    @property
    def tile_metrics(self):
        """Per-layer/tile hit evidence; final-output errors are in checks."""
        return self._tile_metrics

    def __call__(self, index, x, *, native, context=None):
        if torch.is_grad_enabled():
            raise RuntimeError("TiledDenseSwiGLU requires an inference/no-grad context")
        if self._closed:
            raise RuntimeError("TiledDenseSwiGLU group is closed")
        if type(index) is not int or not 0 <= index < len(self.weights):
            raise ValueError("invalid dense FFN index")
        if context is not None and (type(context) is not int or not 0 <= context < 2**31):
            raise ValueError("pipeline context must be a bounded nonnegative shape qualifier")
        if self.disabled or self._group is None:
            self.metrics["native_calls"] += 1
            return native(x)
        gate = self.weights[index][0]
        if (
            x.ndim not in (2, 3)
            or x.shape[-1] != self.shapes[index][0]
            or x.dtype != gate.dtype
            or x.device != gate.device
            or not x.is_contiguous()
        ):
            self.metrics["native_calls"] += 1
            return native(x)
        flat = x.reshape(-1, x.shape[-1])
        rows = flat.shape[0]
        chunks = (
            self._benchmark_chunks if self._benchmark else self.profiles.get((rows, context), 0)
        )
        if chunks == 0 or rows <= artifacts.CHUNK:
            self.metrics["native_calls"] += 1
            return native(x)
        if not self._benchmark and self._input_shapes.get(((rows, context), index)) != tuple(
            x.shape
        ):
            self.metrics["native_calls"] += 1
            return native(x)
        if type(chunks) is not int or not 1 <= chunks <= (rows - 1) // artifacts.CHUNK:
            raise ValueError("ANE must take complete chunks and GPU must retain a row")
        if DenseSwiGLU._version(self.weights[index]) != self._versions[index]:
            self._disable("weights were modified after tiled ANE binding")
            return native(x)
        # Account for the FP32 reduction and tile outputs without relaxing the reserve.
        if psutil.virtual_memory().available < artifacts.RESERVE + flat.numel() * 12:
            self._disable("system memory headroom exhausted")
            return native(x)
        if self._benchmark and not self._audit:
            check = self.checks.get((index, rows, chunks, context))
            if check is None or check["shape"] != list(x.shape):
                raise RuntimeError("timed ANE execution requires a completed audit for this shape")
        return self._execute(index, x, flat, chunks, native, context)

    def _execute(self, index, x, flat, chunks, native, context):
        rows = flat.shape[0]
        offset = sum(len(plan) for plan in self.plans[:index])
        try:
            total = torch.zeros(flat.shape, dtype=torch.float32, device=flat.device)
            for tile, _ in enumerate(self.plans[index]):
                before = self._group.metrics.copy()
                partial = self._group._split(offset + tile, flat, chunks)
                hits = self._group.metrics.copy()
                hits.subtract(before)
                self._tile_metrics.setdefault((index, tile), Counter()).update(hits)
                if self._group.disabled or hits["ane_chunks"] != chunks:
                    raise RuntimeError("incomplete tiled ANE execution")
                total.add_(partial)
            result = total.to(x.dtype).reshape_as(x)
            if not torch.isfinite(result).all():
                raise RuntimeError("non-finite tiled output")
        except BaseException as error:
            self._disable(str(error))
            if isinstance(error, (RuntimeError, ValueError)):
                return native(x)
            raise
        if self._benchmark and self._audit:
            expected = native(x)
            rel, peak = _errors(result, expected)
            key = index, rows, chunks, context
            previous = self.checks.get(key, {})
            if previous and previous["shape"] != list(x.shape):
                self._disable("activation shape changed within one calibration key")
                return expected
            self.checks[key] = {
                "relative_l2": max(rel, previous.get("relative_l2", 0)),
                "scaled_max_abs": max(peak, previous.get("scaled_max_abs", 0)),
                "tiles": len(self.plans[index]),
                "shape": list(x.shape),
            }
            if rel > 0.01 or peak > 0.01:
                self.metrics["numerical_rejections"] += 1
                self._disable(f"final tiled numerical gate failed: rel={rel}, max={peak}")
                return expected
        self.metrics.update(calls=1, ane_chunks=chunks * len(self.plans[index]))
        return result

    def finish_audit(self):
        """Freeze successful warmup evidence before measuring the production path."""
        if not self._benchmark or self.disabled or self._closed or self._group is None:
            raise RuntimeError("a healthy offline calibration group is required")
        for index in range(len(self.weights)):
            if not any(key[0] == index and key[2] == self._benchmark_chunks for key in self.checks):
                raise RuntimeError("every FFN must be audited before timed execution")
        self._audit = False

    def _disable(self, reason):
        self.disabled = True
        self.close()
        self._closed = False
        self.metrics["failures"] += 1
        logger.warning("Tiled ANE disabled: %s; retaining original GPU FFN", reason)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._group is not None:
            self._group.close()
            self._group = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
