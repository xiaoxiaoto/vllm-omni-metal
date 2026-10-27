# SPDX-License-Identifier: Apache-2.0
"""Explicit, inference-only streaming of individual dense linear projections.

Export requires the source projection to fit in CPU memory.  The caller owns
replacement of the original module and must account for shared parameters and
external references before releasing it.
"""

import hashlib
import json
import os
import stat
import threading
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

_DEFAULT_LIMIT = 64 * 1024**2
_MAX_MANIFEST = 16 * 1024**2
_MAX_CHUNKS = 100_000
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
_SIZES = {"float16": 2, "bfloat16": 2, "float32": 4}
_NOFOLLOW = os.O_NOFOLLOW
_DIRECTORY = os.O_DIRECTORY


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _open_parent(path):
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise ValueError("directory must be an absolute Path without '..'")
    if len(path.parts) < 2:
        raise ValueError("directory cannot be the filesystem root")
    fd = os.open(path.anchor, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            next_fd = os.open(part, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd, path.name
    except BaseException:
        os.close(fd)
        raise


def _check_stat(info, *, directory=False):
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    mode = 0o700 if directory else 0o600
    if (
        not expected(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != mode
        or (not directory and info.st_nlink != 1)
    ):
        raise ValueError("artifact has unsafe ownership, type, permissions, or links")


def _open_file(directory_fd, name):
    fd = os.open(name, os.O_RDONLY | _NOFOLLOW | getattr(os, "O_NONBLOCK", 0), dir_fd=directory_fd)
    try:
        _check_stat(os.fstat(fd))
        return fd
    except BaseException:
        os.close(fd)
        raise


def _write_all(fd, data):
    view = memoryview(data).cast("B")
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short artifact write")
        view = view[written:]


def _tensor_bytes(tensor):
    # NumPy is only a view here; including BF16, no tensor-sized copy is made.
    return memoryview(tensor.detach().view(torch.uint8).numpy()).cast("B")


def _has_hooks(obj):
    return (
        any(
            bool(value)
            for name, value in vars(obj).items()
            if "hook" in name and isinstance(value, dict | list)
        )
        or bool(getattr(obj, "_backward_hooks", None))
        or bool(getattr(obj, "_post_accumulate_grad_hooks", None))
    )


def prepare_linear(linear, directory, *, max_chunk_bytes=_DEFAULT_LIMIT):
    """Export one CPU Linear to a newly created, private artifact directory."""
    _positive_int(max_chunk_bytes, "max_chunk_bytes")
    if type(linear) is not nn.Linear:
        raise TypeError("linear must be exactly torch.nn.Linear")
    weight, bias = linear.weight, linear.bias
    if (
        weight.device.type != "cpu"
        or weight.dtype not in _DTYPES.values()
        or not weight.is_contiguous()
        or weight.ndim != 2
        or tuple(weight.shape) != (linear.out_features, linear.in_features)
        or linear.in_features <= 0
        or linear.out_features <= 0
    ):
        raise ValueError("weight must be a nonempty contiguous CPU FP16/BF16/FP32 matrix")
    if bias is not None and (
        bias.device.type != "cpu"
        or bias.dtype != weight.dtype
        or not bias.is_contiguous()
        or bias.ndim != 1
        or bias.shape[0] != linear.out_features
    ):
        raise ValueError("bias must match weight dtype and output dimensions on CPU")
    if _has_hooks(linear) or _has_hooks(weight) or (bias is not None and _has_hooks(bias)):
        raise ValueError("hooks are not supported")
    dtype = str(weight.dtype).removeprefix("torch.")
    row_bytes = (linear.in_features + (bias is not None)) * weight.element_size()
    rows = max_chunk_bytes // row_bytes
    if not rows:
        raise ValueError("max_chunk_bytes cannot fit one complete output row")
    count = (linear.out_features + rows - 1) // rows
    if count > _MAX_CHUNKS:
        raise ValueError("too many chunks")
    detached_weight = weight.detach()
    detached_bias = None if bias is None else bias.detach()

    parent_fd, name = _open_parent(directory)
    directory_fd = None
    created = False
    files = []
    identity = None
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        created = True
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        identity = (info.st_dev, info.st_ino)
        directory_fd = os.open(name, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=parent_fd)
        info = os.fstat(directory_fd)
        if (info.st_dev, info.st_ino) != identity:
            raise ValueError("artifact directory changed during creation")
        os.fchmod(directory_fd, 0o700)
        chunks = []
        for index, start in enumerate(range(0, linear.out_features, rows)):
            end = min(start + rows, linear.out_features)
            filename = f"chunk-{index:06d}.bin"
            fd = os.open(
                filename,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            files.append(filename)
            try:
                os.fchmod(fd, 0o600)
                digest = hashlib.sha256()
                for tensor in (
                    detached_weight[start:end],
                    None if detached_bias is None else detached_bias[start:end],
                ):
                    if tensor is not None:
                        data = _tensor_bytes(tensor)
                        digest.update(data)
                        _write_all(fd, data)
            finally:
                os.close(fd)
            chunks.append(
                {
                    "file": filename,
                    "row_start": start,
                    "row_end": end,
                    "bytes": (end - start) * row_bytes,
                    "sha256": digest.hexdigest(),
                }
            )
        manifest = {
            "schema": 1,
            "dtype": dtype,
            "in_features": linear.in_features,
            "out_features": linear.out_features,
            "bias": bias is not None,
            "chunks": chunks,
        }
        encoded = json.dumps(manifest, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > _MAX_MANIFEST:
            raise ValueError("manifest exceeds size limit")
        fd = os.open(
            "manifest.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        files.append("manifest.json")
        try:
            os.fchmod(fd, 0o600)
            _write_all(fd, encoded)
        finally:
            os.close(fd)
    except BaseException:
        if directory_fd is not None:
            for filename in reversed(files):
                try:
                    os.unlink(filename, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
        if created:
            try:
                info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if identity is not None and (info.st_dev, info.st_ino) == identity:
                    os.rmdir(name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(parent_fd)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _read_bounded(fd, size):
    result = bytearray(size)
    view = memoryview(result)
    position = 0
    while position < size:
        count = os.readv(fd, [view[position:]])
        if not count:
            raise ValueError("artifact file is truncated")
        position += count
    if os.read(fd, 1):
        raise ValueError("artifact file exceeds declared size")
    return result


def _load_manifest(directory_fd, limit):
    fd = _open_file(directory_fd, "manifest.json")
    try:
        size = os.fstat(fd).st_size
        if size <= 0 or size > _MAX_MANIFEST:
            raise ValueError("invalid manifest size")
        raw = _read_bounded(fd, size)
    finally:
        os.close(fd)
    try:
        manifest = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid manifest JSON") from exc
    if type(manifest) is not dict or set(manifest) != {
        "schema",
        "dtype",
        "in_features",
        "out_features",
        "bias",
        "chunks",
    }:
        raise ValueError("invalid manifest fields")
    if type(manifest["schema"]) is not int or manifest["schema"] != 1:
        raise ValueError("unsupported manifest schema")
    dtype = manifest["dtype"]
    if type(dtype) is not str or dtype not in _DTYPES:
        raise ValueError("unsupported artifact dtype")
    inputs = _positive_int(manifest["in_features"], "in_features")
    outputs = _positive_int(manifest["out_features"], "out_features")
    bias = manifest["bias"]
    if type(bias) is not bool:
        raise ValueError("invalid bias flag")
    chunks = manifest["chunks"]
    if type(chunks) is not list or not chunks or len(chunks) > _MAX_CHUNKS:
        raise ValueError("invalid chunks")
    cursor = 0
    for index, chunk in enumerate(chunks):
        if type(chunk) is not dict or set(chunk) != {
            "file",
            "row_start",
            "row_end",
            "bytes",
            "sha256",
        }:
            raise ValueError("invalid chunk fields")
        if chunk["file"] != f"chunk-{index:06d}.bin" or type(chunk["file"]) is not str:
            raise ValueError("invalid chunk filename")
        start, end, size, digest = (
            chunk["row_start"],
            chunk["row_end"],
            chunk["bytes"],
            chunk["sha256"],
        )
        if (
            type(start) is not int
            or type(end) is not int
            or start != cursor
            or end <= start
            or end > outputs
            or type(size) is not int
            or size != (end - start) * (inputs + bias) * _SIZES[dtype]
            or size > limit
            or type(digest) is not str
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("invalid chunk metadata or chunk exceeds max_weight_bytes")
        fd = _open_file(directory_fd, chunk["file"])
        try:
            if os.fstat(fd).st_size != size:
                raise ValueError("chunk size does not match manifest")
        finally:
            os.close(fd)
        cursor = end
    if cursor != outputs:
        raise ValueError("chunks do not cover output rows")
    return manifest, hashlib.sha256(raw).digest()


class StreamedLinear(nn.Module):
    """An MPS-only, serialized, inference-only linear backed by raw chunks."""

    def __init__(self, directory, *, max_weight_bytes=_DEFAULT_LIMIT, device="mps"):
        super().__init__()
        _positive_int(max_weight_bytes, "max_weight_bytes")
        device = torch.device(device)
        if device.type != "mps" or device.index not in (None, 0):
            raise ValueError("StreamedLinear only supports MPS")
        self._lock = threading.Lock()
        self._directory_fd = None
        parent_fd, name = _open_parent(directory)
        try:
            fd = os.open(name, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        try:
            _check_stat(os.fstat(fd), directory=True)
            manifest, manifest_hash = _load_manifest(fd, max_weight_bytes)
        except BaseException:
            os.close(fd)
            raise
        self._directory_fd = fd
        self._manifest = manifest
        self._manifest_hash = manifest_hash
        self._device = device
        self._max_weight_bytes = max_weight_bytes
        self.in_features = manifest["in_features"]
        self.out_features = manifest["out_features"]
        self.training = False

    def train(self, mode=True):
        if type(mode) is not bool:
            raise ValueError("training mode must be boolean")
        if mode:
            raise RuntimeError("StreamedLinear is inference-only")
        return super().train(False)

    def _apply(self, fn, recurse=True):
        raise RuntimeError("StreamedLinear cannot be moved or cast")

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        raise RuntimeError("StreamedLinear cannot be saved in a state_dict")

    def _load_from_state_dict(self, *args, **kwargs):
        raise RuntimeError("StreamedLinear cannot load a state_dict")

    def __getstate__(self):
        raise RuntimeError("StreamedLinear cannot be serialized")

    def close(self):
        with self._lock:
            if self._directory_fd is not None:
                os.close(self._directory_fd)
                self._directory_fd = None

    def __enter__(self):
        with self._lock:
            if self._directory_fd is None:
                raise RuntimeError("StreamedLinear is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        fd = getattr(self, "_directory_fd", None)
        if fd is not None:
            os.close(fd)
            self._directory_fd = None

    @torch.no_grad()
    def forward(self, input):
        with self._lock:
            if self._directory_fd is None:
                raise RuntimeError("StreamedLinear is closed")
            if self.training:
                raise RuntimeError("StreamedLinear is inference-only")
            _check_stat(os.fstat(self._directory_fd), directory=True)
            manifest, manifest_hash = _load_manifest(self._directory_fd, self._max_weight_bytes)
            if manifest != self._manifest or manifest_hash != self._manifest_hash:
                raise ValueError("artifact manifest changed")
            dtype = _DTYPES[self._manifest["dtype"]]
            if (
                not isinstance(input, torch.Tensor)
                or input.device.type != "mps"
                or input.dtype != dtype
                or input.requires_grad
                or input.ndim < 1
                or input.numel() == 0
                or input.shape[-1] != self.in_features
            ):
                raise ValueError(
                    "input must be a nonempty MPS tensor of matching dtype and width without gradients"
                )
            output = torch.empty(
                (*input.shape[:-1], self.out_features), dtype=dtype, device=input.device
            )
            for chunk in self._manifest["chunks"]:
                fd = _open_file(self._directory_fd, chunk["file"])
                try:
                    if os.fstat(fd).st_size != chunk["bytes"]:
                        raise ValueError("chunk size changed")
                    data = _read_bounded(fd, chunk["bytes"])
                finally:
                    os.close(fd)
                if hashlib.sha256(data).hexdigest() != chunk["sha256"]:
                    raise ValueError("chunk checksum mismatch")
                rows = chunk["row_end"] - chunk["row_start"]
                flat = torch.frombuffer(data, dtype=dtype)
                weight_elements = rows * self.in_features
                weight = flat[:weight_elements].view(rows, self.in_features)
                bias = flat[weight_elements:] if self._manifest["bias"] else None
                device_weight = device_bias = result = None
                try:
                    device_weight = weight.to(input.device)
                    if bias is not None:
                        device_bias = bias.to(input.device)
                    result = F.linear(input, device_weight, device_bias)
                    output[..., chunk["row_start"] : chunk["row_end"]].copy_(result)
                finally:
                    torch.mps.synchronize()
                    del device_weight, device_bias, result, weight, bias, flat, data
            return output
