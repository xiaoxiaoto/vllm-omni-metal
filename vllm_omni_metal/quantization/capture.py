# SPDX-License-Identifier: Apache-2.0
"""Bounded real-forward activation capture for offline DynamicEngine search.

Use ``with CaptureSession(root, ["projection"]) as session: ...`` and pass
``session.captures`` to search after exit. Missing calls remain missing; this
utility neither generates inputs nor approves a model's quality contract.
"""

import copy
import hashlib
import os
import stat
import uuid
import weakref
from pathlib import Path

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


class DiskCaptures:
    """Owned raw-byte captures: one complete CPU activation at a time.

    The new absolute directory is private (0700); each non-pickle raw file is
    private (0600). Metadata and source descriptors contain no tensor references.
    ``max_bytes`` bounds each complete host activation, not the total dataset;
    ``max_disk_bytes`` bounds all raw files. Use a context manager or ``close``.
    Unrecognized or replaced filesystem entries are never removed by cleanup.
    """

    def __init__(self, modules, directory, *, max_bytes=128 * 1024**2, max_disk_bytes=1024**3):
        modules = dict(modules)
        if not modules:
            raise ValueError("captures require nonempty candidate modules")
        for label, budget in (("max_bytes", max_bytes), ("max_disk_bytes", max_disk_bytes)):
            if type(budget) is not int or budget <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if not isinstance(directory, Path) or not directory.is_absolute():
            raise ValueError("directory must be an absolute Path")
        if ".." in directory.parts or directory.name in ("", ".", ".."):
            raise ValueError("directory must name a new private directory")
        precisions = {
            name: {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}.get(
                getattr(getattr(module, "weight", None), "dtype", None), "bf16"
            )
            for name, module in modules.items()
        }
        bits = linear._modules(modules, list(modules), precisions)
        self._descriptors = {
            name: linear._descriptor(module, bits[name]) for name, module in modules.items()
        }
        self.source_changed = False
        self.directory = directory
        self.max_bytes = max_bytes
        self.max_disk_bytes = max_disk_bytes
        self.bytes_used = 0
        self._records = {name: {} for name in modules}
        self._closed = False
        self._parent_fd = None
        self._fd = None
        # Pin original inodes until cleanup: unlink/recreate must not let inode
        # reuse turn a user's replacement into an apparently owned file.
        self._file_fds = {}
        # Walk by descriptors: neither ancestors nor the final entry may be links.
        parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        created = False
        try:
            for part in directory.parts[1:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                os.close(parent)
                parent = child
            os.mkdir(directory.name, mode=0o700, dir_fd=parent)
            created = True
            self._fd = os.open(
                directory.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
            )
            os.fchmod(self._fd, 0o700)
            self._identity = self._stat_identity(os.fstat(self._fd))
            self._parent_fd = parent
        except BaseException:
            if self._fd is not None:
                os.close(self._fd)
            if created:
                os.rmdir(directory.name, dir_fd=parent)
            os.close(parent)
            raise

    @staticmethod
    def _stat_identity(info):
        return info.st_dev, info.st_ino

    def _check_directory(self):
        if self._closed:
            raise ValueError("disk captures are closed")
        info = os.stat(self.directory.name, dir_fd=self._parent_fd, follow_symlinks=False)
        if (
            self._stat_identity(info) != self._identity
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise ValueError("capture directory identity or permissions changed")

    @property
    def names(self):
        return tuple(self._records)

    def steps(self, name):
        self._check_directory()
        return sorted(self._records[name])

    def validate_source(self, modules):
        Captures.validate_source(self, modules)

    def record(self, name, step, tensor):
        self._check_directory()
        if name not in self._records:
            raise ValueError("unknown capture candidate")
        if type(step) is not int or step < 0 or step in self._records[name]:
            raise ValueError("captured steps must be distinct nonnegative integers")
        descriptor = self._descriptors[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.requires_grad
            or tensor.layout != torch.strided
            or str(tensor.dtype) != descriptor["dtype"]
            or tensor.ndim < 1
            or tensor.numel() == 0
            or tensor.shape[-1] != descriptor["shape"][1]
        ):
            raise ValueError("capture requires nonempty matching source input without gradients")
        size = tensor.numel() * tensor.element_size()
        if size > self.max_bytes:
            raise ValueError("capture resident byte budget exceeded")
        if size > self.max_disk_bytes - self.bytes_used:
            raise ValueError("capture disk byte budget exceeded")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("capture requires finite input")
        # Allocate exactly one host copy, including for noncontiguous CPU inputs.
        host = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="cpu")
        host.copy_(tensor.detach())
        metadata = {"step": step, "shape": list(host.shape), "dtype": str(host.dtype)}
        raw = memoryview(host.view(torch.uint8).numpy()).cast("B")
        digest = hashlib.sha256(linear._encoded(metadata))
        digest.update(raw)
        filename = uuid.uuid4().hex + ".raw"
        fd = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self._fd,
        )
        identity = self._stat_identity(os.fstat(fd))
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(raw)
                stream.flush()
            info = os.fstat(fd)
            if info.st_size != size:
                raise ValueError("incomplete capture write")
            self._records[name][step] = {
                **metadata,
                "sha256": digest.hexdigest(),
                "bytes": size,
                "file": filename,
                "identity": identity,
            }
            self._file_fds[filename] = fd
            self.bytes_used += size
        except BaseException:
            info = os.stat(filename, dir_fd=self._fd, follow_symlinks=False)
            if self._stat_identity(info) == identity:
                os.unlink(filename, dir_fd=self._fd)
            raise
        finally:
            if filename not in self._file_fds:
                os.close(fd)

    def read(self, name, step):
        self._check_directory()
        if type(step) is not int or step < 0:
            raise ValueError("captured step must be a nonnegative integer")
        record = self._records[name][step]
        # A replaced FIFO must fail validation rather than block waiting for a writer.
        fd = os.open(record["file"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self._fd)
        try:
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or self._stat_identity(before) != record["identity"]
                or before.st_uid != os.getuid()
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_nlink != 1
                or before.st_size != record["bytes"]
                or before.st_size > self.max_bytes
            ):
                raise ValueError("capture file identity, size or permissions changed")
            dtype = {
                "torch.float16": torch.float16,
                "torch.bfloat16": torch.bfloat16,
                "torch.float32": torch.float32,
            }[record["dtype"]]
            host = torch.empty(record["shape"], dtype=dtype, device="cpu")
            raw = memoryview(host.view(torch.uint8).numpy()).cast("B")
            if len(raw) != before.st_size:
                raise ValueError("capture metadata size mismatch")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                if stream.readinto(raw) != len(raw) or stream.read(1):
                    raise ValueError("incomplete capture read")
            after = os.fstat(fd)
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError("capture file changed during read")
            metadata = {key: record[key] for key in ("step", "shape", "dtype")}
            digest = hashlib.sha256(linear._encoded(metadata))
            digest.update(raw)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError("capture file hash mismatch")
            return host
        finally:
            os.close(fd)

    def summary(self):
        self._check_directory()
        return copy.deepcopy(
            {
                "storage": "disk",
                "directory": str(self.directory),
                "bytes_used": self.bytes_used,
                "max_bytes": self.max_bytes,
                "max_disk_bytes": self.max_disk_bytes,
                "source_descriptors": self._descriptors,
                "captures": {
                    name: [
                        {
                            key: records[step][key]
                            for key in ("step", "shape", "dtype", "bytes", "sha256")
                        }
                        for step in sorted(records)
                    ]
                    for name, records in self._records.items()
                },
            }
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            for records in self._records.values():
                for record in records.values():
                    try:
                        info = os.stat(record["file"], dir_fd=self._fd, follow_symlinks=False)
                        if self._stat_identity(info) == record["identity"]:
                            os.unlink(record["file"], dir_fd=self._fd)
                    except FileNotFoundError:
                        pass
            try:
                info = os.stat(self.directory.name, dir_fd=self._parent_fd, follow_symlinks=False)
                if self._stat_identity(info) == self._identity:
                    os.rmdir(self.directory.name, dir_fd=self._parent_fd)
            except OSError:
                # An added user file or a renamed directory is not ours to remove.
                pass
        finally:
            for fd in self._file_fds.values():
                os.close(fd)
            self._file_fds.clear()
            os.close(self._fd)
            os.close(self._parent_fd)

    def __enter__(self):
        self._check_directory()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


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
