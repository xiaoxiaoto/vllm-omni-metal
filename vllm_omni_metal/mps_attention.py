# SPDX-License-Identifier: Apache-2.0
"""Optional upstream Torch-stream launcher; run this module once to build it locally."""

import fcntl
import hashlib
import importlib.util
import os
import shutil
import stat
import sysconfig
import tempfile
from functools import cache
from pathlib import Path

import psutil

_NATIVE = Path(__file__).parent / "_native"


def _artifact_path() -> Path:
    import torch
    from vllm_metal.metal.constants import PA_WINDOW_MAX_HEAD_SIZE, PA_WINDOW_ROWS, PARTITION_SIZE

    digest = hashlib.sha256(Path(__file__).read_bytes())
    for source in sorted(
        (*(_NATIVE / "metal").glob("*.h"), *(_NATIVE / "pytorch_backend").glob("*.mm"))
    ):
        digest.update(source.read_bytes())
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    identity = (
        torch.__version__,
        str(Path(torch.__file__).resolve()),
        suffix,
        PARTITION_SIZE,
        PA_WINDOW_ROWS,
        PA_WINDOW_MAX_HEAD_SIZE,
    )
    digest.update(repr(identity).encode())
    return (
        Path.home() / ".cache/vllm-omni-metal/mps" / digest.hexdigest() / f"_omni_mps_ops{suffix}"
    )


def _check_private_directory(directory: Path) -> None:
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError("MPS launcher cache must be a private, owned directory")


def _check_artifact(path: Path) -> None:
    _check_private_directory(path.parent)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise RuntimeError("unsafe MPS launcher artifact")


@cache
def get_mps_ops():
    """Absent artifact retains the bridge; invalid artifacts fail rather than hide errors."""
    path = _artifact_path()
    if not path.exists() and not path.is_symlink():
        return None
    _check_artifact(path)
    from vllm_metal.metal.build import metallib_path

    spec = importlib.util.spec_from_file_location("_omni_mps_ops", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    library = metallib_path("paged_attention_v2_kern")
    if not library.is_file():
        raise RuntimeError("installed vllm-metal paged attention metallib is missing")
    nax = metallib_path("paged_attention_nax_kern")
    nax_path = str(nax) if module.nax_supported() and nax.is_file() else ""
    return module.PagedAttention(str(library), nax_path, module.gpu_core_count())


def build_mps_launcher() -> Path:
    """Explicit build only, with upstream host code plus released FP32 kernel dispatch."""
    path = _artifact_path()
    if (
        psutil.virtual_memory().available < 3 * 1024**3
        or shutil.disk_usage(Path.home()).free < 512 * 1024**2
    ):
        raise RuntimeError("MPS build budget requires 3 GiB system headroom and 512 MiB disk")
    if not shutil.which("clang++"):
        raise RuntimeError("MPS launcher build requires Apple Command Line Tools")
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required; no CPU fallback")
    # Create only our cache namespace; never modify the installed upstream package.
    for directory in (path.parent.parent.parent, path.parent.parent, path.parent):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _check_private_directory(directory)
    with (path.parent / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() or path.is_symlink():
            _check_artifact(path)
            return path
        from setuptools import Distribution
        from torch.utils.cpp_extension import BuildExtension, CppExtension
        from vllm_metal.metal.constants import (
            PA_WINDOW_MAX_HEAD_SIZE,
            PA_WINDOW_ROWS,
            PARTITION_SIZE,
        )

        extension = CppExtension(
            "_omni_mps_ops",
            sources=[str(_NATIVE / "pytorch_backend/mps_ops.mm")],
            extra_compile_args=[
                "-O3",
                "-mmacosx-version-min=14.0",
                f"-DVLLM_METAL_PARTITION_SIZE={PARTITION_SIZE}",
                f"-DVLLM_METAL_PA_WINDOW_ROWS={PA_WINDOW_ROWS}",
                f"-DVLLM_METAL_PA_WINDOW_MAX_HEAD={PA_WINDOW_MAX_HEAD_SIZE}",
            ],
            extra_link_args=[
                "-mmacosx-version-min=14.0",
                "-framework",
                "Metal",
                "-framework",
                "Foundation",
                "-framework",
                "IOKit",
                "-framework",
                "CoreFoundation",
            ],
        )
        command = BuildExtension(Distribution({"ext_modules": [extension]}))
        command.use_ninja = False
        command.ensure_finalized()
        with tempfile.TemporaryDirectory(prefix="build-", dir=path.parent) as temporary:
            command.build_lib = temporary
            command.build_temp = str(Path(temporary) / "objects")
            command.run()
            os.replace(Path(temporary) / path.name, path)
    get_mps_ops.cache_clear()
    return path


if __name__ == "__main__":
    print(build_mps_launcher())
