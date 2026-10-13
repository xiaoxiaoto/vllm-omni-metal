# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree Apple Silicon platform for vLLM-Omni."""

import os
import platform as py_platform
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vllm-omni-metal")
except PackageNotFoundError:
    __version__ = "0.1.0"


def is_available() -> bool:
    """Detection must not import vLLM or initialize MLX in a parent process."""
    if py_platform.system() != "Darwin" or py_platform.machine() != "arm64":
        return False
    import torch

    return torch.backends.mps.is_available()


def _prepare_runtime() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm_omni_metal.compat import apply_compat_patches

    apply_compat_patches()


def register_platform() -> str | None:
    if not is_available():
        return None
    # Importing pipeline registries here would recurse into Omni's rotary imports.
    _prepare_runtime()
    return "vllm_omni_metal.platform.MetalOmniPlatform"
