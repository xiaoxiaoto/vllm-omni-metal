# SPDX-License-Identifier: Apache-2.0
"""Generic vpipe ports; optional metallibs are built explicitly, never by workers."""

import argparse
import fcntl
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from functools import cache
from pathlib import Path

import psutil
import torch

from vllm_omni_metal.mps_attention import _check_artifact, _check_private_directory

_SOURCE = Path(__file__).parent / "_native/vpipe_elementwise.metal"
_SCALARS = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}


def _artifact_path(dtype):
    digest = hashlib.sha256(Path(__file__).read_bytes() + _SOURCE.read_bytes())
    digest.update(torch.__version__.encode())
    return (
        Path.home()
        / ".cache/vllm-omni-metal/vpipe"
        / digest.hexdigest()
        / f"{_SCALARS[dtype]}.metallib"
    )


@cache
def _library(dtype):
    path = _artifact_path(dtype)
    if not path.exists() and not path.is_symlink():
        return None
    _check_artifact(path)
    return torch.mps.load_metallib(path)


def build_vpipe_kernels():
    """Offline, explicit Metal compiler invocation; no tool installation or dependency changes."""
    if (
        psutil.virtual_memory().available < 3 * 1024**3
        or shutil.disk_usage(Path.home()).free < 512 * 1024**2
    ):
        raise RuntimeError("vpipe build budget requires 3 GiB headroom and 512 MiB disk")
    if not shutil.which("xcrun"):
        raise RuntimeError("vpipe build requires an already installed Apple Metal compiler")
    found = subprocess.run(["xcrun", "--find", "metal"], capture_output=True, text=True, timeout=5)
    if found.returncode:
        raise RuntimeError("Apple Metal compiler unavailable; no tools are installed automatically")
    for dtype, scalar in _SCALARS.items():
        path = _artifact_path(dtype)
        for directory in (path.parent.parent.parent, path.parent.parent, path.parent):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            _check_private_directory(directory)
        with (path.parent / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists() or path.is_symlink():
                _check_artifact(path)
                continue
            with tempfile.TemporaryDirectory(prefix="build-", dir=path.parent) as temporary:
                source = Path(temporary) / "ops.metal"
                air = Path(temporary) / "ops.air"
                target = Path(temporary) / "ops.metallib"
                source.write_text(f"#define VPIPE_ELT {scalar}\n" + _SOURCE.read_text())
                subprocess.run(
                    [
                        "xcrun",
                        "metal",
                        "-std=metal3.1",
                        "-fno-fast-math",
                        "-c",
                        str(source),
                        "-o",
                        str(air),
                    ],
                    check=True,
                    timeout=60,
                )
                subprocess.run(
                    ["xcrun", "metallib", str(air), "-o", str(target)], check=True, timeout=60
                )
                os.replace(target, path)
    _library.cache_clear()
    return _artifact_path(torch.float16).parent


def supports_silu(x):
    return (
        x.device.type == "mps"
        and x.dtype in _SCALARS
        and not x.requires_grad
        and x.ndim in (1, 2, 3)
        and x.is_contiguous()
        and x.shape[-1] % 2 == 0
        and 0 < x.numel() < 2**31
        and _library(x.dtype) is not None
    )


def silu_and_mul(x):
    width = x.shape[-1] // 2
    result = torch.empty(x.shape[:-1] + (width,), dtype=x.dtype, device=x.device)
    _library(x.dtype).swiglu_rs_f16(
        x,
        x[..., width:],
        result,
        x.numel() // (2 * width),
        width,
        0,
        0,
        threads=result.numel(),
    )
    return result


def supports_add_rms_norm(x, residual, weight):
    """AOT add-before-norm domain; independent of the existing SwiGLU predicate."""
    return (
        x.device.type == "mps"
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.ndim > 0
        and 0 < x.numel() < 2**31
        and 0 < x.shape[-1] < 2**31
        and x.is_contiguous()
        and not x.requires_grad
        and residual.shape == x.shape
        and residual.dtype == x.dtype
        and residual.device == x.device
        and residual.is_contiguous()
        and not residual.requires_grad
        and weight is not None
        and weight.shape == (x.shape[-1],)
        and weight.dtype == x.dtype
        and weight.device == x.device
        and weight.is_contiguous()
        and not weight.requires_grad
        and _library(x.dtype) is not None
    )


def _supports_add_rms_norm_epsilon(epsilon):
    # Metal float constants cannot represent overflow or preserve subnormals.
    # Unsupported epsilon values must retain Torch's original semantics.
    return (
        isinstance(epsilon, (float, int))
        and 0 <= epsilon <= torch.finfo(torch.float32).max
        and math.isfinite(epsilon)
        and (epsilon == 0 or epsilon >= torch.finfo(torch.float32).tiny)
    )


def add_rms_norm(x, residual, weight, epsilon):
    """Return (cast-before-weight RMSNorm, cast residual sum), without mutation."""
    if not _supports_add_rms_norm_epsilon(epsilon):
        raise ValueError("vpipe add_rms_norm requires zero or finite normal FP32 epsilon")
    if not supports_add_rms_norm(x, residual, weight):
        raise ValueError("unsupported vpipe add_rms_norm inputs or missing AOT artifact")
    result = torch.empty_like(x)
    summed = torch.empty_like(x)
    # Fixed 256-thread groups, not width threads: wide rows stride safely and
    # require only eight SIMD partials, with no width-sized threadgroup storage.
    _library(x.dtype).add_rms_norm_f16(
        x,
        residual,
        weight,
        result,
        summed,
        x.shape[-1],
        float(epsilon),
        threads=(256, x.numel() // x.shape[-1]),
        group_size=(256, 1),
    )
    return result, summed


def cached_neox_rope(x, cos, sin, rotary_dim):
    tokens, heads, dim = x.shape
    result = x.clone(memory_format=torch.contiguous_format)
    _library(x.dtype).rope_half_part_ftab_inplace_f16(
        result,
        cos.float().contiguous(),
        sin.float().contiguous(),
        heads,
        tokens,
        dim,
        rotary_dim,
        heads * dim,
        0,
        dim,
        threads=(dim, tokens, heads),
        group_size=(dim, 1, 1),
    )
    return result


def cached_neox_rope_qk(query, key, selected_cache, head_size):
    """Full-dimension NeoX Q/K; caller supplies validated contiguous cache rows."""
    tokens = selected_cache.shape[0]
    q_width, k_width = query.numel() // tokens, key.numel() // tokens
    out_q, out_k = torch.empty_like(query), torch.empty_like(key)
    _library(query.dtype).rope_qk_cached(
        query,
        key,
        selected_cache,
        out_q,
        out_k,
        (q_width, k_width, head_size),
        threads=((q_width + k_width) // 2, tokens),
        group_size=(256, 1),
    )
    return out_q, out_k


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(build_vpipe_kernels())
