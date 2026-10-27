# SPDX-License-Identifier: Apache-2.0
"""Explicit offline ANE tools, graph preparation and whole-pipeline approval.

No model loading lives here: independent adapters pass immutable dense weights
and benchmark their own unchanged pipeline before publishing a shape profile.
"""

import argparse
import ctypes
import fcntl
import math
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F

from vllm_omni_metal.ane import cache as artifacts
from vllm_omni_metal.ane.runtime import _Bridge, _errors, _headroom_retry


def _budget(disk, memory=artifacts.RESERVE):
    if (
        psutil.virtual_memory().available < memory
        or shutil.disk_usage(Path.home()).free < disk + 2 * 1024**3
    ):
        raise RuntimeError("ANE preparation requires 4 GiB RAM headroom and 2 GiB disk reserve")


def _lock(root):
    path = root / "prepare.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        artifacts.checked_file(path)
        fcntl.flock(fd, fcntl.LOCK_EX)
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "a")


def build_tools():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("ANE preparation requires native Apple Silicon macOS")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no CPU fallback")
    if not shutil.which("xcrun"):
        raise RuntimeError(
            "Apple Command Line Tools are required; no tools are installed automatically"
        )
    _budget(512 * 1024**2)
    root = artifacts.ensure_cache()
    with _lock(root):
        if (root / "tools.json").exists() or (root / "tools.json").is_symlink():
            _Bridge()
            return root
        with tempfile.TemporaryDirectory(prefix="tools-", dir=root) as temporary:
            temporary = Path(temporary)
            subprocess.run(
                [
                    "xcrun",
                    "clang++",
                    "-std=c++20",
                    "-O2",
                    "-mmacosx-version-min=15.0",
                    str(artifacts.NATIVE / "ane-emitter.cc"),
                    str(artifacts.NATIVE / "emit.cc"),
                    "-o",
                    str(temporary / "emit"),
                ],
                check=True,
                timeout=120,
            )
            subprocess.run(
                [
                    "xcrun",
                    "swiftc",
                    "-O",
                    "-emit-library",
                    "-module-name",
                    "OmniMetalANE",
                    "-target",
                    "arm64-apple-macos15.0",
                    str(artifacts.NATIVE / "bridge.swift"),
                    "-o",
                    str(temporary / "bridge.dylib"),
                ],
                check=True,
                timeout=120,
            )
            library = ctypes.CDLL(str(temporary / "bridge.dylib"))
            library.ffn_device.restype = ctypes.c_char_p
            device = library.ffn_device().decode()
            for name in ("emit", "bridge.dylib"):
                target = root / name
                if target.exists() or target.is_symlink():
                    artifacts.checked_file(target)
                (temporary / name).chmod(0o700 if name == "emit" else 0o600)
                os.replace(temporary / name, target)
            artifacts.write_json(
                root / "tools.json",
                {
                    "identity": root.name,
                    "device": device,
                    "files": {
                        name: artifacts.checksum(root / name) for name in ("emit", "bridge.dylib")
                    },
                },
            )
    return root


def _seal(directory):
    for path in directory.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    directory.chmod(0o700)


def _test_graph(bridge, constant, runtime, packed, down):
    """Check both static and mutable-weight graphs against an independent FP32 FFN."""
    hidden, inner = artifacts.weight_shape(packed, down)
    generator = torch.Generator(device="cpu").manual_seed(42)
    x = (torch.randn((artifacts.CHUNK, hidden), generator=generator, device="cpu") * 0.1).half()
    matrices = (
        packed[:inner].half().contiguous(),
        packed[inner:].half().contiguous(),
        down.half().contiguous(),
    )
    gate, up, wd = matrices
    expected = F.linear(
        F.silu(F.linear(x.float(), gate.float())) * F.linear(x.float(), up.float()), wd.float()
    )
    raw = torch.empty_like(x)
    constant_handle = bridge.load(constant, hidden)
    try:
        runtime_handle = bridge.load(runtime, hidden)
        try:
            for part, value in enumerate(matrices):
                if bridge.lib.ffn_stage(runtime_handle, part, value.data_ptr(), value.numel()) != 0:
                    raise RuntimeError("ANE self-test weight binding failed")
            errors = []
            for initial in (constant_handle, runtime_handle):
                active = initial

                def rescale(scale):
                    nonlocal active
                    active = runtime_handle
                    value = (up.float() / scale).half().contiguous()
                    if bridge.lib.ffn_stage(active, 1, value.data_ptr(), value.numel()) != 0:
                        raise RuntimeError("ANE self-test headroom staging failed")

                def run():
                    if (
                        bridge.lib.ffn_predict(
                            active,  # noqa: B023 - synchronous retries share the mutable handle
                            x.data_ptr(),
                            raw.data_ptr(),
                            x.numel(),
                            raw.numel(),
                        )
                        != 0
                    ):
                        raise RuntimeError("ANE numerical self-test prediction failed")
                    return raw

                rescale(1) if initial == runtime_handle else None
                finite, scale, _ = _headroom_retry(run, rescale, 1)
                if finite is None:
                    raise RuntimeError("ANE self-test exceeds bounded FP16 dynamic range")
                rel, scaled = _errors(finite.float() * scale, expected)
                if rel > 0.01 or scaled > 0.01:
                    raise RuntimeError(
                        f"ANE numerical self-test failed: relative L2={rel}, max={scaled}"
                    )
                errors.append({"relative_l2": rel, "scaled_max_abs": scaled})
            return errors
        finally:
            bridge.lib.ffn_unload(runtime_handle)
    finally:
        bridge.lib.ffn_unload(constant_handle)


def _emit_model(bridge, key, hidden, inner, packed=None, down=None):
    root = artifacts.cache_root()
    destination = artifacts.model_directory(key)
    runtime = packed is None
    if destination.exists() or destination.is_symlink():
        return artifacts.verify_model(key, hidden, inner, bridge.device, runtime=runtime)
    _budget(6 * hidden * inner if not runtime else 0)
    with tempfile.TemporaryDirectory(prefix="graph-", dir=root) as temporary:
        temporary = Path(temporary)
        model = temporary / "module.mlmodelc"
        payload = temporary / "slabs.bin"
        if not runtime:
            with payload.open("wb") as file:
                for matrix in (packed[:inner], packed[inner:], down):
                    for col in range(0, matrix.shape[0], artifacts.TILE):
                        for reduction in range(0, matrix.shape[1], artifacts.TILE):
                            tile = (
                                matrix[
                                    col : col + artifacts.TILE,
                                    reduction : reduction + artifacts.TILE,
                                ]
                                .half()
                                .T.contiguous()
                            )
                            if not torch.isfinite(tile).all():
                                raise ValueError(
                                    "ANE weights cannot be represented finitely as FP16"
                                )
                            tile.numpy().tofile(file)
        args = [
            str(root / "emit"),
            str(artifacts.CHUNK),
            str(hidden),
            str(inner),
            str(payload),
            str(model),
        ]
        subprocess.run(args + (["runtime"] if runtime else []), check=True, timeout=60)
        payload.unlink(missing_ok=True)
        _seal(model)
        if bridge.lib.ffn_plan(str(model).encode()) != 1:
            raise RuntimeError("CoreML does not prefer ANE for every projection; retain GPU")
        tests = None
        if not runtime:
            retry = (
                artifacts.model_directory(artifacts.runtime_key(hidden, inner)) / "module.mlmodelc"
            )
            tests = _test_graph(bridge, model, retry, packed, down)
        names = ["model.mil", "coremldata.bin"] + ([] if runtime else ["weights/weight.bin"])
        artifacts.write_json(
            temporary / "manifest.json",
            {
                "key": key,
                "shape": [artifacts.CHUNK, hidden, inner],
                "device": bridge.device,
                "runtime": runtime,
                "ane_preferred": True,
                "numerical_tests": tests,
                "files": {name: artifacts.checksum(model / name) for name in names},
            },
        )
        # Publish a complete directory only. Never replace existing model/cache assets.
        os.rename(temporary, destination)
    return destination / "module.mlmodelc"


@torch.inference_mode()
def prepare_group(weights, *, scope="dense-swiglu"):
    weights = tuple(weights)
    if not weights or not all(artifacts.eligible(*pair) for pair in weights):
        raise ValueError("ANE requires tile-aligned dense FP16/BF16 SwiGLU weights")
    if "VLLM_OMNI_METAL_ANE_SILU_OP" in os.environ:
        raise ValueError("ANE preparation requires the verified stable SiLU graph")
    root = build_tools()
    bridge = _Bridge()
    _budget(sum(6 * h * n for h, n in map(lambda pair: artifacts.weight_shape(*pair), weights)))
    keys = []
    with _lock(root):
        for packed, down in weights:
            hidden, inner = artifacts.weight_shape(packed, down)
            key = artifacts.weight_key(packed, down)
            _emit_model(bridge, artifacts.runtime_key(hidden, inner), hidden, inner)
            if not artifacts.model_directory(key).exists():
                # Never combine MPS readback with dtype conversion.
                p = packed.detach().to(device="cpu", copy=True)
                d = down.detach().to(device="cpu", copy=True)
                _emit_model(bridge, key, hidden, inner, p, d)
            else:
                artifacts.verify_model(key, hidden, inner, bridge.device)
            keys.append(key)
    return artifacts.group_key(keys, scope)


def publish_profile(group, rows, baseline, candidates, *, context=None):
    """Only an explicit offline pipeline benchmark may approve a runtime profile."""
    if not group._benchmark or group.disabled or group.identity is None or group._bridge is None:
        raise ValueError("a healthy offline calibration group is required")
    record = artifacts.select_profile(rows, baseline, candidates)
    chunks = record["chunks"]
    if chunks and any(
        (index, rows, chunks, context) not in group.checks for index in range(len(group.weights))
    ):
        raise ValueError(
            "every FFN must pass the unchanged live-input numerical gate before approval"
        )
    return _store_profile(group, rows, record, context)


def _store_profile(group, rows, record, context):
    root = artifacts.cache_root()
    with _lock(root):
        path = root / f"{group.identity}.json"
        profiles = {}
        if path.exists() or path.is_symlink():
            old = artifacts.read_json(path)
            if old.get("group") != group.identity or old.get("device") != group._bridge.device:
                raise RuntimeError("existing ANE calibration has a different identity")
            artifacts.validate_profiles(old.get("profiles"))
            profiles.update(old["profiles"])
        profiles[artifacts.profile_label(rows, context)] = record
        artifacts.validate_profiles(profiles)
        artifacts.write_json(
            path, {"group": group.identity, "device": group._bridge.device, "profiles": profiles}
        )
    return record


@torch.inference_mode()
def prepare_tiled_group(weights, *, scope):
    """Prepare separate gate/up/down matrices sequentially within existing limits.

    Published individual graphs are immutable reusable assets, not approvals.
    Failure leaves no partial graph directory or usable pipeline profile.
    """
    weights = tuple(tuple(layer) for layer in weights)
    shapes = [artifacts.tiled_weight_shape(*layer) for layer in weights]
    if not weights or any(
        layer[0].dtype not in (torch.float16, torch.bfloat16) for layer in weights
    ):
        raise ValueError("ANE requires dense FP16/BF16 SwiGLU weights")
    plans = [artifacts.inner_tiles(*shape) for shape in shapes]
    artifacts.group_key([], scope)
    if "VLLM_OMNI_METAL_ANE_SILU_OP" in os.environ:
        raise ValueError("ANE preparation requires the verified stable SiLU graph")
    largest = max(
        6 * hidden * (end - start)
        for (hidden, _), plan in zip(shapes, plans, strict=True)
        for start, end in plan
    )
    _budget(sum(6 * h * n for h, n in shapes), memory=artifacts.RESERVE + 3 * largest)
    root = build_tools()
    bridge = _Bridge()
    keys = []
    with _lock(root):
        for packed, down in artifacts.tiled_weights(weights, plans, host=True):
            hidden, inner = artifacts.weight_shape(packed, down)
            key = artifacts.weight_key(packed, down)
            _emit_model(bridge, artifacts.runtime_key(hidden, inner), hidden, inner)
            _emit_model(bridge, key, hidden, inner, packed, down)
            keys.append(key)
    return artifacts.tiled_group_key(keys, shapes, plans, scope)


def publish_tiled_profile(group, rows, baseline, candidates, *, context=None):
    """Approve only a healthy tiled group with final, live-input error evidence."""
    from vllm_omni_metal.ane.runtime import TiledDenseSwiGLU

    if not isinstance(group, TiledDenseSwiGLU):
        raise ValueError("a tiled offline calibration group is required")
    if not group._benchmark or group.disabled or group.identity is None or group._bridge is None:
        raise ValueError("a healthy offline calibration group is required")
    record = artifacts.select_profile(rows, baseline, candidates)
    chosen = record["chunks"]
    if chosen:
        for index, plan in enumerate(group.plans):
            check = group.checks.get((index, rows, chosen, context))
            if (
                not isinstance(check, dict)
                or check.get("tiles") != len(plan)
                or any(
                    type(check.get(name)) not in (int, float)
                    or not math.isfinite(check[name])
                    or not 0 <= check[name] <= 0.01
                    for name in ("relative_l2", "scaled_max_abs")
                )
                or any(
                    group.tile_metrics.get((index, tile), {}).get("ane_chunks", 0) < chosen
                    for tile in range(len(plan))
                )
            ):
                raise ValueError("every tiled FFN must pass the final live-input 1% numerical gate")
    record["numerical_checks"] = (
        [dict(group.checks[index, rows, chosen, context]) for index in range(len(group.weights))]
        if chosen
        else []
    )
    artifacts.validate_tiled_profiles(
        {artifacts.profile_label(rows, context): record}, group.shapes, group.plans
    )
    return _store_profile(group, rows, record, context)


def main():
    parser = argparse.ArgumentParser(description="显式离线构建通用 ANE host 工具；不下载模型或工具")
    parser.add_argument(
        "--bridge",
        action="store_true",
        required=True,
        help="构建 Swift/CoreML bridge 与 ANE emitter；模型权重和整流水线校准由独立适配器准备",
    )
    parser.parse_args()
    print(build_tools())


if __name__ == "__main__":
    main()
