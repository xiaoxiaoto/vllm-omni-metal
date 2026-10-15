# SPDX-License-Identifier: Apache-2.0
"""Private, content-addressed ANE artifacts shared by the runtime and offline tools."""

import hashlib
import json
import math
import os
import platform
import re
import stat
import statistics
import tempfile
from functools import cache
from pathlib import Path

import torch

SCHEMA = 1
CHUNK = 256
TILE = 1024
RESERVE = 4 * 1024**3
MAX_WEIGHTS = 256 * 1024**2
NATIVE = Path(__file__).parent / "_native/ane"
KEY = re.compile(r"[0-9a-f]{64}\Z")


@cache
def cache_root():
    digest = hashlib.sha256()
    package = Path(__file__).parent
    sources = [
        *package.glob("*.py"),
        *(
            p
            for p in (package / "_native").rglob("*")
            if p.suffix in (".cc", ".h", ".swift", ".metal", ".mm")
        ),
    ]
    for path in sorted(sources):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    digest.update(
        repr(
            (
                platform.machine(),
                platform.mac_ver()[0],
                platform.version(),
                torch.__version__,
                torch.version.git_version,
            )
        ).encode()
    )
    return Path.home() / ".cache/vllm-omni-metal/ane" / digest.hexdigest()


def private_directory(path):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f"ANE cache must be a private owned directory: {path}")


def _check_cache_base():
    base = Path.home() / ".cache"
    if base.exists() or base.is_symlink():
        info = base.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise RuntimeError(f"Unsafe ANE cache base: {base}")


def check_existing_cache():
    _check_cache_base()
    root = cache_root()
    for path in (root.parent.parent, root.parent, root):
        if path.exists() or path.is_symlink():
            private_directory(path)


def ensure_cache():
    _check_cache_base()
    root = cache_root()
    for path in (root.parent.parent, root.parent, root):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        private_directory(path)
    return root


def checked_file(path, max_bytes=MAX_WEIGHTS + 1024**2):
    private_directory(path.parent)
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o022
        or info.st_size > max_bytes
    ):
        raise RuntimeError(f"Unsafe ANE artifact: {path}")
    return path


def checksum(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_json(path):
    checked_file(path, 1024**2)
    try:
        data = json.loads(path.read_text())
    except (ValueError, UnicodeError) as error:
        raise RuntimeError(f"Invalid ANE manifest: {path}") from error
    if (
        not isinstance(data, dict)
        or type(data.get("schema")) is not int
        or data["schema"] != SCHEMA
    ):
        raise RuntimeError(f"Unknown ANE manifest schema: {path}")
    return data


def write_json(path, data):
    private_directory(path.parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump({**data, "schema": SCHEMA}, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_files(directory, expected, names):
    private_directory(directory)
    if not isinstance(expected, dict) or set(expected) != set(names):
        raise RuntimeError("Invalid ANE artifact file list")
    for name in names:
        path = directory / name
        if path.parent != directory:
            private_directory(path.parent)
        digest = expected[name]
        if not isinstance(digest, str) or not KEY.fullmatch(digest):
            raise RuntimeError("Invalid ANE artifact digest")
        if checksum(checked_file(path)) != digest:
            raise RuntimeError(f"ANE artifact checksum mismatch: {path}")


def weight_shape(packed, down):
    if packed.ndim != 2 or down.ndim != 2 or packed.shape[0] % 2:
        raise ValueError("dense packed gate/up and down matrices are required")
    inner, hidden = packed.shape[0] // 2, packed.shape[1]
    if min(inner, hidden) <= 0 or down.shape != (hidden, inner):
        raise ValueError("incompatible SwiGLU projection shapes")
    if (
        packed.dtype != down.dtype
        or packed.device != down.device
        or not packed.is_floating_point()
        or not packed.is_contiguous()
        or not down.is_contiguous()
    ):
        raise ValueError("matching contiguous floating SwiGLU weights are required")
    return hidden, inner


def eligible(packed, down):
    hidden, inner = weight_shape(packed, down)
    return (
        packed.dtype in (torch.float16, torch.bfloat16)
        and hidden % TILE == 0
        and inner % TILE == 0
        and 6 * hidden * inner <= MAX_WEIGHTS
    )


def weight_key(packed, down):
    hidden, inner = weight_shape(packed, down)
    digest = hashlib.sha256(repr((hidden, inner, str(packed.dtype), CHUNK)).encode())
    for tensor in (packed, down):
        # Separate device readback and dtype conversion; never mutate an MPS source view.
        host = tensor.detach().to(device="cpu", copy=True)
        digest.update(memoryview(host.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def group_key(keys, scope):
    if not isinstance(scope, str) or not scope or len(scope) > 1024:
        raise ValueError("a bounded pipeline revision scope is required")
    return hashlib.sha256(
        json.dumps([keys, scope, "fp16-rne"], separators=(",", ":")).encode()
    ).hexdigest()


def model_directory(key):
    if not isinstance(key, str) or not KEY.fullmatch(key):
        raise RuntimeError("Invalid ANE model identity")
    return cache_root() / key


def verify_model(key, hidden, inner, device, *, runtime=False):
    directory = model_directory(key)
    private_directory(directory)
    data = read_json(directory / "manifest.json")
    if (
        data.get("key") != key
        or data.get("shape") != [CHUNK, hidden, inner]
        or data.get("device") != device
        or data.get("ane_preferred") is not True
        or data.get("runtime") is not runtime
    ):
        raise RuntimeError("ANE model manifest does not match the execution contract")
    names = ["model.mil", "coremldata.bin"]
    if not runtime:
        names.append("weights/weight.bin")
    verify_files(directory / "module.mlmodelc", data.get("files"), names)
    tests = data.get("numerical_tests")
    if not runtime and (
        not isinstance(tests, list)
        or len(tests) != 2
        or any(
            not isinstance(t, dict)
            or any(
                type(t.get(k)) not in (int, float)
                or not math.isfinite(t[k])
                or not 0 <= t[k] <= 0.01
                for k in ("relative_l2", "scaled_max_abs")
            )
            for t in tests
        )
    ):
        raise RuntimeError("ANE graph has no passing numerical self-test")
    return directory / "module.mlmodelc"


def runtime_key(hidden, inner):
    return hashlib.sha256(repr(("runtime", CHUNK, hidden, inner)).encode()).hexdigest()


def samples(values):
    if (
        not isinstance(values, list)
        or len(values) < 2
        or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in values)
    ):
        raise ValueError("at least two finite positive full-pipeline latency samples are required")
    return statistics.median(values)


def select_profile(rows, baseline, candidates):
    if type(rows) is not int or rows <= CHUNK:
        raise ValueError("profile must retain at least one GPU row")
    reference = samples(baseline)
    best, best_time, measured = 0, reference, baseline
    for chunks, values in candidates.items():
        if type(chunks) is not int or not 1 <= chunks <= (rows - 1) // CHUNK:
            raise ValueError("invalid ANE chunk count")
        elapsed = samples(values)
        if elapsed < best_time and elapsed < reference * 0.97 and max(values) < min(baseline):
            best, best_time, measured = chunks, elapsed, values
    return {"chunks": best, "baseline": baseline, "candidate": measured}


def profile_label(rows, context=None):
    if type(rows) is not int or rows <= CHUNK:
        raise ValueError("profile must retain a GPU row")
    if context is not None and (type(context) is not int or not 0 <= context < 2**31):
        raise ValueError("pipeline context must be a bounded nonnegative shape qualifier")
    return str(rows) if context is None else f"{rows}:{context}"


def validate_profiles(data):
    if not isinstance(data, dict):
        raise RuntimeError("Invalid ANE pipeline profiles")
    profiles = {}
    for label, record in data.items():
        try:
            parts = label.split(":")
            if len(parts) not in (1, 2):
                raise ValueError("invalid shape key")
            rows = int(parts[0])
            context = int(parts[1]) if len(parts) == 2 else None
            if profile_label(rows, context) != label or not isinstance(record, dict):
                raise ValueError("invalid row key")
            chunks = record["chunks"]
            if type(chunks) is not int or not 0 <= chunks <= (rows - 1) // CHUNK or rows <= CHUNK:
                raise ValueError("invalid row split")
            base = samples(record["baseline"])
            elapsed = samples(record["candidate"])
            if chunks and (
                elapsed >= base * 0.97 or max(record["candidate"]) >= min(record["baseline"])
            ):
                raise ValueError("profile does not pass the 3% whole-pipeline gain gate")
            profiles[rows, context] = chunks
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Invalid ANE whole-pipeline calibration") from error
    return profiles
