# SPDX-License-Identifier: Apache-2.0
"""Benchmark resource/input limits must fail before any GPU allocation."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "args,available",
    [
        (["--batch-size", "0"], 32 * 1024**3),
        (["--seq-len", "1000000"], 32 * 1024**3),
        (["--iterations", "1"], 32 * 1024**3),
        (["--heads", "3"], 32 * 1024**3),
        (["--kv-heads", "0"], 32 * 1024**3),
        (["--batch-size", "32", "--seq-len", "16384", "--heads", "128"], 32 * 1024**3),
        ([], 2 * 1024**3),
    ],
)
def test_benchmark_rejects_unsafe_arguments_before_gpu_allocation(monkeypatch, args, available):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=available))
    monkeypatch.setattr(sys, "argv", ["benchmark_attention.py", *args])

    def forbidden(*args, **kwargs):
        raise AssertionError("unsafe benchmark must not allocate GPU tensors")

    monkeypatch.setattr(torch, "randn", forbidden)
    script = Path(__file__).resolve().parents[1] / "examples/benchmark_attention.py"
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2
