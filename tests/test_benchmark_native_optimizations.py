# SPDX-License-Identifier: Apache-2.0
"""Reject unsafe benchmark requests before model loading or shader compilation."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "args,available",
    [
        (["--prompt-tokens", "0"], 8 * 1024**3),
        (["--output-tokens", "65"], 8 * 1024**3),
        (["--batch-size", "5"], 8 * 1024**3),
        (["--iterations", "3"], 8 * 1024**3),
        ([], 3 * 1024**3),
    ],
)
def test_native_benchmark_stops_before_loading(monkeypatch, args, available):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=available))
    monkeypatch.setattr(sys, "argv", ["benchmark_native_optimizations.py", *args])
    script = Path(__file__).resolve().parents[1] / "examples/benchmark_native_optimizations.py"
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2


def test_native_benchmark_requires_local_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        sys, "argv", ["benchmark_native_optimizations.py", "--model", str(tmp_path)]
    )
    script = Path(__file__).resolve().parents[1] / "examples/benchmark_native_optimizations.py"
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2
