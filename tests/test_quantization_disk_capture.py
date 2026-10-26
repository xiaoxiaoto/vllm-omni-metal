# SPDX-License-Identifier: Apache-2.0
"""Full-byte disk storage regressions; opt-in real MPS search equivalence."""

import gc
import json
import os
import stat
import weakref
from pathlib import Path

import pytest
import torch

from vllm_omni_metal.quantization import engine, linear
from vllm_omni_metal.quantization.capture import DiskCaptures
from vllm_omni_metal.quantization.engine import DynamicEngine

SCOPE = "disk-capture/full-original-activations"
MPS = pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="explicit MPS integration opt-in required",
)


def source(dtype=torch.float16, width=5, device="cpu"):
    generator = torch.Generator().manual_seed(42)
    module = torch.nn.Linear(width, 8, dtype=dtype).requires_grad_(False)
    module.weight.copy_(torch.randn(8, width, generator=generator).to(dtype) * 0.02)
    module.bias.copy_(torch.randn(8, generator=generator).to(dtype) * 0.01)
    return module.to(device)


def raw(tensor):
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def files(directory):
    return list(directory.iterdir())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_complete_noncontiguous_bytes_round_trip_and_metadata(tmp_path, dtype):
    module = source(dtype)
    x = torch.arange(20, dtype=dtype).reshape(5, 4).T
    x[0, 0] = -0.0
    expected = raw(x)
    directory = tmp_path / "capture"
    with DiskCaptures({"proj": module}, directory, max_bytes=80) as captures:
        for step in (24, 0, 16, 8):
            captures.record("proj", step, x)
        assert captures.names == ("proj",)
        assert captures.steps("proj") == [0, 8, 16, 24]
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in files(directory))
        assert all(path.read_bytes() == expected for path in files(directory))
        x.zero_()
        restored = captures.read("proj", 8)
        assert restored.dtype == dtype and restored.shape == (4, 5)
        assert restored.device.type == "cpu" and restored.is_contiguous()
        assert raw(restored) == expected
        restored.zero_()
        assert raw(captures.read("proj", 8)) == expected
        summary = captures.summary()
        json.dumps(summary, allow_nan=False)
        assert summary["bytes_used"] == len(expected) * 4
        assert summary["captures"]["proj"][0]["shape"] == [4, 5]
        summary["source_descriptors"]["proj"]["shape"][0] = 0
        captures.validate_source({"proj": module})  # Summary is not mutable internal state.
    assert not directory.exists()
    captures.close()
    with pytest.raises(ValueError, match="closed"):
        captures.read("proj", 0)


def test_no_source_or_activation_references_survive_record(tmp_path):
    module = source()
    x = torch.ones(2, 5, dtype=torch.float16)
    module_ref, weight_ref, x_ref = weakref.ref(module), weakref.ref(module.weight), weakref.ref(x)
    with DiskCaptures({"proj": module}, tmp_path / "capture") as captures:
        captures.record("proj", 0, x)
        del module, x
        gc.collect()
        assert module_ref() is None and weight_ref() is None and x_ref() is None
        assert captures.read("proj", 0).shape == (2, 5)


def test_resident_budget_is_per_full_sample_and_disk_budget_is_total(tmp_path):
    module = source()
    x = torch.ones(2, 5, dtype=torch.float16)
    with DiskCaptures(
        {"proj": module}, tmp_path / "capture", max_bytes=20, max_disk_bytes=80
    ) as captures:
        for step in range(4):
            captures.record("proj", step, x)
        assert captures.bytes_used == 80  # Four samples exceed the resident limit legally.
        with pytest.raises(ValueError, match="disk byte budget"):
            captures.record("proj", 4, x)
        assert len(files(captures.directory)) == 4
        assert captures.steps("proj") == list(range(4))
    with DiskCaptures({"proj": module}, tmp_path / "too-large", max_bytes=19) as captures:
        with pytest.raises(ValueError, match="resident byte budget"):
            captures.record("proj", 0, x)
        assert captures.bytes_used == 0 and files(captures.directory) == []


@pytest.mark.parametrize("budget", [0, -1, True, 1.0])
@pytest.mark.parametrize("argument", ["max_bytes", "max_disk_bytes"])
def test_invalid_budget_does_not_create_directory(tmp_path, budget, argument):
    directory = tmp_path / "capture"
    with pytest.raises(ValueError, match=argument):
        DiskCaptures({"proj": source()}, directory, **{argument: budget})
    assert not directory.exists()


@pytest.mark.parametrize("step", [-1, True, 1.0, "0", None])
def test_invalid_steps_never_write(tmp_path, step):
    with DiskCaptures({"proj": source()}, tmp_path / "capture") as captures:
        with pytest.raises(ValueError, match="distinct nonnegative"):
            captures.record("proj", step, torch.ones(2, 5, dtype=torch.float16))
        assert captures.bytes_used == 0 and files(captures.directory) == []


def test_duplicate_and_unknown_candidate_do_not_replace_valid_sample(tmp_path):
    with DiskCaptures({"proj": source()}, tmp_path / "capture") as captures:
        x = torch.ones(2, 5, dtype=torch.float16)
        captures.record("proj", 0, x)
        with pytest.raises(ValueError, match="distinct nonnegative"):
            captures.record("proj", 0, x * 2)
        with pytest.raises(ValueError, match="unknown"):
            captures.record("../escape", 1, x)
        assert torch.equal(captures.read("proj", 0), x)
        assert len(files(captures.directory)) == 1


@pytest.mark.parametrize("bad", ["dtype", "width", "empty", "nan", "grad"])
def test_invalid_full_activation_is_not_written(tmp_path, bad):
    x = torch.ones(2, 5, dtype=torch.float16)
    if bad == "dtype":
        x = x.float()
    elif bad == "width":
        x = x[:, :4]
    elif bad == "empty":
        x = x[:0]
    elif bad == "nan":
        x[0, 0] = float("nan")
    else:
        x.requires_grad_(True)
    with DiskCaptures({"proj": source()}, tmp_path / "capture") as captures:
        with pytest.raises(ValueError):
            captures.record("proj", 0, x)
        assert captures.bytes_used == 0 and files(captures.directory) == []


@pytest.mark.parametrize("changed", ["weight", "bias", "names"])
def test_source_mutation_rejected_by_capture_and_engine(tmp_path, changed):
    module = source()
    modules = {"proj": module}
    with DiskCaptures(modules, tmp_path / "capture") as captures:
        for step in range(4):
            captures.record("proj", step, torch.ones(2, 5, dtype=torch.float16))
        if changed == "names":
            modules = {"other": module}
        else:
            getattr(module, changed).add_(1)
        with pytest.raises(ValueError, match="source_changed"):
            engine._captures(modules, captures)
        assert captures.source_changed


@pytest.mark.parametrize(
    "tamper", ["bytes", "truncate", "append", "replace", "link", "fifo", "mode"]
)
def test_read_rejects_tampering_without_touching_replacement(tmp_path, tamper):
    captures = DiskCaptures({"proj": source()}, tmp_path / "capture")
    captures.record("proj", 0, torch.ones(2, 5, dtype=torch.float16))
    path = files(captures.directory)[0]
    if tamper == "bytes":
        with path.open("r+b") as stream:
            stream.write(b"\x00\x00")
    elif tamper == "truncate":
        with path.open("r+b") as stream:
            stream.truncate(1)
    elif tamper == "append":
        with path.open("ab") as stream:
            stream.write(b"x")
    elif tamper == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(path.read_bytes())
        replacement.chmod(0o600)
        replacement.replace(path)
    elif tamper == "link":
        path.unlink()
        path.symlink_to(tmp_path / "user-file")
    elif tamper == "fifo":
        path.unlink()
        os.mkfifo(path, 0o600)
    else:
        path.chmod(0o644)
    with pytest.raises((ValueError, OSError)):
        captures.read("proj", 0)
    captures.close()
    captures.close()
    if tamper in ("replace", "link", "fifo"):
        assert path.is_symlink() or path.exists()
    else:
        assert not captures.directory.exists()


def test_cleanup_keeps_user_entries_and_context_exception_cleans_owned_files(tmp_path):
    directory = tmp_path / "capture"
    with pytest.raises(RuntimeError, match="forward failed"):
        with DiskCaptures({"proj": source()}, directory) as captures:
            captures.record("proj", 0, torch.ones(2, 5, dtype=torch.float16))
            foreign = directory / "user-data"
            foreign.write_text("keep")
            raise RuntimeError("forward failed")
    assert files(directory) == [foreign]
    assert foreign.read_text() == "keep"
    captures.close()


def test_failed_write_removes_only_new_partial_file(tmp_path, monkeypatch):
    with DiskCaptures({"proj": source()}, tmp_path / "capture") as captures:
        captures.record("proj", 0, torch.ones(2, 5, dtype=torch.float16))
        fdopen = os.fdopen

        def fail_write(fd, mode, **kwargs):
            if mode == "wb":
                raise OSError("disk failure")
            return fdopen(fd, mode, **kwargs)

        monkeypatch.setattr(os, "fdopen", fail_write)
        with pytest.raises(OSError, match="disk failure"):
            captures.record("proj", 1, torch.ones(2, 5, dtype=torch.float16))
        assert len(files(captures.directory)) == 1
        assert captures.steps("proj") == [0] and captures.bytes_used == 20
        assert bool(captures.read("proj", 0).all())


def test_directory_must_be_new_absolute_and_without_symlink_ancestors(tmp_path):
    module = source()
    existing = tmp_path / "existing"
    existing.mkdir(mode=0o755)
    sentinel = existing / "user-data"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        DiskCaptures({"proj": module}, existing)
    assert sentinel.read_text() == "keep" and stat.S_IMODE(existing.stat().st_mode) == 0o755
    with pytest.raises(ValueError, match="absolute Path"):
        DiskCaptures({"proj": module}, Path("relative-capture"))
    link = tmp_path / "link"
    link.symlink_to(existing, target_is_directory=True)
    with pytest.raises(OSError):
        DiskCaptures({"proj": module}, link / "capture")
    assert not (existing / "capture").exists()
    with pytest.raises((ValueError, OSError)):
        DiskCaptures({"proj": module}, link)


def test_replaced_directory_is_not_read_or_cleaned(tmp_path):
    directory = tmp_path / "capture"
    captures = DiskCaptures({"proj": source()}, directory)
    captures.record("proj", 0, torch.ones(2, 5, dtype=torch.float16))
    moved = tmp_path / "moved"
    directory.rename(moved)
    directory.mkdir()
    foreign = directory / "user-data"
    foreign.write_text("keep")
    with pytest.raises(ValueError, match="directory identity"):
        captures.read("proj", 0)
    captures.close()
    assert foreign.read_text() == "keep"
    assert files(moved) == []  # Owned file removed through the original descriptor.


def test_search_admission_requires_four_and_matching_actual_steps(tmp_path):
    modules = {"first": source(), "second": source()}
    with DiskCaptures(modules, tmp_path / "capture") as captures:
        for name in modules:
            for step in range(3):
                captures.record(name, step, torch.ones(2, 5, dtype=torch.float16))
        with pytest.raises(ValueError, match="four distinct"):
            engine._captures(modules, captures)
        captures.record("first", 3, torch.ones(2, 5, dtype=torch.float16))
        captures.record("second", 4, torch.ones(2, 5, dtype=torch.float16))
        with pytest.raises(ValueError, match="same captured steps"):
            engine._captures(modules, captures)


def test_streamed_retention_search_releases_each_sample_before_next_read(tmp_path, monkeypatch):
    # Actual CPU source arithmetic, unaligned so no packed candidate. Bypass only
    # the MPS admission gate; real packed equivalence is separately opt-in below.
    validate = linear._linear
    monkeypatch.setattr(
        linear, "_linear", lambda module, *, mps=False, bits=8: validate(module, bits=bits)
    )
    monkeypatch.setattr(linear, "_environment", lambda: {"test_environment": True})
    module = source()
    data = {
        "proj": [(step, torch.full((2, 5), step + 1, dtype=torch.float16)) for step in range(4)]
    }
    expected = DynamicEngine.search({"proj": module}, data, scope=SCOPE)
    with DiskCaptures({"proj": module}, tmp_path / "capture", max_bytes=20) as captures:
        for step, tensor in data["proj"]:
            captures.record("proj", step, tensor)
        original_read = captures.read
        previous = []

        def one_at_a_time(name, step):
            assert all(reference() is None for reference in previous)
            value = original_read(name, step)
            previous.append(weakref.ref(value))
            return value

        monkeypatch.setattr(captures, "read", one_at_a_time)
        result = DynamicEngine.search({"proj": module}, captures, scope=SCOPE)
        assert len(previous) == 4 and all(reference() is None for reference in previous)
        assert result == expected


def test_source_mutation_during_streamed_search_cannot_publish_profile(tmp_path, monkeypatch):
    validate = linear._linear
    monkeypatch.setattr(
        linear, "_linear", lambda module, *, mps=False, bits=8: validate(module, bits=bits)
    )
    module = source()
    with DiskCaptures({"proj": module}, tmp_path / "capture") as captures:
        for step in range(4):
            captures.record("proj", step, torch.ones(2, 5, dtype=torch.float16))
        original_read = captures.read

        def mutate_after_admission(name, step):
            value = original_read(name, step)
            if step == 1:
                with torch.no_grad():
                    module.weight.add_(1)
            return value

        monkeypatch.setattr(captures, "read", mutate_after_admission)
        with pytest.raises(ValueError, match="source_changed"):
            DynamicEngine.search({"proj": module}, captures, scope=SCOPE)


@MPS
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_real_mps_disk_and_memory_search_have_identical_decisions_and_errors(tmp_path, dtype):
    module = source(dtype, width=64, device="mps")
    generator = torch.Generator().manual_seed(79)
    data = {
        "proj": [
            (step, torch.randn(2, 3, 64, generator=generator).to(dtype)) for step in (0, 8, 16, 24)
        ]
    }
    expected = DynamicEngine.search({"proj": module}, data, scope=SCOPE)
    with DiskCaptures({"proj": module}, tmp_path / "capture") as captures:
        for step, tensor in reversed(data["proj"]):
            captures.record("proj", step, tensor)
        result = DynamicEngine.search({"proj": module}, captures, scope=SCOPE)
    # Includes every per-step error, precision, local_pass, capture fingerprint,
    # source descriptor and profile hash, not merely the chosen bit width.
    assert result == expected
    assert result["local_pass"] and result["profile"]["calibration"]["approved"] is False
    for precision in ("fp32",) if dtype == torch.float32 else ("w4a16", "w8a16"):
        assert len(result["audits"]["proj"]["candidates"][precision]["steps"]) == 4
