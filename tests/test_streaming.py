# SPDX-License-Identifier: Apache-2.0
"""Dense streaming artifact validation and opt-in real MPS numerical execution."""

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from vllm_omni_metal.execution.streaming import StreamedLinear, prepare_linear

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
REJECTED = (ValueError, RuntimeError, OSError, TypeError)
MPS = pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="explicit MPS integration opt-in required",
)


def source(dtype=torch.float32, bias=True):
    module = torch.nn.Linear(7, 11, bias=bias, dtype=dtype)
    with torch.no_grad():
        module.weight.copy_((torch.arange(77).reshape(11, 7) - 38).to(dtype) / 64)
        module.weight[0, 0] = -0.0
        if bias:
            module.bias.copy_((torch.arange(11) - 5).to(dtype) / 32)
            module.bias[0] = -0.0
    return module


def raw(tensor):
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def manifest(directory):
    return json.loads((directory / "manifest.json").read_text())


def write_manifest(directory, data):
    (directory / "manifest.json").write_text(json.dumps(data, allow_nan=False))


def exported(tmp_path):
    module = source()
    directory = tmp_path / "projection"
    prepare_linear(module, directory, max_chunk_bytes=3 * 8 * 4)
    return directory


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias_enabled", [False, True])
def test_export_round_trips_complete_rows_and_preserves_source(tmp_path, dtype, bias_enabled):
    module = source(dtype, bias_enabled)
    weight = module.weight
    bias = module.bias
    expected_weight = raw(weight)
    expected_bias = raw(bias) if bias_enabled else b""
    row_bytes = (module.in_features + int(bias_enabled)) * weight.element_size()
    directory = tmp_path / "projection"

    prepare_linear(module, directory, max_chunk_bytes=3 * row_bytes + 1)

    data = manifest(directory)
    assert data["schema"] == 1
    assert data["dtype"] == str(dtype).removeprefix("torch.")
    assert data["in_features"] == 7 and data["out_features"] == 11
    assert data["bias"] is bias_enabled
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in directory.iterdir())
    assert all(path.stat().st_uid == os.getuid() for path in directory.iterdir())
    assert [chunk["row_end"] - chunk["row_start"] for chunk in data["chunks"]] == [3, 3, 3, 2]

    restored_weight, restored_bias = [], []
    cursor = 0
    for chunk in data["chunks"]:
        assert chunk["row_start"] == cursor
        rows = chunk["row_end"] - cursor
        payload = (directory / chunk["file"]).read_bytes()
        assert 0 < len(payload) == chunk["bytes"] == rows * row_bytes <= 3 * row_bytes + 1
        assert hashlib.sha256(payload).hexdigest() == chunk["sha256"]
        values = torch.frombuffer(bytearray(payload), dtype=dtype)
        restored_weight.append(values[: rows * 7].reshape(rows, 7))
        if bias_enabled:
            restored_bias.append(values[rows * 7 :])
        cursor = chunk["row_end"]
    assert cursor == module.out_features
    assert raw(torch.cat(restored_weight)) == expected_weight
    if bias_enabled:
        assert raw(torch.cat(restored_bias)) == expected_bias
    assert module.weight is weight and module.bias is bias
    assert raw(module.weight) == expected_weight
    assert not bias_enabled or raw(module.bias) == expected_bias
    assert module.training and module.weight.requires_grad
    assert not bias_enabled or module.bias.requires_grad
    assert module.weight.grad is None
    if bias_enabled:
        assert module.bias.grad is None


@pytest.mark.parametrize("bias_enabled", [False, True])
def test_budget_smaller_than_one_complete_row_leaves_no_artifact(tmp_path, bias_enabled):
    module = source(bias=bias_enabled)
    row_bytes = (module.in_features + int(bias_enabled)) * module.weight.element_size()
    directory = tmp_path / "projection"
    with pytest.raises(REJECTED):
        prepare_linear(module, directory, max_chunk_bytes=row_bytes - 1)
    assert not directory.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("budget", [0, -1, True, 1.0, "64"])
def test_invalid_export_budget_does_not_publish(tmp_path, budget):
    directory = tmp_path / "projection"
    with pytest.raises(REJECTED):
        prepare_linear(source(), directory, max_chunk_bytes=budget)
    assert not directory.exists()


def test_existing_directory_is_not_overwritten(tmp_path):
    directory = tmp_path / "projection"
    directory.mkdir(mode=0o700)
    sentinel = directory / "manifest.json"
    sentinel.write_bytes(b"user-owned existing artifact")
    with pytest.raises(REJECTED):
        prepare_linear(source(), directory)
    assert sentinel.read_bytes() == b"user-owned existing artifact"
    assert list(directory.iterdir()) == [sentinel]


def test_export_requires_absolute_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = Path("projection")
    with pytest.raises(REJECTED):
        prepare_linear(source(), directory)
    assert not directory.exists()


@pytest.mark.parametrize(
    "fault",
    ["subclass", "dtype", "weight_layout", "bias_layout", "bias_dtype", "bias_shape", "hook"],
)
def test_unsupported_source_is_rejected_before_directory_creation(tmp_path, fault):
    module = source()
    if fault == "subclass":

        class CustomLinear(torch.nn.Linear):
            pass

        module = CustomLinear(7, 11)
    elif fault == "dtype":
        module = module.double()
    elif fault == "weight_layout":
        module.weight = torch.nn.Parameter(module.weight.T.contiguous().T)
    elif fault == "bias_layout":
        module.bias = torch.nn.Parameter(torch.arange(22, dtype=torch.float32)[::2])
    elif fault == "bias_dtype":
        module.bias = torch.nn.Parameter(module.bias.double())
    elif fault == "bias_shape":
        module.bias = torch.nn.Parameter(module.bias[:10])
    else:
        module.register_forward_hook(lambda _module, _args, output: output)
    directory = tmp_path / "projection"
    with pytest.raises(REJECTED):
        prepare_linear(module, directory)
    assert not directory.exists()


def test_artifact_can_be_opened_without_device_execution_and_close_preserves_it(tmp_path):
    directory = exported(tmp_path)
    expected = {path.name: path.read_bytes() for path in directory.iterdir()}
    with StreamedLinear(directory) as streamed:
        assert isinstance(streamed, torch.nn.Module)
        assert not streamed.training
        assert list(streamed.parameters()) == []
        assert list(streamed.buffers()) == []
    streamed.close()
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == expected
    with pytest.raises(REJECTED):
        streamed(torch.ones(7))


@pytest.mark.parametrize(
    "body",
    [
        "{",
        "[]",
        "null",
        '{"schema":1,"schema":1}',
        '{"schema":1,"unexpected":NaN}',
        '{"schema":1,"unexpected":Infinity}',
    ],
)
def test_malformed_or_ambiguous_manifest_is_rejected(tmp_path, body):
    directory = exported(tmp_path)
    (directory / "manifest.json").write_text(body)
    with pytest.raises(REJECTED):
        StreamedLinear(directory)


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "schema_bool",
        "dtype",
        "input_bool",
        "input_float",
        "output_string",
        "bias_int",
        "chunks_object",
        "empty_chunks",
        "missing_field",
        "extra_field",
        "row_bool",
        "row_float",
        "gap",
        "overlap",
        "tail",
        "bytes_bool",
        "bytes_count",
        "hash",
        "parent_path",
        "absolute_path",
        "duplicate_file",
    ],
)
def test_invalid_manifest_contract_is_rejected(tmp_path, fault):
    directory = exported(tmp_path)
    data = manifest(directory)
    chunk = data["chunks"][0]
    if fault == "schema":
        data["schema"] = 2
    elif fault == "schema_bool":
        data["schema"] = True
    elif fault == "dtype":
        data["dtype"] = "float64"
    elif fault == "input_bool":
        data["in_features"] = True
    elif fault == "input_float":
        data["in_features"] = 7.0
    elif fault == "output_string":
        data["out_features"] = "11"
    elif fault == "bias_int":
        data["bias"] = 1
    elif fault == "chunks_object":
        data["chunks"] = {}
    elif fault == "empty_chunks":
        data["chunks"] = []
    elif fault == "missing_field":
        del chunk["sha256"]
    elif fault == "extra_field":
        chunk["unexpected"] = 1
    elif fault == "row_bool":
        chunk["row_start"] = False
    elif fault == "row_float":
        chunk["row_end"] = 3.0
    elif fault == "gap":
        data["chunks"][1]["row_start"] += 1
    elif fault == "overlap":
        data["chunks"][1]["row_start"] -= 1
    elif fault == "tail":
        data["chunks"].pop()
    elif fault == "bytes_bool":
        chunk["bytes"] = True
    elif fault == "bytes_count":
        chunk["bytes"] -= 1
    elif fault == "hash":
        chunk["sha256"] = "not-a-sha256"
    elif fault == "parent_path":
        chunk["file"] = "../outside.bin"
    elif fault == "absolute_path":
        chunk["file"] = str(tmp_path / "outside.bin")
    else:
        data["chunks"][1]["file"] = chunk["file"]
    write_manifest(directory, data)
    with pytest.raises(REJECTED):
        StreamedLinear(directory)


def test_duplicate_nested_json_keys_are_rejected(tmp_path):
    directory = exported(tmp_path)
    path = directory / "manifest.json"
    body = json.dumps(manifest(directory))
    path.write_text(body.replace('"row_start": 0', '"row_start": 0, "row_start": 0', 1))
    with pytest.raises(REJECTED):
        StreamedLinear(directory)


@pytest.mark.parametrize("fault", ["missing", "truncate", "append"])
def test_missing_or_wrong_size_chunk_is_rejected(tmp_path, fault):
    directory = exported(tmp_path)
    path = directory / manifest(directory)["chunks"][0]["file"]
    if fault == "missing":
        path.unlink()
    elif fault == "truncate":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(REJECTED):
        StreamedLinear(directory)


@pytest.mark.parametrize("target", ["directory", "manifest", "chunk"])
@pytest.mark.parametrize("fault", ["permissions", "symlink"])
def test_unsafe_artifact_files_are_rejected(tmp_path, target, fault):
    directory = exported(tmp_path)
    if target == "directory":
        path = directory
    elif target == "manifest":
        path = directory / "manifest.json"
    else:
        path = directory / manifest(directory)["chunks"][0]["file"]
    if fault == "permissions":
        path.chmod(0o755 if target == "directory" else 0o644)
    else:
        original = tmp_path / "original"
        path.rename(original)
        path.symlink_to(original, target_is_directory=target == "directory")
    with pytest.raises(REJECTED):
        StreamedLinear(directory)


@pytest.mark.integration
@MPS
class TestStreamedLinearMPS:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("bias_enabled", [False, True])
    @pytest.mark.parametrize("layout", ["vector", "matrix", "strided"])
    def test_multiple_chunks_tail_and_repeated_forward(self, tmp_path, dtype, bias_enabled, layout):
        module = source(dtype, bias_enabled)
        row_bytes = (7 + int(bias_enabled)) * module.weight.element_size()
        budget = 3 * row_bytes
        directory = tmp_path / "projection"
        prepare_linear(module, directory, max_chunk_bytes=budget)
        artifact_bytes = {path.name: path.read_bytes() for path in directory.iterdir()}
        assert [c["row_end"] - c["row_start"] for c in manifest(directory)["chunks"]] == [
            3,
            3,
            3,
            2,
        ]
        with torch.inference_mode():
            weight = module.weight.detach().to("mps")
            bias = module.bias.detach().to("mps") if bias_enabled else None
            if layout == "vector":
                inputs = (torch.arange(14, device="mps", dtype=dtype) / 16)[::2]
                assert not inputs.is_contiguous()
            elif layout == "matrix":
                inputs = torch.arange(28, device="mps", dtype=dtype).reshape(4, 7) / 32
            else:
                inputs = (torch.arange(84, device="mps", dtype=dtype) / 64).reshape(2, 7, 6)
                inputs = inputs.transpose(1, 2)[:, ::2, :]
                assert not inputs.is_contiguous()
            original_inputs = inputs.clone()
            expected = F.linear(inputs, weight, bias)
            tolerance = {torch.float32: 2e-6, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
            with StreamedLinear(directory, max_weight_bytes=budget, device="mps") as streamed:
                assert not streamed.training
                assert list(streamed.parameters()) == []
                assert list(streamed.buffers()) == []
                for _ in range(3):
                    actual = streamed(inputs)
                    assert actual.shape == (*inputs.shape[:-1], 11)
                    assert actual.device.type == "mps" and actual.dtype == dtype
                    assert not actual.requires_grad
                    assert torch.isfinite(actual).all().item()
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(inputs, original_inputs, rtol=0, atol=0)
            with pytest.raises(REJECTED):
                streamed(inputs)
            streamed.close()
        assert {path.name: path.read_bytes() for path in directory.iterdir()} == artifact_bytes

    @pytest.mark.parametrize("fault", ["dtype", "grad", "cpu", "empty", "width", "scalar"])
    def test_invalid_input_never_falls_back(self, tmp_path, fault):
        directory = exported(tmp_path)
        inputs = torch.ones(2, 7, device="mps")
        if fault == "dtype":
            inputs = inputs.half()
        elif fault == "grad":
            inputs.requires_grad_(True)
        elif fault == "cpu":
            inputs = inputs.cpu()
        elif fault == "empty":
            inputs = inputs[:0]
        elif fault == "width":
            inputs = inputs[:, :6]
        else:
            inputs = inputs[0, 0]
        with StreamedLinear(directory) as streamed, torch.no_grad(), pytest.raises(REJECTED):
            streamed(inputs)

    @pytest.mark.parametrize(
        "operation", ["train", "to_dtype", "to_device", "half", "float", "cpu", "state_dict"]
    )
    def test_training_cast_move_and_torch_serialization_are_rejected(self, tmp_path, operation):
        directory = exported(tmp_path)
        with StreamedLinear(directory) as streamed:
            with pytest.raises(REJECTED):
                if operation == "train":
                    streamed.train()
                elif operation == "to_dtype":
                    streamed.to(dtype=torch.float16)
                elif operation == "to_device":
                    streamed.to("cpu")
                elif operation == "half":
                    streamed.half()
                elif operation == "float":
                    streamed.float()
                elif operation == "cpu":
                    streamed.cpu()
                else:
                    streamed.state_dict()
            with torch.inference_mode():
                inputs = torch.ones(2, 7, device="mps")
                expected = F.linear(inputs, source().weight.to("mps"), source().bias.to("mps"))
                torch.testing.assert_close(streamed(inputs), expected, rtol=2e-6, atol=2e-6)

    @pytest.mark.parametrize("when", ["before_open", "after_forward"])
    @pytest.mark.parametrize("fault", ["bytes", "truncate", "missing"])
    def test_chunk_changes_are_rejected_without_fallback(self, tmp_path, when, fault):
        directory = exported(tmp_path)
        data = manifest(directory)
        path = directory / data["chunks"][-1]["file"]
        streamed = None
        with torch.inference_mode():
            inputs = torch.ones(2, 7, device="mps")
            try:
                if when == "after_forward":
                    streamed = StreamedLinear(directory)
                    reference = source()
                    expected = F.linear(
                        inputs, reference.weight.to("mps"), reference.bias.to("mps")
                    )
                    torch.testing.assert_close(streamed(inputs), expected, rtol=2e-6, atol=2e-6)
                payload = path.read_bytes()
                if fault == "bytes":
                    path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
                elif fault == "truncate":
                    path.write_bytes(payload[:-1])
                else:
                    path.unlink()
                with pytest.raises(REJECTED):
                    if streamed is None:
                        streamed = StreamedLinear(directory)
                    streamed(inputs)
            finally:
                if streamed is not None:
                    streamed.close()

    def test_runtime_budget_is_checked_against_complete_chunk(self, tmp_path):
        directory = exported(tmp_path)
        budget = max(chunk["bytes"] for chunk in manifest(directory)["chunks"])
        with pytest.raises(REJECTED):
            StreamedLinear(directory, max_weight_bytes=budget - 1)
        with StreamedLinear(directory, max_weight_bytes=budget) as streamed, torch.inference_mode():
            inputs = torch.ones(7, device="mps")
            reference = source()
            expected = F.linear(inputs, reference.weight.to("mps"), reference.bias.to("mps"))
            torch.testing.assert_close(streamed(inputs), expected, rtol=2e-6, atol=2e-6)
