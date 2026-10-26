# SPDX-License-Identifier: Apache-2.0
"""Linear graph policy boundaries and opt-in real Neural Engine execution."""

import copy
import json
import os

import pytest
import torch
import torch.nn.functional as F

from vllm_omni_metal.ane import cache, projection


def policy():
    record = cache.select_profile(513, [10.0, 10.1], {1: [9.0, 9.1]})
    record["numerical_check"] = {
        "relative_l2": 0.001,
        "scaled_max_abs": 0.002,
        "shape": [1, 513, 128],
        "ane_chunks": 1,
    }
    return {
        "schema": 1,
        "group": "a" * 64,
        "device": "unit-policy-device",
        "kind": "linear-bias-v1",
        "profiles": {"513:7": record},
    }


def test_disabled_projection_preserves_noncontiguous_native_math(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_METAL_ANE", "off")
    weight = torch.arange(35, dtype=torch.float32).reshape(5, 7) / 64
    bias = torch.arange(5, dtype=torch.float32) / 16
    inputs = torch.arange(42, dtype=torch.float32).reshape(2, 7, 3).transpose(1, 2)
    assert not inputs.is_contiguous()
    with torch.inference_mode():
        expected = F.linear(inputs, weight, bias)
        with projection.DenseProjection(weight, bias, scope="disabled") as group:
            actual = group(inputs, native=lambda value: F.linear(value, weight, bias))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.shape == (2, 3, 5)
            assert group.metrics["ane_chunks"] == 0
        with pytest.raises(RuntimeError, match="closed"):
            group(inputs, native=lambda value: F.linear(value, weight, bias))


def test_projection_rejects_invalid_source_and_staged_cpu(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_METAL_ANE", "auto")
    with pytest.raises(ValueError, match="source weight"):
        projection.DenseProjection(torch.ones(4, 8, dtype=torch.float64), scope="invalid")
    with pytest.raises(ValueError, match="bias"):
        projection.DenseProjection(torch.ones(4, 8), torch.ones(3), scope="invalid")
    with pytest.raises(ValueError, match="contiguous"):
        projection.DenseProjection(torch.ones(4, 8).T, scope="invalid")
    with pytest.raises(ValueError, match="staged replay"):
        projection.DenseProjection(torch.ones(4, 8), scope="invalid", _staged_policy=policy())


@pytest.mark.parametrize("fault", ["error", "shape", "rows", "chunks", "speed", "context"])
def test_profile_rejects_unbound_or_failed_evidence(fault):
    data = policy()
    record = data["profiles"]["513:7"]
    if fault == "error":
        record["numerical_check"]["relative_l2"] = 0.010001
    elif fault == "shape":
        record["numerical_check"]["shape"][-1] = 64
    elif fault == "rows":
        record["numerical_check"]["shape"] = [1, 514, 128]
    elif fault == "chunks":
        record["numerical_check"]["ane_chunks"] = 2
    elif fault == "speed":
        record["candidate"] = [10.0, 10.0]
    else:
        data["profiles"]["513:07"] = data["profiles"].pop("513:7")
    with pytest.raises(RuntimeError):
        projection._profiles(data["profiles"], 128)


@pytest.mark.parametrize(
    "body",
    [
        '{"schema":1,"schema":1}',
        '{"schema":1,"value":NaN}',
        '{"schema":1,"value":1e400}',
    ],
)
def test_policy_rejects_ambiguous_and_nonfinite_json(tmp_path, body):
    path = tmp_path / "policy.json"
    path.write_text(body)
    path.chmod(0o600)
    with pytest.raises(RuntimeError):
        projection._policy(path)


def test_policy_permissions_and_symlinks_are_rejected(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy()))
    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="0600"):
        projection._policy(path)
    path.chmod(0o600)
    linked = tmp_path / "linked.json"
    linked.symlink_to(path)
    with pytest.raises(RuntimeError, match="Unsafe"):
        projection._policy(linked)
    assert projection._policy(path) == policy()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="requires opt-in actual MPS/NE execution",
)
@pytest.mark.parametrize(
    "dtype,bias_enabled,shape,chunks",
    [
        (torch.float16, True, (513, 1024), 1),
        (torch.bfloat16, False, (2, 257, 1024), 1),
        (torch.float32, True, (2, 2, 129, 1024), 2),
    ],
)
def test_real_projection_staged_replay_and_deferred_publication(
    monkeypatch,
    dtype,
    bias_enabled,
    shape,
    chunks,
):
    if not (cache.cache_root() / "tools.json").is_file():
        pytest.skip("ANE tools must be explicitly built before this test")
    monkeypatch.setenv("VLLM_OMNI_METAL_ANE", "auto")
    scope = f"projection-real-regression:{dtype}:{shape}:{chunks}"
    with torch.inference_mode():
        weight = torch.full((2048, 1024), 0.25, dtype=dtype, device="mps")
        bias = torch.full((2048,), 0.125, dtype=dtype, device="mps") if bias_enabled else None
        inputs = torch.full(shape, 0.125, dtype=dtype, device="mps")

        def native(value):
            return F.linear(value, weight, bias)

        expected = native(inputs)
        projection.prepare_projection(weight, bias, scope=scope)
        with projection.DenseProjection(weight, bias, scope=scope, _benchmark=True) as group:
            group.set_benchmark_chunks(chunks)
            actual = group(inputs, native=native, context=7)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            group.finish_audit()
            group.set_benchmark_chunks(0)
            torch.testing.assert_close(
                group(inputs, native=native, context=7), expected, rtol=0, atol=0
            )
            group.set_benchmark_chunks(chunks)
            group(inputs, native=native, context=7)
            assert group._audit is False
            wrong_shape = inputs.reshape(-1, 1024).unsqueeze(0)
            if wrong_shape.shape == inputs.shape:
                wrong_shape = inputs.reshape(1, 1, -1, 1024)
            with pytest.raises(RuntimeError, match="completed audit"):
                group(wrong_shape, native=native, context=7)
            # This is a numerical/approval boundary test, not a speed claim. The
            # explicit timing fixture must never be published as a production policy.
            rows = inputs.numel() // 1024
            record = projection.publish_projection_profile(
                group,
                rows,
                [10.0, 10.1],
                {chunks: [9.0, 9.1]},
                context=7,
                publish=False,
            )
            staged = {
                "schema": 1,
                "group": group.identity,
                "device": group._bridge.device,
                "kind": "linear-bias-v1",
                "profiles": {cache.profile_label(rows, 7): record},
            }
            assert group.metrics["ane_rows"] > 0
            state = cache.cache_root() / f"{group.identity}.json"
            assert not state.exists()
        with projection.DenseProjection(weight, bias, scope=scope, _staged_policy=staged) as replay:
            torch.testing.assert_close(
                replay(inputs, native=native, context=7), expected, rtol=0, atol=0
            )
            assert replay.metrics["ane_chunks"] == chunks
            with pytest.raises(RuntimeError, match="immutable"):
                replay.set_benchmark_chunks(chunks)
        assert not state.exists()
        bad = copy.deepcopy(staged)
        bad["device"] = "wrong-device"
        with pytest.raises(RuntimeError, match="another device"):
            projection.DenseProjection(weight, bias, scope=scope, _staged_policy=bad)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_METAL_INTEGRATION") != "1" or not torch.backends.mps.is_available(),
    reason="requires opt-in actual MPS/NE execution",
)
def test_real_adaptive_allocation_freezes_audited_split(monkeypatch):
    if not (cache.cache_root() / "tools.json").is_file():
        pytest.skip("ANE tools must be explicitly built before this test")
    monkeypatch.setenv("VLLM_OMNI_METAL_ANE", "auto")
    with torch.inference_mode():
        weight = torch.full((1024, 1024), 0.25, dtype=torch.float16, device="mps")
        inputs = torch.full((1025, 1024), 0.125, dtype=torch.float16, device="mps")

        def native(value):
            return F.linear(value, weight)

        expected = native(inputs)
        scope = "real-predictive-allocation-freeze"
        projection.prepare_projection(weight, scope=scope)
        with projection.DenseProjection(weight, scope=scope, _benchmark=True) as group:
            group.set_benchmark_chunks("auto")
            for _ in range(4):
                torch.testing.assert_close(group(inputs, native=native), expected, rtol=0, atol=0)
            frozen = group.allocation_records[-1]["chunks"]
            assert 1 <= frozen <= 4
            group.finish_audit()
            before = group.metrics["ane_chunks"]
            for _ in range(3):
                torch.testing.assert_close(group(inputs, native=native), expected, rtol=0, atol=0)
            assert group.metrics["ane_chunks"] - before == 3 * frozen
            with pytest.raises(RuntimeError, match="open offline audit"):
                group.set_benchmark_chunks("auto")
            with pytest.raises(RuntimeError, match="audited shape/context"):
                group(inputs, native=native, context=1)
