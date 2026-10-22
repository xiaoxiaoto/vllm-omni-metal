# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig, CompilationConfig, OffloadConfig, ParallelConfig

from vllm_omni_metal.native.worker import MetalNativeWorker, validate_native_config
from vllm_omni_metal.platform import MetalOmniPlatform


@pytest.fixture
def config():
    return SimpleNamespace(
        parallel_config=ParallelConfig(distributed_executor_backend="uni"),
        model_config=SimpleNamespace(
            dtype=torch.float32, quantization=None, enable_sleep_mode=False
        ),
        cache_config=CacheConfig(enable_prefix_caching=False),
        offload_config=OffloadConfig(),
        lora_config=None,
        speculative_config=None,
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        attention_config=SimpleNamespace(backend=None),
    )


def test_native_config_boundaries(config):
    validate_native_config(config)
    config.attention_config.backend = "FLASHINFER"
    with pytest.raises(ValueError, match="CUSTOM"):
        validate_native_config(config)
    config.attention_config.backend = None
    config.scheduler_config.async_scheduling = True
    with pytest.raises(NotImplementedError, match="synchronous"):
        validate_native_config(config)
    config.scheduler_config.async_scheduling = False
    config.offload_config.uva.cpu_offload_gb = 1
    with pytest.raises(NotImplementedError, match="offload"):
        validate_native_config(config)


@pytest.mark.parametrize(
    "ops,flag",
    [
        (["all"], "+silu_and_mul"),
        (["none"], "+silu_and_mul"),
        (["none", "+silu_and_mul"], "+silu_and_mul"),
        (["all", "-silu_and_mul"], "-silu_and_mul"),
        (["all", "-rotary_embedding"], "+silu_and_mul"),
    ],
)
def test_native_config_enables_only_safe_oot_activation_and_preserves_opt_out(config, ops, flag):
    from vllm_metal.platform import MetalPlatform

    from vllm_omni_metal.execution.compat import apply_compat_patches

    config.parallel_config.worker_cls = "vllm_omni_metal.native.worker.MetalARWorker"
    config.compilation_config = CompilationConfig(custom_ops=ops)
    apply_compat_patches()
    for _ in range(2):
        MetalPlatform.check_and_update_config(config)
        rotary_flag = "-rotary_embedding" if "-rotary_embedding" in ops else "+rotary_embedding"
        assert config.compilation_config.custom_ops == ["none", flag, rotary_flag]
        assert config.device_config.device_type == "mps"


@pytest.mark.parametrize("operation", ["silu_and_mul", "rotary_embedding"])
def test_native_config_rejects_conflicting_activation_directives(config, operation):
    from vllm_metal.platform import MetalPlatform

    from vllm_omni_metal.execution.compat import apply_compat_patches

    config.parallel_config.worker_cls = "vllm_omni_metal.native.worker.MetalARWorker"
    config.compilation_config = CompilationConfig(custom_ops=["none"])
    # Hooks can mutate the list after the constructor's validation.
    config.compilation_config.custom_ops.extend([f"+{operation}", f"-{operation}"])
    apply_compat_patches()
    with pytest.raises(ValueError, match="both enable and disable"):
        MetalPlatform.check_and_update_config(config)


def test_shutdown_closes_model_operator_resources_before_connectors(monkeypatch):
    calls = []
    worker = object.__new__(MetalNativeWorker)
    worker.model_runner = SimpleNamespace(
        model=SimpleNamespace(close_metal_operators=lambda: calls.append("operators")),
        shutdown_omni_connectors=lambda: calls.append("connectors"),
        requests={"request": 1},
    )
    worker.compilation_config = SimpleNamespace(static_forward_context={"layer": 1})
    monkeypatch.setattr(MetalOmniPlatform, "empty_cache", lambda: None)
    worker.shutdown()
    assert calls == ["operators", "connectors"]
    assert worker.model_runner.model is None and not worker.model_runner.requests
    assert not worker.compilation_config.static_forward_context
    worker.shutdown()
    assert calls == ["operators", "connectors", "connectors"]


def test_kv_memory_reserves_host_headroom_and_respects_explicit_budget(config, monkeypatch):
    gib = 1024**3
    worker = object.__new__(MetalNativeWorker)
    worker.cache_config = config.cache_config
    worker.get_kv_cache_spec = lambda: {"layer": object()}
    monkeypatch.setattr(MetalOmniPlatform, "synchronize", lambda: None)
    monkeypatch.setattr(MetalOmniPlatform, "get_device_memory", lambda: (10 * gib, 30 * gib))
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 8 * gib)
    assert 0 < worker.determine_available_memory() <= 7 * gib
    config.cache_config.kv_cache_memory_bytes = 3 * gib
    assert worker.determine_available_memory() == 3 * gib
    config.cache_config.kv_cache_memory_bytes = 9 * gib
    with pytest.raises(MemoryError, match="headroom"):
        worker.determine_available_memory()
