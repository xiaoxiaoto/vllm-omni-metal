# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig, OffloadConfig, ParallelConfig

from vllm_omni_metal.native_worker import MetalNativeWorker, validate_native_config
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
