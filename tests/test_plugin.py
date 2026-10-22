# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
from importlib.metadata import entry_points
from types import SimpleNamespace

import pytest
import torch
from vllm_omni.diffusion.data import AttentionSpec, OmniDiffusionConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

import vllm_omni_metal
from vllm_omni_metal.diffusion.config import validate_config
from vllm_omni_metal.diffusion.worker import MetalDiffusionModelRunner
from vllm_omni_metal.platform import MetalOmniPlatform


def make_config(**kwargs):
    return OmniDiffusionConfig(diffusion_load_format="diffusers", dtype="float32", **kwargs)


def test_package_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm_omni_metal, sys; assert 'vllm' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        check=True,
    )


def test_entry_points():
    assert any(
        p.load() is vllm_omni_metal.register_platform
        for p in entry_points(group="vllm_omni.platform_plugins")
    )
    assert not any(
        p.value.startswith("vllm_omni_metal") for p in entry_points(group="vllm.general_plugins")
    )
    assert not any(
        p.value.startswith("vllm_omni_metal") for p in entry_points(group="vllm.platform_plugins")
    )


@pytest.mark.parametrize("system,machine", [("Linux", "aarch64"), ("Darwin", "x86_64")])
def test_unsupported_hardware(monkeypatch, system, machine):
    monkeypatch.setattr(vllm_omni_metal.py_platform, "system", lambda: system)
    monkeypatch.setattr(vllm_omni_metal.py_platform, "machine", lambda: machine)
    assert vllm_omni_metal.register_platform() is None


def test_mps_unavailable(monkeypatch):
    monkeypatch.setattr(vllm_omni_metal.py_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(vllm_omni_metal.py_platform, "machine", lambda: "arm64")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert vllm_omni_metal.register_platform() is None


def test_registration_preserves_explicit_start_method(monkeypatch):
    monkeypatch.setattr(vllm_omni_metal, "is_available", lambda: True)
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    assert vllm_omni_metal.register_platform().endswith("MetalOmniPlatform")
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "forkserver")
    vllm_omni_metal.register_platform()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "forkserver"


def test_registration_preserves_model_registries(monkeypatch):
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    pipelines = dict(OMNI_PIPELINES)
    models = dict(DiffusionModelRegistry.models)
    monkeypatch.setattr(vllm_omni_metal, "is_available", lambda: True)
    vllm_omni_metal.register_platform()
    assert OMNI_PIPELINES == pipelines
    assert DiffusionModelRegistry.models == models


def test_sdpa_default_and_idempotence():
    config = make_config()
    validate_config(config)
    validate_config(config)
    assert config.diffusion_attention_config.default.backend == "TORCH_SDPA"
    config.diffusion_attention_config.default = AttentionSpec(backend="torch_sdpa")
    validate_config(config)
    assert config.diffusion_attention_config.default.backend == "TORCH_SDPA"


@pytest.mark.parametrize(
    "option",
    [
        "enable_cpu_offload",
        "enable_layerwise_offload",
        "enable_sleep_mode",
        "step_execution",
        "streaming_output",
        "enforce_eager",
        "enable_prefix_caching",
    ],
)
def test_unsupported_flags(option):
    config = make_config()
    setattr(config, option, True)
    with pytest.raises(NotImplementedError, match=option):
        validate_config(config)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"num_gpus": 2}, "one GPU"),
        ({"distributed_executor_backend": "mp"}, "uni"),
        ({"cache_backend": "tea_cache"}, "TeaCache"),
        ({"diffusion_model_runner_cls": "custom.Runner"}, "default engine"),
    ],
)
def test_unsupported_config(kwargs, match):
    with pytest.raises(NotImplementedError, match=match):
        validate_config(make_config(**kwargs))


def test_native_pipeline_rejected():
    with pytest.raises(NotImplementedError, match="diffusers"):
        validate_config(OmniDiffusionConfig())


def test_dtype_and_cuda_attention_rejected():
    config = make_config()
    config.dtype = torch.float64
    with pytest.raises(ValueError, match="float64"):
        validate_config(config)
    config.dtype = torch.float32
    config.diffusion_attention_config.default = AttentionSpec(backend="FLASH_ATTN")
    with pytest.raises(ValueError, match="TORCH_SDPA"):
        validate_config(config)


def test_local_device_uses_single_mps_device(monkeypatch):
    from vllm_omni.diffusion.distributed import utils

    monkeypatch.setattr(utils, "current_omni_platform", MetalOmniPlatform())
    monkeypatch.setenv("LOCAL_RANK", "7")
    assert utils.get_local_device() == torch.device("mps")


def test_platform_contract():
    p = MetalOmniPlatform()
    assert p.get_torch_device() == torch.device("mps")
    assert p.is_out_of_tree() and not p.is_cuda()
    assert not p.is_pin_memory_available() and not p.supports_float64()
    assert not p.supports_torch_inductor() and not p.supports_cpu_offload()
    with pytest.raises(ValueError):
        p.get_torch_device(1)
    with pytest.raises(ValueError):
        p.set_device(torch.device("cpu"))
    assert p.get_omni_ar_worker_cls() == "vllm_omni_metal.native.worker.MetalARWorker"
    assert (
        p.get_omni_generation_worker_cls() == "vllm_omni_metal.native.worker.MetalGenerationWorker"
    )
    with pytest.raises(ValueError, match="TORCH_SDPA"):
        p.get_diffusion_attn_backend_cls("FLASH_ATTN", 64)
    from vllm.utils.import_utils import resolve_obj_by_qualname

    assert resolve_obj_by_qualname(p.get_diffusion_attn_backend_cls(None, 64)).get_name() == "SDPA"


def test_memory_is_clamped(monkeypatch):
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 100)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 130)
    assert MetalOmniPlatform.get_device_memory() == (0, 100)


def test_seeded_noise_uses_cpu():
    runner = object.__new__(MetalDiffusionModelRunner)
    runner.od_config = make_config()
    runner.device = torch.device("mps")
    first = OmniDiffusionSamplingParams(seed=42)
    second = OmniDiffusionSamplingParams(seed=42)
    runner._initialize_generator(first)
    runner._initialize_generator(second)
    assert first.generator_device == "cpu"
    assert first.generator.device.type == "cpu"
    assert torch.equal(
        torch.randn(8, generator=first.generator), torch.randn(8, generator=second.generator)
    )
    supplied = torch.Generator().manual_seed(7)
    params = OmniDiffusionSamplingParams(generator=supplied)
    runner._initialize_generator(params)
    assert params.generator is supplied
    with pytest.raises(ValueError, match="cpu"):
        runner._initialize_generator(OmniDiffusionSamplingParams(seed=1, generator_device="cuda"))


def test_compat_patch_preserves_llm_hook(monkeypatch):
    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm_metal.platform import MetalPlatform

    from vllm_omni_metal.execution.compat import apply_compat_patches

    calls = []

    def original(cls, config):
        calls.append(config)

    monkeypatch.setattr(MetalPlatform, "check_and_update_config", classmethod(original))
    apply_compat_patches()
    patched = MetalPlatform.check_and_update_config.__func__
    apply_compat_patches()
    assert MetalPlatform.check_and_update_config.__func__ is patched
    llm = SimpleNamespace(model_config=object())
    MetalPlatform.check_and_update_config(llm)
    assert calls == [llm]
    diffusion = SimpleNamespace(
        model_config=None, compilation_config=SimpleNamespace(), parallel_config=SimpleNamespace()
    )
    MetalPlatform.check_and_update_config(diffusion)
    assert calls == [llm]
    assert diffusion.compilation_config.mode == CompilationMode.NONE
    assert diffusion.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert diffusion.parallel_config.disable_custom_all_reduce


def test_memory_profiler_bridge_is_scoped(monkeypatch):
    import vllm.config

    from vllm_omni_metal.execution.compat import _patch_memory_profiler

    class Platform:
        @classmethod
        def get_current_memory_usage(cls, device=None):
            return 99

    _patch_memory_profiler(Platform)
    patched = Platform.get_current_memory_usage.__func__
    _patch_memory_profiler(Platform)
    assert Platform.get_current_memory_usage.__func__ is patched
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: None)
    assert Platform.get_current_memory_usage() == 99
    config = SimpleNamespace(device_config=SimpleNamespace(device_type="cpu"))
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    assert Platform.get_current_memory_usage() == 99
    config.device_config.device_type = "mps"
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 123)
    assert Platform.get_current_memory_usage() == 123
