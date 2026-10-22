# SPDX-License-Identifier: Apache-2.0
"""Model-independent execution contracts; no pretrained weights or downloads."""

import asyncio
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from vllm.config import LoadConfig
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from vllm_omni_metal import is_available
from vllm_omni_metal.diffusion.config import validate_config
from vllm_omni_metal.diffusion.worker import MetalDiffusionModelRunner, MetalDiffusionWorker
from vllm_omni_metal.execution.runtime import pipeline_runtime, resolve_pipeline_class


class NativeSignalPipeline(nn.Module):
    """A synthetic native pipeline using Omni's normal Torch weight loader."""

    support_audio_output = True
    supports_request_batch = False
    dummy_run_num_frames = 0

    def __init__(self, *, od_config):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(1))
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(od_config.model, None, None)
        ]

    def load_weights(self, weights):
        loaded = set()
        with torch.no_grad():
            for name, weight in weights:
                if name != "gain":
                    raise ValueError(f"Unexpected signal weight {name}")
                self.gain.copy_(weight)
                loaded.add(name)
        return loaded

    def forward(self, batch):
        assert len(batch.requests) == 1
        signal = torch.linspace(-1, 1, 1600, device=self.gain.device) * self.gain
        return DiffusionOutput(output={"audio": signal, "audio_sample_rate": 16000})


class SelfManagedPipeline(nn.Module):
    """A minimal MLX facade exercising the self-managed loading contract."""

    metal_runtime = "mlx"

    def __init__(self, *, od_config):
        super().__init__()
        self.loaded = False

    def load_weights(self):
        self.loaded = True


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(DiffusionModelRegistry, "models", dict(DiffusionModelRegistry.models))
    DiffusionModelRegistry.register_model("TestNativeSignal", NativeSignalPipeline)
    DiffusionModelRegistry.register_model("TestSelfManaged", SelfManagedPipeline)
    return DiffusionModelRegistry


@pytest.fixture
def native_config(registry, tmp_path):
    save_file({"gain": torch.tensor([0.25])}, str(tmp_path / "model.safetensors"))
    return OmniDiffusionConfig(
        model=str(tmp_path),
        model_class_name="TestNativeSignal",
        dtype="float32",
        enforce_eager=True,
        distributed_executor_backend="uni",
        max_num_seqs=1,
    )


def test_registered_native_pipeline_and_strict_weights(native_config):
    validate_config(native_config)
    assert resolve_pipeline_class(native_config) is NativeSignalPipeline
    assert pipeline_runtime(native_config) == "torch"
    loader = DiffusersPipelineLoader(LoadConfig(), od_config=native_config)
    pipeline = loader.load_model(load_device="cpu", device=torch.device("cpu"))
    assert torch.equal(pipeline.gain, torch.tensor([0.25]))
    output = pipeline(SimpleNamespace(requests=[None])).output
    assert output["audio_sample_rate"] == 16000
    assert output["audio"].std() > 0
    save_file({"unexpected": torch.ones(1)}, str(native_config.model) + "/model.safetensors")
    with pytest.raises(ValueError, match="Unexpected signal weight"):
        loader.load_model(load_device="cpu", device=torch.device("cpu"))


def test_external_pipeline_config_hook(native_config, monkeypatch):
    calls = []

    def validate(cls, config):
        calls.append(config)
        if config.custom_pipeline_args is not None:
            raise ValueError("unsupported external pipeline arguments")

    monkeypatch.setattr(
        NativeSignalPipeline, "validate_metal_config", classmethod(validate), raising=False
    )
    validate_config(native_config)
    assert calls == [native_config]
    native_config.custom_pipeline_args = {"unknown": True}
    with pytest.raises(ValueError, match="external pipeline arguments"):
        validate_config(native_config)


def test_self_managed_runtime_is_not_model_specific(registry, monkeypatch):
    import mlx.core as mx

    config = OmniDiffusionConfig(model_class_name="TestSelfManaged", dtype="float32")
    validate_config(config)
    assert pipeline_runtime(config) == "mlx"
    memory = iter([10, 26])
    monkeypatch.setattr(mx, "get_active_memory", lambda: next(memory))
    runner = object.__new__(MetalDiffusionModelRunner)
    runner.od_config = config
    runner.pipeline = None
    runner.load_model()
    assert isinstance(runner.pipeline, SelfManagedPipeline) and runner.pipeline.loaded
    assert runner.model_memory_usage == 16
    params = OmniDiffusionSamplingParams(seed=42)
    runner._initialize_generator(params)
    assert params.generator is None
    params.lora_request = object()
    with pytest.raises(ValueError, match="LoRA"):
        runner._initialize_generator(params)
    worker = object.__new__(MetalDiffusionWorker)
    worker.od_config = config
    worker.lora_manager = None
    worker.init_lora_manager()
    assert worker.lora_manager is None


def test_invalid_runtime_rejected_without_loading(registry, monkeypatch):
    monkeypatch.setattr(SelfManagedPipeline, "metal_runtime", "cuda")
    config = OmniDiffusionConfig(model_class_name="TestSelfManaged")
    with pytest.raises(NotImplementedError, match="runtime"):
        validate_config(config)
    monkeypatch.setattr(SelfManagedPipeline, "metal_runtime", "mlx")
    monkeypatch.setattr(SelfManagedPipeline, "load_weights", None)
    with pytest.raises(TypeError, match="load_weights"):
        validate_config(config)


def test_mlx_dummy_and_failed_load_do_not_publish_pipeline(registry, monkeypatch):
    config = OmniDiffusionConfig(model_class_name="TestSelfManaged")
    runner = object.__new__(MetalDiffusionModelRunner)
    runner.od_config = config
    runner.pipeline = None
    runner.load_model(load_format="dummy")
    assert runner.pipeline is None
    with pytest.raises(ValueError, match="self-managed"):
        runner.load_model(custom_pipeline_name="unregistered")

    def fail(self):
        raise RuntimeError("weight load failed")

    monkeypatch.setattr(SelfManagedPipeline, "load_weights", fail)
    with pytest.raises(RuntimeError, match="weight load failed"):
        runner.load_model()
    assert runner.pipeline is None


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("VLLM_OMNI_METAL_INTEGRATION") != "1" or not is_available(),
    reason="requires Apple Silicon and VLLM_OMNI_METAL_INTEGRATION=1",
)
def test_native_pipeline_engine_on_mps(native_config):
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.request import OmniDiffusionRequest

    engine = DiffusionEngine(native_config)
    loop = asyncio.Runner()
    try:
        pipeline = engine.executor.driver_worker.worker.model_runner.pipeline
        assert isinstance(pipeline, NativeSignalPipeline) and pipeline.gain.device.type == "mps"
        assert torch.equal(pipeline.gain.cpu(), torch.tensor([0.25]))

        async def generate():
            request = OmniDiffusionRequest(
                prompt="signal",
                sampling_params=OmniDiffusionSamplingParams(seed=42),
                request_id="native-signal",
            )
            result = []
            async for outputs in engine.step_streaming(request):
                result = outputs
            return result

        outputs = loop.run(asyncio.wait_for(generate(), timeout=10))
        assert len(outputs) == 1
        assert outputs[0].multimodal_output["audio_sample_rate"] == 16000
        audio = outputs[0].multimodal_output["audio"]
        assert torch.isfinite(audio).all() and audio.std() > 0
    finally:
        engine.close()
        loop.close()
