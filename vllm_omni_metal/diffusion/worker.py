# SPDX-License-Identifier: Apache-2.0
"""Reuse Omni's loader, scheduler, output transport and worker lifecycle."""

from vllm_omni.diffusion.registry import initialize_model
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

from vllm_omni_metal.diffusion.config import validate_config
from vllm_omni_metal.execution.runtime import pipeline_runtime
from vllm_omni_metal.quantization.runtime import configure_model_quantization


class MetalDiffusionWorker(DiffusionWorker):
    def init_device(self) -> None:
        validate_config(self.od_config)
        if self.rank != 0 or self.local_rank != 0:
            raise ValueError("Metal diffusion only supports rank 0.")
        super().init_device()

    def init_lora_manager(self) -> None:
        if pipeline_runtime(self.od_config) == "torch":
            super().init_lora_manager()

    def shutdown(self) -> None:
        if getattr(self, "_shutdown_complete", False):
            return
        runner = getattr(self, "model_runner", None)
        pipeline = getattr(runner, "pipeline", None)
        close = getattr(pipeline, "close_metal_operators", None)
        try:
            if callable(close):
                close()
        finally:
            # A private pipeline's failed cleanup must not leak shared process state.
            super().shutdown()


class MetalDiffusionModelRunner(DiffusionModelRunner):
    def load_model(
        self, memory_pool_context_fn=None, load_format="default", custom_pipeline_name=None
    ):
        if pipeline_runtime(self.od_config) == "torch":
            result = super().load_model(memory_pool_context_fn, load_format, custom_pipeline_name)
            configure_model_quantization(self.pipeline, self.od_config, runtime="torch")
            return result
        if load_format == "dummy":
            return
        if load_format != "default" or custom_pipeline_name is not None:
            raise ValueError("MLX pipelines require their registered, self-managed weight loader.")
        import mlx.core as mx

        # MLX facades share Omni's constructor contract but own their weight loader.
        before = mx.get_active_memory()
        pipeline = initialize_model(self.od_config)
        pipeline.load_weights()
        self.pipeline = pipeline
        configure_model_quantization(pipeline, self.od_config, runtime="mlx")
        self.model_memory_usage = max(0, mx.get_active_memory() - before)

    def _sample_peak_memory_mb(self) -> float:
        # shortcut: MPS has no peak allocator API; add telemetry when PyTorch exposes one.
        return 0.0

    def _initialize_generator(self, sampling_params) -> None:
        if pipeline_runtime(self.od_config) == "mlx":
            if sampling_params.generator is not None or sampling_params.lora_request is not None:
                raise ValueError(
                    "MLX pipelines use seed, not a torch.Generator or PyTorch LoRA request."
                )
            return
        # MPS has no torch.Generator; Torch pipelines create noise on CPU then copy it.
        if sampling_params.generator_device not in (None, "cpu"):
            raise ValueError("Metal Torch pipelines require generator_device='cpu'.")
        if sampling_params.generator is not None and sampling_params.generator.device.type != "cpu":
            raise ValueError("Metal Torch pipelines require a CPU torch.Generator.")
        sampling_params.generator_device = "cpu"
        super()._initialize_generator(sampling_params)
