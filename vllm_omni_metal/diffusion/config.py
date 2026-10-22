# SPDX-License-Identifier: Apache-2.0
"""Fail before allocating model weights for unsupported execution modes."""

import torch
from vllm_omni.diffusion.data import AttentionSpec, OmniDiffusionConfig, uses_diffusers_adapter

from vllm_omni_metal.execution.runtime import pipeline_runtime, resolve_pipeline_class


def validate_config(config: OmniDiffusionConfig) -> None:
    cls = resolve_pipeline_class(config)
    runtime = pipeline_runtime(config)
    validate = getattr(cls, "validate_metal_config", None)
    if validate is not None:
        validate(config)
    if runtime == "mlx" and config.lora_path is not None:
        raise NotImplementedError("MLX pipelines cannot use the PyTorch LoRA manager.")
    if config.num_gpus != 1 or config.parallel_config.world_size != 1:
        raise NotImplementedError(
            "Metal supports one GPU and no model/data/CFG/sequence parallelism."
        )
    if config.parallel_config.use_hsdp or config.parallel_config.enable_expert_parallel:
        raise NotImplementedError("Metal does not support HSDP or expert parallelism.")
    if config.distributed_executor_backend not in (None, "uni"):
        raise NotImplementedError(
            "Metal diffusion currently requires distributed_executor_backend='uni'."
        )
    if config.engine_backend != "default" or config.diffusion_model_runner_cls is not None:
        raise NotImplementedError(
            "Metal requires the default engine and its platform model runner."
        )
    if config.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("MPS supports float16, bfloat16 and float32, not float64 or CUDA FP8.")
    if config.dtype == torch.bfloat16:
        import platform

        if platform.system() == "Darwin" and int(platform.mac_ver()[0].split(".")[0]) < 14:
            raise ValueError("MPS bfloat16 requires macOS 14 or later; use float16 or float32.")
    for option in (
        "enable_cpu_offload",
        "enable_layerwise_offload",
        "enable_distributed_layerwise_offload",
        "enable_sleep_mode",
        "step_execution",
        "streaming_output",
        "enable_prefix_caching",
    ):
        if getattr(config, option):
            raise NotImplementedError(
                f"{option}=True is not supported by the Metal generation backend."
            )
    if uses_diffusers_adapter(config) and config.enforce_eager:
        raise NotImplementedError("enforce_eager=True is not supported by the Diffusers adapter.")
    if config.diffusion_offload_config is not None or config.quantization_config is not None:
        raise NotImplementedError(
            "Metal diffusion currently supports unquantized, resident model weights only."
        )
    if config.cache_backend not in (None, "none"):
        raise NotImplementedError("Metal generation does not support TeaCache or Cache-DiT.")
    if config.diffusion_kv_mode.value != "dense_legacy" or config.kv_transfer_config is not None:
        raise NotImplementedError(
            "Metal generation does not support paged KV or native KV transfer."
        )
    attention = config.diffusion_attention_config
    if attention.per_role:
        raise NotImplementedError("Metal generation does not support per-role attention overrides.")
    if attention.default is None:
        attention.default = AttentionSpec(backend="TORCH_SDPA")
    elif attention.default.backend.upper() != "TORCH_SDPA":
        raise ValueError("Metal requires diffusion attention backend TORCH_SDPA.")
    else:
        # Diffusers' adapter matches a case-sensitive backend name.
        attention.default.backend = "TORCH_SDPA"
