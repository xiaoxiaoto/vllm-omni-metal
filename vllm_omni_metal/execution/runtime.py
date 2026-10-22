# SPDX-License-Identifier: Apache-2.0
"""Resolve execution capabilities from Omni's registry, never from model names."""

from torch import nn
from vllm_omni.diffusion.data import uses_diffusers_adapter
from vllm_omni.diffusion.registry import DiffusionModelRegistry


def resolve_pipeline_class(config) -> type[nn.Module] | None:
    if uses_diffusers_adapter(config):
        return None
    if config.diffusion_load_format != "default":
        raise NotImplementedError("Metal pipelines require default or diffusers load format.")
    name = config.model_class_name
    if not name or name not in DiffusionModelRegistry.models:
        raise NotImplementedError(
            "Metal requires a registered native pipeline or diffusion_load_format='diffusers'."
        )
    cls = DiffusionModelRegistry._try_load_model_cls(name)
    if cls is None:
        raise NotImplementedError(
            f"Registered pipeline {name!r} cannot be imported on this platform."
        )
    if not issubclass(cls, nn.Module):
        raise TypeError("Omni generation pipelines must implement the nn.Module pipeline contract.")
    return cls


def pipeline_runtime(config) -> str:
    cls = resolve_pipeline_class(config)
    runtime = getattr(cls, "metal_runtime", "torch")
    if runtime not in ("torch", "mlx"):
        raise NotImplementedError(f"Unknown Metal pipeline runtime {runtime!r}.")
    if runtime == "mlx" and not callable(getattr(cls, "load_weights", None)):
        raise TypeError("MLX pipelines must provide a self-managed load_weights() method.")
    return runtime
