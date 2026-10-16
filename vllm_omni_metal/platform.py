# SPDX-License-Identifier: Apache-2.0
"""Omni uses MPS; vLLM's separate MetalPlatform continues to use MLX."""

from contextlib import nullcontext

import psutil
import torch
from vllm.platforms.interface import PlatformEnum
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum


class MetalOmniPlatform(OmniPlatform):
    _enum = PlatformEnum.OOT
    _omni_enum = OmniPlatformEnum.OOT
    device_name = "mps"
    device_type = "mps"
    dispatch_key = "MPS"
    dist_backend = "gloo"
    device_control_env_var = "VLLM_METAL_VISIBLE_DEVICES"
    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank not in (None, 0):
            raise ValueError("Metal exposes one unified GPU; local_rank must be 0.")
        return torch.device("mps")

    @classmethod
    def get_device_count(cls) -> int:
        return int(torch.backends.mps.is_available())

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        cls.get_torch_device(device_id)
        return "Apple Silicon Metal"

    @classmethod
    def get_device_version(cls) -> str:
        import platform

        return platform.mac_ver()[0]

    @classmethod
    def set_device(cls, device: torch.device | int | None = None) -> None:
        if device in (None, 0):
            return
        if isinstance(device, torch.device) and device.type == "mps" and device.index in (None, 0):
            return
        raise ValueError(f"Expected the single MPS device, got {device!r}.")

    @classmethod
    def synchronize(cls) -> None:
        torch.mps.synchronize()

    @classmethod
    def empty_cache(cls) -> None:
        torch.mps.empty_cache()

    @classmethod
    def reset_peak_memory_stats(cls) -> None:
        # MPS exposes current allocations, but no resettable peak counter.
        return None

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.mps.manual_seed(seed)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        cls.get_torch_device(device_id)
        return torch.mps.recommended_max_memory()

    @classmethod
    def get_device_memory(cls, device: torch.device | None = None) -> tuple[int, int]:
        cls.set_device(device)
        total = cls.get_device_total_memory()
        free = min(
            psutil.virtual_memory().available, max(0, total - torch.mps.driver_allocated_memory())
        )
        return free, total

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        return cls.get_device_memory(device)[0]

    @classmethod
    def memory_reserved(cls, device: torch.device | int | None = None) -> int:
        cls.set_device(device)
        return torch.mps.driver_allocated_memory()

    @classmethod
    def get_current_memory_usage(cls, device: torch.device | None = None) -> int:
        cls.set_device(device)
        return torch.mps.current_allocated_memory()

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def supports_float64(cls) -> bool:
        return False

    @classmethod
    def supports_cpu_offload(cls) -> bool:
        return False

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return False

    @classmethod
    def supports_diffusion_dense_flash_attention(cls) -> bool:
        return False

    @classmethod
    def supports_talker_mtp_graph_capture(cls) -> bool:
        return False

    @classmethod
    def create_autocast_context(cls, *, device_type: str, dtype: torch.dtype, enabled: bool = True):
        # Diffusers loads weights in the requested dtype; avoid CUDA AMP assumptions.
        return nullcontext()

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls, selected_backend: str | None, head_size: int, allow_trtllm_default: bool = False
    ) -> str:
        if selected_backend is not None and selected_backend.upper() != "TORCH_SDPA":
            raise ValueError(
                "Metal diffusion supports TORCH_SDPA only; CUDA kernels cannot run on MPS."
            )
        return "vllm_omni_metal.diffusion_attention.MetalSDPABackend"

    @classmethod
    def get_diffusion_worker_cls(cls) -> str:
        return "vllm_omni_metal.worker.MetalDiffusionWorker"

    @classmethod
    def get_diffusion_model_runner_cls(cls) -> str:
        return "vllm_omni_metal.worker.MetalDiffusionModelRunner"

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni_metal.native_worker.MetalARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni_metal.native_worker.MetalGenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        raise NotImplementedError(
            "Metal has no native AR stage configs; use a registered generation or Diffusers pipeline."
        )
