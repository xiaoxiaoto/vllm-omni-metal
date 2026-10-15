# SPDX-License-Identifier: Apache-2.0
"""Keep vllm-metal's LLM config hook away from model-less diffusion configs."""

from functools import wraps


def apply_compat_patches() -> None:
    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm_metal.platform import MetalPlatform

    _patch_memory_profiler(MetalPlatform)
    _patch_native_attention(MetalPlatform)
    _patch_native_ir_priority(MetalPlatform)
    original = MetalPlatform.check_and_update_config.__func__
    if getattr(original, "_omni_metal_patch", False):
        return

    @classmethod
    @wraps(original)
    def check_and_update_config(cls, config):
        if config.model_config is not None:
            worker = getattr(getattr(config, "parallel_config", None), "worker_cls", "")
            if isinstance(worker, str) and worker.startswith("vllm_omni_metal.native_worker."):
                from vllm.config import DeviceConfig

                from vllm_omni_metal.native_worker import validate_native_config

                validate_native_config(config)
                custom_ops = config.compilation_config.custom_ops
                flags = []
                for operation in ("silu_and_mul", "rotary_embedding"):
                    if f"+{operation}" in custom_ops and f"-{operation}" in custom_ops:
                        raise ValueError(f"custom_ops cannot both enable and disable {operation}")
                    flags.append(
                        f"-{operation}" if f"-{operation}" in custom_ops else f"+{operation}"
                    )
                config.device_config = DeviceConfig("mps")
                config.compilation_config.mode = CompilationMode.NONE
                config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
                config.compilation_config.custom_ops = ["none", *flags]
                config.parallel_config.disable_custom_all_reduce = True
                return None
            return original(cls, config)
        # Omni builds its base VllmConfig before attaching a diffusion model view.
        config.compilation_config.mode = CompilationMode.NONE
        config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        config.parallel_config.disable_custom_all_reduce = True

    check_and_update_config.__func__._omni_metal_patch = True
    MetalPlatform.check_and_update_config = check_and_update_config


def _patch_native_attention(platform_cls) -> None:
    original = platform_cls.get_attn_backend_cls.__func__
    if getattr(original, "_omni_metal_patch", False):
        return

    @classmethod
    @wraps(original)
    def get_attn_backend_cls(cls, *args, **kwargs):
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if config is not None and config.device_config.device_type == "mps":
            return "vllm_omni_metal.native_attention.MetalAttentionBackend"
        return original(cls, *args, **kwargs)

    get_attn_backend_cls.__func__._omni_metal_patch = True
    platform_cls.get_attn_backend_cls = get_attn_backend_cls


def _patch_native_ir_priority(platform_cls) -> None:
    original = platform_cls.get_default_ir_op_priority.__func__
    if getattr(original, "_omni_metal_patch", False):
        return

    @classmethod
    @wraps(original)
    def get_default_ir_op_priority(cls, config):
        worker = getattr(getattr(config, "parallel_config", None), "worker_cls", "")
        # Defaults are resolved before the later hook changes DeviceConfig to MPS.
        if isinstance(worker, str) and worker.startswith("vllm_omni_metal.native_worker."):
            from vllm.config.kernel import IrOpPriorityConfig

            import vllm_omni_metal.normalization  # noqa: F401

            return IrOpPriorityConfig.with_default(
                ["native"],
                rms_norm=["omni_mps", "native"],
                fused_add_rms_norm=["omni_mps", "native"],
            )
        return original(cls, config)

    get_default_ir_op_priority.__func__._omni_metal_patch = True
    platform_cls.get_default_ir_op_priority = get_default_ir_op_priority


def _patch_memory_profiler(platform_cls) -> None:
    original = platform_cls.get_current_memory_usage.__func__
    if getattr(original, "_omni_metal_patch", False):
        return

    @classmethod
    @wraps(original)
    def get_current_memory_usage(cls, device=None):
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if config is not None and config.device_config.device_type == "mps":
            from vllm_omni_metal.platform import MetalOmniPlatform

            return MetalOmniPlatform.get_current_memory_usage()
        return original(cls, device)

    get_current_memory_usage.__func__._omni_metal_patch = True
    platform_cls.get_current_memory_usage = get_current_memory_usage
