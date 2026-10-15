# SPDX-License-Identifier: Apache-2.0
"""Single-device native Omni workers without CUDA runner initialization."""

import gc

import torch
from vllm.config import set_current_vllm_config
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase
from vllm_omni.worker.mixins import OmniWorkerMixin

# Each spawned worker registers its operator providers before loading the model.
import vllm_omni_metal.activation  # noqa: F401
import vllm_omni_metal.normalization  # noqa: F401
import vllm_omni_metal.rotary  # noqa: F401
from vllm_omni_metal.native_runner import MetalNativeModelRunner
from vllm_omni_metal.platform import MetalOmniPlatform


def validate_native_config(config):
    parallel = config.parallel_config
    if parallel.world_size != 1 or parallel.distributed_executor_backend not in (None, "uni"):
        raise NotImplementedError("native Metal requires single-device uni execution")
    if config.lora_config or config.speculative_config or config.kv_transfer_config:
        raise NotImplementedError(
            "native Metal does not support LoRA, speculative or cross-engine KV transfer"
        )
    if config.model_config.dtype not in MetalOmniPlatform.supported_dtypes:
        raise ValueError("unsupported native Metal dtype")
    if config.model_config.quantization or config.cache_config.cache_dtype != "auto":
        raise NotImplementedError("native Metal requires unquantized model and KV weights")
    if config.cache_config.enable_prefix_caching:
        raise NotImplementedError("native Metal model-state prefix caching is not yet implemented")
    if config.scheduler_config.async_scheduling:
        raise NotImplementedError("native Metal requires synchronous scheduling")
    backend = config.attention_config.backend
    if backend is not None and getattr(backend, "name", backend) != "CUSTOM":
        raise ValueError("native Metal requires CUSTOM (MPS SDPA), not a CUDA attention backend")
    offload = config.offload_config
    if (
        offload.offload_backend != "auto"
        or offload.uva.cpu_offload_gb
        or offload.prefetch.offload_group_size
        or config.cache_config.kv_offloading_size
        or config.model_config.enable_sleep_mode
    ):
        raise NotImplementedError("native Metal does not support offload or sleep mode")


class MetalNativeWorker(OmniWorkerMixin, WorkerBase):
    model_runner_cls = MetalNativeModelRunner

    def init_device(self):
        validate_native_config(self.vllm_config)
        MetalOmniPlatform.set_device(self.local_rank)
        self.device = torch.device("mps")
        init_distributed_environment(
            world_size=1,
            rank=self.rank,
            distributed_init_method=self.distributed_init_method,
            local_rank=self.local_rank,
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        from vllm.utils.torch_utils import set_random_seed

        set_random_seed(self.model_config.seed)
        from vllm_omni_metal.shm_connector import register_metal_shm_connector

        register_metal_shm_connector()
        self.model_runner = self.model_runner_cls(self.vllm_config, self.device)

    def get_model(self):
        return self.model_runner.model

    def load_model(self, *, load_dummy_weights=False):
        if load_dummy_weights:
            raise NotImplementedError("native Metal dummy model loading is not supported")
        self.model_runner.load_model()

    def get_kv_cache_spec(self):
        specs = {}
        with set_current_vllm_config(self.vllm_config):
            for name, layer in self.compilation_config.static_forward_context.items():
                spec = layer.get_kv_cache_spec(self.vllm_config)
                if spec is not None:
                    specs[name] = spec
        return specs

    def determine_available_memory(self):
        gc.collect()
        MetalOmniPlatform.synchronize()
        free, total = MetalOmniPlatform.get_device_memory()
        requested = self.cache_config.kv_cache_memory_bytes
        if requested is not None:
            if requested > max(0, free - 3 * 1024**3):
                raise MemoryError("requested KV budget would exhaust system memory headroom")
            return requested
        budget = int(total * self.cache_config.gpu_memory_utilization)
        available = max(0, min(free - 3 * 1024**3, budget - torch.mps.current_allocated_memory()))
        if self.get_kv_cache_spec() and available == 0:
            raise MemoryError("insufficient native Metal KV memory after reserving system headroom")
        return available

    def initialize_from_config(self, kv_cache_config):
        specs = self.get_kv_cache_spec()
        self.model_runner.kv_group_ids = {
            name: index
            for index, group in enumerate(kv_cache_config.kv_cache_groups)
            for name in group.layer_names
        }
        for tensor in kv_cache_config.kv_cache_tensors:
            if not tensor.layers or tensor.host_resident:
                raise ValueError("KV allocation requires device-resident attention layers")
            raw = torch.empty(tensor.size, dtype=torch.uint8, device=self.device)
            for index, name in enumerate(tensor.layers):
                spec = specs[name]
                itemsize = torch.empty((), dtype=spec.dtype).element_size()
                offset = tensor.offset + index * tensor.layer_stride
                if offset % itemsize or tensor.block_stride % itemsize:
                    raise ValueError("KV allocation offset/stride is not dtype aligned")
                page_elements = spec.block_size * spec.num_kv_heads * spec.head_size
                cache = torch.as_strided(
                    raw.view(spec.dtype),
                    (
                        2,
                        kv_cache_config.num_blocks,
                        spec.block_size,
                        spec.num_kv_heads,
                        spec.head_size,
                    ),
                    (
                        page_elements,
                        tensor.block_stride // itemsize,
                        spec.num_kv_heads * spec.head_size,
                        spec.head_size,
                        1,
                    ),
                    offset // itemsize,
                )
                self.compilation_config.static_forward_context[name].kv_cache = cache
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

    def get_supported_kv_cache_layouts(self):
        return ["NHD"]

    def compile_or_warm_up_model(self):
        MetalOmniPlatform.synchronize()
        return CompilationTimes(0.0, 0.0)

    def get_supported_tasks(self):
        return ("generate",)

    def execute_model(self, scheduler_output):
        return self.model_runner.execute_model(scheduler_output)

    def update_max_model_len(self, max_model_len):
        if max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        self.model_config.max_model_len = max_model_len

    def get_cache_block_size_bytes(self):
        return sum(spec.page_size_bytes for spec in self.get_kv_cache_spec().values())

    def shutdown(self):
        if self.model_runner is not None:
            self.model_runner.shutdown_omni_connectors()
            self.model_runner.requests.clear()
            self.model_runner.model = None
        self.compilation_config.static_forward_context.clear()
        gc.collect()
        MetalOmniPlatform.empty_cache()

    def set_omni_connector_output_sink(self, sink):
        self.model_runner.set_omni_connector_output_sink(sink)

    def check_encoder_loaded(self):
        check = getattr(self.get_model(), "encoder_loaded", None)
        return bool(check()) if callable(check) else False


class MetalARWorker(MetalNativeWorker):
    pass


class MetalGenerationWorker(MetalNativeWorker):
    pass
