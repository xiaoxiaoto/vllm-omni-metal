# 通用 Metal 后端协议

## 责任边界

后端负责 MPS 执行，复用上游模型 registry、原生 topology、权重加载、scheduler、阶段处理器和 HTTP 服务。没有模型白名单、模型名分支或模型专用融合实现。

目前可执行原生 `LLM_AR` 和 `LLM_GENERATION`，以及原有完整 generation / Diffusers pipeline。支持范围限单卡、`uni`、同步调度、未量化 dense decoder；这不是任意 CUDA/Triton 模型或所有 Omni 功能的移植。

## 原生阶段

### Worker 与配置

平台返回 `MetalARWorker` / `MetalGenerationWorker`，共用 `MetalNativeModelRunner`。不继承 GPUModelRunner 的 CUDA stream/event 初始化。模型使用上游 `get_model(vllm_config=...)`，保留自身权重完整性检查，验证实际参数在 MPS。

原生阶段配置需明确：`attention_backend=CUSTOM`、`enable_prefix_caching=false`、`async_scheduling=false`、单设备 `uni`。资源限制可通过正常 deploy profile 调整，不替换模型 topology。

配置桥接仅对本插件 native worker 路径生效；原有 vllm-metal 文本/VLM 配置继续委托原始 hook。MPS attention 桥接只在 MPS config 下选择本地 backend。

### KV 与 attention

遵守 0.31 的 `KVCacheTensor.layers / offset / layer_stride / block_stride` 合同，用 device byte backing 和显式 strided view 映射物理页，而不是假定所有层共享同一 contiguous tensor。

按请求 block table 写入当前 token K/V，读取实际 sequence 长度；支持 GQA 和 sliding-window causal mask。decode 使用历史长度作为 causal offset，不丢弃过去的 KV、不读取无效尾部。不同 cache group 使用对应的 block table。

任何 KV 写入前验证整个 batch 的 Q/K/V、缓存、输出 shape/device/dtype、query 边界和 block 范围。非正长度、重叠缓存 strides、请求内重复或跨请求共享 block 均拒绝；当前 prefix cache 关闭，不能通过共享物理页绕过请求隔离。

MPS FP32/FP16/BF16、head size 64/96/128/256/512、block size 8/16/32、页内连续 NHD 时，`metal_attention.py` 调用上游 `reshape_and_cache` 和 `paged_attention_primitive`。K/V scatter 是批量原生算子，attention 直接读取分页缓存并映射 GQA head，不 gather 历史 KV、不复制 GQA K/V。页与 K/V bank 的 stride、storage offset 保留 allocator 原值。其他已支持 geometry/布局继续走原有 MPS SDPA；这是按算子参数选择实现，不是 CPU fallback，也不放宽 ALiBi/softcap/sinks 等现有不支持项。

Torch/MLX 通过上游 DLPack bridge 零拷贝共享存储：导出等待 Torch 写入，保留上游 scatter 的缓存输出作为 attention 输入以维持 lazy graph 写读依赖；`mx.eval` 完成写读后才导回 Torch。MLX 只执行这些共享算子，模型加载、权重、RNG 与 worker 合同仍是 Torch，未引入 MLX 模型 facade。两套框架有同步开销，MLX 分配不在 MPS 内存指标内；原生 split scratch 保守估计超过每调用 256 MiB 时，在写 KV 前报错。

KV 预算基于 MPS 用量及系统可用内存，保留 3 GiB 系统 headroom；显式 `kv_cache_memory_bytes` 不会被忽略或静默超额分配。

### 已发布算子来源与验证

复用 [vllm-metal v0.31.0](https://github.com/vllm-project/vllm-metal/releases/tag/v0.31.0)，tag commit `e7a0c90bec180e5dd4275c473fd7a36e709a930e`，Apache-2.0。安装器已经固定对应 wheel 的 SHA256；本次未复制第三方源码、未添加新依赖、未修改已安装包。不是 GitHub 未合并 PR，也不是自行编写的 shader。

上游源码及测试：

- [原生 launcher / MLX primitive](https://github.com/vllm-project/vllm-metal/blob/v0.31.0/vllm_metal/metal/paged_ops.cpp)、[共享 Metal kernel](https://github.com/vllm-project/vllm-metal/blob/v0.31.0/vllm_metal/metal/kernels_v2/pagedattention.metal)。
- [官方 DLPack bridge](https://github.com/vllm-project/vllm-metal/blob/v0.31.0/vllm_metal/pytorch_backend/tensor_bridge.py) 与 [bridge 测试](https://github.com/vllm-project/vllm-metal/blob/v0.31.0/tests/test_tensor_bridge.py)。
- [分页 attention 测试](https://github.com/vllm-project/vllm-metal/blob/v0.31.0/tests/test_paged_attention.py)、[GQA native 测试](https://github.com/vllm-project/vllm-metal/blob/v0.31.0/tests/test_gqa_paged_decode.py)。

本项目 `tests/test_metal_attention.py` 用独立 CPU FP64 softmax/reference 核对 dtype、分块、split-KV、非连续物理页/offset、窗口和多请求，禁止测试路径调用 SDPA 或 repeat_interleave；另外验证输入不被 donation 改写、连续 prefill/decode 和返回存储生命周期。`tests/test_native_runner.py` 的随机 Llama/Qwen3 同时覆盖 SDPA 与原生 kernel 支持域，保留 greedy/seed 行为检查。上游 v0.31.0 bridge 测试本机复跑 16 项通过，GQA native specialization 数值矩阵及库存在性检查 41 项通过（其余 309 项未运行，未下载模型）。

```bash
VLLM_OMNI_METAL_INTEGRATION=1 .venv-vllm-omni-metal/bin/python -m pytest -q \\
  tests/test_metal_attention.py tests/test_native_attention.py tests/test_native_runner.py
.venv-vllm-omni-metal/bin/python examples/benchmark_attention.py --require-speedup
```

基准只做小张量，设置 256 MiB 张量估计上限和 3 GiB 系统 headroom；同步计时包含 bridge 与 KV 写入。Apple M4 Pro 的默认 BF16 4×2048、16 Q / 4 KV heads、head size 128，20 次中位数为旧路径 5.956 ms / 原生路径 0.512 ms（11.6×）。该结果不代表模型级吞吐、生产负载或其他硬件；真实大权重/听感未验收。

### 请求与模型 hook

处理 new/cached/resumed/finished 请求、外部 request ID、token history 和独立 MPS RNG。普通模型只接收其 forward 签名接受的 kwargs；原生模型仍可接收 Omni runtime information、seq counts 和 query boundaries。

复用模型的 `preprocess`、`prepare_runner_inputs`、`update_decode_step_metadata`、`prefer_model_sampler` / `sample`、`postprocess`、`make_omni_output` 及结束 hook。`prepare_runner_inputs` 获得实际请求 token span 和 computed/scheduled counts；声明 `accepts_runner_sampling_extra_args` 时才传递请求级 `extra_args` 和 partial-prefill discard mask。返回 tensor 不得改变调度长度、设备或整数 dtype。标准 sampler 保留 penalty、top-k/top-p、allowed tokens、bad words、logit bias 和 min-token stop 约束。

pre/postprocess 按实际请求 token span 运行，不把全 batch 或单个 logits row 误当完整请求。上游 state-free payload builder 负责拆分输出；hidden states 和 multimodal tensor 迁至 CPU 后跨阶段传输。输出浮点 payload 拒绝 NaN/Inf。

最终客户端 `OmniEngineCoreOutput.multimodal_output` 必须是 `dict[str, torch.Tensor]`。runner 在拆分客户端/阶段间 payload 后复用上游 `_ensure_tensor_values`：数值标量与可转换的列表/元组转为张量，None 省略，不可转换值按上游规则警告后省略。转换后再次检查有限值；阶段间 payload 的字符串、标量与嵌套类型不受此转换影响。空输出和异步分区的 None 保持原语义。此合同不识别模型名、采样率或元数据键。

`make_omni_output` 在模型 sampler 之前建立当前步输出，sampler 追加的终结音频保留在同一步。稀疏音频复用上游 marker/请求 ID 解析和 payload builder，按 `meta.req_id` 对齐而非 batch 下标；拒绝非法、重复、未知 ID 和不匹配的列表长度。空稀疏步和未输出的请求返回空 payload，不附带 hidden states 或其他请求的共享字段。平台层不识别具体模型名称。

encoder multimodal 输入、prompt embeddings、auxiliary Talker MTP、logprobs、structured decoding、prefix/model-state cache、LoRA、跨引擎 KV、sleep/offload 未实现；不要只解除错误检查来“支持”它们。

### Shared memory 与 capture 查询

上游阶段 processor 保留采样/codec 语义，sender scheduler 负责 chunk 发送，runner 不重复发送。Metal worker 中替换 SHM constructor：

- 锁在用户私有临时目录，不依赖 `/dev/shm`。
- 逻辑 key 映射为 28 字节物理名，满足 Darwin POSIX SHM 名字限制。
- 8 字节长度头避免 Darwin page-rounded segment 大小被当成真实序列化数据长度。
- 明确释放 flock，避免首个 resource tracker 启动时继承描述符造成死锁；消费后 unlink，关闭时回收未消费对象。

CPU wheel 的 CUDA capture 查询会抛 dummy-base 错误。`native_compat.py` 仅克隆 MPS 模型实例内含该查询的方法，使用独立 globals 的 Torch view 返回 `capture=false`。没有改动 class、模型模块、全局 `torch.cuda` 或安装源码，且模型代码范围校验继续执行；其他 CUDA API 不被模拟。

## 完整 pipeline

使用上游 `register_diffusion_model()`，不新增 registry。Torch pipeline 默认运行时是 `torch`：`nn.Module`、`__init__(*, od_config)`、`weights_sources`、`load_weights(weights)` 和 `forward(batch)` 返回 `DiffusionOutput`。

MLX facade 声明 `metal_runtime='mlx'`，自行实现无参数 `load_weights()`；通用 runner 只构造并调用一次，加载成功后发布实例，用 MLX allocator 统计权重用量。此路径不经过 PyTorch loader，不创建 Torch noise generator 或 LoRA manager。本项目不提供具体 MLX facade。

外部类可以提供 `validate_metal_config(config)`，模型约束不硬编码到后端。Diffusers 使用上游 adapter、CPU noise generator 和 MPS SDPA。

## 可执行验收

```bash
.venv-vllm-omni-metal/bin/python -m pytest -q -m 'not integration'
.venv-vllm-omni-metal/bin/ruff check .
.venv-vllm-omni-metal/bin/ruff format --check .

VLLM_OMNI_METAL_INTEGRATION=1 \
VLLM_OMNI_METAL_NATIVE_MODEL=/path/to/local/higgs-checkpoint \
  .venv-vllm-omni-metal/bin/python -m pytest -q
```

- 数学 gate：分块 prefill/decode 对完整 causal SDPA、GQA、不连续页、跨请求和无效页检查。
- 不同模型 gate：本地随机 Llama 与 Qwen3，真实 MPS 合批 6 步 token 与独立 Transformers CPU reference 相同；stochastic seed 在单请求与合批间隔离。
- 原生多阶段 gate：上游 Higgs Talker → Code2Wav、原生 async chunk，通过 HTTP 生成可解码 24 kHz 非静音 PCM16 WAV，同 seed 字节一致；没有 mlx-audio 融合路径。
- 回归 gate：随机 tiny SD 和完整原生合成 pipeline，以及平台/registry/兼容隔离、输入与内存边界、SHM 生命周期。

测试只调整上游 profile 的资源/后端参数，保留 topology/connector/processor。随机小模型不代表大模型质量，WAV 不代表 ASR 或听感通过；未进行生产负载、真实 encoder 模型或高吞吐基准。
