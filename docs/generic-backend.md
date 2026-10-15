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

bridge 路径通过上游 DLPack 共享 Torch/MLX 存储：导出等待 Torch 写入，保留上游 scatter 的缓存输出作为 attention 输入以维持 lazy graph 写读依赖；`mx.eval` 完成写读后才导回 Torch。MLX 只执行这些共享算子，模型加载、权重、RNG 与 worker 合同仍是 Torch，未引入 MLX 模型 facade。两套框架有同步开销，MLX 分配不在 MPS 内存指标内；原生 split scratch 保守估计超过每调用 256 MiB 时，在写 KV 前报错。

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

### 可选 Torch-stream attention

固定 vllm-metal 0.31.0 wheel 缺少新的 MPS launcher。`_native/` 带入已合并 [#858](https://github.com/vllm-project/vllm-metal/pull/858) 的 host-only [`mps_ops.mm`](https://github.com/vllm-project/vllm-metal/blob/fb9fbc3bd857bee3f7a74f59ee46cc068fa7267c/vllm_metal/pytorch_backend/mps_ops.mm)、`metal_device.h` 与 `paged_attention_kernels.h`，immutable commit `fb9fbc3bd857bee3f7a74f59ee46cc068fa7267c`，Apache-2.0。两个共享 header 原样保留；host source 仅补齐 FP32 dtype guard/name 和 half-only tiled 路由限制，回归将这三处适配归一化后核对原始 SHA256。不修改 attention shader、已安装上游源码或模型 registry；加载已安装的 0.31 metallib，仍使用同一套 paged/varlen kernels。

```bash
# 仅显式构建一次，需要 Apple Command Line Tools；不下载、不自动安装工具。
.venv-vllm-omni-metal/bin/python -m vllm_omni_metal.mps_attention
# 构建后启动新的 worker；正在运行的引擎不会自动切换。
.venv-vllm-omni-metal/bin/python examples/benchmark_attention.py --require-speedup
```

`mps_attention.py` 复用上游 PyTorch CppExtension/BuildExtension 方案，不依赖 Ninja。构建前要求系统 3 GiB headroom、512 MiB 空闲磁盘；产物放在 `~/.cache/vllm-omni-metal/mps/<hash>/`，私有 owned 目录、文件安全检查、flock、临时目录和原子发布。key 只绑定 host sources/headers、builder、编译常量、Torch 版本/环境路径及 Python ABI；普通 wheel 只打包源码，常规安装不强制编译。Torch/插件源码变化后须重新显式构建。已有不安全或损坏产物报错，不静默吞掉错误；缺少匹配产物时保留现有 bridge。

通过原生 attention 的完整 batch 校验与 scratch 256 MiB budget 后，支持域 FP32/FP16/BF16 选择匹配的 launcher；FP32 prefill 复用已发布分页 kernel，其他 dtype 可用 tiled/NAX，缺产物才保留 bridge。C++ int32 stride 参数另在 KV 写入前检查；官方 storage offsets 与 bank/block/head strides 保持。slots/int32 scheduler metadata 经非阻塞拷贝送入同一 Torch stream，int32 metadata 合并一次传输。该 stream 排序 projection → KV scatter → attention → Torch consumer，省去 attention 的 MLX/DLPack 和显式 GPU synchronize；这是 attention 路径，不承诺 SwiGLU 等其他路径无桥接。不解除 prefix cache、async scheduling、quantization 等限制。

`tests/test_mps_attention.py` 覆盖 cache 身份/缺失、权限与 symlink、CPU 预算、原始 source hashes（仅归一化三处 FP32 host 适配）、int32 边界；fresh subprocess 屏蔽 MLX、运行时编译和 Python MPS synchronize，验证连续 Torch producer/consumer 与多请求历史 KV。独立 FP64 数值回归同时强制验证 native/bridge 的 FP16/BF16/FP32、五种 head sizes、滑窗、interleaved/padded cache/nonzero offset、输入不变；native prefill/decode 和输出生命周期保留。测试专用 spawn worker 探针确认支持几何的 FP32/BF16 Llama/Qwen3 实际选择 native attention、HF greedy 与 seed 行为不变。

M4 Pro、BF16 batch4/context2048、Q/KV heads16/4、head128，20 次同步中位数：gather 5.936 ms、bridge 0.515 ms、Torch-stream 0.477 ms（比 bridge 快1.08×）。三路包含 Python dispatch、scheduler metadata、KV 写入与最终同步；收益小且受形状/负载影响，不是模型吞吐结果。最初的同步 metadata 拷贝更慢，已通过非阻塞传输修正；M5 NAX、生产负载与真实 Higgs/YuE2 未验收。

### RMSNorm 与残差归一化

`normalization.py` 移植上游已合并 [vllm-metal #858](https://github.com/vllm-project/vllm-metal/pull/858) 的 [normalization adapter](https://github.com/vllm-project/vllm-metal/blob/fb9fbc3bd857bee3f7a74f59ee46cc068fa7267c/vllm_metal/pytorch_backend/normalization.py)，commit `fb9fbc3bd857bee3f7a74f59ee46cc068fa7267c`，Apache-2.0。当前固定的 release wheel 尚不包含该 adapter，因此只移植小型 Python 适配，不复制 Metal/C++ kernel、不升级 vllm-metal。底层使用已安装 PyTorch 的 `torch.nn.functional.rms_norm`；本机复现了上游 FP16/BF16 plain/residual 四组精度用例。

原生 worker 的 IR 默认优先级是 `omni_mps → native`，仅按 worker 合同选择，不按模型名分支。平台 defaults 在 DeviceConfig 转为 MPS 之前解析，因此以原生 worker 类路径限定；每个 spawn worker 导入 adapter 后，由上游 WorkerBase 安装配置的优先级。普通 LLM 默认 hook 保留，用户显式优先级仍由上游 KernelConfig 合并规则优先保留。注册的是 IR 算子实现，不是模型 registry 或模型方法替换。

优化只选择 MPS FP16/BF16、非空输入、同 dtype/device 且宽度匹配的一维 learned weight、无 variance override。残差另外要求相同 shape/device/dtype。CPU、FP32、无 weight、部分 variance、混合权重或广播等原生合法参数仍交给上游 native；不把 MPS tensor 搬到 CPU，不忽略其他不支持配置。

残差使用 FP32 sum 计算 variance，再按 vLLM 顺序先 cast normalized 值后乘 learned weight，并单独返回 cast 后的 residual sum；输入不就地改写。它使用融合 RMSNorm，但 add、cast、multiply 仍是独立操作，不宣称整个 residual+norm 是单 kernel 融合。

`tests/test_normalization.py` 检查 dispatch、用户优先级与作用域、spawn 导入、精度边界、非连续多维输入、maybe_inplace 语义和原生后备。`tests/test_native_runner.py` 新增 BF16 随机 Llama/Qwen3，检查真实配置已选择新 IR 优先级，保持 CPU HF greedy 和 seed 隔离对照。

```bash
VLLM_OMNI_METAL_INTEGRATION=1 .venv-vllm-omni-metal/bin/python -m pytest -q \\
  tests/test_normalization.py tests/test_native_runner.py
.venv-vllm-omni-metal/bin/python examples/benchmark_normalization.py --require-speedup
```

基准限制估计张量 64 MiB、保留系统 headroom 3 GiB、不加载模型。M4 Pro、BF16 4×4096、30 次同步中位数的一次实测：plain 0.182 / 0.152 ms（1.19×），residual 0.191 / 0.170 ms（1.12×）。包含 IR dispatch 与 GPU 同步，不等于生产吞吐；真实 Higgs/YuE2、大上下文和完整生产负载未在该路径验收。

### SiLU×gate（SwiGLU）

`activation.py` 使用已安装 MLX v0.32.1 的 [编译 `nn.silu`](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/python/mlx/nn/layers/activations.py)，commit `3a6219917e4535575ce5bce2fc2ba27a483a709b`，MIT。复用公开 `mx.compile(shapeless=True)` 将 SiLU、dtype 舍入和 up 分支乘法编译为一个 elementwise primitive；本地最小 Python adapter 处理 Torch 精度顺序、作用域及官方 DLPack bridge。这不是移植一个已合并的上游 Torch SwiGLU adapter，也不复制/自写 Metal shader；[已发布 stateless MLP 编译方案](https://github.com/vllm-project/vllm-metal/blob/e7a0c90bec180e5dd4275c473fd7a36e709a930e/vllm_metal/compiled_mlp.py) 的模型专用 wrappers 不采用。

通过 vLLM `SiluAndMul.register_oot` 注册通用算子子类，不修改原始类的方法、in-tree operator registry 或模型 registry。OOT 查找使用类名 `SiluAndMul`，配置启停名保留原始 `silu_and_mul`。每个 spawn worker 在加载模型前导入该 provider，导入阶段不初始化 MLX。原生配置以 `none` 禁用其他 CUDA ops，只选择安全的 `+silu_and_mul` 与可选 `+rotary_embedding`；各自显式 opt-out 保留，冲突指令明确报错。普通 worker 或无当前 engine 配置不选择桥接。

仅 MPS FP32/FP16/BF16、连续 1–3 维、无梯度、最后一维偶数且输出元素数 ≥262144 的推理输入启用。这是测得的跨框架开销门槛，不是模型白名单；小 decode、非连续、空输入、梯度和其他输入继续走上游实现，保留原设备、异常与 autograd 语义。MLX SiLU 使用 FP32 计算，先舍入到输入 dtype，再乘 up 分支，不能只在最终结果 cast。借用的输入保持至 `mx.eval` 完成以防 buffer donation，输出通过官方 bridge 回到 Torch；不接管权重、模型加载或 RNG。

`tests/test_activation.py` 覆盖无 MLX 的 fresh-worker 注册、原类/模型映射不变、作用域、预算、FP32/FP16/BF16 的 1D/2D/3D 数值、输入不变、输出生命周期、梯度及关闭后备。FP16/BF16 判别样例分别包含 81/72 个不能省略中间舍入的位置，逐值精确对照；MLX 导出的图确认同一个 compiled primitive 包含 sigmoid、两次乘法和两次 cast。`tests/test_native_runner.py` 增加 BF16 大 MLP 随机 Llama/Qwen3，与 CPU HF greedy 和 seeded batch/solo 对照；使用测试专属 worker extension 和字符串 RPC 在真实子进程检查路径，不关闭 spawn、不放开不安全序列化。

```bash
VLLM_OMNI_METAL_INTEGRATION=1 .venv-vllm-omni-metal/bin/python -m pytest -q \
  tests/test_activation.py tests/test_native_runner.py
.venv-vllm-omni-metal/bin/python examples/benchmark_activation.py --require-speedup
```

基准预算 64 MiB、系统 headroom 3 GiB；M4 Pro、BF16 32×8192、30 次同步中位数：原生 0.343 / 编译桥接 0.170 ms（2.0×），含 Python OOT dispatch 与双框架同步。小块桥接不稳定，保留原路径；不是模型吞吐或其他机器/形状的加速保证。可选缓存表 NeoX RoPE 的源合同已通过隔离验证，见下节；真实 Higgs/YuE2 和生产负载未在这些路径验收。

### vpipe 通用片段

固定参考 `tgo-app-dev/vpipe` commit `6a721d5c3b74ded70001f7c02121f14ae8f5321a`，Apache-2.0，源码/适配说明打包为 `_native/VPIPE.md`。只采用 elementwise SwiGLU 和 table-driven partial NeoX RoPE；没有移植模型/codec/视频/quantization，也没有采用会重算频率或使用专用 ring KV 布局的融合。上游有 unit tests，未发现 pinned revision 的公开 GitHub Actions，不宣称全套上游 CI 通过。

原版 SwiGLU 只做最终 cast，与 Torch 的中间舍入不等价（FP16/BF16 的257元素判别样例有81/72处差异）；移植先 cast SiLU，再乘 up，并读取连续 packed `[gate,up]`。缓存 NeoX RoPE 同样保留每个乘法的 dtype 舍入，再加减；上游 `cos_sin_cache` 构建、缩放、dtype 匹配和 `index_select` 输入边界校验不变，不重算频率。只注册通用 `RotaryEmbedding` OOT 子类；其他派生 RoPE、GPTJ style、梯度/非连续/其他合法输入保留原生实现，模型映射与原类方法不变。返回新 storage，partial tail 和 query/key 输入保持不变。

`vpipe_kernels.py` 的生产路径仅 `load_metallib`，没有隐式 shader JIT。显式构建需已有 Apple Metal compiler，私有 owned 目录、artifact 校验、flock、临时目录、原子发布与系统 headroom 保持；不安装工具或改变 MPS 水位。缺产物回退，unsafe/corrupt 产物明确报错。本机 `metal` 不可用，**AOT 和真实 worker 新路径仍未验收**。

```bash
python -m vllm_omni_metal.vpipe_kernels
# 只在隔离测试进程显式编译（不是生产/worker JIT）：
python examples/benchmark_vpipe.py --compile-for-test --operator rope --rows 4 --require-speedup
python examples/benchmark_vpipe.py --compile-for-test --operator silu --rows 1
```

测试覆盖三种 dtype、packed rows、257元素精度判别、cache 随机覆盖（不能替换成自算频率）、partial tail、key=None、GPTJ fallback、输入不变、artifact/预算安全。M4 Pro BF16 RoPE 4×hidden4096（head128、Q/KV heads32/8）0.239/0.203ms（1.18×）；单行 width4096 SwiGLU 0.138/0.138ms，无收益，数值片段**不接入默认小 decode**。FP32 attention 4×context2048、Q/KV16/4、head128：gather/bridge/Torch-stream 6.377/0.617/0.597ms（bridge改进1.03×）。基准含最终同步，预算64MiB（vpipe）/256MiB（attention）、headroom3GiB；不能外推模型吞吐。

未完成 residual-add+RMSNorm 单 kernel：vpipe `rms_add` 是 norm 后加 residual，语义不符；其他检索候选缺可辨许可证、等价 dtype 合同或 norm 验证，不能靠自写新 kernel/放宽精度来冒充移植。Higgs 缓存 index 指向的完整权重缺失，YuE2 缓存无权重；未下载，不宣称真实模型新路径通过。M5 NAX 需真机，当前是 M4 Pro。

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
