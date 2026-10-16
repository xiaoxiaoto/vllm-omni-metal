# 通用 Metal 后端协议

## 责任边界

后端负责 MPS 执行，复用上游模型 registry、原生 topology、权重加载、scheduler、阶段处理器和 HTTP 服务。没有模型白名单、模型名分支或模型专用融合实现。

目前可执行原生 `LLM_AR` 和 `LLM_GENERATION`，以及原有完整 generation / Diffusers pipeline。支持范围限单卡、`uni`、同步调度、未量化 dense decoder；这不是任意 CUDA/Triton 模型或所有 Omni 功能的移植。

完整 generation pipeline 的 diffusion attention 使用 `MetalSDPABackend`：仅覆盖 backend/impl 的选择与 `forward` 入口，直接复用上游 `SDPAImpl._forward_impl` 的 Torch 实现，保留 BSHD 布局、mask、GQA、causal 和 scale 语义。上游默认入口只派发内置 CUDA/ROCm/NPU/XPU/MUSA 平台，因此 OOT Metal 不能直接返回原 `SDPABackend`。不修改上游类、平台枚举或模型 registry，不伪装 CUDA，也不转移张量到 CPU；Diffusers 自己的 attention 路径不受影响。

2026-10-16 本次回归与部署验收：

```bash
# 保留修复代码，仅在此测试进程恢复旧 backend 选择：应因缺少 Metal forward 而失败。
VLLM_OMNI_METAL_INTEGRATION=1 .venv-vllm-omni-metal/bin/python -c 'import vllm_omni; from vllm_omni_metal.platform import MetalOmniPlatform; MetalOmniPlatform.get_diffusion_attn_backend_cls = classmethod(lambda cls, *a, **k: "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend"); import pytest; raise SystemExit(pytest.main(["-q", "tests/test_diffusion_attention.py::test_platform_forward_matches_independent_reference[fp32-mha]"]))'

VLLM_OMNI_METAL_INTEGRATION=1 .venv-vllm-omni-metal/bin/python -m pytest -q \
  tests/test_diffusion_attention.py tests/test_integration.py
.venv-vllm-omni-metal/bin/python -m pytest -q -m 'not integration'
.venv-vllm-omni-metal/bin/ruff check .
.venv-vllm-omni-metal/bin/ruff format --check .
.venv-vllm-omni-metal/bin/python -m pip check
```

旧 backend 对照为 1 项预期失败；修复后 8 项真实 MPS attention（FP32/FP16/BF16、MHA/GQA、padding/additive mask、causal、非连续 BSHD、输入不变和非法 head geometry）与 1 项随机 tiny Stable Diffusion 双生命周期生成通过，独立 CPU FP64 reference 检查数值，不用模型名特判。非集成测试 260 项、Ruff、依赖检查通过。

Apple M4 Pro / 48 GiB、macOS 26.6.2（25G83），Torch 2.13.0、Transformers 5.14.1、Diffusers 0.40.0、vLLM/Metal 0.31.0、Omni/Omni Metal 0.31.0.dev0；部署 Omni main 为 `2176457b3d8d621e53bdbfd22a42997c8814704f`。相邻部署运行 `./start.sh qwenimage`，经共享准备器从干净副本构建非 editable wheel（SHA256 `c49ae51c756284f7f68ebebc04a909667315ea2c065a41d13a9b37c868cd2735`），实际安装路径和新实现 MPS 执行已用 `-I` 新进程检查。`install.sh`、Formula 和 setuptools 自动分发新增 Python 模块，无需改依赖或 native 资源；源码指纹自动触发新后端 `0794b39fd3ae1c44`。

统一入口聚合/单模型身份检查及真实 `qwen-image-2.1-turbo` 512×512、8 步、CFG 1、seed 42 请求通过：HTTP 200、`image/png`、1,049,508 字节、27.70 秒，512×512 RGBA 可解码且非单色，目视符合窗台橘猫/蓝天提示。请求命令与部署边界见相邻部署根目录及模型 README；此结果不证明 2K、编辑、seed 字节一致性或生产并发。未重装 Homebrew、未推送远端，其他模型未启动；原有 `platform.py` / `test_plugin.py` 工作区改动保留、不归入本次提交。

通用 dense SwiGLU 的 GPU + ANE 合作 contract、显式 CoreML 准备、完整 pipeline 校准和 FP16 误差边界见[独立协议](ane.md)。该路径不替代完整模型适配入口；未知形状、设备或校准配置保留 GPU。

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

残差使用 FP32 sum 计算 variance，再按 vLLM 顺序先 cast normalized 值后乘 learned weight，并单独返回 cast 后的 residual sum；输入不就地改写。有匹配的显式 vpipe AOT 产物时，连续、无梯度 FP16/BF16 输入选择改编自 `rms_norm_fast_f16` 的 add-before-norm 单 kernel；保留非连续、梯度、缺产物以及不适合 Metal FP32 常量的 epsilon（负数、非有限、溢出或非零 subnormal）的原 Torch 分解，不扩大原 IR provider 支持域。

`tests/test_normalization.py` 检查 dispatch、用户优先级与作用域、spawn 导入、精度边界、非连续多维输入、maybe_inplace 语义和原生后备。`tests/test_native_runner.py` 新增 BF16 随机 Llama/Qwen3，检查真实配置已选择新 IR 优先级，保持 CPU HF greedy 和 seed 隔离对照。

```bash
VLLM_OMNI_METAL_INTEGRATION=1 .venv-vllm-omni-metal/bin/python -m pytest -q \\
  tests/test_normalization.py tests/test_native_runner.py
.venv-vllm-omni-metal/bin/python examples/benchmark_normalization.py --require-speedup
```

基准限制估计张量 64 MiB、保留系统 headroom 3 GiB、不加载模型。M4 Pro、BF16 4×4096、30 次同步中位数的一次实测：plain 0.182 / 0.152 ms（1.19×），residual 0.191 / 0.170 ms（1.12×）。包含 IR dispatch 与 GPU 同步，不等于生产吞吐；真实 Higgs/YuE2、大上下文和完整生产负载未在该路径验收。

### SiLU×gate（SwiGLU）

`activation.py` 通过上游 OOT 算子合同执行 SiLU×gate。有匹配 vpipe AOT 产物且满足既有大输入门槛时，选择同一 Torch stream 的 packed SwiGLU；否则使用已安装 MLX v0.32.1 的[编译 `nn.silu`](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/python/mlx/nn/layers/activations.py)（MIT）及官方 DLPack bridge。MLX 路径复用 `mx.compile(shapeless=True)`，不接管权重；[已发布 stateless MLP 编译方案](https://github.com/vllm-project/vllm-metal/blob/e7a0c90bec180e5dd4275c473fd7a36e709a930e/vllm_metal/compiled_mlp.py) 的模型专用 wrappers 不采用。这不是 GEMM epilogue 融合。

通过 vLLM `SiluAndMul.register_oot` 注册通用算子子类，不修改原始类的方法、in-tree operator registry 或模型 registry。OOT 查找使用类名 `SiluAndMul`，配置启停名保留原始 `silu_and_mul`。每个 spawn worker 在加载模型前导入该 provider，导入阶段不初始化 MLX。原生配置以 `none` 禁用其他 CUDA ops，只选择安全的 `+silu_and_mul` 与可选 `+rotary_embedding`；各自显式 opt-out 保留，冲突指令明确报错。普通 worker 或无当前 engine 配置不选择桥接。

仅 MPS FP32/FP16/BF16、连续 1–3 维、无梯度、最后一维偶数且输出元素数 ≥262144 的推理输入启用。这是测得的跨框架开销门槛，不是模型白名单；小 decode、非连续、空输入、梯度和其他输入继续走上游实现，保留原设备、异常与 autograd 语义。MLX SiLU 使用 FP32 计算，先舍入到输入 dtype，再乘 up 分支，不能只在最终结果 cast。借用的输入保持至 `mx.eval` 完成以防 buffer donation，输出通过官方 bridge 回到 Torch；不接管权重、模型加载或 RNG。

`tests/test_activation.py` 覆盖无 MLX 的 fresh-worker 注册、原类/模型映射不变、作用域、预算、FP32/FP16/BF16 的 1D/2D/3D 数值、输入不变、输出生命周期、梯度及关闭后备；显式测试编译另外验证无 MLX 的 Torch producer→activation→consumer 数值。FP16/BF16 判别样例逐值检查不能省略的中间舍入。`tests/test_native_runner.py` 使用 BF16 大 MLP 随机 Llama/Qwen3 对照 CPU HF greedy 及 seeded batch/solo；真实子进程通过字符串 RPC 检查路径，不关闭 spawn、不放开不安全序列化。

```bash
VLLM_OMNI_METAL_INTEGRATION=1 .venv-vllm-omni-metal/bin/python -m pytest -q \
  tests/test_activation.py tests/test_native_runner.py
.venv-vllm-omni-metal/bin/python examples/benchmark_activation.py --require-speedup
```

基准预算 64 MiB、系统 headroom 3 GiB；M4 Pro、BF16 32×8192、30 次同步中位数：原生 0.343 / 编译桥接 0.170 ms（2.0×），含 Python OOT dispatch 与双框架同步。小块桥接不稳定，保留原路径；不是模型吞吐或其他机器/形状的加速保证。可选缓存表 NeoX RoPE 的源合同已通过隔离验证，见下节；真实 Higgs/YuE2 和生产负载未在这些路径验收。

### vpipe 通用片段

固定参考 `tgo-app-dev/vpipe` commit `6a721d5c3b74ded70001f7c02121f14ae8f5321a`，Apache-2.0，源码/适配说明打包为 `_native/VPIPE.md`。采用 elementwise SwiGLU、table-driven partial NeoX RoPE，以及 `rms_norm_fast_f16` 的 per-row SIMD reduction（改编为符合 vLLM 合同的 add-before-norm 双输出）。没有移植模型/codec/视频/quantization，也没有采用会重算频率或使用专用 ring KV 布局的融合。上游有 unit tests，未发现 pinned revision 的公开 GitHub Actions，不宣称全套上游 CI 通过。

原版 SwiGLU 只做最终 cast，与 Torch 的中间舍入不等价（FP16/BF16 的257元素判别样例有81/72处差异）；移植先 cast SiLU，再乘 up，并读取连续 packed `[gate,up]`。缓存 NeoX RoPE 同样保留每个乘法的 dtype 舍入，再加减；上游 `cos_sin_cache` 构建、缩放、dtype 匹配和 `index_select` 输入边界校验不变，不重算频率。只注册通用 `RotaryEmbedding` OOT 子类；其他派生 RoPE、GPTJ style、梯度/非连续/其他合法输入保留原生实现，模型映射与原类方法不变。返回新 storage，partial tail 和 query/key 输入保持不变。

`vpipe_kernels.py` 的生产路径仅 `load_metallib`，没有隐式 shader JIT。显式构建需已有 Apple Metal compiler，私有 owned 目录、artifact 校验、flock、临时目录、原子发布与系统 headroom 保持；不安装工具或改变 MPS 水位。缺产物回退，unsafe/corrupt 产物明确报错。本机 `metal` 不可用，**AOT 和真实 worker 新路径仍未验收**。

```bash
python -m vllm_omni_metal.vpipe_kernels
# 只在隔离测试进程显式编译（不是生产/worker JIT）：
python examples/benchmark_vpipe.py --compile-for-test --operator rope --rows 4 --require-speedup
python examples/benchmark_vpipe.py --compile-for-test --operator silu --rows 1
```

测试覆盖三种 dtype、packed rows、257元素精度判别、cache 随机覆盖（不能替换成自算频率）、partial tail、key=None、GPTJ fallback、输入不变、artifact/预算安全。M4 Pro BF16 RoPE 4×hidden4096（head128、Q/KV heads32/8）0.239/0.203ms（1.18×）；单行 width4096 SwiGLU 0.138/0.138ms，无收益，数值片段**不接入默认小 decode**。FP32 attention 4×context2048、Q/KV16/4、head128：gather/bridge/Torch-stream 6.377/0.617/0.597ms（bridge改进1.03×）。基准含最终同步，预算64MiB（vpipe）/256MiB（attention）、headroom3GiB；不能外推模型吞吐。

add-before-norm 单 kernel 已完成数值及显式测试 worker 验证，但不是原样采用语义不同的 vpipe `rms_add`。GEMM+SwiGLU 仍缺合法的跨算子接入：linear OOT 不知道下一步 activation，SiluAndMul OOT 拿不到原输入与投影权重；vpipe 原实现另有交错布局和省略中间舍入的问题。QK norm+RoPE+KV 的上游 backend 合同存在，但 MPS 图融合 pass 未开放，当前 eager attention 入口拿不到 positions/cache/norm weights；不能以未命中的方法宣称完成。ModelScope 本地有 Higgs/YuE2 权重候选，HF 对应缺失不能概括全部缓存；本轮未做全量权重完整性校验，也未完成这些模型的新路径验收。

### 成熟方案续移植：packed Q/K RoPE

采用 [vllm-metal `f2e26b3` 的融合 RoPE](https://github.com/vllm-project/vllm-metal/blob/f2e26b342c93d6c705ebee0950a0baf8025ee46d/vllm_metal/metal/kernels_v2/rope.metal)（Apache-2.0）：Q/K 共用一次 dispatch，每线程旋转一对元素，直接写新输出，省去原先两次 clone、两次旋转及 cos/sin FP32 转换。仍先通过 Torch `index_select` 校验并收集上游缓存，不在 shader 中直接使用未校验 positions。每个乘法先舍入到输入 dtype，再加减；FP16/BF16、GQA、修改后的缓存及输入不变已逐位验证。仅完整 NeoX、同时有 Q/K 时采用；partial、FP32、key=None 仍走既有实现。没有扩大支持范围，没有融合 projection/QK norm/KV 写入。

本轮有界检索排除了图形/游戏/语言运行时；只细读 vllm-metal 与 [MLX RMSNorm](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/kernels/rms_norm.metal)（MIT）。两项目均有活跃维护和 CI。MLX 的 FP32 SIMD 归约与当前 Torch/vpipe norm 重复，未再引入桥接或第二套运行时；MRv2 输入兼容替换、模型专用融合、量化和 M5 NAX 不属于本次合法加速落点。不是穷尽所有 Metal 项目的结论。

M4 Pro，Q/KV heads32/8、head128、hidden4096；现有 `examples/benchmark_vpipe.py` 的 **30 次逐调用同步中位数**，包含 OOT dispatch、缓存选择和同步：

| dtype / rows | 上游 native（ms） | packed Q/K OOT（ms） | 耗时比 |
| --- | ---: | ---: | ---: |
| BF16 / 32 | 0.3334 | 0.1689 | 1.97× |
| FP16 / 1 | 0.2191 | 0.1449 | 1.51× |

```bash
python examples/benchmark_vpipe.py --compile-for-test --operator rope --rows 32 --dtype bfloat16 --require-speedup
python examples/benchmark_vpipe.py --compile-for-test --operator rope --rows 1 --dtype float16 --require-speedup
```

另存 `outputs/metal-packed-rope-current/report.json`：对上一版 separate-Q/K shader 做交错 7 轮、每轮100次异步调用后同步的隔离 A/B；6个 dtype/rows 组合均逐位等于 Torch，测得1.81–3.29×，有负载波动。计时口径不同，不能与上表混算。

**以上均为显式测试编译，不是生产 AOT 或 YuE2 新增整模型收益。** 生产仍只加载显式 metallib；本机缺 Metal compiler，默认 worker 不会命中新融合。新增源码随 wheel、源码安装及 Formula 一并分发，不自动装工具，不在启动/请求 JIT。

### 2026-10-15 逐项实测

本次为 M4 Pro 16 GPU cores、48 GiB、Torch 2.13。Metal compiler 缺失，生产 AOT 构建/加载未验收；以下新 shader 均通过显式 `--compile-for-test` 编译，不能冒充默认生产 worker 已启用。生产不增加 JIT fallback。

算子基准各 10 次 warmup、50 次交错正反序测量，计入 dispatch 与最终同步。与原后端比较，而不是只与更慢的原生分解比较：

| BF16 算子与形状 | 原后端中位 ms | 候选中位 ms | 原后端 / 候选 |
| --- | ---: | ---: | ---: |
| residual-add+RMSNorm，4×4096 | 0.3341 | 0.2043 | 1.635× |
| SwiGLU，32×8192 输出 | 0.1355（MLX bridge） | 0.1388（Torch stream） | 0.976× |

`examples/benchmark_native_optimizations.py` 通过真正 spawned native worker、scheduler、分页 attention、采样和 CPU token 返回执行完整生成；同一模型/worker 切换 baseline、仅 activation、仅 normalization、两者组合。禁用可选 RoPE OOT，避免第三项优化混入对照。四层、hidden512、intermediate8192、head64、约5309万**本地随机参数**，不是预训练权重或真实 TTS 验收。每次 batch2、每请求64 prompt tokens+16 output tokens；2轮warmup、8次交错计时：

| 模型 | 当前后端 ms | 仅 Torch-stream SwiGLU ms | 仅 add+RMSNorm ms | 两者组合 ms |
| --- | ---: | ---: | ---: | ---: |
| 随机 Llama | 417.427 | 411.675 | 374.262 | 383.231 |
| 随机 Qwen3 | 269.673 | 375.234 | 403.330 | 286.521 |

所有模式实际命中计数已核对：每次生成候选 activation 8次、候选 norm136次；greedy与seed42输出跨模式一致。系统干扰很大，Llama baseline样本297–543ms、Qwen3 baseline71–472ms，候选区间也重叠；**不能据此认定模型级稳定加速，更不能只挑 Llama 的有利结果外推**。SwiGLU 不降低小 decode 门槛；新路径保持显式 AOT 准备边界。

首轮真实预训练 Qwen3-VL-2B 两次均在加载前被预算拒绝：available分别9.62/7.20GiB，按本地权重字节和系统余量计算需要10.93GiB；没有生成数据。当时独立 YuE2 完整 ANE 校准入口也被启动前16GiB available门槛拒绝；释放内存后的真实 YuE2 结果见下节。现有 host extension 实测 `nax_supported=False`，M5 NAX不可验证；不停止用户服务、不调高水位。

```bash
# 小张量；显式测试编译，不创建生产 worker JIT 路径。
.venv-vllm-omni-metal/bin/python examples/benchmark_normalization.py --compile-for-test --iterations 50
.venv-vllm-omni-metal/bin/python examples/benchmark_activation.py --compile-for-test --iterations 50
# 实际模型执行层，随机权重，不下载。
.venv-vllm-omni-metal/bin/python examples/benchmark_native_optimizations.py --compile-for-test --architecture llama --iterations 8
.venv-vllm-omni-metal/bin/python examples/benchmark_native_optimizations.py --compile-for-test --architecture qwen3 --iterations 8
# 已有本地预训练权重可用 --model /absolute/checkpoint；加载前按文件大小检查预算。
```

上述随机模型检查：全部非集成223项通过；normalization、vpipe kernels、activation、native runner 的显式 MPS 回归190项通过（含独立 HF greedy 与 seed/request isolation）；另实际运行16-token短prompt/2-token输出烟测，确认短请求不一定合批、不能按总token数假定激活命中。Ruff check、format check及pip check通过。预训练模型、真实媒体与生产AOT不计入这批测试。

### 真实 YuE2：当前 Torch-stream attention 对照

释放内存后，使用本地真实 YuE2-3B / YuE2-Vae 权重和独立 `0.31.0.dev0+cpu.yue2mps2` overlay，加载当前 checkout 的通用后端，通过上游 `AsyncOmni` / `Yue2Adapter` 完成 AR→NAR→VAE 生成。该 overlay 包含独立模型适配修复，**不是随包未补丁 Omni 已自动支持 YuE2**，也未修改上游安装、registry 或现有服务。

M4 Pro、BF16 AR/NAR、FP32 VAE；同一 worker、seed42、512帧预算、相同歌词/风格/原生采样。每路独立 warmup，再按 bridge→Torch→Torch→bridge 计时，包含采样、合成及最终 CPU 音频返回，不含模型加载和 warmup。两路仅切换通用 attention 的执行 stream；其他当前后端设置相同。资源配置仅缩小至 context6144、显式 KV1GiB，保留同步 uni、CUSTOM、关闭 prefix cache、原生拓扑及输入语义。

| Attention 路径 | 样本1 s | 样本2 s | 中位 s |
| --- | ---: | ---: | ---: |
| 原 MLX bridge | 54.953 | 61.322 | 58.138 |
| 当前 Torch stream | 41.130 | 41.079 | 41.105 |

本次中位耗时比为 **1.414×**，延迟降低 **29.30%**；每路仅两次正式样本，不推广到其他模型、输入或长期热负载。每次真实请求实际命中14364次 paged attention：Torch 路径全部命中 host launcher，bridge 路径0次 Torch forward。

生成 **20.4787秒、48kHz、双声道 PCM16 WAV**；有限、非静音、幅度检查通过。各路同seed重复 hash 一致，跨路浮点音频也完全一致；不等于听感或歌词准确率验收。所有请求 pending 清零，engine正常退出，本次进程已结束；最低系统可用内存10.48GiB，无资源停止线触发。

**本次没有启用 ANE 或 vpipe AOT shader。** 新 RoPE/SwiGLU/add+RMSNorm metallib 仍缺编译工具，不能将本表归因于这些融合。真实 ANE 图自检 relative L2=5.499455%、峰值归一误差0.794141%，L2超过1%门槛，正确拒绝且不发布 profile。独立校准脚本已修复失败RPC被当成功、准备失败关闭原组及TorchVersion不能安全序列化的问题；11项行为回归通过，真实失败复验在 prepare 阶段立即退出，没有继续生成或启用不安全 pickle。

独立部署目录复现入口（`PYTHONPATH` 指向此 checkout；只使用本地权重，不调用服务启停脚本）：

```bash
VLLM_PLUGINS=metal,vllm_omni_register_models <yue2-private-overlay-python> prepare_ane.py \
  --gpu-only --attention-ab --frames 512 --deploy-config /path/to/resource-profile.yaml
```

本机原始记录、配置、artifact哈希、全部WAV及拒绝证据保存在 `outputs/yue2-real-current-20261015-204628/`；该目录被Git忽略，不是发行包资产。


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
