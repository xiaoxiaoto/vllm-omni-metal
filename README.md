# vllm-omni-metal

Apple Silicon 上的 **vLLM-Omni out-of-tree Metal 后端**，复用上游模型注册、原生多阶段拓扑、调度、权重加载及 HTTP 服务。

不捆绑模型专用实现、采样/codec 或 speech adapter，不覆盖模型 registry，也不注册第二套基础 vLLM 硬件平台。

## 支持与验证

| 路径 | 后端 | 已验证 |
| --- | --- | --- |
| 原生 `LLM_AR` / Talker | MPS、分页 KV、causal SDPA/GQA、上游模型 sampler | Higgs 4B Talker；本地随机 Llama / Qwen3 合批、6步 decode 与独立 HF greedy 结果一致 |
| 原生 `LLM_GENERATION` | MPS、上游 forward 与标准 multimodal 输出 | Higgs Code2Wav，原生双阶段 shared-memory 传输和 24 kHz WAV |
| 完整 Torch generation pipeline | 标准 registry、loader、MPS runner | 本地合成 pipeline |
| Diffusers | 上游通用 adapter、MPS、Torch SDPA | 本地随机 tiny Stable Diffusion |
| MLX 完整 pipeline facade | 自管理加载合同和内存统计 | 合同单元测试；无内置模型 facade |
| 基础 vLLM 文本/VLM | 现有 vllm-metal | 不改其默认 worker 或路由 |

原生 Higgs 验收使用上游 Talker → Code2Wav，**不是 mlx-audio 融合 facade**。语音接口返回 24 kHz、3.08 秒有限非静音 PCM16 WAV，同 seed 输出字节一致；这不等于听感或内容准确率评估通过。

**通用执行后端不等于所有 Omni 模型和功能都已移植。** 当前原生路径限单 MPS 设备、`uni`、同步调度、未量化 dense decoder attention。encoder multimodal 输入、prompt embeddings、auxiliary Talker MTP、prefix/model-state caching、logprobs、structured decoding、LoRA、跨引擎 KV、睡眠和 offload 未支持。CUDA/Triton 专用模型算子仍需独立移植；不自动 CPU fallback。

原生 generation pipeline 的 diffusion SDPA 通过本插件的 `MetalSDPABackend` 进入上游通用 Torch 实现，避免上游 `AttentionImpl.forward` 缺少 OOT 平台分派而报 `No forward implementation for platform`；不改变模型 registry、采样或上游 attention 类。mask、GQA、causal、scale 与布局的真实 MPS 数值回归见 `tests/test_diffusion_attention.py` 和[执行合同](docs/generic-backend.md)。

`diffusion/ops.py` 另提供显式、通用的 Diffusers RMSNorm、独立 gate/up SwiGLU、interleaved complex RoPE，复用 MPS fused norm、已有 MLX bridge 与可选 Torch-stream AOT。它们不会全局改写上游模型，需独立适配器接入；不是把 AR 分页 KV／NeoX 算子直接套给 diffusion。`TiledDenseSwiGLU` 保持原256MiB与1%门槛，支持一般大 inner 的有界分片。QwenImage 私有适配已完成小算子、生命周期和真实单层权重验证，但完整图片 A/B 受内存门禁、生产 AOT 受完整 Xcode 缺失阻塞；不能宣称整模型已提速或默认 ANE 已启用。见[diffusion 合同](docs/generic-backend.md#diffusion-显式算子接入)和[分片 ANE](docs/ane.md#有界-inner-分片独立-gateupdown)。

原生 runner 已支持通用 `prepare_runner_inputs`（含 opt-in 请求级采样参数）、采样期间追加终结输出，以及按请求 ID 路由的稀疏音频输出。稀疏协议的非法 marker、重复/未知 ID 和列表长度不一致均拒绝传输；无输出的请求不会收到其他请求的音频或 hidden states。

**随包未补丁 Omni 的 YuE2 仍未完成端到端 Metal 支持。** 其 CPU generator/MPS 张量不匹配和 CUDA pinned-memory/event 假设需要上游或独立模型适配，本插件不硬编码。独立 `yue2mps2` overlay 配合当前后端已完成真实512帧歌曲及 attention A/B，结果见[真实 YuE2 对照](docs/generic-backend.md#真实-yue2当前-torch-stream-attention-对照)；这不等于默认安装开箱可用。兼容性与升级边界见 [YuE2 接入状态](docs/yue2-readiness.md)。安装固定 Omni main 开发快照，不伪装稳定 release。

通用 `vllm_omni_metal.quantization` 提供离线 `DynamicEngine`、真实激活
`CaptureSession`、`DenseModelAdapter` 与加载期 `configure_model_quantization`。
全局 `VLLM_OMNI_METAL_QUANTIZATION=off|dynamic` 默认 `off`；
dynamic 要求 `VLLM_OMNI_METAL_QUANTIZATION_PROFILE` 为绝对路径、当前用户 owned0600
的已批准 profile。支持单个 schema2 或 `{"profiles": {实际完整执行scope: profile}}`
集合，不按模型名称选配置。模型私有 mode/profile 参数分别优先于全局参数，显式 off
可覆盖全局 dynamic；无匹配批准、非法配置或不支持的执行合同明确报错，不静默 fallback。

已接入原生 runner、Torch diffusion 和 MLX 自管理 diffusion 加载入口：
普通 `nn.Linear`、TP1 未量化 vLLM 全投影以及 Diffusers 未注册 `components`
使用通用适配器；MLX 自管理对象必须显式提供 scope/apply hook。模型融合算路仍需
私有适配；YuE2、Higgs、SeedVR2 和 Qwen-Image 的当前私有 dynamic 适配与校准器
随各自 Omni wheel 分发，不在通用包注册模型。W4A16／W8A16 使用 MLX affine group64，
FP16/BF16 保留源 dtype，FP32 只支持 identity；不隐式 cast。
旧 `ops.quantization` 路径无 shim，改用 `quantization.linear`。
这些是执行路径和离线校准合同，不是任意 Omni 模型已移植或四个真实模型质量／速度
通过的声明；没有自动启用、迁移服务或发布。完整合同见
[通用 DynamicEngine](docs/generic-backend.md#通用-dynamicengine-加载合同)。
历史真实 YuE2 单个低敏感 NAR down 投影对照为有损、未证明加速，未批准默认量化；
完整样本、音频误差和单投影内存口径见[历史实测](docs/generic-backend.md#2026-10-17真实-yue2-选择性-w8a16-对照)。

## 通用算子优化

原生 attention 复用 GitHub 已发布的 [vllm-metal v0.32.0](https://github.com/vllm-project/vllm-metal/releases/tag/v0.32.0) `reshape_and_cache` / `paged_attention_primitive`，不另写 Metal kernel；核心升级仅为兼容 vLLM 0.32 的必要配套，不改变本插件算子实现。通过上游 DLPack bridge 共享 Torch KV，直接读取物理分页，避免历史 KV gather 与 GQA 复制；选择条件只有设备、dtype、head/block size 和布局，没有模型名分支。

优化域为 FP32/FP16/BF16、head size 64/96/128/256/512、block size 8/16/32，以及连续 NHD 页内布局；其他已支持形状保留原有 MPS SDPA，不自动转 CPU。覆盖 prefill/decode、sliding window、MHA/GQA/MQA、strided KV offset 和请求隔离。默认 wheel 的 MLX bridge 有跨框架同步开销；下述可选 Torch-stream launcher 消除 attention 的桥接。FP32 同样可使用显式构建的 Torch-stream launcher（prefill 复用分页 kernel，不使用 half-only tiled shader），详见 [协议与来源](docs/generic-backend.md#已发布算子来源与验证)。

复现小张量 decode 基准（不加载模型，包含跨框架同步）：

```bash
.venv-vllm-omni-metal/bin/python examples/benchmarks/attention.py --require-speedup
```

Apple M4 Pro、BF16、batch=4、context=2048、Q/KV heads=16/4、head size=128，20 次同步计时中位数：旧 gather+SDPA **5.956 ms**，上游分页算子 **0.512 ms**，约 **11.6×**。这是指定形状的算子端到端微基准，不是模型吞吐或所有输入的加速承诺；真实 Higgs 大权重未在新路径重跑。

第二阶段已接入通用 **RMSNorm / 残差归一化**：移植上游已合并 [vllm-metal #858](https://github.com/vllm-project/vllm-metal/pull/858) 的小型 IR adapter，调用 PyTorch 已有的融合 MPS RMSNorm，不写新 kernel、不升级依赖。原生 Metal worker 的 FP16/BF16、同 dtype/device 的一维 learned weight、完整 variance 默认选择 `omni_mps`；FP32、无 weight、variance override 或其他合法不匹配参数保留上游 `native`（MPS 输入仍在 MPS）。用户显式 IR 优先级保持优先，普通 LLM 配置不受影响。

残差路径保留 **FP32 相加、未舍入 sum 的 variance、归一化先 cast 再乘 learned weight**。现有 Torch 分解继续作为后备；显式预编译 vpipe 产物可启用连续无梯度 FP16/BF16 的 **add+RMSNorm 单 kernel**，缺产物、不支持布局或 epsilon 保持原路径。新 kernel 已通过显式测试编译的数值及随机模型生成对照，生产 AOT 构建/加载尚未验收。

```bash
.venv-vllm-omni-metal/bin/python examples/benchmarks/normalization.py --require-speedup
```

Apple M4 Pro、BF16、4×4096、30 次同步计时：普通 RMSNorm 0.182 → 0.152 ms（1.19×），残差归一化 0.191 → 0.170 ms（1.12×）。这是一次指定形状的算子微基准，不代表整模型吞吐；复现会有负载波动。

第三阶段接入通用 **SiLU×gate（SwiGLU）融合**：通过上游 CustomOp OOT 合同，缺少显式 vpipe AOT 产物时复用 MLX 编译 SiLU 与官方 DLPack bridge，有产物则可选择同一 Torch stream 的 packed kernel，不替换模型方法。仅原生 worker 的 FP32/FP16/BF16、连续 1–3 维、无梯度、最后一维为偶数且输出元素数 ≥262144 时启用；小 decode、其他合法输入保留原设备上的上游实现。保留 SiLU 的 FP32 计算及乘 gate 前的 dtype 舍入；可用 `custom_ops=["none", "-silu_and_mul"]` 显式关闭。这不是 GEMM+SwiGLU 投影级融合。

```bash
.venv-vllm-omni-metal/bin/python examples/benchmarks/activation.py --require-speedup
```

M4 Pro、BF16、32×8192、30 次同步中位数：原生 0.343 → 编译路径 0.170 ms（约 **2.0×**），包含 OOT dispatch、桥接和同步。小块桥接收益不稳定，因此不默认优化小 decode。随机大 MLP Llama/Qwen3 已确认真实 worker 命中新路径、greedy 与 seed 回归通过；这仍不是整模型吞吐结果。可选缓存表 NeoX RoPE 的 vpipe 移植及限制见下节；隔离算子验证不等于默认 worker 已启用。

GPU + ANE 合作提供独立、通用的 dense SwiGLU contract；默认 auto 只消费绑定源码/设备/权重/完整 pipeline 配置的离线校准，缺少匹配准备产物即保留 GPU，不在 worker/request 编译。FP16 math 必须由外部模型适配器显式选择，不是 BF16 等价替代；本插件不接管模型 registry 或捆绑 YuE2 实现。本次真实 YuE2 图自检因 relative L2 5.50% 超过1%门槛被拒绝，没有可用的本次 ANE 加速结论。准备与误差限制见 [GPU + ANE 协议](docs/ane.md)。

联合量化另提供单投影 `ane.projection.DenseProjection`：支持非方阵、bias 和显式
FP16 ANE 近似，保留原生 gate/up 与 DynamicEngine down 的独立执行，不持有已量化
down 的浮点回退。真实 NE 数值烟测不等于模型批准；Higgs、Qwen Image、SeedVR2
必须分别完成整任务质量、实际双路径命中、相对同量化 GPU 基线的3%收益及释放重放，
不能拼接两份独立批准或只改 flags 设默认。详见[联合投影合同](docs/ane.md#单线性投影与联合动态量化)。

单投影离线校准可通过 `set_benchmark_chunks("auto")` 使用 vpipe 式预测分配：
按同步实测的 GPU 秒/行与 ANE 秒/块更新速率，枚举合法 split，仅预测改善超过5%
才调整。`finish_audit()` 冻结最后实际审核过的分配；生产不在线改变已批准策略。
这不解除256行分块限制，也不替代完整任务的1%质量及3%收益门槛。

第四阶段提供可选 **纯 Torch-stream paged attention**：复用已合并 [vllm-metal #858](https://github.com/vllm-project/vllm-metal/pull/858) 的 host launcher 和已安装的 0.31 shader。固定 wheel 缺少该 launcher，因此需在选定环境中显式编译一次（Apple Command Line Tools）；host adapter 仅补齐已发布 FP32 shader 的 dtype/路由选择，不升级依赖、不写 attention shader、不修改上游安装，运行时不自动编译。

```bash
.venv-vllm-omni-metal/bin/python -I -m vllm_omni_metal.ops.mps_attention
.venv-vllm-omni-metal/bin/python examples/benchmarks/attention.py --require-speedup
```

编译产物在用户私有 cache，绑定源文件、Torch 环境及 Python ABI；随后新 worker 的支持域 FP32/FP16/BF16 自动选择 Torch-stream。无匹配产物保留原 bridge；已存在但损坏/不安全的产物明确报错。attention 的元数据异步送入 Torch stream，KV 写读按同一 stream 排序，无 MLX 导入或每层显式 GPU wait；**不代表整个模型没有跨框架操作**，缺少 vpipe AOT 时大输入 SwiGLU 仍使用 bridge。

M4 Pro、BF16、batch=4、context=2048、16/4 heads、head size=128，20 次同步实测：gather+SDPA 5.936 ms、MLX bridge 0.515 ms、Torch-stream 0.477 ms；相对 bridge 约 **1.08×**，相对原 gather 约 12.45×。仅为指定形状微基准，新路径已验证独立数值对照、KV strides/offset、请求隔离与真实随机模型生成；真实大模型与 M5 NAX 尚未验收。

### vpipe 通用算子移植与当前未完成项

固定参考 [tgo-app-dev/vpipe](https://github.com/tgo-app-dev/vpipe/tree/6a721d5c3b74ded70001f7c02121f14ae8f5321a)（Apache-2.0），移植通用 packed SwiGLU、缓存表 NeoX RoPE，并将 `rms_norm_fast_f16` 的 SIMD reduction 改编为等价 add-before-norm 双输出；做必要的 dtype/layout 适配，不移植模型包装、频率重算、视频/量化或 Gemma 专用 residual 逻辑。来源和精度合同见 [执行协议](docs/generic-backend.md#vpipe-通用片段)及打包的 `_native/VPIPE.md`。

生产 worker **仅加载显式预编译 metallib**，缺失则保留原路径；运行 `python -I -m vllm_omni_metal.ops.vpipe_kernels` 需要已安装的 Apple Metal compiler。本机 `xcrun --find metal` 失败，未安装工具。因此本次仅验证隔离 shader、OOT cache 合同及显式测试基准，不宣称新 RoPE 已在真实 worker 中加速。

M4 Pro / BF16：缓存 RoPE（4 rows、hidden 4096、Q/KV heads32/8、head128）0.239 → 0.203 ms（1.18×）；单行 SwiGLU（width4096）0.138 → 0.138 ms（无收益），**不接入默认小 decode 路径**。FP32 attention batch4/context2048/head128 的三路为 6.377 / 0.617 / 0.597 ms，Torch-stream 比 bridge 快约1.03×。这些都不是模型吞吐。

新增三路交错算子基准和 spawned native worker 四模式生成对照；随机 Llama/Qwen3 的新 SwiGLU/norm 整模型结果波动大，未证明稳定收益，见[逐项实测](docs/generic-backend.md#2026-10-15-逐项实测)。释放内存后，独立私有 overlay 已完成[真实 YuE2 attention 对照](docs/generic-backend.md#真实-yue2当前-torch-stream-attention-对照)，该结果不包含 ANE 或缺少产物的 vpipe AOT 融合。仍未完成：GEMM+SwiGLU、QK norm+RoPE+分页 KV 的通用跨算子接入，小 decode 稳定加速，生产 vpipe AOT、Higgs 新路径复验、生产负载和 M5 NAX。没有自动下载、停止现有服务或放宽水位；本地代码不能冒充已发布入口。

本轮继续采用 [vllm-metal packed Q/K RoPE](https://github.com/vllm-project/vllm-metal/blob/f2e26b342c93d6c705ebee0950a0baf8025ee46d/vllm_metal/metal/kernels_v2/rope.metal)（Apache-2.0），完整 FP16/BF16 NeoX 的 Q/K 共用一次 dispatch，省去输入 clone，保留上游缓存边界和每次乘法舍入。M4 Pro BF16、32 rows、hidden4096 的实际 OOT 同步中位数 0.3334→0.1689 ms（1.97×）；这是显式测试编译，不是生产 AOT 或 YuE2 整模型加速。详细来源、对照口径及不采用方案见[成熟方案续移植](docs/generic-backend.md#成熟方案续移植packed-qk-rope)。

## 环境与安装

参考 [vllm-metal 安装方式](https://github.com/vllm-project/vllm-metal/blob/main/docs/installation.md)：**独立 Python 环境 + 预编译 macOS arm64 wheel**，不把 Omni 追加到已有 vllm-metal 的 Homebrew 私有环境。

需要 Apple Silicon、**macOS 15+**、原生 Python 3.12。当前固定栈：

| 项目 | 安装版本 |
| --- | --- |
| vLLM | `0.32.0+cpu`，官方 macOS arm64 cp38-abi3 wheel |
| vllm-metal | `0.32.0`，官方 macOS 15+ arm64 wheel |
| vLLM-Omni | 官方 main `d6db19b41f8c081506468859cd8f3910f77f1e5c`，开发版 `0.32.0.dev0+cpu` |
| vllm-omni-metal | `0.32.0.dev0`，与 Omni 开发版公开版本一致（不含 `+cpu` 构建标记） |
| Transformers | 由 vLLM 0.32 解析至 `>=5.16.1,<5.20.0` |

Omni 开发版号允许本地 `+cpu` 构建标签；插件依赖保留 `>=0.32.0.dev0`，不限制本地构建标签。Omni `+cpu` 与 Metal 插件开发版号明确标记为开发快照，不冒充稳定发布版。

### 源码安装

```bash
cd /path/to/vllm-omni-metal
./install.sh
source .venv-vllm-omni-metal/bin/activate
vllm-omni --help
```


```bash
cd /path/to/vllm-omni-metal  # 包含本次新增 install.sh 的 checkout
./install.sh
source .venv-vllm-omni-metal/bin/activate
vllm-omni --help
```

默认创建源码目录中的 `.venv-vllm-omni-metal` 并 editable 安装；可用 `./install.sh --venv /absolute/path/to/new-env` 指定独立环境。原有开发 `.venv` 保持不变。

`./install.sh --with-mps-attention` 显式构建并验证 host launcher；`--with-vpipe` 需要已有 Metal compiler，`--with-ane` 只准备 CoreML host 工具，不表示 ANE 已通过模型校准。普通安装不编译。安装校验/构建使用所选环境 `-I`，忽略 cwd/PYTHONPATH；源码、curl 脚本与 Formula 分发同一组 native 资源，但未发布远端入口不会自动包含本地改动。

### Homebrew 与 curl

本地 `Formula/vllm-omni-metal.rb` 与源码/curl 安装器已固定 vLLM / Metal 0.32、Omni `0.32.0.dev0+cpu` 和插件 `0.32.0.dev0`。本次未完成全依赖重新安装或远端发布；下列远端命令仅在配套资源发布后使用，不能把本地 wheel 烟测当作新安装入口端到端通过。

```bash
brew tap xiaoxiaoto/vllm-omni-metal https://github.com/xiaoxiaoto/vllm-omni-metal
brew install --HEAD --build-from-source xiaoxiaoto/vllm-omni-metal/vllm-omni-metal
vllm-omni --help  # Homebrew libexec，不必激活环境，也不覆盖 vllm 命令

# 或安装到 ~/.venv-vllm-omni-metal，再激活：
curl -fsSL https://raw.githubusercontent.com/xiaoxiaoto/vllm-omni-metal/main/install.sh | bash
source ~/.venv-vllm-omni-metal/bin/activate
```

Formula 暂仅支持 HEAD：旧 `0.1.0` 稳定源码快照要求 0.30，不能与新栈混装。curl 模式安装本插件 main，源码模式 editable 安装当前目录；native wheels 和 Omni commit 始终固定。已有 Homebrew 安装在停止所有共享服务后用 `brew reinstall --HEAD --build-from-source` 切换新配方；仅看插件 commit 是否变化不足以确认依赖已经升级。

平台通过 `vllm_omni.platform_plugins` 自动发现。若设置 `VLLM_PLUGINS`，保留 `metal` 和上游所需插件（如 `vllm_omni_register_models`）。不需要本项目的 general-plugin 注册函数。

Omni 构建使用 **`VLLM_OMNI_TARGET_DEVICE=cpu`**，不是 `VLLM_TARGET_DEVICE`。默认进程启动方式为 `spawn`；不要 fork 已初始化的 Metal/Objective-C。服务运行安装环境的 `vllm-omni`，不是基础 `vllm`。原有 `.venv` 不动；以下命令使用本次完整安装的 `.venv-vllm-omni-metal`，其他环境替换路径。

## 原生多阶段使用

使用上游模型的正常 topology 和 `--deploy-config`，为每个原生阶段明确设置：

```yaml
# 这是 engine 参数示意，不是完整部署文件。
max_num_seqs: 1
attention_backend: CUSTOM
enable_prefix_caching: false
async_scheduling: false
gpu_memory_utilization: 0.3
```

保留模型自己的 stage IDs、执行类型、connectors、stage processors 和采样参数。不要直接照搬含 FLASHINFER / prefix caching 的 CUDA 高吞吐 profile；`CUSTOM` 对应本插件的 Metal paged attention / MPS SDPA，不是 Torch Diffusers 的 `TORCH_SDPA` 名称。

```bash
.venv-vllm-omni-metal/bin/vllm-omni serve /path/to/local/checkpoint --omni \
  --deploy-config /path/to/metal-resource-profile.yaml \
  --host 127.0.0.1 --port 18094
```

`tests/test_native_integration.py` 演示从上游部署配置生成只调整资源/后端参数的临时 profile，保留原生双阶段和 async chunk。模型文件必须已存在，不自动下载大权重。

## Diffusers 示例
按用途分包：性能与执行层基准位于 `examples/benchmarks/`，端到端生成示例位于 `examples/generation/`。


```bash
.venv-vllm-omni-metal/bin/python examples/generation/image.py \
  --model hf-internal-testing/tiny-stable-diffusion-pipe \
  --prompt 'a small red robot' --steps 2 --width 64 --height 64 \
  --output outputs/robot.png
```

脚本接受任意 `--model`，不含模型专用适配。Python API 使用 `diffusion_load_format='diffusers'`、`num_gpus=1`、`distributed_executor_backend='uni'`，不是 `model_impl='diffusers'`；Diffusers attention 为 `TORCH_SDPA`。

未缓存模型首次使用会下载。高层 Omni 有一次 512×512 warmup，小请求的启动内存也可能较高；tiny smoke test 不代表正式模型质量。

## 检查与验收

```bash
.venv-vllm-omni-metal/bin/python -m pytest -q -m 'not integration'
.venv-vllm-omni-metal/bin/ruff check .
.venv-vllm-omni-metal/bin/ruff format --check .

# 本地随机模型：图像、完整 pipeline、Llama 和 Qwen3；无需下载。
VLLM_OMNI_METAL_INTEGRATION=1 .venv-vllm-omni-metal/bin/python -m pytest -q

# 加入已有本地原生 Higgs checkpoint；HTTP 真正生成音频并检查 seed。
VLLM_OMNI_METAL_INTEGRATION=1 \
VLLM_OMNI_METAL_NATIVE_MODEL=/path/to/local/higgs-checkpoint \
  .venv-vllm-omni-metal/bin/python -m pytest -q

.venv-vllm-omni-metal/bin/python -m pip wheel --no-deps --no-build-isolation -w dist .
```

测试涵盖不连续物理页、增量 causal offset、GQA、请求隔离、标准模型签名、采样约束、pre/postprocess、hidden-state 输出、配置/内存边界，以及 macOS shared-memory 的长名字、帧长度、锁和资源清理。

## 实现与边界

`vllm_omni_metal/` 按功能分包：`execution/` 管执行与兼容层，`native/` 管原生 scheduler/runner/attention，`diffusion/` 管 diffusion 配置、worker 与显式算子，`ops/` 管通用 GPU 算子及构建入口，`ane/` 管 ANE runtime、离线构建与缓存。包初始化保持惰性，不重新导出旧模块；顶层仅保留 `__init__.py`、`platform.py` 和原位 `_native/` 资源。外部适配器的迁移要求见[安装说明](docs/installation.md#内部-api-功能分包切换)。

- `platform.py`：MPS 设备、内存和 worker 选择。
- `native/worker.py` / `native/runner.py`：原生 scheduler-driven worker、加载、KV 布局、请求状态、采样 hook、CPU payload 与上游传输。
- `native/attention.py` / `ops/metal_attention.py`：全 batch 输入校验后按算子参数选择上游分页 Metal kernel 或 MPS SDPA；保留实际 KV offset/strides、causal offset、GQA 和请求隔离。
- `execution/shm_connector.py`：Metal worker 中的 POSIX SHM 兼容实现；私有临时锁目录、短物理名字和长度帧，避开 Linux `/dev/shm` 假设。
- `native/compat.py`：只在 MPS 模型实例上克隆包含 CUDA capture 查询的方法；原始 class/module 和 `torch.cuda` 不变，范围检查继续执行。
- `execution/runtime.py` / `diffusion/config.py` / `diffusion/worker.py`：完整 Torch/MLX pipeline 和 Diffusers 的共用执行层。
- `ops/mps_attention.py` / `_native/`：可选私有 cache 构建、已合并上游 host launcher 和原样 shared headers；支持域 attention 不经过 MLX。
- `ops/normalization.py`：通过上游 IR 合同复用融合 MPS RMSNorm，保留残差精度顺序与参数后备。
- `ops/activation.py`：通过上游 OOT 算子合同复用 MLX 编译 SiLU×gate；按输入规模选择，不修改模型 registry 或原始算子类。
- `execution/compat.py`：幂等且 scoped 的配置、attention、原生 worker IR 默认优先级和 MPS profiler 桥接，保留普通 LLM hook。

协议与限制见 [通用后端设计](docs/generic-backend.md)。没有修改上游安装文件、MPS 内存水位或全局 `torch.cuda`。MPS peak-memory 的 `0` 表示指标不可用；MLX 内存不计入 MPS 指标。

插件 Apache-2.0 许可不改变外部模型授权；Higgs 权重使用前须核对 Boson 研究/非商业及创作者条款。
