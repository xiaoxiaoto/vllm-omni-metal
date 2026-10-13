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

原生 runner 已支持通用 `prepare_runner_inputs`（含 opt-in 请求级采样参数）、采样期间追加终结输出，以及按请求 ID 路由的稀疏音频输出。稀疏协议的非法 marker、重复/未知 ID 和列表长度不一致均拒绝传输；无输出的请求不会收到其他请求的音频或 hidden states。

**YuE2 尚未完成端到端 Metal 支持。** 最新 Omni 有原生实现，但其模型采样仍有 CPU generator/MPS 张量不匹配和 CUDA pinned-memory/event 假设；这些修复属于上游模型，不在本插件硬编码。兼容性核对、实际测试及版本升级边界见 [YuE2 接入状态](docs/yue2-readiness.md)。安装入口已升级到 0.31 native wheels + 固定 Omni main 快照，Omni 明确标为开发版，不把它伪装成稳定 release。

## 环境与安装

参考 [vllm-metal 安装方式](https://github.com/vllm-project/vllm-metal/blob/main/docs/installation.md)：**独立 Python 环境 + 预编译 macOS arm64 wheel**，不把 Omni 追加到已有 vllm-metal 的 Homebrew 私有环境。

需要 Apple Silicon、**macOS 15+**、原生 Python 3.12。当前固定栈：

| 项目 | 安装版本 |
| --- | --- |
| vLLM | 最新稳定 `0.31.0+cpu`，官方 macOS arm64 wheel |
| vllm-metal | 最新稳定 `0.31.0`，官方 macOS arm64 wheel |
| vLLM-Omni | main `2ec5f265c69520cdd5e5f0a19db485d38885843e`，标记 `0.31.0.dev0+cpu` |
| PyTorch / Transformers | 本次完整安装为 `2.13.0` / `5.14.1` |

Omni 最新稳定 tag 仍为 0.30；需要 YuE2 等新集成时使用上述固定 main 快照，不能与 0.30 vLLM wheel 混装。native wheel 固定 SHA256，Omni 固定 commit，不在每次启动时自动追踪最新依赖，不下载模型。无需编译本项目或 vllm-metal 的原生 kernel。

已在独立 `.venv-vllm-omni-metal` 完整安装并通过依赖检查、CLI 和 **72 项测试**（含真实 MPS 的随机 Llama/Qwen3、tiny SD 及原生 generation pipeline）；真实 Higgs 大权重验收未在新栈重跑，YuE2 音乐生成仍未通过。

### 源码安装（当前 checkout 可用）

```bash
cd /path/to/vllm-omni-metal  # 包含本次新增 install.sh 的 checkout
./install.sh
source .venv-vllm-omni-metal/bin/activate
vllm-omni --help
```

默认创建源码目录中的 `.venv-vllm-omni-metal` 并 editable 安装；可用 `./install.sh --venv /absolute/path/to/new-env` 指定独立环境。原有开发 `.venv` 保持不变。

### Homebrew 与 curl

已提供 `Formula/vllm-omni-metal.rb` 和支持 `curl | bash` 的安装器。**本次 0.31 升级尚未发布到远端，远端 tap/curl 不会自动包含本地改动**；发布 Formula、安装器和插件源码后：

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

保留模型自己的 stage IDs、执行类型、connectors、stage processors 和采样参数。不要直接照搬含 FLASHINFER / prefix caching 的 CUDA 高吞吐 profile；`CUSTOM` 对应本插件的 MPS SDPA，不是 Torch Diffusers 的 `TORCH_SDPA` 名称。

```bash
.venv-vllm-omni-metal/bin/vllm-omni serve /path/to/local/checkpoint --omni \
  --deploy-config /path/to/metal-resource-profile.yaml \
  --host 127.0.0.1 --port 18094
```

`tests/test_native_integration.py` 演示从上游部署配置生成只调整资源/后端参数的临时 profile，保留原生双阶段和 async chunk。模型文件必须已存在，不自动下载大权重。

## Diffusers 示例

```bash
.venv-vllm-omni-metal/bin/python examples/generate_image.py \
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

- `platform.py`：MPS 设备、内存和 worker 选择。
- `native_worker.py` / `native_runner.py`：原生 scheduler-driven worker、加载、KV 布局、请求状态、采样 hook、CPU payload 与上游传输。
- `native_attention.py`：按 block table 读写物理 KV 页，正确 prefill/decode causal mask 和 GQA；当前实现以正确性为先，不是专用高吞吐 Metal kernel。
- `shm_connector.py`：Metal worker 中的 POSIX SHM 兼容实现；私有临时锁目录、短物理名字和长度帧，避开 Linux `/dev/shm` 假设。
- `native_compat.py`：只在 MPS 模型实例上克隆包含 CUDA capture 查询的方法；原始 class/module 和 `torch.cuda` 不变，范围检查继续执行。
- `runtime.py` / `config.py` / `worker.py`：完整 Torch/MLX pipeline 和 Diffusers 的共用执行层。
- `compat.py`：幂等且 scoped 的配置、attention 和 MPS profiler 桥接，保留普通 LLM hook。

协议与限制见 [通用后端设计](docs/generic-backend.md)。没有修改上游安装文件、MPS 内存水位或全局 `torch.cuda`。MPS peak-memory 的 `0` 表示指标不可用；MLX 内存不计入 MPS 指标。

插件 Apache-2.0 许可不改变外部模型授权；Higgs 权重使用前须核对 Boson 研究/非商业及创作者条款。
