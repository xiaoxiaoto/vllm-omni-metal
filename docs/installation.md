# 安装与 0.31 升级

使用独立 Python 3.12 环境和官方 macOS arm64 wheel，不修改系统 Python、原有 `.venv` 或 vllm-metal 的 Homebrew 私有环境。

## 固定依赖

- 原生 Apple Silicon、macOS **15+**，不支持 Rosetta。
- vLLM **0.31.0+cpu** 和 vllm-metal **0.31.0**（本次核对的最新稳定版），官方预编译 wheel，SHA256 固定在 Formula 和安装器。
- Omni 最新稳定 tag 仍为 0.30.0。为使用 YuE2 等新集成，固定 main **`2ec5f265c69520cdd5e5f0a19db485d38885843e`**，构建为 **`0.31.0.dev0+cpu`**，明确是开发快照，不冒充稳定 release。
- 插件版本为 **`0.31.0.dev0`**，跟随固定 Omni 的公开版本；`+cpu` 仅为 Omni 构建标记。后续升级须同步插件元数据、源码回退版本及 Omni 依赖，安装器和 Formula 测试会校验版本一致。
- 构建设置 `VLLM_OMNI_TARGET_DEVICE=cpu` 和 `VLLM_OMNI_VERSION_OVERRIDE=0.31.0.dev0+cpu`；前者不是 `VLLM_TARGET_DEVICE`。
- 本次完整解析得到 PyTorch 2.13.0、Transformers 5.14.1、MLX 0.32.1、diffusers 0.40.0。Omni 限定 Transformers `>=5.13,<5.15`；不能绕过这些约束只升级某个包。

本插件的默认安装无需编译 Metal kernel；可选 Torch-stream attention 复用上游 host-only launcher（仅补齐已发布 FP32 kernel 的 host dtype/路由选择），需用户显式编译 Objective-C++ host extension，不编译新 shader。安装需要联网下载依赖和 Git 源码，不下载模型。不要用 PyPI CUDA vLLM wheel 替代 macOS CPU wheel，也不能把最新 Omni main 与 vLLM 0.30 混装。

## 1. 当前源码：已实际安装验证

```bash
cd /path/to/vllm-omni-metal
./install.sh
source .venv-vllm-omni-metal/bin/activate
vllm-omni --help
python -m pip check
```

默认创建当前 checkout 的 `.venv-vllm-omni-metal`，editable 安装当前插件。也可指定新的独立路径：

```bash
./install.sh --venv "$HOME/.venv-vllm-omni-metal"
```

安装器只接受新目录或它自己标记创建的环境，不接管任意已有 venv/Homebrew libexec，不使用 `--clear`。失败时保留环境和缓存供诊断、重试。依赖和插件在同一次 uv 解析中安装，保持 vllm-metal 声明的 mlx-lm Git URL 可见。

总部署预检可以设置 `VLLM_OMNI_COMMIT=<完整 40 位 SHA>`，让安装器从查询到的官方 Omni 主干 commit 构建；省略时仍使用上述固定快照。非法 commit 会在安装前拒绝。开发版公开版本与 CPU 构建标记保持明确，不伪装稳定发布。

`docker-compose/start.sh` 的模型启动现在统一调用 `prepare_backend.py`：查询官方最新核心标签与 Omni main，构建独立 Application Support 后端，并从本仓库构建非 editable wheel。四个模型使用同一版本组，YuE2/MOSS 私有扩展只覆盖各自模型适配；不修改 Homebrew。新官方核心超出当前兼容矩阵时明确拒绝启动，须先更新 native wheel pins、插件依赖和部署校验并重新验证，不能逐包盲升。

更新一个已使用的环境前，先停止所有使用它的服务。本次安装使用新的独立环境，保留运行中的 Higgs 及旧 Homebrew 环境；没有自动迁移服务。

## 2. Homebrew：发布本次改动后

0.31 固定栈及此前源码已发布，但本次新增 vpipe/FP32 算子与验收改动尚未提交/推送；远端 HEAD 不会自动包含工作树。新 Formula **仅支持 HEAD**：旧 0.1.0 源码快照要求 0.30，不能作为新栈的稳定源码冒用；暂无匹配新栈的已发布插件 release。

发布配套源码与 Formula 后：

```bash
brew tap xiaoxiaoto/vllm-omni-metal https://github.com/xiaoxiaoto/vllm-omni-metal
brew update
brew install --HEAD --build-from-source xiaoxiaoto/vllm-omni-metal/vllm-omni-metal
```

已有安装切换新栈时，先停止共享该安装的服务，再运行：

```bash
brew reinstall --HEAD --build-from-source xiaoxiaoto/vllm-omni-metal/vllm-omni-metal
brew test xiaoxiaoto/vllm-omni-metal/vllm-omni-metal
```

配方使用独立 `libexec`，仅链接 `vllm-omni`，不接管已有 `vllm`。保留 `preserve_rpath` 和 TVM-FFI/xgrammar RECORD。Python 位于 `$(brew --prefix vllm-omni-metal)/libexec/bin/python`，无需激活。

不要只比较插件 HEAD 是否变化来判断依赖已升级。总部署已改用统一隔离后端，不再依赖上述 Homebrew Formula；单独使用 Formula 的项目仍需按发布、停止、重装的顺序迁移，不能复用旧 0.30 栈。

本次没有运行实际 `brew install/reinstall`，没有改 Homebrew libexec 或重启已有服务。

## 3. curl：发布本次改动后

```bash
curl -fsSL https://raw.githubusercontent.com/xiaoxiaoto/vllm-omni-metal/main/install.sh | bash
source ~/.venv-vllm-omni-metal/bin/activate
```

无本地 checkout 时安装本插件 Git main；它与 Homebrew HEAD 一样是源码入口，不再使用要求 0.30 的旧插件 archive。native wheels 和 Omni commit 仍固定。没有 uv 时采用固定 uv bootstrap。远端 main 必须同时包含本次依赖与代码变更，当前未验证此远端入口。

## 可选：本地 Torch-stream attention

固定的 0.31 wheel 不含 #858 的新 MPS launcher。默认路径无需编译并保留 MLX bridge；若已有 Apple Command Line Tools，可在**选定独立环境**内运行一次：

```bash
.venv-vllm-omni-metal/bin/python -m vllm_omni_metal.mps_attention
# curl 环境或 Homebrew：使用各自环境的 python -m 同一模块，不混用基础 vllm 环境。
```

这是本地显式构建，不升级依赖、不安装工具、不修改上游 package 或当前服务。产物在用户私有 `~/.cache/vllm-omni-metal/mps/`，绑定源码、Torch 环境与 Python ABI；构建后新 worker 的支持域 FP32/FP16/BF16 attention 使用 Torch stream，缺少匹配产物仍走 bridge；FP32 prefill 不使用仅支持 half/bfloat 的 tiled shader。引擎启动/请求处理不触发编译；环境或插件源码改变后需重建并在正常服务维护流程下启动新 worker。本次在开发环境及全新 `/tmp` 私有安装器环境完成真实 source install、CLI、pipcheck 和显式 host launcher 编译；Homebrew/curl 新入口仍未完整验收，不覆盖已有安装/服务；详见[执行合同](generic-backend.md#可选-torch-stream-attention)。

## 可选：vpipe 源算子 metallib

```bash
python -m vllm_omni_metal.vpipe_kernels
```

需要**已安装的 Apple Metal compiler**（`xcrun --find metal`）；它与仅能编译 host Objective-C++ 的 Command Line Tools 不等同。构建在本地私有 cache、离线、显式执行，不安装工具、不修改上游依赖。生产 worker 缺少匹配 metallib 时保留原路径，绝不在 startup/request 自动编译。本机缺 compiler，所以本次仅验证了隔离源 shader/数值/benchmark，未宣称 AOT 或真实 worker 已可用；单行小 SwiGLU 无稳定收益，仍不接入默认路径。来源和限制见[执行协议](generic-backend.md#vpipe-通用片段)。

## 使用与验证

```bash
.venv-vllm-omni-metal/bin/vllm-omni serve /path/to/local/checkpoint --omni \
  --deploy-config /path/to/metal-resource-profile.yaml \
  --host 127.0.0.1 --port 18094

VLLM_OMNI_METAL_INTEGRATION=1 .venv-vllm-omni-metal/bin/python -m pytest -q
```

原生 profile 要求 CUSTOM、单设备、同步调度、关闭 prefix cache，见 [通用后端协议](generic-backend.md)。安装器不启动模型，不替换 registry、模型 topology 或请求语义。

本轮从新的私有安装器环境完成真实联网解析/安装（223 packages）、`pip check`、版本与 MPS generator 校验、CLI `--help`、显式 attention host 编译和 **277 passed / 1 skipped**。开发环境另通过166个非集成测试及同一277/1完整回归。测试包括随机 Llama/Qwen3、tiny SD、原生 generation、FP32 paged attention 和隔离 vpipe shader；跳过的是需显式提供大权重的 Higgs 验收。vpipe AOT 仍因 Metal compiler 不可用阻塞，不能把隔离 shader 测试算成真实 worker 新路径成功。mock 安装回归另覆盖源码/curl 命令、安全边界、固定版本与 checksum 同步。

YuE2 的模型级 MPS 障碍尚未修复；更新依赖不等于音乐生成可用，见 [接入状态](yue2-readiness.md)。PyAV/OpenCV/系统 FFmpeg 存在重复 Objective-C 类警告，本次 CLI 与测试可通过，但媒体视频路径未额外验收。

卸载仅移除确认不再被服务使用的安装器环境或 `brew uninstall vllm-omni-metal`；不删除模型缓存、生成文件或仍被其他项目使用的 vllm-metal。
