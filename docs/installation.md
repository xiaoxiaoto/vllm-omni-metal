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

安装时可显式准备所选工具，选项可以组合：

```bash
./install.sh --venv /absolute/path/to/new-env --with-mps-attention
# 可选：--with-vpipe（已有 Metal compiler）、--with-ane（已有 clang++/Swift）
```

默认不构建。构建失败即退出，不打印安装成功；不安装工具、不下载权重、不转换模型或校准。安装验证及构建以所选 Python 的 `-I` 隔离模式执行，忽略调用方 cwd/PYTHONPATH；模块 `--help` 和非法参数只解析参数，不触发构建。

安装器只接受新目录或它自己标记创建的环境，不接管任意已有 venv/Homebrew libexec，不使用 `--clear`。失败时保留环境和缓存供诊断、重试。依赖和插件在同一次 uv 解析中安装，保持 vllm-metal 声明的 mlx-lm Git URL 可见。

总部署预检可以设置 `VLLM_OMNI_COMMIT=<完整 40 位 SHA>`，让安装器从查询到的官方 Omni 主干 commit 构建；省略时仍使用上述固定快照。非法 commit 会在安装前拒绝。开发版公开版本与 CPU 构建标记保持明确，不伪装稳定发布。

`docker-compose/start.sh` 的模型启动统一调用 `prepare_backend.py`：查询官方最新核心标签与 Omni main，构建独立后端，并从本仓库干净副本构建非 editable wheel。准备器显式选择 `--with-mps-attention`，在选定环境的全新 `-I` 进程中验证 launcher 能加载；源码/native资源、安装脚本及准备器内容均进入指纹，校验通过且构建期间指纹未变才写 `.ready` 并激活。YuE2 私有 overlay 在自己的解释器重复准备与加载验证，不能借基础环境的缓存冒充可用；已有 ready 也会重新验证，失败先撤销 ready。部署模型使用同一版本组，模型私有扩展不进入通用插件、不修改 Homebrew。新官方核心超出兼容矩阵时明确拒绝，不能逐包盲升。

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

Formula 的校验、可选工具准备及 CLI help 均通过该 libexec Python 的 `-I` 模式执行，并断言插件来自此环境内的非 editable 安装，避免调用方 cwd/PYTHONPATH 命中开发源码或其他安装。

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

## 可选：GPU + ANE CoreML host 工具

普通安装只安装代码和 Apache-2.0 emitter/bridge 源码；不需要 CoreML Python wheel，也不编译工具。用户可在安装时显式选择 `./install.sh --with-ane`，或之后在所选隔离环境运行：

```bash
.venv-vllm-omni-metal/bin/python -m vllm_omni_metal.ane_build --bridge
```

需要现有 Apple Command Line Tools（clang++/Swift）及 macOS 15+；没有时明确失败，不下载工具。Homebrew 命令见 Formula caveats。该步骤只构建 host 工具，不下载/转换模型，也不代表 ANE 可用；模型适配器还须显式准备权重图并以完整 pipeline 延迟批准 profile。默认 auto 无匹配 profile 时保留 GPU；当前真实 YuE2 图 relative L2 5.499% 超过1%门槛被拒绝，不得放宽门槛。FP16近似精度与生命周期见 [GPU + ANE 协议](ane.md)。没有重启当前服务或发布远端入口。

## 可选：vpipe 源算子 metallib

```bash
python -m vllm_omni_metal.vpipe_kernels
```

需要**已安装的 Apple Metal compiler**（`xcrun --find metal`）；它与仅能编译 host Objective-C++ 的 Command Line Tools 不等同。也可安装时显式选择 `--with-vpipe`。构建在本地私有 cache、离线执行，不安装工具、不修改上游依赖。生产 worker 缺少匹配 metallib 时保留原路径，绝不在 startup/request 自动编译。本机缺 compiler，因此 packed Q/K RoPE 等新增源码仅完成隔离 shader/OOT 数值与计时，不宣称 AOT 或真实 worker 已可用；单行小 SwiGLU 无稳定收益，仍不接入默认路径。来源和限制见[执行协议](generic-backend.md#成熟方案续移植packed-qk-rope)。

## 使用与验证

```bash
.venv-vllm-omni-metal/bin/vllm-omni serve /path/to/local/checkpoint --omni \
  --deploy-config /path/to/metal-resource-profile.yaml \
  --host 127.0.0.1 --port 18094

VLLM_OMNI_METAL_INTEGRATION=1 .venv-vllm-omni-metal/bin/python -m pytest -q
```

原生 profile 要求 CUSTOM、单设备、同步调度、关闭 prefix cache，见 [通用后端协议](generic-backend.md)。安装器不启动模型，不替换 registry、模型 topology 或请求语义。

**既有完整后端验收（历史记录，非本次重跑）：**

本轮在全新私有安装器环境实际解析/安装223个包，验证固定版本、MPS generator、随包10项 native资源，并显式准备 attention host launcher；额外以污染的 cwd/PYTHONPATH 重跑，隔离验证仍通过。随后实际调用部署准备器的干净副本 wheel 安装路径：非 editable 导入、`pip check`、三个可选模块及服务CLI help通过；禁用MLX、请求期编译和显式同步后，FP32/FP16/BF16六个序列、每序列四步 attention 正确执行，记录在 `outputs/metal-install-current/report.json`。最终非集成回归 **259 passed / 143 deselected**，旋转/vpipe针对性回归（含实际MPS）**90 passed**，Ruff、格式、shell和Ruby语法通过。安装mock/CPU准备器回归不冒充真实安装；此前277/1是历史集成结果，不作为本轮完整重跑结论。未实际运行会停服务的总部署激活流程，未发布远端curl或重装Homebrew。

**本次安装资源同步验收：** 从干净副本重新构建 wheel，并在临时非 editable 环境安装；依赖通过显式 `.pth` 复用所选开发环境，没有重新解析完整依赖。安装后的全部30个包文件（含10项 native资源）与当前源码逐字节一致。使用仅替代 Homebrew 声明 DSL 的临时执行器，实际运行 Formula 的测试块：cwd/PYTHONPATH 污染场景修复前失败，改用所选 Python `-I` 后通过，覆盖版本、MPS generator、资源及四个 CLI help；这不是实际 `brew install/test`。安装/部署回归54项通过，Ruff、格式与shell/Ruby语法通过。持续同步指令已写入 `AGENTS.md`，临时环境与执行器已清理；未发布、切换服务或重跑模型生成。

未补丁 Omni 的 YuE2 模型级 MPS 障碍仍需独立适配；私有 `yue2mps2` overlay 已有真实512帧歌曲与 attention 对照，但不代表普通安装开箱可用，见 [接入状态](yue2-readiness.md)。PyAV/OpenCV/系统 FFmpeg 存在重复 Objective-C 类警告，本轮安装/CLI未因此失败，媒体视频路径未额外验收。

卸载仅移除确认不再被服务使用的安装器环境或 `brew uninstall vllm-omni-metal`；不删除模型缓存、生成文件或仍被其他项目使用的 vllm-metal。
