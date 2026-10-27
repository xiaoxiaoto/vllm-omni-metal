# 安装与 vLLM / Metal / Omni 0.32 兼容栈

使用独立 Python 3.12 环境和官方 macOS arm64 wheel，不修改系统 Python、原有 `.venv` 或 vllm-metal 的 Homebrew 私有环境。

## 固定依赖

- 原生 Apple Silicon、macOS **15+**，不支持 Rosetta。
- vLLM **0.32.0+cpu**（官方 macOS arm64 `cp38-abi3` wheel）和 vllm-metal **0.32.0**（官方 macOS 15+ arm64 wheel）；SHA256 固定在 Formula 和安装器。
- Omni 官方 main commit **`d6db19b41f8c081506468859cd8f3910f77f1e5c`**，以 `VLLM_OMNI_VERSION_OVERRIDE=0.32.0.dev0+cpu` 构建；插件为 `0.32.0.dev0`。
- vLLM / Metal 0.32 要求 Transformers `>=5.16.1,<5.20.0`；Omni 0.31 的 `<5.15` 冲突已通过升级 Omni 与插件版本合同解决。Omni 构建含本地版本标签 `+cpu`，插件依赖以 `>=0.32.0.dev0` 接受该构建标记。
- vLLM 0.32.0 和 vllm-metal 0.32.0 按官方配套发布对齐；vllm-metal 发布说明要求该版本对应 vLLM 0.32.0。

本版本合同来自官方 main 开发快照，不将开发版号伪装成稳定发布版。安装器和 Formula 均固定同一 Omni commit、版本覆盖和 wheel SHA256；`pip check` 与新隔离解释器验证必须通过后才可使用。

0.32 的跨框架桥接位于 `vllm_metal.tensor_bridge`，不再使用
`vllm_metal.pytorch_backend.tensor_bridge`。本插件的 activation、diffusion、
分页 attention 和量化调用点统一按固定0.32合同构建，不提供0.31布局 shim。
这些 Python 模块由现有 setuptools、源码/curl 安装器及 Formula 自动分发；
仅修改 checkout 不会更新已有部署。独立 YuE2 可以通过
`VLLM_OMNI_METAL_WHEEL` 选择新构建的非 editable wheel，复用既有共享依赖，
而不切换共享 `current` 或升级其他模型；私有 ready 与 attention launcher
仍须在该 overlay 的 `-I` 解释器下重新校验。

## 显式磁盘后备投影安装验收

`execution.streaming` 是纯 Python 显式接口，现有 setuptools 自动发现、
源码／curl 安装器和 Formula 分发同一实现；两处安装校验已加入该模块。
没有新增依赖、native 源码、编译入口、环境变量或 entry point，
`pyproject.toml` 与许可证资源无需改变。接口及接入限制见
[磁盘后备 dense 投影](generic-backend.md#磁盘后备-dense-投影)。
安装插件不会自动启用 streaming，私有加载器仍须显式逐投影导出和写回。

本次使用临时 Python3.12 venv，复用已安装0.32共享环境的依赖路径；
只向临时环境安装测试／构建工具和非 editable 插件，没有改变共享环境或部署。
从排除 `.git`、build、dist、egg-info、环境与缓存的干净源码副本构建 wheel：

```bash
<python> -I -m pip wheel --no-deps --no-build-isolation -w <temporary-wheels> <clean-source>
<python> -I -m pip install --ignore-installed --no-deps <wheel>
<python> -I -m pip check
VLLM_OMNI_METAL_INTEGRATION=1 <python> -I -m pytest --import-mode=importlib -q \
  <checkout>/tests/test_streaming.py <checkout>/tests/test_install.py \
  <checkout>/tests/test_native_worker.py <checkout>/tests/test_plugin.py
```

wheel SHA256：`3a352f5012ce92113c494180b6762dd0a85b1b26f849d3f53738631f711365b7`。
47个随包源码／native／许可文件逐一匹配 wheel、checkout 与临时安装，
`-I` 实际执行两处完整安装 Python 校验片段通过；
安装后的双投影三 dtype MPS 流式流水线各重放三次，输出与 resident 完全相同。
上述安装与执行边界回归179项通过，新增流式专项99项通过；
Ruff check、变更 Python format check、Shell／Ruby 语法通过。
临时工具最初安装的 setuptools84 不满足 vLLM 约束，已仅在临时 venv
对齐78.1.1，最终 `pip check` 通过。

导入完整上游安装栈时仍输出 FFmpeg/PyAV/OpenCV 的重复 Objective-C 类警告；
本次未处理此独立依赖问题，也没有将无崩溃视为媒体路径已验证。
这不是全依赖新安装、物理 SSD 性能、超 RAM 整模或完整媒体生成验收。
未执行 Homebrew 重装、远端 curl/tap 发布、私有模型接入或服务迁移／重启。

## 内部 API 功能分包切换

本次将内部模块干净切换至 `vllm_omni_metal.execution`、`.native`、`.diffusion`、`.ops` 和 `.ane` 功能包；各包 `__init__.py` 保持惰性，不重新导出迁移模块，**不提供旧路径兼容 shim**。外部模型适配器、独立 overlay 和部署脚本必须同步更新直接导入、`import_module` 字符串、worker/backend 动态类路径及构建命令；例如通用算子从 `vllm_omni_metal.diffusion.ops` 导入，ANE API 从 `vllm_omni_metal.ane.runtime` 导入，离线构建入口为 `vllm_omni_metal.ane.build`。

顶层 `platform.py` 和 `vllm_omni_metal:register_platform` entry point 不变，依赖 pins 不变；所有 native 资源仍位于包根的 `_native/`，不移入功能包。源码安装和 curl 安装共用 `install.sh` 的新模块检查与显式构建路径，Formula 使用相同路径；校验覆盖五个功能包的全部21个迁移模块，并实际解析AR、generation、diffusion worker、diffusion runner及SDPA backend类路径，同时检查native源码与许可证。继续使用所选解释器的 `-I` 隔离模式，不加载模型或构建可选产物。

切换前停止使用目标环境的服务；从本次源码重新构建并安装非 editable wheel（editable 环境也须重新安装当前源码），让外部适配器一起迁移。源码指纹变化后重新构建隔离后端与所需的 MPS host launcher、通用 AOT metallib、ANE host/权重图缓存，并重新准备对应校准/profile；不要复用旧 wheel、ready 标记或把旧缓存当作新路径已可用的证据。不删除模型缓存或接管其他环境。

本次分包验证：`.venv-vllm-omni-metal/bin/python -m pytest -q -m 'not integration'` 为479项通过、207项未选择；相邻部署 `python -m unittest discover -s ../docker-compose/yue2-3b -p test_prepare_ane.py -v` 为11项通过。Ruff检查、`bash -n install.sh`、`ruby -c Formula/vllm-omni-metal.rb` 和相邻YuE2启动脚本语法检查通过。ANE缓存新增嵌套Python源码与native源码变化／恢复的指纹回归，避免分包后漏掉源码身份。

从不含build／egg-info的干净副本，以所选Python `-I -m pip wheel --no-deps --no-build-isolation` 构建wheel，在临时venv非editable安装；复用所选环境现有依赖，不是全依赖重新安装。检查全部39个随包文件无遗漏、顶层无旧模块、五个功能包惰性导入、所有模块真实安装路径、平台注册与动态worker/backend类解析、native资源定位和ANE缓存身份均通过；安装后的MPS SwiGLU／RMSNorm实际数值烟测，以及三个离线构建模块的 `-I -m ... --help` 均通过。未重新编译可选native产物、运行真实模型或做性能验收；未重装Homebrew、激活隔离部署、重启服务或发布远端curl/tap。历史模型／构建验收不能代替新布局的生产验证。

安装资源同步复验：`python -m pytest -q tests/test_install.py tests/test_deployment_prepare.py` 为69项通过，覆盖本地／管道安装、安全边界及独立部署准备器；Shell／Ruby语法、Ruff check与format check通过。从干净副本再次用所选Python `-I -m pip wheel --no-deps --no-build-isolation` 构建并在临时venv非editable安装（复用现有依赖），直接执行 `install.sh` 和Formula中实际的Python校验片段均通过；39个包文件完整、三个构建入口的隔离help通过。移除临时安装中的 `execution/shm_connector.py` 后，实际安装校验因模块缺失而失败，未误报成功。未执行联网全依赖安装、Homebrew安装或远端curl入口；未编译可选产物、迁移环境或重启服务。


## 通用 AOT 命名切换

通用离线构建模块为 `vllm_omni_metal.ops.aot_kernels`，elementwise 构建函数为 `build_elementwise_kernels`，diffusion 构建函数仍为 `build_diffusion_kernels`。安装 CLI 使用 `--with-aot-kernels`；不提供旧模块、函数或 CLI alias/shim。外部适配器、私有 overlay 和部署准备器须同步更新 import、动态模块字符串及构建命令，再从当前源码重新安装并显式构建。

源码和许可检查统一定位 `_native/elementwise.metal`、`_native/diffusion_rotary.metal` 与 `_native/THIRD_PARTY_NOTICES.md`；来源、版权、许可证及上游 URL 保留准确引用。新私有缓存 namespace 为 `aot/elementwise`、`aot/diffusion`：保留旧缓存，不删除、不复用，也不搬移或改名冒充新身份；必须在目标解释器下显式重建。算子基准入口为 `examples/benchmarks/aot_kernels.py`。

这是命名与缓存身份的干净切换，不新增算子、不改变数值／布局／精度／dispatch，也不声明性能收益。下方历史验收不能作为新模块或缓存身份的验证；本次迁移验收单独记录如下。未运行 `prepare_backend.py`、激活 ready/current 或部署、重启服务、修改 topology/model policy、发布远端 curl/tap。

### 本次命名迁移验收

在临时venv使用现有0.32依赖的独立快照（Omni源码`9842ebc17cf1d9c6b440ecb6a79039162f68a1dd`），
从干净源码副本构建并非editable安装插件wheel，SHA256为
`f85e5b35dfd49a4a6e0dcf78900850a8d61181cbac6864d7e8c8a3c1fc3fb341`。
46个包内源码／native资源／许可文件逐一匹配checkout、wheel和实际安装目录；
旧模块不可导入，旧安装选项被拒绝，所有变更模块均从本次临时安装加载。
install.sh和Formula的完整Python校验片段、`pip check`及通用CLI的`--help`实际通过。
这是复用依赖的隔离安装，不是全依赖重新安装或Homebrew/curl入口验收。

实际执行以下显式构建，随后在新进程确认三种dtype的elementwise/diffusion产物加载：

```bash
<python> -I -m vllm_omni_metal.ops.aot_kernels --diffusion
<python> -I -m vllm_omni_metal.ane.build --bridge
```

通用AOT、activation、normalization、rotary、diffusion、ANE及安装专项回归483项通过，
含真实MPS数值及更新后native host路径；不代表真实模型整任务通过。
完整非集成一次为1065通过、9跳过、7失败：其中1项仅锁定异常文案，已删除该文案断言，
保留缺产物拒绝且不JIT的行为检查，并在上述专项回归通过；
另6项为相邻部署准备器既有current保留／ready重用合同失败，未改写该准备器策略。
Ruff、变更文件格式、Shell/Ruby语法检查通过。

真实权重局部路径另保存于`outputs/operator-fusions-real-20261027-100610/`：
Higgs首层1024-token prefill与8次decode的旧／新P0结果逐值相同；
Qwen首步图像token的真实投影、时间调制、LayerNorm和QKV之后，4个seed均命中
`qk_norm_rotary:aot`并通过数值对照。这些是局部真实权重验证，不是完整TTS／图片生成；
完整Higgs双阶段生成实际启动尝试在加载模型前被资源门禁拒绝：当时可用11.48 GiB，
低于权重、系统余量与KV预算合计要求的13.17 GiB；记录见同目录`native-generation/`。
Qwen完整生成仍不满足32 GiB可用内存门槛；
未降低资源或精度要求，也未把没有匹配调用点的gated RMS AdaLN宣称为真实模型通过。

## 1. 当前源码：0.32 本地资源，完整安装尚未复验
本地安装器、Formula 和包元数据已对齐 vLLM / vllm-metal 0.32.0、Omni `0.32.0.dev0+cpu` 与插件 `0.32.0.dev0`。复用既有0.32依赖的独立 wheel 安装校验不等于全依赖新安装通过；下列完整安装流程尚未在本次重新执行，远端资源也未发布。

```bash
cd /path/to/vllm-omni-metal
./install.sh
source .venv-vllm-omni-metal/bin/activate
vllm-omni --help
python -I -m pip check
```

默认创建当前 checkout 的 `.venv-vllm-omni-metal`，editable 安装当前插件。也可指定新的独立路径：

```bash
./install.sh --venv "$HOME/.venv-vllm-omni-metal"
```

安装时可显式准备所选工具，选项可以组合：

```bash
./install.sh --venv /absolute/path/to/new-env --with-mps-attention
# 可选：--with-aot-kernels（已有 Metal compiler）、--with-ane（已有 clang++/Swift）
```

默认不构建。构建失败即退出，不打印安装成功；不安装工具、不下载权重、不转换模型或校准。安装验证及构建以所选 Python 的 `-I` 隔离模式执行，忽略调用方 cwd/PYTHONPATH；模块 `--help` 和非法参数只解析参数，不触发构建。

`--with-aot-kernels` 同时准备通用 elementwise 算子、diffusion interleaved RoPE以及P1
gated RMS AdaLN／QK norm+RoPE融合；手动命令为
`python -I -m vllm_omni_metal.ops.aot_kernels --diffusion`。
P0复用attention launcher构建入口，native源码变化会生成新缓存身份，须显式重建。
Python模块与更新后的`_native/diffusion_rotary.metal`、`mps_ops.mm`通过现有setuptools
规则随源码、wheel与Formula分发，依赖、entry point、package-data及许可证无需新增。
安装器源码/curl路径与Formula均执行两个新API的CPU数值合同，不触发Metal编译。
模型适配仍在独立部署的私有wheel中，不修改上游registry或自动激活部署。

安装器只接受新目录或它自己标记创建的环境，不接管任意已有 venv/Homebrew libexec，不使用 `--clear`。失败时保留环境和缓存供诊断、重试。依赖和插件在同一次 uv 解析中安装，保持 vllm-metal 声明的 mlx-lm Git URL 可见。

总部署预检可以设置 `VLLM_OMNI_COMMIT=<完整 40 位 SHA>`，让安装器从查询到的官方 Omni 主干 commit 构建；省略时仍使用上述固定快照。非法 commit 会在安装前拒绝。开发版公开版本与 CPU 构建标记保持明确，不伪装稳定发布。

`docker-compose/start.sh` 的模型启动统一调用 `prepare_backend.py`：查询官方最新核心标签与 Omni main，构建独立后端，并从本仓库干净副本构建非 editable wheel。准备器显式选择 `--with-mps-attention`，在选定环境的全新 `-I` 进程中验证 launcher 能加载；源码/native资源、安装脚本及准备器内容均进入指纹，校验通过且构建期间指纹未变才写 `.ready` 并激活。YuE2 私有 overlay 在自己的解释器重复准备与加载验证，不能借基础环境的缓存冒充可用；已有 ready 也会重新验证，失败先撤销 ready。部署模型使用同一版本组，模型私有扩展不进入通用插件、不修改 Homebrew。新官方核心超出兼容矩阵时明确拒绝，不能逐包盲升。

2026-10-16 通用 diffusion SDPA 的 Metal OOT 分派修复已通过上述共享准备器构建非 editable wheel 并完成 Qwen-Image-2.1-Turbo 真实文生图验收，详见[执行合同与本次命令](generic-backend.md#责任边界)。对应模块现为 `diffusion/attention.py`，由现有 setuptools 包发现与源码安装/Formula 自动包含，无须变更版本 pins 或 native 资源；源码指纹会触发新隔离后端。仅本地部署已安装，远端未推送，未运行 Homebrew/curl 安装。

本次diffusion GPU／分片ANE接入使用共享准备器构建非editable wheel。完整校准基于 `runtime/a46a54e37c719e65`、Omni main `c8ccb8aed59e1db273d9dce6ee91b4f09d30db18`、QwenImage overlay `779257e7887353fb`。其后直接部署时，联网预检更新至 Omni main `082eaf235cfc373322bc5f8a1a3398915446a82d`，当前运行时 `runtime/c076adaf5074c803`、QwenImage overlay `overlays/qwen-image-2.1-turbo/bf033dac54505e3e`；依赖校验、统一模型列表和真实512×512推理通过。依赖下载遇到TLS故障时使用 `UV_OFFLINE=true` 与已有兼容缓存，官方Git预检未跳过，未改核心pins或关闭TLS校验。

此前diffusion／ANE验收时生产AOT尚未构建：当时只有Command Line Tools，`xcrun --find metal`／`metallib`失败，Apple官方 `xcodebuild -downloadComponent metalToolchain` 因缺完整Xcode被拒绝。该历史结论不代表本次P0/P1的工具链状态；本次已找到Metal compiler并显式构建新产物，见下方验收记录。此前 Omni main `082eaf235cfc373322bc5f8a1a3398915446a82d` 的QwenImage真实512×512 GPU off/auto A/B输出一致，但auto慢约3.6%；第0层ANE协同整图scaled max 12.94%超过1%门禁，未发布profile。没有借本次算子验证改动服务GPU／ANE默认值；完整证据见[ANE记录](ane.md)。本次代码仍是本地工作区，不把它描述成已推送的远端入口。

更新一个已使用的环境前，先停止所有使用它的服务。本次安装使用新的独立环境，保留运行中的 Higgs 及旧 Homebrew 环境；没有自动迁移服务。

### 2026-10-27：P0/P1 非 editable 与真实 AOT 验收

本次开发`.venv-vllm-omni-metal`仍为0.31，不能据此宣称源码固定的0.32兼容栈通过。
最初使用现有0.32依赖的临时环境，随后该共享runtime在验证期间被替换；
最终在临时venv单独安装固定vLLM/Metal0.32、固定`d6db19b4` Omni源码的CPU wheel，
并复用既有Torch2.13等依赖。Omni从缓存源码的独立副本以版本覆盖重新构建，
避免uv已有0.31版本元数据冒充0.32；没有改动共享runtime或原开发环境。
最终Transformers5.19.0、MLX0.32.2，`python -I -m pip check`通过。

从不含build／egg-info的干净插件副本执行
`python -I -m pip wheel --no-deps --no-build-isolation`，在临时venv非editable安装。
wheel SHA256为`8d4aab8f15596546f07539addf69f4a01747da0eb35aa3f07ff1cbe653685c8c`。
46个包内源码／native资源／许可文件逐一匹配checkout、wheel和安装目录；
变更模块真实路径均位于临时site-packages，不借cwd或PYTHONPATH导入。
实际执行install.sh和Formula中的完整Python校验片段，均通过，含新增CPU融合数值合同。
源码/curl共用同一校验；Formula、依赖、entry point和package-data已逐项核对。

以下为历史实际命令对应的当前入口（历史使用迁移前模块名，非本次重跑）；生产不增加JIT：

```bash
<python> -I -m vllm_omni_metal.ops.mps_attention
<python> -I -m vllm_omni_metal.ops.aot_kernels --diffusion
VLLM_OMNI_METAL_INTEGRATION=1 <python> -m pytest -q \
  tests/test_native_attention.py tests/test_mps_attention.py tests/test_metal_attention.py \
  tests/test_diffusion_fusions.py tests/test_diffusion_ops.py \
  tests/test_native_runner.py::test_two_native_language_architectures_on_mps
<python> -m pytest -q tests/test_install.py
<python> -m pytest -q -m 'not integration'
```

真实AOT独立烟测确认三种dtype的两个`:aot`路径实际命中、rounded residual逐值一致、
GQA与batched频率数值对照，以及context1024的attention偏移输出直写／四次无中间同步
重放／非法alias不改KV。FP32归约误差和低精度末位差异保留原有精度门槛。
290项专项回归（含随机Llama/Qwen3多步生成）及42项安装回归通过。
全量非集成为1066通过、9跳过、6失败：失败均在相邻`prepare_backend.py`的
current保留／ready重用合同，不是新增算子数值失败；未改该部署准备器。
Ruff check、本次8个Python文件format check及Shell/Ruby语法检查通过。

性能采样受并行负载影响明显，不将单次算子计时写成稳定整模型收益。
完整`install.sh`入口另作尝试：离线缺wheel缓存，联网解析／下载在600秒超时，
因此不宣称全依赖新安装通过。上述非editable安装、实际校验和AOT执行均已完成，
不等于远端curl/tap已发布、Homebrew已重装或现有部署已迁移；未重启服务、
下载模型权重或修改模型GPU／ANE／量化默认值。

## 2. Homebrew：发布本次改动后

配套源码与 Formula 发布后可按下列流程安装；本次未重新安装 Homebrew，也未验证远端 tap/curl，不能将本地资源同步等同于已发布。

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
.venv-vllm-omni-metal/bin/python -I -m vllm_omni_metal.ops.mps_attention
# curl 环境或 Homebrew：使用各自环境的 python -I -m 同一模块，不混用基础 vllm 环境。
```

这是本地显式构建，不升级依赖、不安装工具、不修改上游 package 或当前服务。产物在用户私有 `~/.cache/vllm-omni-metal/mps/`，绑定源码、Torch 环境与 Python ABI；构建后新 worker 的支持域 FP32/FP16/BF16 attention 使用 Torch stream，缺少匹配产物仍走 bridge；FP32 prefill 不使用仅支持 half/bfloat 的 tiled shader。引擎启动/请求处理不触发编译；环境或插件源码改变后需重建并在正常服务维护流程下启动新 worker。本次在开发环境及全新 `/tmp` 私有安装器环境完成真实 source install、CLI、pipcheck 和显式 host launcher 编译；Homebrew/curl 新入口仍未完整验收，不覆盖已有安装/服务；详见[执行合同](generic-backend.md#可选-torch-stream-attention)。

## 可选：GPU + ANE CoreML host 工具

普通安装只安装代码和 Apache-2.0 emitter/bridge 源码；不需要 CoreML Python wheel，也不编译工具。用户可在安装时显式选择 `./install.sh --with-ane`，或之后在所选隔离环境运行：

```bash
.venv-vllm-omni-metal/bin/python -I -m vllm_omni_metal.ane.build --bridge
```

需要现有 Apple Command Line Tools（clang++/Swift）及 macOS 15+；没有时明确失败，不下载工具。Homebrew 命令见 Formula caveats。该步骤只构建 host 工具，不下载/转换模型，也不代表 ANE 可用；模型适配器还须显式准备权重图并以完整 pipeline 延迟批准 profile。默认 auto 无匹配 profile 时保留 GPU；当前真实 YuE2 图 relative L2 5.499% 超过1%门槛被拒绝，不得放宽门槛。FP16近似精度与生命周期见 [GPU + ANE 协议](ane.md)。没有重启当前服务或发布远端入口。

## 可选：通用 elementwise metallib

```bash
python -I -m vllm_omni_metal.ops.aot_kernels
```

需要**已安装的 Apple Metal compiler**（`xcrun --find metal`）；它与仅能编译 host Objective-C++ 的 Command Line Tools 不等同。也可安装时显式选择 `--with-aot-kernels`，同时准备 diffusion AOT。构建在本地私有 cache、离线执行，不安装工具、不修改上游依赖。生产 worker 缺少匹配 metallib 时保留原路径，绝不在 startup/request 自动编译。早期缺 compiler 时，packed Q/K RoPE 等源码仅完成隔离 shader/OOT 数值与计时；后续 P0/P1 生产 AOT 验收见上方记录，不能据此跳过命名切换后的显式重建与验证。单行小 SwiGLU 无稳定收益，仍不接入默认路径。来源和限制见[执行协议](generic-backend.md#成熟方案续移植packed-qk-rope)。

## 显式选择性 W8A16

普通安装随包提供 `quantization/linear.py`、`engine.py`、`capture.py`、`adapter.py`
和 `runtime.py`。源码与curl共用安装器烟测，Formula test使用同一组模块导入；
共享后端和四个模型私有overlay的ready检查同步导入新API，均使用选定Python `-I`。
安装器还在CPU小Linear上构建并验证schema2 W4/W8/BF16描述符；这是安装合同，
不是GPU量化、完整模型推理或质量批准，不下载权重、不编译量化kernel。

已有 `[tool.setuptools.packages.find] include=["vllm_omni_metal*"]` 自动包含全部
量化Python子模块，无须新增package-data、依赖或entry point；现有native资源规则不变。
共享/私有准备器递归指纹包含这些源码，Higgs、SeedVR2、Qwen-Image与YuE2的
离线校准器身份资源继续随各自Omni wheel分发并按源码SHA和RECORD归属校验；
离线CLI仍从部署源码目录执行，不能在安装后手工复制到site-packages。

全局 `VLLM_OMNI_METAL_QUANTIZATION=off|dynamic` 默认off；dynamic要求
`VLLM_OMNI_METAL_QUANTIZATION_PROFILE=/absolute/approved.json`。共享
`load_profile(path, *, scope)` 只接受当前用户owned0600、非符号链接、4MiB以内普通文件，
支持单个schema2或 `{"profiles": {实际完整执行scope: profile}}`。多阶段文件按实际
scope选取，不按模型名；缺失scope、未批准、旧实现/环境/权重均报错，不静默fallback。
四个私有模型的mode/profile参数分别覆盖全局参数，显式off可覆盖全局dynamic：
`YUE2_METAL_`、`HIGGS_METAL_`、`SEEDVR2_METAL_`、`QWEN_IMAGE_METAL_`
前缀后接 `QUANTIZATION` / `QUANTIZATION_PROFILE`。旧W8-only运行入口不再使用，
历史策略、profile、权重和媒体仍保留，不能自动继承为动态批准。

执行入口已接原生runner、Torch diffusion和MLX自管理diffusion；普通nn.Linear、
TP1未量化vLLM完整dense投影、Diffusers未注册components使用通用适配器。
MLX自管理对象必须提供显式scope/apply hook，不支持的融合/分片/共享存储合同明确错误。
FP16/BF16搜索和执行保留源dtype，FP32只支持identity；不强制转BF16，允许全部identity，
但不能声称量化收益。私有模型仍自行负责真实执行捕获与整任务质量批准。
完整接口、managed元数据和hook合同见[通用DynamicEngine](generic-backend.md#通用-dynamicengine-加载合同)。
新的单线性投影 ANE 与联合量化安装资源、实测边界见下节；没有真实四模型
质量/速度的共同批准，不自动启用、迁移部署、重启服务或发布远端。之后的旧记录均为独立历史证据。

### 单投影 ANE 与联合量化安装资源

普通 wheel 自动包含 `ane/projection.py`、更新后的 C++ emitter 和 Swift bridge；
现有 package-data 已包含全部 native 源码与许可证，不打包本机编译产物、权重或 profile。
源码/curl 的 `install.sh` 和 Formula 烟测均导入新 API；共享准备器核验
`DenseProjection`／`prepare_projection`，三个私有准备器从当前干净后端源码构建 wheel。
模型私有联合适配器及校准器仍归各自正常 Omni wheel 所有，按源码 SHA、RECORD、
来源指纹与 ready 校验，不能向已安装上游复制补丁。

当前已在复用既有0.32依赖的临时 venv 中非 editable 安装干净 wheel，并以所选
Python `-I` 执行显式 ANE host 构建及真实 BF16 biased 线性投影：
256行 ANE、257行 GPU，relative L2 0.10183%、scaled max 0.48309%，没有发布批准。
最终干净wheel复验：完整非集成906项通过（5 skipped、334 deselected），小型真实
MPS／ANE及三个私有适配器回归131项通过。Ruff check／format check、安装器及三个
启动脚本的独立 `bash -n`、Formula `ruby -c` 均通过。三个正常私有准备器的构建、
非 editable 安装、来源/资源/RECORD/许可证及依赖校验通过；依赖由显式 `.pth` 复用，
这不是全依赖重新安装或服务迁移。
旧开发 `.venv-vllm-omni-metal` 仍为0.31，不用于0.32桥接验收，也没有被本次升级。

完整模型没有联合批准：Higgs真实TTS四层down均局部选W8，但候选656次量化调用、
ANE chunks=0；短输入不填充，候选与原精度波形长度也不同。独立核验6个24kHz单声道
WAV与7个有限值FP32 NPY，原精度双语同seed逐位重复，零ANE候选与quant-only逐位相同。
Qwen真实联合CLI在启动前available27.31GiB时被32GiB门槛拒绝；后续资源复核仅19.24GiB。
Seed联合CLI在模型worker创建前因缺7B索引拒绝，当前仅有3B缓存且未提供真实输入。
因此三者均不设联合默认，不下载权重、不制造输入、不停止其它服务。
本轮证据：`outputs/joint-ane-dynamic-20261022-222638/`；保留既有依赖警告，
不声称完整安装、真实图片/超分、听感/ASR或速度收益通过，未发布远端资源。

### 2026-10-23：Qwen真实启动接口修复

实际启动先暴露上游0.32 `initialize_lora` 加载回调签名变化；共享diffusion runner
按原合同接收并传递给Torch加载路径，自管理MLX不执行Torch回调，未增加LoRA能力。
`diffusion/worker.py` 由现有包发现/安装器/Formula校验分发，入口和依赖无新增；
所有私有准备器递归源码指纹包含此次变更，旧ready不能替代当前源码校验。

从干净后端副本构建非editable wheel，在临时独立环境复用既有0.32依赖，
真实MPS小流水线和MLX生命周期回归6项通过。正常Qwen准备器成功构建并校验
独立 `77b751e47c56c325`，重跑已越过原签名错误进入真实权重加载。
随后Qwen的NPU专用SDPA覆写错误触发Metal继承校验；私有补丁仅在NPU注册该覆写，
不绕过验证，正常重建 `feb09f3c459a18e5` 并通过真实MPS attention数值回归。
上述环境只显式 `.pth` 复用既有基础依赖，不是全依赖重装；共享current与服务未切换。
ANE host工具已在选定私有Python `-I` 下显式构建；完整联合结果须另看本轮真实记录，
不能将安装、加载权重或小型回归视为图片质量/速度批准。

最终完整非集成958项通过（5 skipped、336 deselected），Ruff check/format及
Shell/Ruby语法通过。以官方0.32基础依赖实际执行安装器/Formula Python校验段，
并验证新backend wheel完整源码/资源/许可证与RECORD，均通过。
完整Qwen重跑在修复后的环境被启动资源门槛拒绝，最终复核available24.70GiB；
未生成图片或发布联合批准。详细证据为
`outputs/qwen-joint-direct-20261023-100747-verification.json`及对应三个真实尝试目录。

### 2026-10-23：行数平衡预测分配复验（参考 vpipe）

新增 `ane/allocation.py` 随现有包发现规则分发，无额外依赖、entry point或native资源变化。
源码/curl共用安装器及Formula的实际Python校验清单均导入该模块。共享/私有准备器
递归源码指纹自然覆盖分配器与投影变更；已有ready、图缓存或profile不能当成本轮批准。
外部适配器须显式接入离线 `set_benchmark_chunks("auto")`，生产冻结profile语义不变。

从干净副本构建0.32 wheel，在临时独立venv非editable安装，显式 `.pth` 复用既有0.32
依赖；所选Python `-I` 构建ANE工具并实际运行1025×1024 BF16 biased投影。
8次自适应审计中真实分配经历2→3→2→1块，随后2次冻结执行；总计19个ANE块、
4864行ANE及5386行GPU。实际各split完整投影relative L2最高0.12357%、
scaled max最高0.45663%，未发布profile，不据此声称模型提速。

预测边界和真实NE冻结回归68项通过；完整非集成958项通过（5 skipped、335 deselected）。
实际执行安装器和Formula内嵌Python校验段、完整源码/资源/许可证/RECORD检查均通过；
Ruff check/format、`bash -n install.sh`、`ruby -c Formula/vllm-omni-metal.rb`通过。
未执行完整Homebrew安装或远端curl，不修改既有环境、服务或部署默认。
证据：`outputs/higgs-joint-smallrows-20261023-091229/allocation-verification.json`。
分配预测不解除256行分块，不代表Higgs短输入联合执行或完整音频质量已通过；
本轮按用户新优先级只实施通用分配策略，未继续修改Higgs小图/选层流程。

### 2026-10-22：DynamicEngine 子包与动态部署切换

量化已独立为 `vllm_omni_metal.quantization` 子包，普通 wheel 同时分发惰性
`__init__.py`、`linear.py`、`engine.py`、`capture.py`、`adapter.py`和`runtime.py`。旧 `ops.quantization` 文件／导入路径移除，
所有本仓库及相邻模型消费者已迁移，不提供兼容别名；依赖 pins、entry point、
native 源码／许可证及 setuptools 包发现规则无需改变。

YuE2 私有 Omni wheel 正常分发 `yue2_dynamic_quantization.py` 生产适配器和
`prepare_dynamic_quantization.py` 的只读源码身份资源。离线准备器仍从部署源码目录执行，
复用同目录侧车模块；不把这个只读资源当作独立安装的CLI。指纹／ready同时核对两份
源码的SHA、安装归属和Omni RECORD，源码变更需重新准备，不继承旧manual策略。
`YUE2_METAL_QUANTIZATION` 只接受 `off|dynamic`，默认off；
动态加载使用 `YUE2_METAL_QUANTIZATION_PROFILE` 指向匹配环境的0600已批准profile。
本次没有激活部署或迁移已有服务／用户策略。

本次实际验证：

- 显式MPS命令：
  `VLLM_OMNI_METAL_INTEGRATION=1 .venv-vllm-omni-metal/bin/python -m pytest -q tests/test_dynamic_quantization.py tests/test_yue2_dynamic_engine.py tests/test_yue2_dynamic_preparation.py tests/test_deployment_prepare.py`，
  111项通过，包含真实W4/W8搜索、BF16保留和私有hook捕获／应用／释放；不是实际YuE2质量证据。
- `.venv-vllm-omni-metal/bin/python -m pytest -q -m 'not integration'`：
  724项通过、2项显式MPS未启用而跳过、245项未选择，15条Torch／上游buffer警告未隐藏。
- 从干净后端副本以所选Python `-I -m pip wheel --no-deps --no-build-isolation`
  构建，全部41个包文件逐字节一致，旧模块未残留。
- 从本地已有Omni源码缓存（目录标识 `ebe2c1827`）的干净副本，
  实际检查／应用当前patch并构建私有wheel；两份私有源码及patched模型源码逐字节一致。
  缺失的构建依赖仅在临时build-env中离线安装，遵守上游setuptools<81约束，不修改运行依赖。
- 临时非editable环境实际执行敏感度搜索，选择W4后完成MPS forward、未批准拒绝和源权重释放；
  指定小投影源8192字节、实际持久状态2304字节、4行成功命中，不代表模型内存或速度。
  安装器／Formula完整Python片段、两份真实ready源码／归属／RECORD校验及`pip check`均通过。
  依赖通过显式`.pth`复用现有环境，非全依赖重装；未运行完整联网安装器或实际Homebrew安装。
  同一非editable环境以 `python -I check_metal.py` 实跑seed／合批隔离、同步读回、
  终止清理、随机tiny NAR与FP32 MPS VAE，通过；随机小权重不算实际歌曲验收。
- 后端wheel SHA256：
  `6285d44fd0694a1e8db579e844803cd8cd526aedf7d47d8947c087f6904015ab`；
  私有Omni wheel SHA256：
  `6ecd6e17e810fa943684c46f41f2e92a928fe6b782d667d562e6d29bf195ebc6`。

随后用上述非editable环境实际运行新真实YuE2准备入口，启动available9.823GiB，
小于保留的16GiB门槛，模型进程未创建。没有本次真实歌曲、组合延迟／音频误差或批准profile；
证据保存在`outputs/yue2-dynamic-20261022/{preflight,verification}.json`。
安装导入仍观察到PyAV／系统ffmpeg／OpenCV重复Objective-C类告警及缺失可选
Triton／DeepSelect／FlashInfer提示，校验正常退出；未通过改依赖或抑制提示规避。

### 2026-10-22：显式混合精度验收

- `VLLM_OMNI_METAL_INTEGRATION=1 .venv-vllm-omni-metal/bin/python -m pytest -q tests/test_quantization.py`：
  189 项通过，含真实 MPS W4／W8 独立 affine 数值参考和混合 BF16 pipeline。
- `.venv-vllm-omni-metal/bin/python -m pytest -q -m 'not integration'`：
  628 项通过、243 项未选择；覆盖安装／部署和四个私有 W8-only consumer 的拒绝边界。
  观察到 15 条 Torch／上游 buffer 警告，未修改或隐藏它们。
- 两处事务测试加上严格参数数量比较后，使用同一显式 MPS 命令加
  `-k 'apply_failure_leaves or construct_failure_after_first'` 复验，2 项通过。
- 在临时干净副本用所选 Python `-I -m pip wheel --no-deps --no-build-isolation`
  构建 wheel，逐字节核对全部 39 个包文件；临时 venv 非 editable 安装后，用
  `python -I` 实际执行 FP16 W4/W8 与 BF16 W4/W8/BF16 三层算路，独立 CPU 解包参考、
  原 BF16 模块／参数身份、量化源权重释放、重复结果与调用计数均通过。
  两个量化投影的持久状态合计 27136 字节，源权重／bias 合计 66048 字节；
  不代表全模型或进程内存节省，也不是速度基准。
- 在同一临时非 editable 环境，实际执行安装器全部隔离 Python 校验片段与 Formula
  的 Python 校验片段，均通过；`python -I -m pip check` 无依赖冲突。
  依赖通过显式 `.pth` 复用已安装环境，未执行其 editable hooks，不是全依赖干净重装。
  wheel SHA256：`01d64f760c8e42d0a7fba81121b5f973032b1051a8a8f2e294c624d4aaa6953a`。
- 安装片段导入上游时观察到 PyAV／系统 ffmpeg／OpenCV 的重复 Objective-C 类告警，
  以及缺失可选 Triton／DeepSelect／FlashInfer 的提示；校验进程均正常退出，未改依赖处理。

本次没有真实大模型混合精度质量／速度结果，没有构建或激活私有 overlay，
未运行完整联网安装器或实际 `brew install/test`，未发布远端 curl／tap 或重启服务。
四个私有 W8-only consumer 只补精度门禁，不将其历史 W8 批准扩展至 W4／BF16；
YuE2 历史手动候选须重新采集 schema 2 及当前实现证据，不能自动继承授权。

静态检查：`.venv-vllm-omni-metal/bin/ruff check` 与 `ruff format --check` 覆盖本仓库及
五个修改的私有 consumer Python 文件，通过；`bash -n install.sh` 与
`ruby -c Formula/vllm-omni-metal.rb` 通过。

### 历史单投影 W8 验收

2026-10-17历史验收：通用量化CPU／显式MPS回归81项通过；全部非集成392项通过，
旧YuE2准备器11项通过。执行命令为
`VLLM_OMNI_METAL_INTEGRATION=1 .venv-vllm-omni-metal/bin/python -m pytest -q tests/test_quantization.py`、
`.venv-vllm-omni-metal/bin/python -m pytest -q -m 'not integration'`，
以及部署目录的 `python -m unittest -v test_prepare_ane`；Ruff check／format check、
`bash -n install.sh` 和 `ruby -c Formula/vllm-omni-metal.rb` 通过。

用所选Python `-I` 执行一次性安装烟测器，从不含build／egg-info的干净副本以
`python -I -m pip wheel --no-deps --no-build-isolation` 构建wheel，安装到临时非editable
venv；依赖仅通过显式`.pth`复用开发环境，没有重新解析完整依赖。污染PYTHONPATH下
用临时Python `-I` 核对全部34个包文件逐字节一致，并实际执行BF16量化Linear：
16行成功命中、相对原精度L2为0.583820%、持久状态8832／源16512字节，`pip check`通过。
wheel SHA256为 `4528b2ff678a198a051c35aa721faa52d8831c5c95bdb24dbb04d82de2c91e19`，
原始报告位于 `outputs/yue2-w8-install-report.json`。这不是完整安装器依赖重装或实际
`brew install/test`；真实YuE2另用现有私有overlay与当前源码完成生成，结果有损且未证明
加速，未批准量化profile。未发布远端curl／tap、重装Homebrew、切换部署或重启服务。

### 随后授权的 YuE2 手动部署

用户随后明确要求应用有损候选。只在 YuE2 私有 overlay 安装当前通用后端 wheel
（SHA256 `a04510b6e943a4132203c352285fb368c5aedd841d43fac27dcbf17dae7800b5`），
不修改共享环境或 `current`。私有 Omni wheel 通过正常 setuptools `py-modules` 打包
生产适配器，SHA256 `18be10dc9939b70c70c96f592f59a18a988e5f858236173f33de4abb211d01f0`。
基础依赖复用现有冻结环境；Omni沿用其commit，不在线升级核心版本。所有安装／校验
使用所选Python `-I`，ready绑定私有源码、backend wheel与安装归属。

本次实际构建并安装到非editable overlay `9c8b39c6da1f48c2`，核对全部34个后端包文件
与wheel逐字节一致，并在污染PYTHONPATH下执行真实MPS量化Linear／加载attention launcher。
全部非集成422项通过、部署针对性58项通过、Ruff和启动器shell语法通过；随后实际启动
8096并完成两次真实HTTP歌曲片段生成及量化forward命中验证，服务保持运行。策略明确
`manual_override=true, approved=false`，不伪装为自动批准或无损加速。

新增部署模块、policy准备器、wheel指纹／ready和启动说明均保留在独立部署，不进入
通用模型registry或默认安装。证据在 `outputs/yue2-manual-deployment-20261017/`；
这不是全依赖重新安装、Homebrew重装或远端发布。未修改／停止其他模型，未下载权重。
完整歌曲质量、并发、长热负载与独立冷启动峰值仍未验收。

用户当时反馈真实听感很差并要求还原：此前启动器默认量化off，8096恢复原精度，
资源配置恢复量化前备份，ANE关闭；实验overlay／代码／policy保留但未启用，不删除
模型或生成媒体。原精度HTTP生成及与离线原精度参考的数值对照通过，部署边界回归30项
通过，证据为 `outputs/yue2-restored-float-20261017/`。以上手动部署记录是历史验收，
不能作为当前启用状态或听感质量验收；vpipe实际AR量化路线与本次NAR实验的区别
见[执行合同中的纠正](generic-backend.md#撤销与-vpipe-yue2-路径纠正)。

2026-10-19用户重新明确授权后，本地YuE2启动器默认与持久plist均开启
`manual-w8a16`，继续复用overlay `9c8b39c6da1f48c2`、既有私有策略及原资源配置，
ANE保持off。实际1024帧HTTP请求成功并命中量化forward，PCM与同日本次临时W8对照
一致，服务保持运行；证据在 `outputs/yue2-w8-enabled-20261019/`。
仅NAR第0层down量化，不是通用安装默认开启或其它模型支持声明。
本次更改为独立启动器／plist和说明，不涉及通用包或安装器资源；未重建wheel、
完整重装依赖、重装Homebrew或发布远端。质量边界见
[本次记录](generic-backend.md#2026-10-19重新授权开启-yue2-单投影-w8a16)。

## 2026-10-22：通用 DynamicEngine 接入验收

本次执行结果：

- 全部非集成回归：`864 passed, 5 skipped, 331 deselected`。
- 显式 MPS 专项：`162 passed, 416 deselected`，覆盖 FP16/BF16 W4/W8、
  TP1 vLLM 完整投影的 Tensor/tuple、skip/return-bias 和浮点引用释放；
  同次验证原生随机 Llama/Qwen3、tiny Diffusers 原有生成与生命周期。
- 独立运行真实 MPS capture → search → 未批准拒绝 → 微型完整输出对照 → profile
  加载 → forward；另验证原生 MLX 微型 facade 的 scope/apply hook 与量化 forward。
  这些数值烟测不是任何真实大模型的任务质量批准。
- Ruff check／format、Formula Ruby 语法及六个脚本逐个 `bash -n` 通过。

使用选定 Python `-I`，从不含旧 build／egg-info 的干净副本构建通用 wheel，
SHA256 为 `f6fcb1306396461a2618f48cc94b997ffb784a9d23bc70ac072362883691a0b9`。
在五个临时非 editable 环境安装通用 wheel，其中四个另安装重新构建的私有 Omni
wheel；全部44个后端文件及10个私有适配／校准资源核对源码 SHA 和 RECORD hash/size。
五个环境均实际执行上述 FP16/BF16 MPS 算路；私有 off 覆盖全局 dynamic、
Higgs 原精度非连续输入和 Qwen 量化投影也实际执行通过。
安装器和 Formula 的 Python 校验主体、三个可选工具 CLI help、私有 Omni 服务 CLI
help 及临时后端环境 `python -I -m pip check` 通过。

典型命令（临时路径以本次报告为准）：

```bash
.venv-vllm-omni-metal/bin/python -m pytest -q -m 'not integration'
VLLM_OMNI_METAL_INTEGRATION=1 .venv-vllm-omni-metal/bin/python -m pytest -q \
  -m integration tests/test_quantization.py tests/test_dynamic_quantization.py \
  tests/test_quantization_capture.py tests/test_quantization_adapter.py \
  tests/test_quantization_runtime.py tests/test_higgs_quantization.py \
  tests/test_seedvr2_quantization.py tests/test_qwen_image_quantization.py \
  tests/test_yue2_dynamic_engine.py tests/test_yue2_dynamic_preparation.py \
  tests/test_native_runner.py tests/test_runtime.py tests/test_integration.py
/temporary/buildenv/bin/python -I -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir /temporary/wheels /temporary/clean-source
```

依赖通过显式 `.pth` 复用开发环境，不是完整依赖重装或实际 `brew install/test`。
YuE2／Higgs 私有轮基于官方 `2ec5f265c69520cdd5e5f0a19db485d38885843e`；
SeedVR2 保持其固定 `8201f614b4c213494e276f5eecd099d2b79b0090`；
Qwen 使用既有补丁匹配的缓存 `ebe2c18271db4fdd12221fbb4def6061e14a9379`。
Qwen 补丁在开发基线 `2ec5f265` 的检查失败，未自动迁移或升级该基线；详见其独立说明。

本次 YuE2 真实生成前 available 为9.8457 GiB，未达到原16 GiB启动门槛，故未启动生成、
未批准 profile；另外三个真实模型整任务量化 A/B 也未执行。证据为
`outputs/dynamic-all-models-20261022/`。未改变已有安装、current/ready、服务或用户资产，
未发布远端 curl/tap；本地资源同步不代表部署启用。

## 使用与验证

```bash
.venv-vllm-omni-metal/bin/vllm-omni serve /path/to/local/checkpoint --omni \
  --deploy-config /path/to/metal-resource-profile.yaml \
  --host 127.0.0.1 --port 18094

VLLM_OMNI_METAL_INTEGRATION=1 .venv-vllm-omni-metal/bin/python -m pytest -q
```

原生 profile 要求 CUSTOM、单设备、同步调度、关闭 prefix cache，见 [通用后端协议](generic-backend.md)。安装器不启动模型，不替换 registry、模型 topology 或请求语义。

**既有完整后端验收（历史记录，非本次重跑）：**

本轮在全新私有安装器环境实际解析/安装223个包，验证固定版本、MPS generator、随包10项 native资源，并显式准备 attention host launcher；额外以污染的 cwd/PYTHONPATH 重跑，隔离验证仍通过。随后实际调用部署准备器的干净副本 wheel 安装路径：非 editable 导入、`pip check`、三个可选模块及服务CLI help通过；禁用MLX、请求期编译和显式同步后，FP32/FP16/BF16六个序列、每序列四步 attention 正确执行，记录在 `outputs/metal-install-current/report.json`。最终非集成回归 **259 passed / 143 deselected**，旋转／通用 elementwise 针对性回归（含实际MPS）**90 passed**，Ruff、格式、shell和Ruby语法通过。安装mock/CPU准备器回归不冒充真实安装；此前277/1是历史集成结果，不作为本轮完整重跑结论。未实际运行会停服务的总部署激活流程，未发布远端curl或重装Homebrew。

**本次安装资源同步验收：** 从干净副本重新构建 wheel，并在临时非 editable 环境安装；依赖通过显式 `.pth` 复用所选开发环境，没有重新解析完整依赖。安装后的全部30个包文件（含10项 native资源）与当前源码逐字节一致。使用仅替代 Homebrew 声明 DSL 的临时执行器，实际运行 Formula 的测试块：cwd/PYTHONPATH 污染场景修复前失败，改用所选 Python `-I` 后通过，覆盖版本、MPS generator、资源及四个 CLI help；这不是实际 `brew install/test`。安装/部署回归54项通过，Ruff、格式与shell/Ruby语法通过。持续同步指令已写入 `AGENTS.md`，临时环境与执行器已清理；未发布、切换服务或重跑模型生成。

未补丁 Omni 的 YuE2 模型级 MPS 障碍仍需独立适配；私有 `yue2mps2` overlay 已有真实512帧歌曲与 attention 对照，但不代表普通安装开箱可用，见 [接入状态](yue2-readiness.md)。PyAV/OpenCV/系统 FFmpeg 存在重复 Objective-C 类警告，本轮安装/CLI未因此失败，媒体视频路径未额外验收。

### 2026-10-23：原生扩散MPS随机数与HTTP验收

扩散runner不再把seed-only请求强制成CPU generator：沿用上游runner设备，
原生MPS张量使用MPS RNG；显式CPU噪声路径仍可使用CPU generator，已提供的
generator状态不被重置，MLX合同不变。源码/curl安装与Formula自动打包同一
`diffusion/worker.py`；依赖、入口和native资源无需变更，现有安装检查已要求
PyTorch 2.13与真实MPS generator。默认随机设备改变后须重新建立seeded基线与批准。

本次从干净源码清单构建后端wheel，临时非editable环境通过显式`.pth`复用
选定0.32基础依赖，没有重装完整依赖；实际执行当前`install.sh`校验Python块
及Formula测试Python块均通过。不是实际Homebrew重装，也未运行会切服务的
共享准备器。全量非集成回归959项通过、6项跳过、337项排除。

SeedVR2正常私有准备器已构建`b923e9a3bd43ea71`，核对完整RECORD、资源、许可证
和依赖后，用私有CLI在独立18100端口通过默认warmup；图片与5帧派生视频均完成
原生HTTP生成、同seed像素重复验证及服务关闭。证据在
`outputs/seedvr2-http-full-20261023/`，这不等于ANE＋DynamicEngine已获批。
未发布远端curl/tap，未重装Homebrew、切共享current或重启既有服务。

### 有界落盘捕获与流式DynamicEngine

`DiskCaptures`使用现有包内模块及标准库文件IO，不新增依赖、入口、native资源
或模型白名单；`pyproject.toml`的包发现规则保持不变。安装器源码/curl校验和
Formula测试块均显式导入`CaptureSession`与`DiskCaptures`，私有SeedVR2 ready
校验也验证新接口，递归源码指纹与完整RECORD覆盖实际实现。

本次临时非editable后端wheel已执行更新后的安装器/Formula Python检查、
`pip check`和全量非集成回归：1002通过、9跳过、338排除；流式捕获及真实MPS
内存/磁盘完整搜索等价等定向回归118项通过。临时环境通过明确`.pth`复用选定
0.32依赖，setuptools保持vLLM要求的78.1.1；不是完整依赖重装或真实brew安装。
默认内存捕获上限不变，磁盘路径逐样本保留完整激活；Engine实现hash变化会使
旧批准失效，必须重新校准，不提供历史批准迁移或请求期搜索。

流式SeedVR2环境`32d4415b8e04479a`的Omni构建遇到GitHub传输中断，本次未冒充
全新Omni源码构建：复核此前同一固定commit、同一7B补丁的私有纯Pythonwheel
整包SHA256与完整RECORD后，标准`wheel unpack/pack`更新当前随包校准器资源，
安装到本次尚无ready的独立环境。正常准备器随后完成ready复用校验、
完整资源/RECORD/许可证和`pip check`，ANE工具也已显式准备。
通用后端wheel仍从当前干净源码构建；共享安装和现有服务未修改。
完整离线来源记录在`outputs/seedvr2-stream-install-20261023/offline-wheel-source.json`，
不把该路径记作重新下载源码、完整依赖重装或远端发布。

卸载仅移除确认不再被服务使用的安装器环境或 `brew uninstall vllm-omni-metal`；不删除模型缓存、生成文件或仍被其他项目使用的 vllm-metal。
