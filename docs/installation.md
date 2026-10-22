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

## 内部 API 功能分包切换

本次将内部模块干净切换至 `vllm_omni_metal.execution`、`.native`、`.diffusion`、`.ops` 和 `.ane` 功能包；各包 `__init__.py` 保持惰性，不重新导出迁移模块，**不提供旧路径兼容 shim**。外部模型适配器、独立 overlay 和部署脚本必须同步更新直接导入、`import_module` 字符串、worker/backend 动态类路径及构建命令；例如通用算子从 `vllm_omni_metal.diffusion.ops` 导入，ANE API 从 `vllm_omni_metal.ane.runtime` 导入，离线构建入口为 `vllm_omni_metal.ane.build`。

顶层 `platform.py` 和 `vllm_omni_metal:register_platform` entry point 不变，依赖 pins 不变；所有 native 资源仍位于包根的 `_native/`，不移入功能包。源码安装和 curl 安装共用 `install.sh` 的新模块检查与显式构建路径，Formula 使用相同路径；校验覆盖五个功能包的全部21个迁移模块，并实际解析AR、generation、diffusion worker、diffusion runner及SDPA backend类路径，同时检查native源码与许可证。继续使用所选解释器的 `-I` 隔离模式，不加载模型或构建可选产物。

切换前停止使用目标环境的服务；从本次源码重新构建并安装非 editable wheel（editable 环境也须重新安装当前源码），让外部适配器一起迁移。源码指纹变化后重新构建隔离后端与所需的 MPS host launcher、vpipe metallib、ANE host/权重图缓存，并重新准备对应校准/profile；不要复用旧 wheel、ready 标记或把旧缓存当作新路径已可用的证据。不删除模型缓存或接管其他环境。

本次分包验证：`.venv-vllm-omni-metal/bin/python -m pytest -q -m 'not integration'` 为479项通过、207项未选择；相邻部署 `python -m unittest discover -s ../docker-compose/yue2-3b -p test_prepare_ane.py -v` 为11项通过。Ruff检查、`bash -n install.sh`、`ruby -c Formula/vllm-omni-metal.rb` 和相邻YuE2启动脚本语法检查通过。ANE缓存新增嵌套Python源码与native源码变化／恢复的指纹回归，避免分包后漏掉源码身份。

从不含build／egg-info的干净副本，以所选Python `-I -m pip wheel --no-deps --no-build-isolation` 构建wheel，在临时venv非editable安装；复用所选环境现有依赖，不是全依赖重新安装。检查全部39个随包文件无遗漏、顶层无旧模块、五个功能包惰性导入、所有模块真实安装路径、平台注册与动态worker/backend类解析、native资源定位和ANE缓存身份均通过；安装后的MPS SwiGLU／RMSNorm实际数值烟测，以及三个离线构建模块的 `-I -m ... --help` 均通过。未重新编译可选native产物、运行真实模型或做性能验收；未重装Homebrew、激活隔离部署、重启服务或发布远端curl/tap。历史模型／构建验收不能代替新布局的生产验证。

安装资源同步复验：`python -m pytest -q tests/test_install.py tests/test_deployment_prepare.py` 为69项通过，覆盖本地／管道安装、安全边界及独立部署准备器；Shell／Ruby语法、Ruff check与format check通过。从干净副本再次用所选Python `-I -m pip wheel --no-deps --no-build-isolation` 构建并在临时venv非editable安装（复用现有依赖），直接执行 `install.sh` 和Formula中实际的Python校验片段均通过；39个包文件完整、三个构建入口的隔离help通过。移除临时安装中的 `execution/shm_connector.py` 后，实际安装校验因模块缺失而失败，未误报成功。未执行联网全依赖安装、Homebrew安装或远端curl入口；未编译可选产物、迁移环境或重启服务。


## 1. 当前源码：已实际安装验证

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
# 可选：--with-vpipe（已有 Metal compiler）、--with-ane（已有 clang++/Swift）
```

默认不构建。构建失败即退出，不打印安装成功；不安装工具、不下载权重、不转换模型或校准。安装验证及构建以所选 Python 的 `-I` 隔离模式执行，忽略调用方 cwd/PYTHONPATH；模块 `--help` 和非法参数只解析参数，不触发构建。

`--with-vpipe` 现在同时准备原vpipe算子和通用diffusion interleaved RoPE；手动命令为 `python -I -m vllm_omni_metal.ops.vpipe_kernels --diffusion`。`diffusion/ops.py`、`TiledDenseSwiGLU` 和 `_native/diffusion_rotary.metal` 通过现有setuptools规则随源码、wheel与Formula分发，安装烟测检查其导入和资源存在，不修改上游模型registry。模型适配仍在独立部署的私有wheel中。

安装器只接受新目录或它自己标记创建的环境，不接管任意已有 venv/Homebrew libexec，不使用 `--clear`。失败时保留环境和缓存供诊断、重试。依赖和插件在同一次 uv 解析中安装，保持 vllm-metal 声明的 mlx-lm Git URL 可见。

总部署预检可以设置 `VLLM_OMNI_COMMIT=<完整 40 位 SHA>`，让安装器从查询到的官方 Omni 主干 commit 构建；省略时仍使用上述固定快照。非法 commit 会在安装前拒绝。开发版公开版本与 CPU 构建标记保持明确，不伪装稳定发布。

`docker-compose/start.sh` 的模型启动统一调用 `prepare_backend.py`：查询官方最新核心标签与 Omni main，构建独立后端，并从本仓库干净副本构建非 editable wheel。准备器显式选择 `--with-mps-attention`，在选定环境的全新 `-I` 进程中验证 launcher 能加载；源码/native资源、安装脚本及准备器内容均进入指纹，校验通过且构建期间指纹未变才写 `.ready` 并激活。YuE2 私有 overlay 在自己的解释器重复准备与加载验证，不能借基础环境的缓存冒充可用；已有 ready 也会重新验证，失败先撤销 ready。部署模型使用同一版本组，模型私有扩展不进入通用插件、不修改 Homebrew。新官方核心超出兼容矩阵时明确拒绝，不能逐包盲升。

2026-10-16 通用 diffusion SDPA 的 Metal OOT 分派修复已通过上述共享准备器构建非 editable wheel 并完成 Qwen-Image-2.1-Turbo 真实文生图验收，详见[执行合同与本次命令](generic-backend.md#责任边界)。对应模块现为 `diffusion/attention.py`，由现有 setuptools 包发现与源码安装/Formula 自动包含，无须变更版本 pins 或 native 资源；源码指纹会触发新隔离后端。仅本地部署已安装，远端未推送，未运行 Homebrew/curl 安装。

本次diffusion GPU／分片ANE接入使用共享准备器构建非editable wheel。完整校准基于 `runtime/a46a54e37c719e65`、Omni main `c8ccb8aed59e1db273d9dce6ee91b4f09d30db18`、QwenImage overlay `779257e7887353fb`。其后直接部署时，联网预检更新至 Omni main `082eaf235cfc373322bc5f8a1a3398915446a82d`，当前运行时 `runtime/c076adaf5074c803`、QwenImage overlay `overlays/qwen-image-2.1-turbo/bf033dac54505e3e`；依赖校验、统一模型列表和真实512×512推理通过。依赖下载遇到TLS故障时使用 `UV_OFFLINE=true` 与已有兼容缓存，官方Git预检未跳过，未改核心pins或关闭TLS校验。

生产AOT尚未构建：本机只有Command Line Tools，`xcrun --find metal`／`metallib`失败，Apple官方 `xcodebuild -downloadComponent metalToolchain` 因缺完整Xcode被拒绝。用户已允许安装工具链，但此入口缺少前置Xcode，未安装或下载工具、未重装Homebrew；隔离shader测试不代替AOT。当前 Omni main `082eaf235cfc373322bc5f8a1a3398915446a82d` 的QwenImage真实512×512 GPU off/auto A/B输出一致，但auto慢约3.6%；第0层ANE协同整图scaled max 12.94%超过1%门禁，未发布profile。当前服务GPU优化off、ANE未启用；完整证据见[ANE记录](ane.md)。本次代码仍是本地工作区，不把它描述成已推送的远端入口。

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

## 可选：vpipe 源算子 metallib

```bash
python -I -m vllm_omni_metal.ops.vpipe_kernels
```

需要**已安装的 Apple Metal compiler**（`xcrun --find metal`）；它与仅能编译 host Objective-C++ 的 Command Line Tools 不等同。也可安装时显式选择 `--with-vpipe`。构建在本地私有 cache、离线执行，不安装工具、不修改上游依赖。生产 worker 缺少匹配 metallib 时保留原路径，绝不在 startup/request 自动编译。本机缺 compiler，因此 packed Q/K RoPE 等新增源码仅完成隔离 shader/OOT 数值与计时，不宣称 AOT 或真实 worker 已可用；单行小 SwiGLU 无稳定收益，仍不接入默认路径。来源和限制见[执行协议](generic-backend.md#成熟方案续移植packed-qk-rope)。

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
本次安装资源改动未运行构建、安装或回归，没有真实四模型质量/速度新结论；
没有自动启用、迁移部署、重启服务或发布远端。以下数字均为独立历史记录。

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

本轮在全新私有安装器环境实际解析/安装223个包，验证固定版本、MPS generator、随包10项 native资源，并显式准备 attention host launcher；额外以污染的 cwd/PYTHONPATH 重跑，隔离验证仍通过。随后实际调用部署准备器的干净副本 wheel 安装路径：非 editable 导入、`pip check`、三个可选模块及服务CLI help通过；禁用MLX、请求期编译和显式同步后，FP32/FP16/BF16六个序列、每序列四步 attention 正确执行，记录在 `outputs/metal-install-current/report.json`。最终非集成回归 **259 passed / 143 deselected**，旋转/vpipe针对性回归（含实际MPS）**90 passed**，Ruff、格式、shell和Ruby语法通过。安装mock/CPU准备器回归不冒充真实安装；此前277/1是历史集成结果，不作为本轮完整重跑结论。未实际运行会停服务的总部署激活流程，未发布远端curl或重装Homebrew。

**本次安装资源同步验收：** 从干净副本重新构建 wheel，并在临时非 editable 环境安装；依赖通过显式 `.pth` 复用所选开发环境，没有重新解析完整依赖。安装后的全部30个包文件（含10项 native资源）与当前源码逐字节一致。使用仅替代 Homebrew 声明 DSL 的临时执行器，实际运行 Formula 的测试块：cwd/PYTHONPATH 污染场景修复前失败，改用所选 Python `-I` 后通过，覆盖版本、MPS generator、资源及四个 CLI help；这不是实际 `brew install/test`。安装/部署回归54项通过，Ruff、格式与shell/Ruby语法通过。持续同步指令已写入 `AGENTS.md`，临时环境与执行器已清理；未发布、切换服务或重跑模型生成。

未补丁 Omni 的 YuE2 模型级 MPS 障碍仍需独立适配；私有 `yue2mps2` overlay 已有真实512帧歌曲与 attention 对照，但不代表普通安装开箱可用，见 [接入状态](yue2-readiness.md)。PyAV/OpenCV/系统 FFmpeg 存在重复 Objective-C 类警告，本轮安装/CLI未因此失败，媒体视频路径未额外验收。

卸载仅移除确认不再被服务使用的安装器环境或 `brew uninstall vllm-omni-metal`；不删除模型缓存、生成文件或仍被其他项目使用的 vllm-metal。
