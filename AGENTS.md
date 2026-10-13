# AGENTS.md

## 项目与范围

本项目是 Apple Silicon 上的 vLLM-Omni out-of-tree 平台插件，目标是通用 Metal 后端，不是独立推理服务器。当前支持原生单卡 dense AR / LLM_GENERATION 与完整 generation pipeline 的共用执行层；已验证原生 Higgs 双阶段和随机 Llama/Qwen3，不能宣称所有 Omni 模型/功能已移植。

只维护平台、运行时、设备能力、通用执行合同和兼容桥接。不要重新加入模型专用实现、部署拓扑、采样/codec 逻辑、HTTP adapter 或模型白名单；这些属于上游或独立模型插件。不得覆盖上游模型 registry。

先阅读 `README.md`、`pyproject.toml`、`docs/generic-backend.md` 和涉及模块。

## 环境与命令

从项目根目录运行，优先使用现有 `.venv`。实测基线是 Python 3.12、vllm-metal / vLLM / vLLM-Omni 0.30.0、PyTorch 2.13、diffusers 0.40.0。原生 seeded 采样依赖 MPS generator。不要未经验证升级依赖或以 CUDA wheel 覆盖 macOS 的 vLLM CPU 构建。

项目安装参考 vllm-metal：`install.sh` 创建独立 `.venv-vllm-omni-metal`（本地源码 editable）或 `~/.venv-vllm-omni-metal`（curl）；`Formula/vllm-omni-metal.rb` 创建 Homebrew libexec 并只链接 `vllm-omni`。查看 `docs/installation.md`；不把 Omni 装入已有 vllm-metal 共享环境。安装时固定 CPU/MPS 0.30.0 栈，Omni 构建变量是 `VLLM_OMNI_TARGET_DEVICE=cpu`。变更版本/commit/checksum 时同步 Formula、脚本及安装测试；未经完整安装验证，不宣称新入口端到端可用或已发布。

以下命令仅适用于配套依赖和构建工具已经准备好的既有开发 `.venv`：

```bash
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m pip check

# 常规代码检查，不加载真实模型。
.venv/bin/python -m pytest -q -m 'not integration'
.venv/bin/ruff check .
.venv/bin/ruff format --check .

.venv/bin/python -m pip wheel --no-deps --no-build-isolation -w dist .
```

服务使用选定独立环境的 `vllm-omni`（Homebrew、安装器 venv 或开发 `.venv`），不要误用 Homebrew 的基础 `vllm`。元数据/entry point 修改后重新 editable 安装；构建前清理过期 build 产物，防止已删除模块残留在 wheel。

## 架构约束

- `__init__.py`：只注册 Omni 平台。检测阶段不导入模型 registry 或初始化 MLX，保持包导入惰性。
- `platform.py` / `config.py`：设备 hook 和后端支持范围校验；不支持的配置明确报错，不静默退回 CUDA 或 CPU。
- `runtime.py` / `worker.py`：从注册类解析 `metal_runtime`（默认 Torch），共用生命周期。MLX facade 自行加载权重，不能经过 PyTorch checkpoint loader，也不能使用 Torch generator 或 PyTorch LoRA manager。
- 外部注册类可以通过 `validate_metal_config(config)` 声明模型约束；后端不硬编码模型名称、采样率或 checkpoint 精度。
- `native_worker.py` / `native_runner.py`：原生模型加载、scheduler 请求生命周期、MPS RNG、pre/postprocess、模型 sampler 与 CPU payload。模型特有逻辑留在上游。
- `native_attention.py`：按官方 KV allocation offset/strides 分配和读取；保持 causal offset、GQA、请求隔离及 block 边界校验，不能丢弃历史 KV。
- `shm_connector.py`：仅 Metal worker 注册 POSIX SHM constructor；私有目录、短物理名、长度帧、flock 和 unlink 生命周期必须保持。
- `native_compat.py`：仅模型实例克隆含 CUDA capture 查询的方法；class/module/global torch.cuda 不得改变，其他 CUDA API 不模拟，输入范围校验不能被绕过。
- `compat.py`：窄范围幂等配置/attention/内存桥接，保留普通 LLM hook。不要修改已安装的上游源码或全局替换 `torch.cuda`。
- HTTP、输出处理和请求语义交给上游 Omni，不在平台插件中维护 speech factory 补丁。

保留 `spawn` 启动方式及 `if __name__ == '__main__'` 入口保护；不要 fork 已初始化 Metal/Objective-C 的进程。Diffusers 使用 `diffusion_load_format='diffusers'`，不是 `model_impl='diffusers'`。

## 代码与测试

- 沟通和交付说明使用简体中文；标识符保持项目已有风格。
- 遵循 `pyproject.toml` 的 Ruff 配置：Python 3.12 目标、100 字符格式化行宽；Python 文件保留 Apache-2.0 SPDX 头。
- 选择最小完整改动，复用现有代码、标准库和已安装依赖，不增加无需求的抽象或配置。
- 修改共享函数前搜索调用者，同步更新测试、打包配置及支持文档；删除功能时清理入口和依赖，不能只删实现文件。
- 非平凡逻辑与缺陷修复补充小型回归测试；保留验证、错误处理和精度要求。
- 注释只解释必要原因；暂时限制使用 `shortcut: <限制>, <升级条件>`。

单元测试覆盖安装器命令/安全边界、platform/runtime/native worker/runner/attention/compat/SHM；GPU 回归覆盖 `tests/test_integration.py`、`tests/test_runtime.py`、`tests/test_native_runner.py`（随机 Llama/Qwen3）和 `tests/test_native_integration.py`（上游原生 Higgs HTTP）。修改注册、worker 或兼容补丁时运行全部非集成测试；保持注册不改写上游模型映射的检查。

以下真机测试仅在用户授权且内存充足时运行，使用本地随机权重，无需下载：

```bash
VLLM_OMNI_METAL_INTEGRATION=1 .venv/bin/python -m pytest -q
```

原生真实模型测试用 `VLLM_OMNI_METAL_NATIVE_MODEL=/path/to/local/higgs-checkpoint` 显式启用；从上游 profile 只调整资源/后端参数，保留原生 topology/connectors/processors。不自动下载大权重。

模型识别、加载或 health 200 不能等同于生成成功。验证实际输出的格式、有限值、seed 行为及资源清理；合成信号不算真实 TTS 验证。扩展能力前阅读上游合同，不得只解除 unsupported 检查。

## 安全与交付

- 保留配置、dtype、attention、输入与输出边界校验，不静默忽略不支持的选项；原生要求 CUSTOM、同步 uni、关闭 prefix cache。未实现的 encoder/prompt embeddings、辅助 MTP、稀疏路由、logprobs/structured decode、量化/LoRA/offload 等必须明确报错。
- KV 分配保留系统 headroom，显式 memory budget 必须尊重；长 GPU 任务前复核资源，达到预算停止，不能加大内存水位强行跑。
- 使用空闲测试端口，不停止或覆盖现有 8000 服务；只清理本次创建的进程并关闭 Omni 引擎。
- 不修改 MPS 内存水位或自动启用 CPU fallback，不把 vLLM KV-cache 预算当作 MLX 权重预算。MPS peak-memory 的 `0` 表示指标不可用。
- 不输出凭据或提交 `.env`、权重、虚拟环境、缓存、日志及 `outputs/`、`dist/`、`build/`；遵守 `.gitignore`。
- 删除代码不等于删除用户模型缓存或生成媒体；未明确授权时保留这些资产。插件许可不改变外部模型授权。
- 未要求时不要提交或推送，不删除或重置用户更改。完成后说明修改、实际检查和未验证项，不能把历史结果当成本次结果。
