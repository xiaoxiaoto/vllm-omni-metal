# 安装

参考 vllm-metal 的 Homebrew / curl / 源码三种入口，运行时统一采用独立 Python 3.12 环境和预编译 macOS arm64 wheel，不改系统 Python或已安装的 vllm-metal 私有环境。

## 依赖基线

- 原生 Apple Silicon、macOS **15+**，不支持 Rosetta x86_64。
- vLLM `0.30.0+cpu` 与 vllm-metal `0.30.0` 的官方预编译 wheel，固定 SHA256。
- vllm-omni 固定 `v0.30.0` 对应 commit，设置 `VLLM_OMNI_TARGET_DEVICE=cpu` 和 `VLLM_OMNI_VERSION_OVERRIDE=0.30.0+cpu`。**前者不是 `VLLM_TARGET_DEVICE`**；漏掉时隔离构建可能误选 CUDA。
- PyTorch 2.13 提供 MPS generator。不追踪上游 latest/dev，PyPI CUDA wheel 不是 macOS 安装方案。

本插件是纯 Python；vllm-metal 自带已编译的 native artifacts，不需要按它的源码开发流程重新编译 kernel。Omni CPU 包及少量 Python 依赖仍可能需要构建工具。安装需要联网获取 Python/依赖和 Git 源码，但不下载模型。

## 1. 源码开发

在包含本次新增 `install.sh` 的 checkout 中：

```bash
./install.sh
source .venv-vllm-omni-metal/bin/activate
vllm-omni --help
```

此流程创建源码目录下 `.venv-vllm-omni-metal`、安装上述依赖，再 editable 安装当前目录。原有 `.venv` 不动。可指定新路径：

```bash
./install.sh --venv "$HOME/.venv-vllm-omni-metal"
source "$HOME/.venv-vllm-omni-metal/bin/activate"
```

安装器只接受新目录或它自己标记创建的环境，不接管任意既有 venv/Homebrew libexec，不使用 `--clear`。失败后保留自己创建的环境以便诊断和重试。更新前先停止使用该环境的服务，再运行安装器。

已有项目开发 `.venv` 不需要迁移或重新下载依赖；它仍可使用原来的 `pip install --no-deps --no-build-isolation -e .` 和检查命令。

## 2. Homebrew

`Formula/vllm-omni-metal.rb` 使用独立 `libexec`，安装所有配套依赖，仅链接 **`vllm-omni`**，不接管 `vllm-metal` 已有的 `vllm` 命令。采用上游 Formula 的 `preserve_rpath` 和 TVM-FFI/xgrammar RECORD 保留处理。

目前 Formula 和安装脚本是本地新增文件，尚未推送/发布。当前 checkout 可以创建本地 tap：

```bash
brew tap-new local/vllm-omni-metal
cp Formula/vllm-omni-metal.rb \
  "$(brew --repository)/Library/Taps/local/homebrew-vllm-omni-metal/Formula/"
brew install local/vllm-omni-metal/vllm-omni-metal
```

本地 tap 名称若已存在，不重复 `tap-new`。不要安装到共享的 `$(brew --prefix vllm-metal)/libexec`。

发布到远端仓库后才可使用：

```bash
brew tap xiaoxiaoto/vllm-omni-metal https://github.com/xiaoxiaoto/vllm-omni-metal
brew install xiaoxiaoto/vllm-omni-metal/vllm-omni-metal
vllm-omni --help
```

Formula 当前安装已发布的 `0.1.0` 源码快照，不假装存在尚未发布的插件 release wheel。固定源码 archive 与 native wheel 的 SHA256，Omni 源码也固定 commit。升级需要维护这组匹配版本及 checksum，不能简单改成 latest。

Python 位于 `$(brew --prefix vllm-omni-metal)/libexec/bin/python`。无需激活环境。移除使用 `brew uninstall vllm-omni-metal`；不要卸载仍被其他服务使用的 vllm-metal。

## 3. curl 安装（脚本发布后）

```bash
curl -fsSL https://raw.githubusercontent.com/xiaoxiaoto/vllm-omni-metal/main/install.sh | bash
source ~/.venv-vllm-omni-metal/bin/activate
```

没有 uv 时安装器使用上游同样的 uv bootstrap。无本地 checkout 时安装固定已发布源码 archive，不需要本项目 release wheel。默认环境为 `~/.venv-vllm-omni-metal`；当前远端还没有本次新脚本，因此此命令暂不能执行。

删除该环境即可卸载；确认它仅由本安装器使用后再操作，不删除 Hugging Face / ModelScope 模型缓存或生成文件。

## 使用与验证

```bash
vllm-omni serve /path/to/local/checkpoint --omni \
  --deploy-config /path/to/metal-resource-profile.yaml \
  --host 127.0.0.1 --port 18094
```

原生 AR/generation profile 需要 CUSTOM、单设备、同步调度及关闭 prefix cache，见 [通用后端协议](generic-backend.md)。安装入口不替换模型 topology 或注册表，也不启动服务。

本次验证覆盖安装器命令/安全边界的 mock 回归、Formula Ruby 语法和 Homebrew DSL 加载、源码 archive checksum。**尚未进行联网全新安装或实际 `brew install`**，不能把已有开发环境的成功当成这些入口的端到端安装验证。
