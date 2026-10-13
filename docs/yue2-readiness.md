# YuE2 接入状态：通用后端已补齐，模型端尚未就绪

## 责任边界

本插件只维护通用平台/worker/runner。YuE2 的 token 协议、采样、NAR flow matching、VAE、pipeline、HTTP adapter 均由上游 vLLM-Omni 维护；没有新增模型类、模型名判断、白名单、registry 覆盖或全局 torch/CUDA monkey patch。

本次新增的通用合同：

- `prepare_runner_inputs`：请求 ID、computed/scheduled counts、token IDs/positions；模型显式 opt-in 后获取 sampling extra args 和 partial-prefill discard mask。
- `make_omni_output` 先建立当前步输出，模型 sampler 后续追加的终结音频不会被丢失。
- 使用上游稀疏声明解析与 payload builder，按请求 ID 对齐输出；空步、仅部分请求输出和非 batch 顺序输出均隔离。
- 非法 marker、重复/未知请求 ID、错误列表长度、非有限浮点和 hook 返回 shape/device/dtype 变化仍明确报错。

这些合同不专属于 YuE2，也能用于其他标准原生 AR 模型。

## 核对的上游版本

- YuE2 checkpoint：`m-a-p/YuE2-3B`，HF revision `8f24312f187763b9854adeec87e437e2b03bbff1`。
- Omni main 快照：`2ec5f265c69520cdd5e5f0a19db485d38885843e`。这是本次核对时的 main，不承诺之后一直是最新。
- 已含 `Yue2ForCausalLM`、配置、tokenizer、`yue2.yaml` 和 `/v1/audio/speech` adapter。
- checkpoint 自称 `YuE2ForCausalLM`，上游 deploy 用 `hf_overrides.architectures: [Yue2ForCausalLM]` 选择包内原生类，不需要执行 checkpoint 的 remote code。
- 模型 AR/NAR 为 BF16；单独的 `m-a-p/YuE2-Vae` decoder 为 FP32，最终为 48 kHz 双声道音频。权重 CC BY-NC 4.0，非商业。

## 不能仅修改 Homebrew 的 Omni commit

将上面的 Omni main 与现有 vLLM 0.30 wheel 混用，在导入阶段报错：

```text
ImportError: cannot import name 'MultiModalProcessingResult'
from 'vllm.multimodal.processing.processor'
```

临时隔离环境改用官方 macOS arm64 0.31 wheel 后，通用回归可以运行：

| wheel | SHA256（实际下载核对） |
| --- | --- |
| `vllm-0.31.0+cpu-cp312-cp312-macosx_11_0_arm64.whl` | `21378516a51ff15029f71168e827b94660e21b39484eb52277166f01ac7dd686` |
| `vllm_metal-0.31.0-cp312-cp312-macosx_15_0_arm64.whl` | `e2ab31296f8c42d8b65c8d5f5490f160087d1070178444870a5672e44aaacf6e` |

最初的隔离检查复用了已有环境的其他依赖。随后已用安装器在新的 `.venv-vllm-omni-metal` 完成全新依赖解析与安装：vLLM 0.31.0+cpu、vllm-metal 0.31.0、Omni 固定上述 main commit，明确标记为 0.31.0.dev0+cpu。Formula、安装器和插件依赖已同步，但尚未发布到远端。没有更改 Homebrew libexec、系统 Python 或正在运行的 Higgs；没有把开发源码伪装为稳定发布版。

## 上游模型仍存在的 Metal 障碍

在上述 Omni 快照直接调用真实 `sample_rows`，MPS logits + 模型当前创建的 CPU generator，真机报错：

```text
Expected a 'mps' device type for generator but found 'cpu'
```

源码还存在以下非 CUDA 路径的假设，需在上游模型处理而非平台层兜底：

- staging/readback buffer 无条件 `pin_memory=True`；事件使用 `torch.Event()`，需要验证非 CUDA event 生命周期或明确同步路径。
- VAE 设备选择仅 CUDA/CPU，不能因此宣称完整 pipeline 都在 Metal 上执行。
- NAR/VAE 的实权重数值、有限值、结束 token、frame cap、取消和 seed 行为尚未完成 Apple Silicon 验收。

不能通过替换全局 `torch.cuda`、忽略异常、把模型专用 sampler 搬入 Metal 插件，或只看到 health 200 来绕过这些问题。

## 本次实际验证

- 原有 0.30 环境：全部非集成测试 **67 passed**。
- 隔离的 vLLM/vllm-metal 0.31 + 上述 Omni 源码：全部非集成测试 **67 passed**。
- 两个环境分别执行全部 native runner 测试（包含真实 MPS hook/sparse 合同与随机 Llama/Qwen3 加载、decode、seed 隔离）：各 **18 passed**。
- 对上游 YuE2 采样做了实际 MPS 小张量检查，确认 CPU-generator 阻塞；没有加载 YuE2 大权重。
- 升级后的全新 0.31 安装环境：`pip check`、版本/MPS generator 和 CLI 检查通过，全部测试 **72 passed / 1 skipped**（跳过真实 Higgs 大权重）；不再仅依赖隔离环境中的旧依赖。

## 完成部署前的验收门槛

1. 上游 YuE2 实现提供明确的 MPS generator、同步采样/readback 和 NAR/VAE 设备路径；保留 seed、错误和请求清理语义。
2. 配套 0.31 native wheels 与 Omni 开发快照已在独立环境完整安装、通过通用/MPS 回归，Formula、安装器、依赖和 Higgs 启动脚本版本检查已同步；仍需发布这些改动并在新栈重跑真实 Higgs 非回归验收。不自动停止其他项目。
3. 加载真实 YuE2-3B + YuE2-Vae，通过 HTTP 获得可解码、有限、非静音的 48 kHz 双声道 WAV；检查同 seed、截断、结束与取消后资源回收。
4. 再交付 `docker-compose/yue2-3b` 的缓存准备、launchctl、Nginx 和启停流程。当前没有交付假定可运行的部署脚本。

未提交、推送、修改模型缓存或重启现有服务。通用能力已实现不等于 YuE2 端到端已支持。
