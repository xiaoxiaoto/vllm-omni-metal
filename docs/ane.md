# GPU + ANE：离线校准的默认 auto

`VLLM_OMNI_METAL_ANE=auto` 是默认策略，**不是强制 ANE，也不会自动改写模型 FFN**。仅显式接入通用 `DenseSwiGLU` 的模型适配器可使用；现有 activation OOT 只看到 SiLU，不包含完整投影，不能作为完整 FFN 卸载入口。

## 安装与准备

```bash
# 普通安装提供全部源码/API，缺少准备产物时仍使用 GPU。
./install.sh
# 显式同时构建 host 工具；不下载编译工具或模型。
./install.sh --with-ane
# 也可在选定独立环境中单独执行：
python -m vllm_omni_metal.ane_build --bridge
```

Homebrew 使用 `$(brew --prefix vllm-omni-metal)/libexec/bin/python -m vllm_omni_metal.ane_build --bridge`。需要现有 Apple Command Line Tools 的 clang++/Swift 和 macOS 15+ CoreML；不需要额外安装 Python CoreML 包，亦不要求 `metal` shader compiler。Formula 不在安装过程中自动编译这些可选工具。

**host 工具不等于模型加速已准备完成。** 独立模型适配器还须调用 `ane_build.prepare_group(weights, scope=...)`，准备静态权重图和用于溢出恢复的 runtime-weight 图，然后测量自身真实完整生成流程，调用 `publish_profile` 批准对应形状。普通安装不下载权重、不转换任意模型、不运行生成校准。

准备产物位于私有 `~/.cache/vllm-omni-metal/ane/<源码/系统/Torch摘要>/`，绑定设备身份、原始权重内容及 dtype、流水线 revision 和执行配置。模型可用 `execution_scope(revision, vllm_config)` 绑定上下文/KV 预算、attention provider 和 IR 设置。通过 `context=` 补充流水线形状，例如关联 attention 的历史长度；不同 context 不共用批准记录。环境、系统、后端源码或权重变化后须显式重新准备。

缺少产物/批准记录、形状不匹配或预算不足时保留 GPU；现有文件的权限、schema、设备或 checksum 错误明确报错。工具与图在私有目录中锁定、临时构建并原子发布，执行前校验；不会调用模型 registry、修改上游安装文件或全局 Torch/CUDA。

## 执行与选择

- 仅连续二维、同设备同 dtype、bias-free 的 dense packed gate/up + down；MPS FP16/BF16；隐藏与中间维均为 1024 倍数，每组投影不超过 256 MiB。
- 固定 256 行 ANE chunk，不填充短输入，GPU 至少保留一行。行数不超过 256 或未批准形状均原生执行。
- 每组不可变权重共用一个持久 worker，CoreML `.cpuAndNeuralEngine`，GPU head 与 ANE tail 并行，合并前 join。CPU 读回与 dtype 转换分开，避免已复现的 Torch 2.13 MPS view 源数据破坏。
- 离线按完整流水线计时，必须包含 AR/attention/VAE 等原生阶段及读回、转换、等待和合并；先独立 warmup，再平衡顺序测量每候选至少两次。只有中位延迟降低超过 3%，且候选最慢样本仍快于 GPU 最快样本才采用；否则发布 GPU。
- 请求内不跑性能探针、不搜索比例或编译图。此版本采用冻结的离线 profile，**不是** vpipe 的在线 9 样本/48 block 控制器，也不保证其他并发、热状态或所有输入的最优比例。
- 系统至少保留 4 GiB headroom。离线全流水线校准的 live 层/形状/context 首样本相对 L2 与峰值归一误差均不得超过 1%；逐 chunk 检查有限值。正式请求不额外重跑完整 GPU FFN 做在线对照，避免抵消收益。数值、设备或内存失败后该组锁定 GPU；已完成 chunk 保留，丢失 tail 在 GPU 重算。

只有 CoreML 计划中每个投影都首选 ANE 才发布图。这是公开 compute plan，不是 Instruments 硬件 trace。后端不在 worker/request 调用 emitter、工具链或模型编译 API；CoreML 自身的模型加载与设备缓存成本仍存在，不能承诺冷启动零成本。

## 精度与生命周期

API 默认 `precision='native'`，不允许透明地把 BF16 解释为 FP16。适配器须显式声明 `precision='fp16'`，并通过离线校准后才能参与默认 auto。**FP16 不是 BF16 位等价或无损模式**；首样本 1% 检查不是所有步/所有输入/最终媒体的误差上界。

溢出遵循 vpipe：稳定 SiLU，只有 UP 权重除以 4、16…4096，重跑当前 chunk，FP32 恢复比例后用 Torch RNE 写回源 dtype；静态图不能修改权重，故使用预先准备的 runtime-weight 图恢复。超过上限/恢复后仍非有限则 GPU 重算，不 clipping 或 `nan_to_num`，不放宽门槛。这仅恢复动态范围，不能消除普通舍入误差。

```python
from vllm_omni_metal.ane import DenseSwiGLU

# 生命周期由已注册模型/独立适配器持有；不是每次 forward 构建。
group = DenseSwiGLU([(packed_gate_up, down)], scope=pipeline_revision, precision="fp16")
with torch.inference_mode():
    output = group(0, normalized_input, context=attention_history_length)
group.close()
```

权重绑定后必须不可变（包括 inference tensor 的 alias）；更换权重/device/dtype 必须关闭并重新绑定。普通 tensor 的版本变化会禁用协同。原生 worker 在关闭时调用模型可选的 `close_metal_operators()`，其余 facade 自行持有/释放。失败回 GPU 会改变算路，不承诺跨设备模式或故障前后同 seed 位等价；健康固定配置的复现应由模型适配器验证。

## 当前证据边界

此前独立 YuE2 512 帧真实生成原型：GPU 37.698 s、GPU+ANE 36.113 s（约 4.2%），最终 PCM relative L2 约 **5.18%**；每种仅两次测量，准备约 205.5 s、额外图约 2 GiB。半分模式反而更慢，不能用局部 FFN/ANE 利用率替代完整延迟。

这些是**此前原型**结果，不是本次安装、正式 API 或任意模型的性能验收。独立 YuE2 部署的 `prepare_ane.py` 用原生流程重新准备/校准并核对默认 reader 与同 seed 输出；本后端不捆绑该模型适配。现有 200 帧样本 NAR 只有 202 行，不协同；未知输入形状/context 仍使用 GPU。未做听感、所有 ODE 步审计、跨模型完整生成、长热负载或真实 ANE trace，不宣称这些能力通过。

源码来自固定 vpipe Apache-2.0 版本，见 [`_native/VPIPE.md`](../vllm_omni_metal/_native/VPIPE.md)。emitter 直接生成 vpipe 格式的 compiled bundle，不是 Apple 公共转换器；系统更新可能不兼容，须重建并重新通过计划与数值检查。
