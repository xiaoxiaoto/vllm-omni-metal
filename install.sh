#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Match stable 0.31 native wheels with a frozen Omni main snapshot.
vllm_url='https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl#sha256=21378516a51ff15029f71168e827b94660e21b39484eb52277166f01ac7dd686'
metal_url='https://github.com/vllm-project/vllm-metal/releases/download/v0.31.0/vllm_metal-0.31.0-cp312-cp312-macosx_15_0_arm64.whl#sha256=e2ab31296f8c42d8b65c8d5f5490f160087d1070178444870a5672e44aaacf6e'
omni_commit="${VLLM_OMNI_COMMIT:-2ec5f265c69520cdd5e5f0a19db485d38885843e}"
[[ "$omni_commit" =~ ^[0-9a-f]{40}$ ]] || { echo 'VLLM_OMNI_COMMIT 必须是完整 Git commit' >&2; exit 1; }
omni_source="vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@${omni_commit}"
plugin_url='vllm-omni-metal @ git+https://github.com/xiaoxiaoto/vllm-omni-metal.git@main'

source_dir=''
if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
  candidate="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "$candidate/pyproject.toml" && -d "$candidate/vllm_omni_metal" ]]; then
    source_dir="$candidate"
  fi
fi
venv="${source_dir:-$HOME}/.venv-vllm-omni-metal"
with_mps_attention=false
with_vpipe=false
with_ane=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { echo '--venv 需要路径' >&2; exit 1; }
      venv="$2"; shift 2 ;;
    --with-mps-attention) with_mps_attention=true; shift ;;
    --with-vpipe) with_vpipe=true; shift ;;
    --with-ane) with_ane=true; shift ;;
    -h|--help)
      printf '%s\n' '用法：install.sh [--venv PATH] [--with-mps-attention] [--with-vpipe] [--with-ane]' \
        '安装匹配的 vLLM CPU / vllm-metal 0.31.0 wheel、Omni CPU 包和本插件。' \
        '源码运行默认创建 .venv-vllm-omni-metal（editable）；curl 运行创建 ~/.venv-vllm-omni-metal。' \
        '不修改系统 Python、Homebrew libexec、已有 .venv 或运行中的服务。' \
        '默认不编译可选工具；所有显式构建均在选定私有 Python 安装完成后离线执行。' \
        '--with-mps-attention 构建并验证 Torch-stream attention host launcher，需要现有 Apple Command Line Tools。' \
        '--with-vpipe 构建 vpipe metallib，需要已安装的 Apple Metal compiler。' \
        '--with-ane 构建 CoreML host 工具；不代表 ANE 或模型已可用。' \
        '以上构建不下载工具/权重，不自动转换模型或校准；失败即退出，不重启服务。'
      exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 1 ;;
  esac
done
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo '需要原生 Apple Silicon macOS，不支持 Linux 或 Rosetta x86_64。' >&2
  exit 1
fi
if [[ "$(sw_vers -productVersion | cut -d. -f1)" -lt 15 ]]; then
  echo '预编译 vllm-metal wheel 需要 macOS 15+。' >&2
  exit 1
fi
case "$venv" in /*) ;; *) venv="$PWD/$venv" ;; esac
# Never repurpose an unrelated or shared environment, even on a failed rerun.
if [[ -e "$venv" && ! -f "$venv/.vllm-omni-metal-installer" ]]; then
  echo "拒绝修改未由本安装器创建的环境：${venv}；请选择新的 --venv 路径。" >&2
  exit 1
fi
command -v curl >/dev/null
command -v git >/dev/null || { echo '需要 git；请先安装 macOS Command Line Tools。' >&2; exit 1; }
temporary=''
cleanup() { [[ -z "$temporary" ]] || rm -f -- "$temporary"; }
trap cleanup EXIT
if ! command -v uv >/dev/null; then
  temporary="$(mktemp)"
  curl -fsSL https://astral.sh/uv/0.9.18/install.sh -o "$temporary"
  sh "$temporary"
  export PATH="$HOME/.local/bin:$PATH"
fi
if [[ ! -x "$venv/bin/python" ]]; then
  uv venv "$venv" --python 3.12 --seed
  touch "$venv/.vllm-omni-metal-installer"
fi
python="$venv/bin/python"
"$python" -I - <<'PY'
import platform
import sys
assert sys.version_info[:2] == (3, 12), '需要 Python 3.12 wheel ABI'
assert platform.machine() == 'arm64', 'Python 必须是原生 arm64'
assert sys.prefix != sys.base_prefix, '拒绝安装到系统 Python'
PY
# Omni's actual selector is VLLM_OMNI_TARGET_DEVICE, not VLLM_TARGET_DEVICE.
export VLLM_OMNI_TARGET_DEVICE=cpu VLLM_OMNI_VERSION_OVERRIDE=0.31.0.dev0+cpu
if [[ -n "$source_dir" ]]; then
  plugin_args=(-e "$source_dir")
else
  plugin_args=("$plugin_url")
fi
# Keep the native wheel's Git dependencies visible in the same resolution.
uv pip install --python "$python" "$vllm_url" "$metal_url" "$omni_source" "${plugin_args[@]}"
uv pip check --python "$python"
"$python" -I - <<'PY'
from importlib import import_module
from importlib.metadata import entry_points, version
import torch
from vllm_omni_metal import is_available
from vllm_omni_metal.ane.runtime import DenseSwiGLU, TiledDenseSwiGLU, execution_scope
from importlib.resources import files
for module in ('execution.compat', 'execution.runtime', 'execution.shm_connector',
               'native.attention', 'native.compat', 'native.runner', 'native.worker',
               'diffusion.attention', 'diffusion.config', 'diffusion.ops', 'diffusion.worker',
               'ops.mps_attention', 'ops.metal_attention', 'ops.vpipe_kernels',
               'ops.normalization', 'ops.activation', 'ops.rotary',
               'quantization.linear', 'quantization.engine',
               'quantization.capture', 'quantization.adapter', 'quantization.runtime',
               'ane.cache', 'ane.runtime', 'ane.build'):
    import_module(f'vllm_omni_metal.{module}')
# CPU-only mixed-precision profile contract (no packing or model execution).
from vllm_omni_metal.quantization.linear import build_profile, validate_profile
quantization_modules = {
    'w4': torch.nn.Linear(64, 8, dtype=torch.bfloat16).requires_grad_(False),
    'w8': torch.nn.Linear(64, 8, dtype=torch.bfloat16).requires_grad_(False),
    'bf16': torch.nn.Linear(7, 3, dtype=torch.bfloat16).requires_grad_(False),
}
quantization_profile = build_profile(
    quantization_modules, list(quantization_modules), scope='installation-contract-v2',
    calibration={'purpose': 'installation-contract-only'},
    precisions={'w4': 'w4a16', 'w8': 'w8a16', 'bf16': 'bf16'},
)
assert quantization_profile['schema'] == 2
assert [
    (item['bits'], item['mode'], item['group_size'])
    for item in quantization_profile['modules'].values()
] == [(4, 'affine', 64), (8, 'affine', 64), (16, 'identity', None)]
assert validate_profile(
    quantization_profile, quantization_modules, scope='installation-contract-v2',
) == list(quantization_modules)
# End CPU-only profile contract.
from vllm_omni_metal.platform import MetalOmniPlatform
for path in (
    MetalOmniPlatform.get_omni_ar_worker_cls(),
    MetalOmniPlatform.get_omni_generation_worker_cls(),
    MetalOmniPlatform.get_diffusion_worker_cls(),
    MetalOmniPlatform.get_diffusion_model_runner_cls(),
    MetalOmniPlatform.get_diffusion_attn_backend_cls(None, 64),
):
    module, _, symbol = path.rpartition('.')
    assert isinstance(getattr(import_module(module), symbol), type), path
for resource in (
    '_native/pytorch_backend/mps_ops.mm',
    '_native/metal/metal_device.h',
    '_native/metal/paged_attention_kernels.h',
    '_native/vpipe_elementwise.metal',
    '_native/diffusion_rotary.metal',
    '_native/VPIPE.md',
    '_native/ane/ane-emitter.cc',
    '_native/ane/ane-emitter.h',
    '_native/ane/emit.cc',
    '_native/ane/bridge.swift',
    '_native/ane/LICENSE',
):
    assert files('vllm_omni_metal').joinpath(resource).is_file(), resource
assert version('vllm') == '0.31.0+cpu'
assert version('vllm-metal') == '0.31.0'
assert version('vllm-omni') == '0.31.0.dev0+cpu'
assert version('vllm-omni-metal') == version('vllm-omni').split('+', 1)[0]
assert is_available(), 'MPS 不可用'
assert torch.Generator(device='mps') is not None
assert any(e.name == 'metal' and e.value == 'vllm_omni_metal:register_platform'
           for e in entry_points(group='vllm_omni.platform_plugins'))
PY
if [[ "$with_mps_attention" == true ]]; then
  "$python" -I -m vllm_omni_metal.ops.mps_attention
  "$python" -I -c 'from vllm_omni_metal.ops.mps_attention import get_mps_ops; assert get_mps_ops() is not None, "Torch-stream attention launcher 未加载"'
fi
if [[ "$with_vpipe" == true ]]; then
  "$python" -I -m vllm_omni_metal.ops.vpipe_kernels --diffusion
fi
if [[ "$with_ane" == true ]]; then
  "$python" -I -m vllm_omni_metal.ane.build --bridge
fi
printf '\n可选工具仅显式离线准备；分别使用当前环境：\n  "%s" -I -m vllm_omni_metal.ops.mps_attention\n  "%s" -I -m vllm_omni_metal.ops.vpipe_kernels --diffusion\n  "%s" -I -m vllm_omni_metal.ane.build --bridge\n启动和请求期间不编译，不自动安装工具、下载权重、转换模型或校准。\n' "$python" "$python" "$python"
printf '\nANE auto 无匹配权重图/整流水线批准 profile 时保留 GPU；host 工具不等于 ANE 可用。\n真实 YuE2 ANE 图 relative L2=5.499%% 超过 1%% 门槛，已拒绝，未批准 ANE。\nYuE2 实测依赖独立模型 overlay；通用安装不捆绑模型补丁/registry/媒体，不保证未补丁 Omni 可运行 YuE2。\n未修改其他环境或重启已有服务；维护已有环境前请先停止使用它的服务。\n'
printf '\n安装完成。激活环境：\n  source "%s/bin/activate"\n或直接运行：\n  "%s/bin/vllm-omni" serve /path/to/model --omni --deploy-config /path/to/metal-profile.yaml\n' "$venv" "$venv"
