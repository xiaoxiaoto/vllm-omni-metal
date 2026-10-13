#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Match stable 0.31 native wheels with a frozen Omni main snapshot.
vllm_url='https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl#sha256=21378516a51ff15029f71168e827b94660e21b39484eb52277166f01ac7dd686'
metal_url='https://github.com/vllm-project/vllm-metal/releases/download/v0.31.0/vllm_metal-0.31.0-cp312-cp312-macosx_15_0_arm64.whl#sha256=e2ab31296f8c42d8b65c8d5f5490f160087d1070178444870a5672e44aaacf6e'
omni_source='vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@2ec5f265c69520cdd5e5f0a19db485d38885843e'
plugin_url='vllm-omni-metal @ git+https://github.com/xiaoxiaoto/vllm-omni-metal.git@main'

source_dir=''
if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
  candidate="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "$candidate/pyproject.toml" && -d "$candidate/vllm_omni_metal" ]]; then
    source_dir="$candidate"
  fi
fi
venv="${source_dir:-$HOME}/.venv-vllm-omni-metal"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      [[ $# -ge 2 && -n "$2" ]] || { echo '--venv 需要路径' >&2; exit 1; }
      venv="$2"; shift 2 ;;
    -h|--help)
      printf '%s\n' '用法：install.sh [--venv PATH]' \
        '安装匹配的 vLLM CPU / vllm-metal 0.31.0 wheel、Omni CPU 包和本插件。' \
        '源码运行默认创建 .venv-vllm-omni-metal（editable）；curl 运行创建 ~/.venv-vllm-omni-metal。' \
        '不修改系统 Python、Homebrew libexec、已有 .venv 或运行中的服务。'
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
"$python" - <<'PY'
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
"$python" - <<'PY'
from importlib.metadata import entry_points, version
import torch
from vllm_omni_metal import is_available
assert version('vllm') == '0.31.0+cpu'
assert version('vllm-metal') == '0.31.0'
assert version('vllm-omni') == '0.31.0.dev0+cpu'
assert is_available(), 'MPS 不可用'
assert torch.Generator(device='mps') is not None
assert any(e.name == 'metal' and e.value == 'vllm_omni_metal:register_platform'
           for e in entry_points(group='vllm_omni.platform_plugins'))
PY
printf '\n安装完成。激活环境：\n  source "%s/bin/activate"\n或直接运行：\n  "%s/bin/vllm-omni" serve /path/to/model --omni --deploy-config /path/to/metal-profile.yaml\n' "$venv" "$venv"
