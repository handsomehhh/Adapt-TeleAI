#!/usr/bin/env bash
set -euo pipefail
deploy_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$deploy_root"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/adapt-g1-uv-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/adapt-g1-check-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/adapt-g1-matplotlib}"
printf '[deploy] 仓库：%s\n' "$deploy_root"
if [[ -n "${ADAPT_PYTHON:-}" ]]; then
  deploy_python="$ADAPT_PYTHON"
elif [[ -x "$deploy_root/.venv/bin/python" ]]; then
  deploy_python="$deploy_root/.venv/bin/python"
elif [[ -x "$deploy_root/../Adapt-TeleAI/.venv/bin/python" ]]; then
  deploy_python="$deploy_root/../Adapt-TeleAI/.venv/bin/python"
else
  printf '[deploy] 正在通过 uv 加载项目 Python 环境……\n'
  exec "${ADAPT_UV:-uv}" run python -u -m deploy "$@"
fi
printf '[deploy] Python：%s\n' "$deploy_python"
printf '[deploy] 正在加载运行库，请稍候；就绪后会打印网页地址。\n'
exec "${ADAPT_UV:-uv}" run --no-project --no-python-downloads --python "$deploy_python" python -u -m deploy "$@"
