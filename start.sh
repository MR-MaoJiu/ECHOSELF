#!/usr/bin/env bash
# EchoSelf 一键启动（macOS / Linux）
# 在仓库根目录执行，自动使用 .venv；否则依次尝试 python3 / python

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY=""
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  echo "错误：未找到 Python。请先安装 Python 3.10+，或在本目录创建虚拟环境：" >&2
  echo "  uv venv .venv && uv pip install -e \".\"" >&2
  echo "（全功能训练见 README「快速开始」中的 uv pip install -e \".[all]\"）" >&2
  exit 1
fi

if [[ ! -f "$ROOT/app.py" ]]; then
  echo "错误：在 $ROOT 未找到 app.py，请勿移动本脚本。" >&2
  exit 1
fi

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "提示：未检测到 .venv，将使用: $PY" >&2
  echo "      若缺少依赖，请先按 README「快速开始」创建虚拟环境并安装。" >&2
  echo "" >&2
fi

exec "$PY" "$ROOT/app.py"
