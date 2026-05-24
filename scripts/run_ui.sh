#!/usr/bin/env bash
# Web UI 录制一键启动包装：
#   cd 到 repo 根 → 可选加载 .env → 用 FRANKA_TELEOP_VENV/bin/python 起 UI。
# 用法:
#   ./scripts/run_ui.sh                     # 默认 scripts/config/record_cfg_unityvr.yaml
#   ./scripts/run_ui.sh path/to/cfg.yaml    # 指定 yaml
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# 可选 .env (若存在则覆盖当前 shell 变量; 不想用 .env 就别建)
if [ -f "$REPO_ROOT/.env" ]; then
  set -a; source "$REPO_ROOT/.env"; set +a
fi

# 必需变量缺失 fail-loud (语义同 preflight.py:384)
: "${POLYMETIS_ENV:?POLYMETIS_ENV 未设, 参考 .env.example 或 docs/QUICKSTART.md §5.1}"
: "${FRANKA_TELEOP_VENV:?FRANKA_TELEOP_VENV 未设, 参考 .env.example 或 docs/QUICKSTART.md §5.1}"

CFG="${1:-scripts/config/record_cfg_unityvr.yaml}"
exec "$FRANKA_TELEOP_VENV/bin/python" scripts/core/run_record_hdf5_ui.py --config "$CFG"
