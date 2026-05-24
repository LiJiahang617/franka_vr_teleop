#!/bin/bash
# 一键安装脚本 — 开源用户 git clone 后运行
# 假设: 已装 miniconda3, 已有 libfranka 系统包, lerobot 仓库 clone 到 ../lerobot

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "REPO_ROOT=$REPO_ROOT"

echo "=== 1. 检查 prerequisites ==="
command -v conda >/dev/null || { echo "缺 conda; 装 miniconda3 后重试"; exit 1; }
[ -e /dev/cpu_dma_latency ] || { echo "warn: /dev/cpu_dma_latency 不存在 (PREEMPT_RT 实时内核才有, 普通内核可忽略)"; }

echo "=== 2. 创建 conda env franka-teleop (python 3.10) ==="
if conda env list | grep -q '^franka-teleop '; then
  echo "franka-teleop env 已存在, 跳过创建"
else
  conda create -n franka-teleop python=3.10 -y
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate franka-teleop

echo "=== 3. 装 Python 依赖 (requirements.txt) ==="
pip install -r "$REPO_ROOT/requirements.txt"

echo "=== 4. 装 lerobot (本项目期望在 ../lerobot 目录) ==="
if [ -d "$REPO_ROOT/../lerobot" ]; then
  echo "找到 $REPO_ROOT/../lerobot, 装 editable 模式"
  pip install -e "$REPO_ROOT/../lerobot"
else
  echo "未找到 ../lerobot, 请先 git clone:"
  echo "  cd $(dirname "$REPO_ROOT") && git clone https://github.com/huggingface/lerobot.git"
  echo "然后重运行 setup.sh"
  exit 2
fi

echo "=== 5. 装 lerobot_robot_franka + lerobot_teleoperator_franka ==="
pip install -e "$REPO_ROOT/lerobot_robot_franka"
pip install -e "$REPO_ROOT/lerobot_teleoperator_franka"

echo "=== 6. 装本项目自身 (entry_points: franka-replay / franka-visualize / ...) ==="
pip install -e "$REPO_ROOT"

echo "=== 7. 提示 polymetis 编译 ==="
echo "polymetis-local 需单独编译 (~10 分钟), 见 docs/POLYMETIS_BUILD.md"

echo "=== 8. 提示 env vars ==="
cat <<HINT
请把下面几行加到 ~/.bashrc (按你的实际路径调整):
  export FRANKA_TELEOP_ROOT=$REPO_ROOT
  export POLYMETIS_ENV=\$HOME/miniconda3/envs/polymetis-local
  export POLYMETIS_SOURCE=\$HOME/fairo-franka

然后 source ~/.bashrc, 启动 polymetis server (见 docs/POLYMETIS_BUILD.md), 跑:
  conda activate franka-teleop
  python scripts/core/run_record_hdf5_ui.py --config scripts/config/record_cfg_unityvr.yaml
HINT

echo "=== 完成! ==="
