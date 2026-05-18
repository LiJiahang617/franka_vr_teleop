#!/bin/bash
set -euo pipefail

cd /home/ubuntu/Desktop/jhli/lerobot_franka_teleop
export PATH="/home/ubuntu/Desktop/jhli/platform-tools:${PATH}"
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ubuntu/Desktop/jhli/envs/franka-teleop
exec python stage3_teleop.py "$@"
