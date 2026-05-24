#!/bin/bash
# 干净有序启动 step-2: 后台拉起 zerorpc 接口层 (_run_zerorpc_iface.sh, 端口 4242)
set -u
cd /home/ubuntu/Desktop/jhli || exit 2
: > _zerorpc_iface_live.log
setsid bash /home/ubuntu/Desktop/jhli/franka_vr_teleop/scripts/services/_run_zerorpc_iface.sh </dev/null >_zerorpc_iface_live.log 2>&1 &
disown
echo "ZERORPC_LAUNCHED pid=$! log=/home/ubuntu/Desktop/jhli/_zerorpc_iface_live.log"
