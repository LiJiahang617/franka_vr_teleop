#!/bin/bash
# 干净有序启动 step-1: 后台拉起臂栈 (_run_polymetis_rw.sh, 端口 50051)
# 脚本内 setsid+disown -> 即使 ssh 断开服务存活; 立即返回, 轮询由后续单独 ssh 做
set -u
cd /home/ubuntu/Desktop/jhli || exit 2
: > _polymetis_rw_live.log
setsid bash /home/ubuntu/Desktop/jhli/franka_vr_teleop/scripts/services/_run_polymetis_rw.sh </dev/null >_polymetis_rw_live.log 2>&1 &
disown
echo "ARM_LAUNCHED pid=$! log=/home/ubuntu/Desktop/jhli/_polymetis_rw_live.log"
