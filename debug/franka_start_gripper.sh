#!/bin/bash
# 步骤4: 后台拉起夹爪 (_run_gripper.sh -> launch_gripper.py gripper=franka_hand, 端口 50052)
# _run_gripper.sh 内部无 setsid/lock, 由本脚本 setsid+disown 包起; 立即返回
set -u
cd /home/ubuntu/Desktop/jhli || exit 2
# 清残留 launch_gripper / franka_hand_client (按真名, 不碰臂/zerorpc)
pkill -9 -f 'launch_gripper\.py|franka_hand_client' 2>/dev/null
sleep 1
: > _gripper_live.log
setsid bash /home/ubuntu/Desktop/jhli/franka_vr_teleop/scripts/services/_run_gripper.sh </dev/null >_gripper_live.log 2>&1 &
disown
echo "GRIPPER_LAUNCHED pid=$! log=/home/ubuntu/Desktop/jhli/_gripper_live.log"
