#!/bin/bash
# 清掉失败的 launch_gripper 包装(防 hydra 重试残留), 不碰臂/zerorpc; 复核臂栈仍健康
set -u
echo "=== kill 失败的 gripper 包装 ==="
pkill -9 -f 'launch_gripper\.py|franka_hand_client' 2>/dev/null
sleep 1
pgrep -af 'launch_gripper\.py|franka_hand_client' && echo "(仍有残留!)" || echo "gripper 进程已清"

echo "=== 臂栈/zerorpc 未受影响复核 ==="
ss -ltn 2>/dev/null | grep -E ':50051 |:4242 |:50052 ' || echo "(无监听)"
echo "--- arm log 末尾(确认无新 comm-violation) ---"
grep -cE 'communication_constraints_violation' /home/ubuntu/Desktop/jhli/_polymetis_rw_live.log 2>/dev/null | sed 's/^/comm_violation count: /'
tail -n 3 /home/ubuntu/Desktop/jhli/_polymetis_rw_live.log

echo "=== Desk 侧诊断佐证: 臂 FCI 成功 vs 夹爪 refused ==="
grep -nE 'Connected\.|Connecting to robot_ip|FCI refused|franka_hand_client|franka_panda_client' /home/ubuntu/Desktop/jhli/_polymetis_rw_live.log 2>/dev/null | tail -4
