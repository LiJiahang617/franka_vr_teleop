#!/bin/bash
# 只停夹爪 client(让出 FCI 给 Desk 做 homing); 绝不碰臂 50051 / zerorpc 4242
set -u
echo "=== 停夹爪 client (按真名, 仅 launch_gripper/franka_hand_client) ==="
pkill -TERM -f 'launch_gripper\.py|franka_hand_client' 2>/dev/null
sleep 2
pkill -9 -f 'launch_gripper\.py|franka_hand_client' 2>/dev/null
sleep 1
pgrep -af 'launch_gripper\.py|franka_hand_client' | grep -v pgrep && echo "(残留!)" || echo "夹爪 client 已停"
echo "=== 复核臂/zerorpc 未受影响 ==="
ss -ltn 2>/dev/null | grep -E ':50051 |:4242 ' || echo "(!! 臂/zerorpc 端口异常)"
pgrep -f 'franka_panda_client' >/dev/null && echo "臂 franka_panda_client 仍在 ✓" || echo "(!! 臂进程没了)"
echo "=== 50052 应已释放(让出给 Desk) ==="
ss -ltn 2>/dev/null | grep ':50052 ' && echo "(50052 仍占?)" || echo "50052 已释放 ✓"
