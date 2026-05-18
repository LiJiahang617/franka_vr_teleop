#!/bin/bash
# 轮询夹爪: 检查 50052 + 日志 FCI refused / 连接成功
set -u
LOG=/home/ubuntu/Desktop/jhli/_gripper_live.log
for i in $(seq 1 20); do
    if grep -qiE 'FCI refused|enable FCI mode|Connection to FCI refused' "$LOG" 2>/dev/null; then
        echo "GRIPPER_FCI_REFUSED at iter=$i"
        break
    fi
    if ss -ltn 2>/dev/null | grep -q ':50052 '; then
        echo "50052 LISTEN at iter=$i (~$((i*2))s)"
        break
    fi
    sleep 2
done
echo "---- 50052 状态 ----"
ss -ltn 2>/dev/null | grep ':50052 ' || echo "(50052 未 LISTEN)"
echo "---- launch_gripper / franka_hand 进程 ----"
pgrep -af 'launch_gripper\.py|franka_hand_client' || echo "(无进程)"
echo "---- _gripper_live.log 全文 ----"
cat "$LOG"
echo "---- verdict ----"
if grep -qiE 'FCI refused|enable FCI mode|Connection to FCI refused' "$LOG" 2>/dev/null; then
    echo "GRIPPER_FAIL_FCI_REFUSED"
    exit 1
fi
if ss -ltn 2>/dev/null | grep -q ':50052 ' && pgrep -f 'franka_hand_client' >/dev/null 2>&1; then
    echo "GRIPPER_UP"
    exit 0
fi
echo "GRIPPER_UNCLEAR"
exit 2
