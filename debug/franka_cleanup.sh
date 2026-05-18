#!/bin/bash
# 干净有序启动 step-0: 按真名杀净 + 删锁 + 验证端口/进程清空
# 失败 exit 非 0 -> 接续 agent 必须停下报告, 不靠重启硬试
set -u
echo "=== [cleanup] kill by real names (TERM then KILL) ==="
PAT='franka_panda_client|franka_hand_client|launch_robot\.py|launch_gripper\.py|launch_server\.py|run_server.*-p 50051'
pkill -TERM -f "$PAT" 2>/dev/null
sleep 4
pkill -9 -f "$PAT" 2>/dev/null
sleep 1

echo "=== [cleanup] rm stale locks ==="
rm -f /tmp/polymetis_rw.lock /tmp/zerorpc_iface.lock
echo "locks removed"

echo "=== [cleanup] verify clean ==="
# 排除本清理脚本/grep 自身
NPROC=$(pgrep -af "$PAT" | grep -v 'franka_cleanup.sh' | grep -vc 'pgrep' || true)
NPROC=$(pgrep -f "$PAT" 2>/dev/null | wc -l)
echo "matching procs still alive: $NPROC"
P50051=$(ss -ltn 2>/dev/null | grep -c ':50051 ' || true)
P4242=$(ss -ltn 2>/dev/null | grep -c ':4242 ' || true)
P50052=$(ss -ltn 2>/dev/null | grep -c ':50052 ' || true)
echo "ports listening -> 50051:$P50051 4242:$P4242 50052:$P50052"

if [ "$NPROC" -eq 0 ] && [ "$P50051" -eq 0 ] && [ "$P4242" -eq 0 ] && [ "$P50052" -eq 0 ]; then
    echo "CLEANUP_OK"
    exit 0
else
    echo "CLEANUP_FAIL (procs=$NPROC ports 50051=$P50051 4242=$P4242 50052=$P50052)"
    exit 1
fi
