#!/bin/bash
# 只读快照: 不重启任何东西. 判断臂是"启动瞬态已自愈"还是"持续退化"
set -u
cd /home/ubuntu/Desktop/jhli || exit 2
LOGA=_polymetis_rw_live.log
echo "=== 进程 / 端口 ==="
pgrep -af 'franka_panda_client|launch_robot\.py' | grep -v pgrep || echo "(臂进程没了!)"
ss -ltn 2>/dev/null | grep ':50051 ' || echo "(50051 未 LISTEN)"
echo "=== comm-violation 计数 (总数) ==="
grep -c 'communication_constraints_violation' "$LOGA" 2>/dev/null || echo 0
echo "=== 每次 comm-violation 的时间戳 (看是否还在新增) ==="
grep -nE 'communication_constraints_violation|Robot operation recovered|unable to be controlled|Reverting to default' "$LOGA" 2>/dev/null | tail -10
echo "=== 日志最后 8 行 (看当前是否稳定) ==="
tail -n 8 "$LOGA"
echo "=== 距今: 日志最后修改时间 ==="
stat -c '%y' "$LOGA"
date '+%Y-%m-%d %H:%M:%S now'
