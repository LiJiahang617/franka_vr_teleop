#!/bin/bash
# 轮询臂栈: 等 50051 LISTEN, 然后检查日志 Connected. 且无 comm-violation / FCI refused
set -u
LOG=/home/ubuntu/Desktop/jhli/_polymetis_rw_live.log
for i in $(seq 1 40); do
    if ss -ltn 2>/dev/null | grep -q ':50051 '; then
        echo "50051 LISTEN at iter=$i (~$((i*3))s)"
        break
    fi
    sleep 3
done
if ! ss -ltn 2>/dev/null | grep -q ':50051 '; then
    echo "ARM_FAIL: 50051 never LISTEN after 120s"
    echo "---- tail log ----"; tail -n 40 "$LOG"
    exit 1
fi
# 50051 起来后再给笛卡尔阻抗/Connected 一点时间
sleep 8
echo "---- grep 关键状态 ----"
CONN=$(grep -c 'Connected\.' "$LOG" 2>/dev/null || echo 0)
CV=$(grep -c 'communication_constraints_violation' "$LOG" 2>/dev/null || echo 0)
FCI=$(grep -c 'FCI refused\|enable FCI mode' "$LOG" 2>/dev/null || echo 0)
CART=$(grep -c 'auto-cart-imp' "$LOG" 2>/dev/null || echo 0)
echo "Connected.=$CONN  comm_violation=$CV  FCI_refused=$FCI  auto-cart-imp_lines=$CART"
echo "---- tail -n 50 log ----"
tail -n 50 "$LOG"
echo "---- verdict ----"
if [ "$CV" -gt 0 ] || [ "$FCI" -gt 0 ]; then
    echo "ARM_FAIL: comm-violation 或 FCI refused 出现 (真机 RT/FCI 仍僵, 需用户复位, 勿churn)"
    exit 1
fi
if [ "$CONN" -gt 0 ]; then
    echo "ARM_OK"
    exit 0
fi
echo "ARM_UNCLEAR: 50051 LISTEN 但日志暂无 'Connected.' — 可能仍在抱闸/起阻抗, 见上方 tail"
exit 2
