#!/bin/bash
# 一次干净有序重起 (Desk 切 profile 带掉臂栈后): cleanup -> arm -> poll -> zerorpc -> verify
# 失败即 exit 非0 停下报告; 所有后台服务脚本内 setsid+disown
set -u
cd /home/ubuntu/Desktop/jhli || exit 2
LOGA=_polymetis_rw_live.log
LOGZ=_zerorpc_iface_live.log

echo "########## [1/5] cleanup ##########"
PAT='franka_panda_client|franka_hand_client|launch_robot\.py|launch_gripper\.py|launch_server\.py|run_server.*-p 50051'
pkill -TERM -f "$PAT" 2>/dev/null; sleep 4; pkill -9 -f "$PAT" 2>/dev/null; sleep 1
rm -f /tmp/polymetis_rw.lock /tmp/zerorpc_iface.lock
NPROC=$(pgrep -f "$PAT" 2>/dev/null | wc -l)
P50051=$(ss -ltn 2>/dev/null | grep -c ':50051 ')
P4242=$(ss -ltn 2>/dev/null | grep -c ':4242 ')
P50052=$(ss -ltn 2>/dev/null | grep -c ':50052 ')
echo "procs=$NPROC ports 50051=$P50051 4242=$P4242 50052=$P50052"
[ "$NPROC" -eq 0 ] && [ "$P50051" -eq 0 ] && [ "$P4242" -eq 0 ] && [ "$P50052" -eq 0 ] || { echo "CLEANUP_FAIL"; exit 1; }
echo "CLEANUP_OK"

echo "########## [2/5] launch arm ##########"
: > "$LOGA"
setsid bash /home/ubuntu/Desktop/jhli/franka_vr_teleop/scripts/services/_run_polymetis_rw.sh </dev/null >"$LOGA" 2>&1 & disown
echo "arm launched pid=$!"

echo "########## [3/5] poll arm 50051 + log ##########"
for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ':50051 ' && { echo "50051 LISTEN ~$((i*3))s"; break; }; sleep 3; done
ss -ltn 2>/dev/null | grep -q ':50051 ' || { echo "ARM_FAIL 50051 never up"; tail -n 30 "$LOGA"; exit 1; }
sleep 8
CONN=$(grep -c 'Connected\.' "$LOGA" 2>/dev/null || true)
CV=$(grep -c 'communication_constraints_violation' "$LOGA" 2>/dev/null || true)
FCI=$(grep -c 'FCI refused\|enable FCI mode' "$LOGA" 2>/dev/null || true)
CART=$(grep -c 'auto-cart-imp.*已起' "$LOGA" 2>/dev/null || true)
echo "Connected.=$CONN comm_violation=$CV FCI_refused=$FCI cart_imp_up=$CART"
if [ "${CV:-0}" -gt 0 ] || [ "${FCI:-0}" -gt 0 ]; then
    echo "ARM_FAIL comm-violation/FCI refused"; tail -n 25 "$LOGA"; exit 1
fi
[ "${CONN:-0}" -gt 0 ] || { echo "ARM_UNCLEAR no Connected. yet"; tail -n 25 "$LOGA"; exit 2; }
echo "ARM_OK"

echo "########## [4/5] launch zerorpc ##########"
: > "$LOGZ"
setsid bash /home/ubuntu/Desktop/jhli/franka_vr_teleop/scripts/services/_run_zerorpc_iface.sh </dev/null >"$LOGZ" 2>&1 & disown
echo "zerorpc launched pid=$!"
for i in $(seq 1 25); do ss -ltn 2>/dev/null | grep -q ':4242 ' && { echo "4242 LISTEN ~$((i*2))s"; break; }; sleep 2; done
ss -ltn 2>/dev/null | grep -q ':4242 ' || { echo "ZERORPC_FAIL 4242 never up"; tail -n 20 "$LOGZ"; exit 1; }
sleep 2

echo "########## [5/5] verify 5x RPC ##########"
envs/franka-teleop/bin/python - <<'PYEOF'
import sys, numpy as np, zerorpc
c = zerorpc.Client(heartbeat=20, timeout=15); c.connect("tcp://localhost:4242")
ok=0
for k in range(5):
    try:
        q=np.array(c.robot_get_joint_positions()); assert q.shape==(7,)
        print(f"[rpc {k+1}/5] OK q={np.round(q,4).tolist()}"); ok+=1
    except Exception as e:
        print(f"[rpc {k+1}/5] FAIL {type(e).__name__}: {e}")
print("ZERORPC_OK" if ok==5 else f"ZERORPC_FAIL ({ok}/5)")
sys.exit(0 if ok==5 else 1)
PYEOF
RC=$?
[ "$RC" -eq 0 ] && echo "RESTART_ALL_OK" || echo "RESTART_FAIL_AT_RPC"
exit $RC
