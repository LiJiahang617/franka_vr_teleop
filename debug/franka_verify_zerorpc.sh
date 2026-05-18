#!/bin/bash
# 验证 zerorpc 接口层: 等 4242 LISTEN, 然后 venv zerorpc 连续 5 次 robot_get_joint_positions 全过
set -u
cd /home/ubuntu/Desktop/jhli || exit 2
LOG=_zerorpc_iface_live.log
for i in $(seq 1 30); do
    if ss -ltn 2>/dev/null | grep -q ':4242 '; then
        echo "4242 LISTEN at iter=$i (~$((i*2))s)"
        break
    fi
    sleep 2
done
if ! ss -ltn 2>/dev/null | grep -q ':4242 '; then
    echo "ZERORPC_FAIL: 4242 never LISTEN after 60s"
    echo "---- tail log ----"; tail -n 30 "$LOG"
    exit 1
fi
sleep 2
envs/franka-teleop/bin/python - <<'PYEOF'
import sys, numpy as np, zerorpc
c = zerorpc.Client(heartbeat=20, timeout=15)
c.connect("tcp://localhost:4242")
ok = 0
for k in range(5):
    try:
        q = np.array(c.robot_get_joint_positions())
        assert q.shape == (7,), f"bad shape {q.shape}"
        print(f"[rpc {k+1}/5] OK q={np.round(q,4).tolist()}")
        ok += 1
    except Exception as e:
        print(f"[rpc {k+1}/5] FAIL {type(e).__name__}: {e}")
print("ZERORPC_OK" if ok == 5 else f"ZERORPC_FAIL ({ok}/5 passed)")
sys.exit(0 if ok == 5 else 1)
PYEOF
RC=$?
echo "---- zerorpc server log tail ----"
tail -n 20 "$LOG"
exit $RC
