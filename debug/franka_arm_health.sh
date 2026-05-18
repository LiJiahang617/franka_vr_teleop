#!/bin/bash
# 复核臂栈/zerorpc 是否被 Desk 切 EE profile 带掉
set -u
cd /home/ubuntu/Desktop/jhli || exit 2
echo "=== 端口 ==="
ss -ltn 2>/dev/null | grep -E ':50051 |:4242 |:50052 ' || echo "(无监听)"
echo "=== 进程 ==="
pgrep -af 'franka_panda_client|launch_robot\.py|launch_server\.py' | grep -v pgrep || echo "(无臂/zerorpc 进程)"
echo "=== arm log: comm-violation / FCI refused 计数 ==="
CV=$(grep -cE 'communication_constraints_violation' _polymetis_rw_live.log 2>/dev/null || true)
FCI=$(grep -cE 'FCI refused|enable FCI mode' _polymetis_rw_live.log 2>/dev/null || true)
echo "comm_violation=$CV  FCI_refused=$FCI"
echo "--- arm log 末尾 6 行 ---"
tail -n 6 _polymetis_rw_live.log
echo "=== zerorpc 实连测试 (venv, 单次 robot_get_joint_positions) ==="
timeout 20 envs/franka-teleop/bin/python - <<'PYEOF'
import sys, numpy as np, zerorpc
try:
    c = zerorpc.Client(heartbeat=20, timeout=10)
    c.connect("tcp://localhost:4242")
    q = np.array(c.robot_get_joint_positions())
    assert q.shape == (7,), f"bad shape {q.shape}"
    print("ZERORPC_OK q=", np.round(q,4).tolist())
    sys.exit(0)
except Exception as e:
    print(f"ZERORPC_FAIL {type(e).__name__}: {e}")
    sys.exit(1)
PYEOF
RC=$?
echo "--- verdict ---"
if ss -ltn 2>/dev/null | grep -q ':50051 ' && [ "$RC" -eq 0 ] && [ "${CV:-0}" -eq 0 ] && [ "${FCI:-0}" -eq 0 ]; then
    echo "ARM_STACK_HEALTHY"
    exit 0
else
    echo "ARM_STACK_DEGRADED (需先干净重起臂栈再验夹爪)"
    exit 1
fi
