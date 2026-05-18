#!/bin/bash
# HANDOFF §4 步骤4: 经 zerorpc 4242 实测夹爪受控真动
# gripper_initialize -> get_state -> goto 0.04 -> get_state -> goto 0.0 -> get_state -> 复位 0.04
set -u
cd /home/ubuntu/Desktop/jhli || exit 2
envs/franka-teleop/bin/python - <<'PYEOF'
import sys, time, zerorpc
c = zerorpc.Client(heartbeat=20, timeout=40)
c.connect("tcp://localhost:4242")

def st(tag):
    s = c.gripper_get_state()
    print(f"  [{tag}] width={s['width']:.4f} is_moving={s['is_moving']} "
          f"is_grasped={s['is_grasped']} prev_ok={s['prev_command_successful']} err={s['error_code']}")
    return s

try:
    print("1) gripper_initialize ...")
    c.gripper_initialize()
    time.sleep(1.0)
    s0 = st("init")
    w0 = s0["width"]

    print("2) goto 0.04 (半开) ...")
    c.gripper_goto(0.04, 0.05, 20.0, -1.0, -1.0, True)
    time.sleep(0.5)
    s1 = st("after 0.04")

    print("3) goto 0.0 (闭合) ...")
    c.gripper_goto(0.0, 0.05, 20.0, -1.0, -1.0, True)
    time.sleep(0.5)
    s2 = st("after 0.0")

    print("4) 复位 goto 0.04 (留半开中性) ...")
    c.gripper_goto(0.04, 0.05, 20.0, -1.0, -1.0, True)
    time.sleep(0.5)
    s3 = st("after reopen 0.04")

    moved = abs(s1["width"] - s2["width"]) > 0.01 or abs(w0 - s1["width"]) > 0.005
    allok = all(x["prev_command_successful"] for x in (s1, s2, s3))
    print(f"\nwidth 轨迹: init={w0:.4f} -> 0.04={s1['width']:.4f} -> 0.0={s2['width']:.4f} -> reopen={s3['width']:.4f}")
    print(f"width 真变={moved}  全部 prev_command_successful={allok}")
    if moved and allok:
        print("GRIPPER_VERIFY_OK")
        sys.exit(0)
    print("GRIPPER_VERIFY_FAIL")
    sys.exit(1)
except Exception as e:
    print(f"GRIPPER_VERIFY_EXC {type(e).__name__}: {e}")
    sys.exit(2)
PYEOF
RC=$?
echo "---- _gripper_live.log 末尾 ----"
tail -n 8 _gripper_live.log
exit $RC
