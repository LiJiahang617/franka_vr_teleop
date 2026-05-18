#!/bin/bash
# 正确测量版: 每次 goto 后轮询到 is_moving=False(或超时)再读 settle width, 比是否跟到目标
set -u
cd /home/ubuntu/Desktop/jhli || exit 2
envs/franka-teleop/bin/python - <<'PYEOF'
import sys, time, zerorpc
c = zerorpc.Client(heartbeat=20, timeout=40)
c.connect("tcp://localhost:4242")

def get():
    return c.gripper_get_state()

def settle(tag, max_s=8.0):
    t0 = time.time()
    s = get()
    while s["is_moving"] and time.time() - t0 < max_s:
        time.sleep(0.3)
        s = get()
    print(f"  [{tag}] settled width={s['width']:.4f} is_moving={s['is_moving']} "
          f"prev_ok={s['prev_command_successful']} err={s['error_code']} ({time.time()-t0:.1f}s)")
    return s

def goto(w):
    c.gripper_goto(w, 0.05, 20.0, -1.0, -1.0, True)

try:
    c.gripper_initialize(); time.sleep(1.0)
    s0 = settle("init")
    targets = [0.06, 0.02, 0.06]
    res = []
    for w in targets:
        print(f"goto {w} ...")
        goto(w)
        s = settle(f"target={w}")
        res.append((w, s["width"], s["prev_command_successful"]))

    print("\n--- 判定 ---")
    tol = 0.008  # Franka Hand 定位容差 ~几mm
    track_ok = all(abs(meas - tgt) <= tol for tgt, meas, _ in res)
    cmd_ok = all(ok for _, _, ok in res)
    span = max(m for _, m, _ in res) - min(m for _, m, _ in res)
    print(f"init={s0['width']:.4f}; " + "; ".join(f"目标{t:.2f}→实测{m:.4f}" for t, m, _ in res))
    print(f"跟随目标(±{tol*1000:.0f}mm)={track_ok}  全 prev_ok={cmd_ok}  行程跨度={span:.4f}m")
    if track_ok and cmd_ok and span > 0.02:
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
tail -n 6 _gripper_live.log
exit $RC
