#!/bin/bash
# 只读+良好间隔诊断: 全日志 + 单 get_state + 3 次充分间隔 goto(命令间不轮询, 各等6s再读一次)
set -u
cd /home/ubuntu/Desktop/jhli || exit 2
echo "######## _gripper_live.log 全文 ########"
cat -n _gripper_live.log
echo "######## 进程 ########"
pgrep -af 'launch_gripper\.py|franka_hand_client' | grep -v pgrep || echo "(夹爪进程没了!)"
echo "######## 间隔测试 (命令间不打 server, 各 goto 后单等6s再单读) ########"
envs/franka-teleop/bin/python - <<'PYEOF'
import sys, time, zerorpc
c = zerorpc.Client(heartbeat=30, timeout=60)
c.connect("tcp://localhost:4242")
c.gripper_initialize(); time.sleep(1.0)
s = c.gripper_get_state()
print(f"  baseline width={s['width']:.4f} is_moving={s['is_moving']} prev_ok={s['prev_command_successful']} err={s['error_code']}")
seq = [0.07, 0.01, 0.05]
prev = s['width']
moved_any = False
for w in seq:
    t0 = time.time()
    try:
        c.gripper_goto(w, 0.05, 20.0, -1.0, -1.0, True)   # blocking
    except Exception as e:
        print(f"  goto({w}) EXC {type(e).__name__}: {e}")
        continue
    dt = time.time() - t0
    time.sleep(6.0)   # 充分 settle, 期间不打 server
    s = c.gripper_get_state()
    delta = abs(s['width'] - prev)
    if delta > 0.01: moved_any = True
    print(f"  goto({w}) 返回耗时={dt:.1f}s -> settle width={s['width']:.4f} (Δ前次={delta:+.4f}) "
          f"is_moving={s['is_moving']} prev_ok={s['prev_command_successful']} err={s['error_code']}")
    prev = s['width']
print("DIAG_MOVED" if moved_any else "DIAG_NO_MOVE")
sys.exit(0 if moved_any else 1)
PYEOF
RC=$?
echo "######## goto 后 _gripper_live.log 末尾 ########"
tail -n 10 _gripper_live.log
exit $RC
