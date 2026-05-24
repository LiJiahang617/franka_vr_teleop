#!/bin/bash
# polymetis-rw 启动包装 (强化版 v3 - 用 setsid + process group, 保证带走所有子进程)
# 开源可配置: 通过 env var 覆盖路径
: "${POLYMETIS_ENV:=/home/ubuntu/Desktop/jhli/envs/polymetis-local}"
: "${POLYMETIS_SOURCE:=/home/ubuntu/Desktop/jhli/fairo-franka}"

PORT=50051
LOCK_FILE=/tmp/polymetis_rw.lock
LOG_TAG="[polymetis-rw]"

log() { echo "${LOG_TAG} $*"; }

# ============ Lock ============
exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
    log "另一个 _run_polymetis_rw.sh 已运行 (lock=${LOCK_FILE}), 退出"
    exit 1
fi
log "lock 获取"

_PATTERN_LAUNCH='launch_robot\.py.*franka_hardware'
_PATTERN_SERVER="run_server.*-p ${PORT}"
_port_busy() { ss -tln 2>/dev/null | grep -q ":${PORT} "; }

# 暴力清理: TERM → 5s → KILL → 1s
_kill_all_residual() {
    local wait_term=$1
    pkill -TERM -f "${_PATTERN_LAUNCH}" 2>/dev/null
    pkill -TERM -f "${_PATTERN_SERVER}" 2>/dev/null
    sleep "${wait_term}"
    pkill -9 -f "${_PATTERN_LAUNCH}" 2>/dev/null
    pkill -9 -f "${_PATTERN_SERVER}" 2>/dev/null
    sleep 1
}

# ============ Pre-flight ============
log "pre-flight 清理残留..."
_kill_all_residual 5
if _port_busy; then
    log "端口 ${PORT} 仍占用, 再清"
    _kill_all_residual 3
fi
if _port_busy; then
    log "❌ pre-flight 失败: 端口 ${PORT} 持续被占"
    exit 1
fi
log "pre-flight ✓"

# ============ Cleanup ============
PGID=""
_cleaned=0
cleanup() {
    [[ $_cleaned -eq 1 ]] && return
    _cleaned=1
    log "cleanup: 停 polymetis (process group)..."
    if [[ -n "$PGID" ]]; then
        # 向整个 process group 发 SIGTERM (会同时到达 python 和它 spawn 的 run_server)
        kill -TERM -- "-${PGID}" 2>/dev/null
        log "已发 SIGTERM 至 pgid=${PGID}, 等 6s 让 libfranka 抱刹车"
        sleep 6
        kill -9 -- "-${PGID}" 2>/dev/null
        sleep 1
    fi
    # 双保险: 按名字再清一次任何漏网
    _kill_all_residual 2
    if _port_busy; then
        log "⚠️  退出时端口 ${PORT} 仍占用"
    else
        log "cleanup ✓ (端口 ${PORT} 释放)"
    fi
}
trap cleanup EXIT INT TERM HUP

# ============ 环境 ============
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH COLCON_PREFIX_PATH \
      ROS_PACKAGE_PATH RMW_IMPLEMENTATION CMAKE_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH PKG_CONFIG_PATH
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate "${POLYMETIS_ENV}"
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -vE '^/opt/ros' | paste -sd: -)"
export POLYMETIS_REALTIME_NO_SUDO=1
cd "${POLYMETIS_SOURCE}/polymetis/polymetis/python/scripts"

# ============ 启动 (setsid 独立进程组, 便于 kill -- -PGID 一锅端) ============
log "启动 launch_robot.py (独立 process group)"
setsid python launch_robot.py robot_client=franka_hardware \
    robot_client.executable_cfg.readonly=false use_real_time=true &
PYTHON_PID=$!
# setsid 让 child 成为新 session 的 pgid leader, PGID = PYTHON_PID
PGID=$PYTHON_PID
log "launch_robot PID=${PYTHON_PID}, PGID=${PGID}"

# ====== 自动起默认笛卡尔阻抗 (Connected 后台自动; 软档, 与遥操作同 Kx) ======
(
  _ci_ok=0
  for _i in $(seq 1 40); do
    if _port_busy; then
      if python - <<'PYEOF'
import sys
try:
    import torch
    from polymetis import RobotInterface
    r = RobotInterface(ip_address="localhost", enforce_version=False)
    r.start_cartesian_impedance(Kx=torch.Tensor([100,100,100,40,40,40]),
                                Kxd=torch.Tensor([1,1,1,0.2,0.2,0.2]))
    sys.exit(0)
except Exception as e:
    sys.stderr.write("auto-cart-imp not-ready: %s\n" % e); sys.exit(1)
PYEOF
      then
        log "[auto-cart-imp] 默认笛卡尔阻抗已起 Kx=[100,100,100,40,40,40] Kxd=[1,1,1,0.2,0.2,0.2]"
        _ci_ok=1; break
      fi
    fi
    sleep 3
  done
  [ "$_ci_ok" = 1 ] || log "[auto-cart-imp] WARN 超时未能起默认笛卡尔阻抗(launch 可能失败)"
) &


wait "${PYTHON_PID}"
EXIT_CODE=$?
log "launch_robot 退出 code=${EXIT_CODE}"
exit $EXIT_CODE
