#!/bin/bash
# zerorpc 启动包装 (强化版 v3)
# 开源可配置: 通过 env var 覆盖路径
: "${POLYMETIS_ENV:=/home/ubuntu/Desktop/jhli/envs/polymetis-local}"
: "${POLYMETIS_SOURCE:=/home/ubuntu/Desktop/jhli/fairo-franka}"

PORT=4242
LOCK_FILE=/tmp/zerorpc_iface.lock
LOG_TAG="[zerorpc]"
log() { echo "${LOG_TAG} $*"; }

exec 201>"${LOCK_FILE}"
if ! flock -n 201; then
    log "另一个 _run_zerorpc_iface.sh 已运行, 退出"
    exit 1
fi

_PATTERN='launch_server\.py'
_port_busy() { ss -tln 2>/dev/null | grep -q ":${PORT} "; }

_kill_all_residual() {
    local wait_term=$1
    pkill -TERM -f "${_PATTERN}" 2>/dev/null
    sleep "${wait_term}"
    pkill -9 -f "${_PATTERN}" 2>/dev/null
    sleep 1
}

# ============ Pre-flight ============
log "pre-flight 清理残留..."
_kill_all_residual 2
if _port_busy; then
    log "端口 ${PORT} 仍占用, 再清"
    _kill_all_residual 2
fi
if _port_busy; then
    log "❌ pre-flight 失败"
    exit 1
fi
log "pre-flight ✓"

# ============ Cleanup ============
PGID=""
_cleaned=0
cleanup() {
    [[ $_cleaned -eq 1 ]] && return
    _cleaned=1
    log "cleanup: 停 zerorpc..."
    if [[ -n "$PGID" ]]; then
        kill -TERM -- "-${PGID}" 2>/dev/null
        sleep 3
        kill -9 -- "-${PGID}" 2>/dev/null
        sleep 1
    fi
    _kill_all_residual 1
    log "cleanup ✓"
}
trap cleanup EXIT INT TERM HUP

unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH COLCON_PREFIX_PATH \
      ROS_PACKAGE_PATH RMW_IMPLEMENTATION CMAKE_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH PKG_CONFIG_PATH
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate "${POLYMETIS_ENV}"
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -vE '^/opt/ros' | paste -sd: -)"
cd "${POLYMETIS_SOURCE}/polymetis/polymetis/python/scripts"

log "启动 launch_server.py (独立 process group)"
setsid python launch_server.py &
PYTHON_PID=$!
PGID=$PYTHON_PID
log "launch_server PID=${PYTHON_PID}, PGID=${PGID}"

wait "${PYTHON_PID}"
EXIT_CODE=$?
log "launch_server 退出 code=${EXIT_CODE}"
exit $EXIT_CODE
