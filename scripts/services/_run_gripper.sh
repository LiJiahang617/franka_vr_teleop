#!/bin/bash
# 开源可配置: 通过 env var 覆盖路径
: "${POLYMETIS_ENV:=/home/ubuntu/Desktop/jhli/envs/polymetis-local}"
: "${POLYMETIS_SOURCE:=/home/ubuntu/Desktop/jhli/fairo-franka}"

unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH RMW_IMPLEMENTATION CMAKE_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH PKG_CONFIG_PATH
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate "${POLYMETIS_ENV}"
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -vE '^/opt/ros' | paste -sd: -)"
cd "${POLYMETIS_SOURCE}/polymetis/polymetis/python/scripts"
exec python launch_gripper.py gripper=franka_hand
