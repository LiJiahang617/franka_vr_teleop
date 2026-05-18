#!/bin/bash
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH RMW_IMPLEMENTATION CMAKE_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH PKG_CONFIG_PATH
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ubuntu/Desktop/jhli/envs/polymetis-local
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -vE '^/opt/ros' | paste -sd: -)"
cd /home/ubuntu/Desktop/jhli/fairo-franka/polymetis/polymetis/python/scripts
exec python launch_gripper.py gripper=franka_hand
