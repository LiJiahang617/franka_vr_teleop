"""集中可配路径/端口常量。默认=现有绝对路径, 可被环境变量覆盖。

整合期"jhli→jhli/lerobot_franka_teleop"路径机械改写遗留散落硬编码,
本模块为单一真值; 消费方(入口/服务/调试脚本)统一从此取。
"""
import os

JHLI_ROOT = os.environ.get("FRANKA_JHLI_ROOT", "/home/ubuntu/Desktop/jhli")
REPO_ROOT = JHLI_ROOT + "/lerobot_franka_teleop"
SCRIPTS_DIR = REPO_ROOT + "/scripts"
SERVICES_DIR = REPO_ROOT + "/scripts/services"
HDF5_EPISODES_DIR = os.environ.get(
    "FRANKA_HDF5_EPISODES_DIR", JHLI_ROOT + "/_hdf5_episodes")
LEROBOT_OUT = os.environ.get("FRANKA_LEROBOT_OUT", JHLI_ROOT + "/_lerobot_out")
OC2ARM_R_PATH = os.environ.get(
    "FRANKA_OC2ARM_R", REPO_ROOT + "/.stage3_oc2arm_R.npy")

ARM_PORT = 50051
ZERORPC_PORT = 4242
GRIPPER_PORT = 50052
