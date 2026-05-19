"""hdf5 录制入口：复用 run_record 的 robot/teleop/相机构造，sink=HDF5EpisodeWriter。

与既有 run_record.py 并存，不改其逻辑。
读取同一份 record_cfg.yaml，用 RecordConfig 解析配置，
把 LeRobotDataset sink 替换为 HDF5EpisodeWriter 写 franka-hdf5-v1。

观测字段对齐说明（来自 franka.py get_observation 实读）：
  - joint 位置: joint_1.pos ... joint_7.pos (float，单独 key)
  - joint 速度: get_observation 内已注释掉，写盘时补零
  - ee pose:   ee_pose.x/y/z/rx/ry/rz (float，单独 key)
  - 夹爪状态:  gripper_state_norm ([0,1]), gripper_max_open 来自 cfg
  - 夹爪指令:  gripper_cmd_bin (get_action 返回)
  - 相机图像:  cam.read() 返回 numpy array，用 cv2.imencode 编码为 jpeg bytes
"""
import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop")
sys.path.insert(0, "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop/scripts")

from core.hdf5_writer import HDF5EpisodeWriter
from core.record_params import resolve_record_fps, extract_joint_vel

# 复用既有 run_record 的 RecordConfig 和构造工具
from run_record import RecordConfig

from lerobot_robot_franka import FrankaConfig, Franka
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot_teleoperator_franka import create_teleop

log = logging.getLogger("rec_hdf5")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def build_robot_and_teleop(record_cfg: RecordConfig, fps: float):
    """按既有 run_record.py 同款构造 robot 和 teleop。

    Returns:
        (robot, teleop, gripper_max_open)
    """
    # 相机配置（与 run_record.py 完全一致）
    wrist_image_cfg = RealSenseCameraConfig(
        serial_number_or_name=record_cfg.wrist_cam_serial,
        fps=fps,
        width=record_cfg.width,
        height=record_cfg.height,
        color_mode=ColorMode.RGB,
        use_depth=False,
        rotation=Cv2Rotation.NO_ROTATION,
    )
    exterior_image_cfg = RealSenseCameraConfig(
        serial_number_or_name=record_cfg.exterior_cam_serial,
        fps=fps,
        width=record_cfg.width,
        height=record_cfg.height,
        color_mode=ColorMode.RGB,
        use_depth=False,
        rotation=Cv2Rotation.NO_ROTATION,
    )
    camera_config = {"wrist_image": wrist_image_cfg, "exterior_image": exterior_image_cfg}

    # 机器人配置（与 run_record.py 完全一致）
    robot_config = FrankaConfig(
        robot_ip=record_cfg.robot_ip,
        cameras=camera_config,
        debug=record_cfg.debug,
        close_threshold=record_cfg.close_threshold,
        use_gripper=record_cfg.use_gripper,
        gripper_reverse=record_cfg.gripper_reverse,
        gripper_bin_threshold=record_cfg.gripper_bin_threshold,
        gripper_max_open=record_cfg.gripper_max_open,
        control_mode=record_cfg.control_mode,
        execute_mode=record_cfg.execute_mode,
    )
    robot = Franka(robot_config)

    # teleop 配置（与 run_record.py 完全一致）
    teleop_config = record_cfg.create_teleop_config()
    teleop = create_teleop(teleop_config)

    robot.connect()
    teleop.connect()

    return robot, teleop, record_cfg.gripper_max_open


def _encode_jpg(img: np.ndarray) -> np.ndarray:
    """将 numpy 图像数组编码为 jpeg bytes (uint8 array)。"""
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg') 失败")
    return np.frombuffer(buf.tobytes(), np.uint8)


def record_episode(robot, teleop, writer: HDF5EpisodeWriter,
                   fps: float, max_sec: float, gripper_max_open: float,
                   cam_names: list):
    """录制一个 episode，每 tick 收 action/obs 拼 frame 写入 writer。

    Args:
        robot: Franka 实例
        teleop: teleop 实例
        writer: HDF5EpisodeWriter（已初始化，未 close）
        fps: 目标帧率
        max_sec: 最长录制时间（秒）
        gripper_max_open: 夹爪最大开度（米），用于将 norm 转换为 gripper_m
        cam_names: 相机名列表，需与 writer cam_names 一致
    """
    period = 1.0 / fps
    t_end = time.monotonic() + max_sec
    while time.monotonic() < t_end:
        t0 = time.monotonic()

        # 采集 teleop action
        action = teleop.get_action()

        # 发送 action 到机器人
        robot.send_action(action)

        # 采集机器人观测（包含相机图像）
        obs = robot.get_observation()

        # 拼接 joint 位置数组（joint_1.pos ... joint_7.pos）
        joints = np.array([obs[f"joint_{i+1}.pos"] for i in range(7)], dtype=np.float64)

        # joint_vel 在 get_observation 中已注释掉，补零
        joint_vel = np.zeros(7, dtype=np.float64)

        # ee_pose 数组
        ee_pose = np.array(
            [obs[f"ee_pose.{ax}"] for ax in ["x", "y", "z", "rx", "ry", "rz"]],
            dtype=np.float64,
        )

        # 夹爪状态：gripper_state_norm * gripper_max_open → gripper_m
        gripper_norm = float(obs.get("gripper_state_norm") or 0.0)
        gripper_m = gripper_norm * gripper_max_open

        # 夹爪指令
        gripper_cmd = float(action.get("gripper_cmd_bin", 0.0))

        # delta_ee_pose action 数组
        delta_ee_pose = np.array(
            [action.get(f"delta_ee_pose.{ax}", 0.0) for ax in ["x", "y", "z", "rx", "ry", "rz"]],
            dtype=np.float64,
        )

        # 相机图像：cam.read() 返回 numpy array，用 cv2.imencode 编码
        cams = {}
        for cn in cam_names:
            img = obs.get(cn)
            if img is not None and isinstance(img, np.ndarray):
                cams[cn] = _encode_jpg(img)
            else:
                # 占位（相机未就绪时）
                cams[cn] = np.zeros((4,), np.uint8)

        writer.add(dict(
            ts=time.monotonic(),
            joints=joints,
            joint_vel=joint_vel,
            ee_pose=ee_pose,
            gripper_m=gripper_m,
            gripper_norm=gripper_norm,
            gripper_cmd=gripper_cmd,
            delta_ee_pose=delta_ee_pose,
            cams=cams,
        ))

        dt = period - (time.monotonic() - t0)
        if dt > 0:
            time.sleep(dt)


def main():
    ap = argparse.ArgumentParser(description="hdf5 录制入口（franka-hdf5-v1）")
    ap.add_argument("--config", required=True, help="record_cfg.yaml 路径")
    ap.add_argument("--fps", type=float, default=None, help="录制帧率(默认读 cfg.fps; 给了则临时覆盖)")
    ap.add_argument("--episodes", type=int, default=1, help="录制 episode 数")
    ap.add_argument("--episode-sec", type=float, default=60.0, help="每 episode 最长时间（秒）")
    ap.add_argument("--out-dir", default="/home/ubuntu/Desktop/jhli/_hdf5_episodes",
                    help="输出目录")
    ap.add_argument("--task-name", default="task", help="任务名称写入 hdf5")
    # 标定文件（oc2base_R），Task3 用；此处允许缺失并用单位矩阵占位
    ap.add_argument("--oc2base-R", default=None,
                    help="oc2base_R .npy 路径（缺失则用单位矩阵）")
    a = ap.parse_args()

    with open(a.config) as fh:
        raw = yaml.safe_load(fh)
    record_cfg = RecordConfig(raw["record"])
    fps = resolve_record_fps(a.fps, record_cfg.fps)
    log.info(f"[REC] 录制频率单一来源 fps={fps}（相机/循环/写盘同源）")

    # 标定矩阵
    if a.oc2base_R and os.path.exists(a.oc2base_R):
        R = np.load(a.oc2base_R)
    else:
        log.warning("[REC] oc2base_R 未提供，使用单位矩阵占位")
        R = np.eye(3)

    robot, teleop, gripper_max_open = build_robot_and_teleop(record_cfg, fps)
    os.makedirs(a.out_dir, exist_ok=True)

    # 相机名与 HDF5 schema 对应：wrist_image, exterior_image
    cam_names = list(robot.cameras.keys())
    log.info(f"[REC] 检测到相机: {cam_names}")

    try:
        for ep in range(a.episodes):
            path = f"{a.out_dir}/ep{ep:04d}_{int(time.time())}.h5"
            w = HDF5EpisodeWriter(
                path=path,
                task_name=a.task_name,
                target_fps=fps,
                oc2base_R=R,
                quality={},
                vr_source=record_cfg.control_mode,
                cam_names=cam_names,
            )
            log.info(f"[REC] episode {ep} → {path}，录制 {a.episode_sec}s")
            record_episode(robot, teleop, w, fps, a.episode_sec,
                           gripper_max_open, cam_names)
            w.close()
            log.info(f"[REC] episode {ep} 写盘+自检通过")
    finally:
        robot.disconnect()
        teleop.disconnect()


if __name__ == "__main__":
    main()
