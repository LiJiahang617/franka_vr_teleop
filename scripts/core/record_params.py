"""录制超参纯函数(可单测, 不依赖硬件/lerobot)。"""
import math

import numpy as np


def resolve_record_fps(cli_fps, cfg_fps):
    """录制频率单一来源: CLI 给了用 CLI(临时覆盖), 否则用 cfg(唯一真值)。
    相机 fps / 循环节拍 / hdf5 target_fps 都应取本函数结果, 保证同源一致。
    """
    fps = float(cli_fps) if cli_fps is not None else float(cfg_fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"record fps 必须为有限正数, 得到 {fps}")
    return fps


def extract_joint_vel(obs, dof=7):
    """从 get_observation 的 obs 取 joint 速度; 缺失(未接通)则零填(向后兼容)。"""
    if all(f"joint_{i+1}.vel" in obs for i in range(dof)):
        return np.array([float(obs[f"joint_{i+1}.vel"]) for i in range(dof)],
                        dtype=np.float64)
    return np.zeros(dof, dtype=np.float64)
