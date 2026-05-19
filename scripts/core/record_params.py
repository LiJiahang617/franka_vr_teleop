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


def realsense_fps(fps):
    """RealSense 相机帧率适配: pyrealsense2 enable_stream 要求 framerate 为 int,
    而单一来源 fps 是 float(循环节拍 1.0/fps 与 hdf5 元数据需要)。在相机边界转 int。
    RealSense 硬件仅支持整数帧率(15/30/60/90), 非整数 fps 本就非法。
    """
    return int(round(float(fps)))


def parse_reset_config(rec_raw: dict):
    """从 record yaml 的 raw dict 解析 reset 配置（纯函数, 可离线单测）。

    Returns (reset_between_episodes: bool, reset_wait: float)。
    严格解析: 防 yaml 引号字符串 "false" 被 bool() 误判为 True 而在用户
    以为关闭时仍执行真机 robot.reset() 回 HOME（高危静默误动作）。

    Raises ValueError: 非法 reset_between_episodes / reset_wait（非有限或<0）。
    """
    v = rec_raw.get("reset_between_episodes", True)
    if isinstance(v, bool):
        rbe = v
    elif v is None:
        rbe = True
    elif isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "on"):
            rbe = True
        elif s in ("false", "0", "no", "off"):
            rbe = False
        else:
            raise ValueError(
                f"record.reset_between_episodes 非法: {v!r}（应为 bool 或 "
                "true/false/1/0/yes/no/on/off）"
            )
    else:
        raise ValueError(
            f"record.reset_between_episodes 必须为 bool, got {type(v).__name__}={v!r}"
        )

    w = rec_raw.get("reset_wait", 1.0)
    if w is None:
        rw = 1.0
    else:
        try:
            rw = float(w)
        except (TypeError, ValueError):
            raise ValueError(f"record.reset_wait 非数字: {w!r}")
        import math as _math
        if not _math.isfinite(rw) or rw < 0:
            raise ValueError(
                f"record.reset_wait 必须为有限非负数, got {rw!r}"
            )
    return rbe, rw
