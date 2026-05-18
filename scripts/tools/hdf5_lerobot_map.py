"""franka-hdf5-v1 → lerobot frame 的纯映射（无 lerobot 依赖, TDD）。

lerobot hw_to_dataset_features 将所有 float 键聚合为向量：
  action float keys  → features["action"]         shape=(N_action,)
  obs float keys     → features["observation.state"]  shape=(N_obs_float,)
  obs image keys     → features["observation.images.{cam}"]  shape=(H,W,3)

本模块按此规范产出 frame dict，hdf5_to_lerobot.py 直接传给 add_frame。
"""
import cv2
import numpy as np

_AX = ["x", "y", "z", "rx", "ry", "rz"]
DOF = 7

# action 键顺序（与 action numpy array 的列顺序一致）
ACTION_KEYS = [f"delta_ee_pose.{a}" for a in _AX] + ["gripper_cmd_bin"]
# obs float 键顺序
OBS_STATE_KEYS = [f"joint_{i+1}.pos" for i in range(DOF)] + [f"ee_pose.{a}" for a in _AX] + ["gripper_norm"]


def build_feature_specs(cam_names, cam_hw=None):
    """返回 (action_hw, obs_hw)：传给 lerobot hw_to_dataset_features 的 hw 规格。

    Args:
        cam_names: 相机名称列表，例如 ["wrist", "exterior"]
        cam_hw: 各相机图像尺寸 dict，例如 {"wrist": (480, 640, 3)}。
                None 则默认 (480, 640, 3)。
    """
    action_hw = {k: float for k in ACTION_KEYS}
    obs_hw = {k: float for k in OBS_STATE_KEYS}
    for c in cam_names:
        shape = (cam_hw or {}).get(c, (480, 640, 3))
        obs_hw[c] = shape
    return action_hw, obs_hw


def _decode(jpeg_bytes):
    """解码 vlen jpeg bytes → RGB HWC numpy array。"""
    arr = np.frombuffer(bytes(jpeg_bytes), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR HWC
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def hdf5_frame_to_lerobot(h5, i, cam_names, task="task"):
    """取 franka-hdf5-v1 第 i 帧 → lerobot add_frame 用的 frame dict。

    frame dict 键格式（与 hw_to_dataset_features 输出的 features keys 一致）：
      "action"                      → np.float32 shape (N_action,)
      "observation.state"           → np.float32 shape (N_obs_float,)
      "observation.images.{cam}"    → np.uint8 HWC
      "task"                        → str
    """
    # action 向量
    d = h5["action/delta_ee_pose"][i]          # (6,)
    g = float(h5["action/gripper_cmd"][i][0])
    action = np.array(list(d) + [g], dtype=np.float32)

    # obs state 向量
    j = h5["observations/arm/joints"][i]       # (7,)
    pe = h5["observations/arm/pose"][i]        # (6,)
    gn = float(h5["observations/effector/position_norm"][i][0])
    obs_state = np.array(list(j) + list(pe) + [gn], dtype=np.float32)

    fr = {
        "action": action,
        "observation.state": obs_state,
        "task": task,
    }
    for c in cam_names:
        fr[f"observation.images.{c}"] = _decode(h5[f"observations/camera/rgb/{c}/images"][i])
    return fr
