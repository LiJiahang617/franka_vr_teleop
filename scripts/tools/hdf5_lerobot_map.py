"""franka-hdf5-v2 aligned dict → lerobot frame 的纯映射（无 lerobot 依赖, TDD）。

v2 接口变更（相对 v1）：
  - 核心接口从 per-frame 改为 per-episode：接收 align_offline 产出的 aligned dict
  - observation.state 和 action 均为 realman 14D 布局
  - action = next-state：action[i] = state[i+1]，末帧复制末帧 state

realman 14D 布局（observation.state 与 action 字段名/顺序完全一致）：
  [0-6]  joint_1_rad..joint_7_rad   ← aligned["arm_joints"][:, 0:7]
  [7]    gripper_open               ← aligned["gripper_position_norm"][:, 0]
  [8-10] eef_pos_x_m..eef_pos_z_m  ← aligned["arm_pose"][:, 0:3]
  [11-13] eef_rot_euler_x_rad..eef_rot_euler_z_rad  ← aligned["arm_pose"][:, 3:6]

lerobot hw_to_dataset_features 将所有 float 键聚合为向量：
  action float keys  → features["action"]              shape=(14,)
  obs float keys     → features["observation.state"]   shape=(14,)
  obs image keys     → features["observation.images.{cam}"]  shape=(H,W,3)

本模块按此规范产出 frame dict 和 episode 级数组，hdf5_to_lerobot*.py 调用。
"""
import cv2
import numpy as np

# observation.state 字段名（顺序即此，14D，与 realman 逐字一致）
OBS_STATE_NAMES = [
    "joint_1_rad", "joint_2_rad", "joint_3_rad", "joint_4_rad",
    "joint_5_rad", "joint_6_rad", "joint_7_rad",
    "gripper_open",
    "eef_pos_x_m", "eef_pos_y_m", "eef_pos_z_m",
    "eef_rot_euler_x_rad", "eef_rot_euler_y_rad", "eef_rot_euler_z_rad",
]

# action 字段名与 observation.state 完全相同（action = next-state）
ACTION_NAMES = list(OBS_STATE_NAMES)

# 维度常量
STATE_DIM = 14
ACTION_DIM = 14

# 兼容旧接口名（供外部仍引用的代码平滑过渡，不建议新代码使用）
OBS_STATE_KEYS = OBS_STATE_NAMES
ACTION_KEYS = ACTION_NAMES


def build_feature_specs(cam_names, cam_hw=None):
    """返回 (action_hw, obs_hw)：传给 lerobot hw_to_dataset_features 的 hw 规格。

    Args:
        cam_names: 相机名称列表，例如 ["wrist", "exterior"]
        cam_hw: 各相机图像尺寸 dict，例如 {"wrist": (480, 640, 3)}。
                None 则默认 (480, 640, 3)。

    Returns:
        (action_hw, obs_hw) 元组，各为 {key: float 或 (H,W,C)} dict
    """
    action_hw = {k: float for k in ACTION_NAMES}
    obs_hw = {k: float for k in OBS_STATE_NAMES}
    for c in cam_names:
        shape = (cam_hw or {}).get(c, (480, 640, 3))
        obs_hw[c] = shape
    return action_hw, obs_hw


def _decode(jpeg_bytes):
    """解码 vlen jpeg bytes → RGB HWC numpy array。

    Raises:
        ValueError: jpeg 数据损坏，cv2.imdecode 返回 None 时抛出。
    """
    arr = np.frombuffer(bytes(jpeg_bytes), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR HWC
    if img is None:
        raise ValueError(
            f"cv2.imdecode 返回 None：jpeg 数据损坏或格式不支持（数据长度 {len(arr)} 字节）"
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_state_array(aligned: dict) -> np.ndarray:
    """从 aligned dict 构建 observation.state 数组 (N, 14) float32（realman 14D 布局）。

    布局（索引对应 OBS_STATE_NAMES）：
      [0:7]   arm_joints[:, 0:7]             ← joint_1_rad..joint_7_rad
      [7]     gripper_position_norm[:, 0]    ← gripper_open
      [8:11]  arm_pose[:, 0:3]              ← eef_pos_xyz（位置，单位 m）
      [11:14] arm_pose[:, 3:6]              ← eef_rot_euler_xyz（欧拉角，rad）

    Args:
        aligned: align_offline.align_by_image_timestamp 返回的 aligned dict

    Returns:
        state_array (N, 14) float32

    Raises:
        ValueError: arm_joints/gripper_position_norm/arm_pose 长度不一致，或 gripper 不是 (N,1)
    """
    joints = aligned["arm_joints"]           # (N, 7)
    gripper = aligned["gripper_position_norm"]  # (N, 1)
    pose = aligned["arm_pose"]               # (N, 6) [px, py, pz, rx, ry, rz]
    N = len(joints)

    # 轻量 shape 校验
    if len(gripper) != N or len(pose) != N:
        raise ValueError(
            f"aligned 各模态长度不一致：arm_joints={N}, "
            f"gripper_position_norm={len(gripper)}, arm_pose={len(pose)}"
        )
    if gripper.ndim != 2 or gripper.shape[1] != 1:
        raise ValueError(
            f"gripper_position_norm 应为 (N, 1) 二维数组，实际 shape: {gripper.shape}"
        )

    state = np.empty((N, STATE_DIM), dtype=np.float32)
    state[:, 0:7] = joints.astype(np.float32)
    state[:, 7] = gripper[:, 0].astype(np.float32)
    state[:, 8:11] = pose[:, 0:3].astype(np.float32)
    state[:, 11:14] = pose[:, 3:6].astype(np.float32)
    return state


def build_action_array(state: np.ndarray) -> np.ndarray:
    """从 state (N, 14) 构建 next-state action (N, 14) float32。

    action[i] = state[i+1]（i < N-1）；
    action[N-1] = state[N-1]（末帧复制，与 realman 行为一致）。
    N=0 时直接返回空数组，不执行末帧复制。

    Args:
        state: observation.state 数组 (N, 14) float32

    Returns:
        action_array (N, 14) float32
    """
    N = state.shape[0]
    assert state.shape[1] == ACTION_DIM, (
        f"state 列数应为 ACTION_DIM={ACTION_DIM}，实际: {state.shape[1]}"
    )
    if N == 0:
        return np.empty((0, ACTION_DIM), dtype=np.float32)
    action = np.empty_like(state)
    if N > 1:
        action[:-1] = state[1:]   # next-state
    action[-1] = state[-1]        # 末帧复制
    return action


def episode_to_lerobot_arrays(aligned: dict, h5, cam_names: list, task: str = "task"):
    """将 aligned dict + hdf5 图像数据转换为整个 episode 的 lerobot 数组。

    图像读取使用 aligned["anchor_indices"] 作为原始 hdf5 图像数组的下标索引，
    确保 drop 模式下图像与 state/action 时间上对齐（每个输出帧 i 的图像来自
    原始下标 anchor_indices[i]，而非简单的 ds[i]）。

    这是核心 per-episode 接口（替代旧版 per-frame 的 hdf5_frame_to_lerobot）。
    调用方负责打开 h5py.File 并传入。

    Args:
        aligned: align_offline.align_by_image_timestamp 返回的 aligned dict
        h5: 已打开的 h5py.File 对象（用于读取相机图像）
        cam_names: 相机名称列表，例如 ["wrist", "exterior"]
        task: 任务描述字符串（写入 lerobot task 字段）

    Returns:
        dict with keys:
          "state"  : np.ndarray (N, 14) float32
          "action" : np.ndarray (N, 14) float32
          "images" : {cam_name: list of np.ndarray HWC uint8} (N frames per cam)
          "task"   : str
          "N"      : int 帧数

    Raises:
        ValueError: 某相机 hdf5 图像帧数不足以覆盖 anchor_indices 最大值时抛出。
    """
    state = build_state_array(aligned)
    action = build_action_array(state)
    N = state.shape[0]

    # anchor_indices：各输出帧在原始 hdf5 图像数组中的下标
    # 旧 aligned dict（无此键）兼容：退化为 0..N-1
    anchor_indices = aligned.get("anchor_indices", np.arange(N, dtype=np.intp))

    images = {}
    for c in cam_names:
        ds = h5[f"observations/camera/rgb/{c}/images"]
        n_ds = ds.shape[0]
        # 检查原始帧数足以覆盖所有 anchor_indices
        if len(anchor_indices) > 0 and int(anchor_indices.max()) >= n_ds:
            raise ValueError(
                f"相机 {c!r} 的 hdf5 图像帧数 {n_ds} 不足以覆盖 "
                f"anchor_indices 最大值 {int(anchor_indices.max())}（0-based）"
            )
        imgs = []
        for idx in anchor_indices:
            imgs.append(_decode(ds[int(idx)]))
        images[c] = imgs

    return {
        "state": state,
        "action": action,
        "images": images,
        "task": task,
        "N": N,
    }
