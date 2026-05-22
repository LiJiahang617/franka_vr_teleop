"""tests/test_hdf5_lerobot_map.py — Task 6 TDD：hdf5_lerobot_map v2 接口测试。

测试覆盖：
  - build_feature_specs 返回 14D action_hw + 14D obs_hw + 相机 hw
  - build_state_array 从 aligned dict 组装 (N, 14) float32（realman 布局）
  - build_action_array 实现 next-state 语义
  - episode_to_lerobot_arrays 返回完整 episode 数组
  - OBS_STATE_NAMES / ACTION_NAMES 字段名正确（与 realman 逐字一致）
"""
import sys
import numpy as np
import h5py
import cv2
import pytest

sys.path.insert(0, "/home/ubuntu/Desktop/jhli")
sys.path.insert(0, "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop/scripts")

from tools.hdf5_lerobot_map import (
    build_feature_specs,
    build_state_array,
    build_action_array,
    episode_to_lerobot_arrays,
    OBS_STATE_NAMES,
    ACTION_NAMES,
    STATE_DIM,
    ACTION_DIM,
)


# ──────────────────────────────────────────────────────────────────────────────
# 合成 v2 hdf5 生成器（供测试用）
# ──────────────────────────────────────────────────────────────────────────────

def _mk_v2(p, N=4, cams=("wrist",), img_hw=(8, 8)):
    """生成最小合规 franka-hdf5-v2 文件。img_hw=(H,W)。"""
    import franka_hdf5_schema as S

    H, W = img_hw
    img = np.zeros((H, W, 3), np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    jb = np.frombuffer(enc.tobytes(), np.uint8)
    ts = np.arange(N, dtype=np.float64)  # 时间轴：0..N-1，严格递增

    with h5py.File(p, "w") as f:
        inf = f.create_group("infos")
        inf.create_dataset("schema_version", data=np.bytes_(S.SCHEMA_VERSION))
        ti = inf.create_group("task_info")
        ti.create_dataset("task_name", data=np.bytes_("pick"))
        ti.create_dataset("collection_frequency", data=np.array([30.0, 30.0], np.float64))
        ti.create_dataset("total_frames", data=np.int64(N))
        ti.create_dataset("robot", data=np.bytes_("franka_panda"))
        inf.create_group("camera_params")
        cal = inf.create_group("calibration")
        cal.create_dataset("oc2base_R", data=np.eye(3, dtype=np.float64))
        cal.create_dataset("quality", data=np.bytes_("{}"))
        cal.create_dataset("vr_source", data=np.bytes_("unity"))

        obs = f.create_group("observations")

        arm = obs.create_group("arm")
        arm.create_dataset("joints", data=np.arange(N * 7, dtype=np.float64).reshape(N, 7) * 0.1)
        arm.create_dataset("joint_vel", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("pose", data=np.arange(N * 6, dtype=np.float64).reshape(N, 6) * 0.2)
        arm.create_dataset("timestamp", data=ts.copy())
        arm.create_dataset("stale", data=np.zeros(N, dtype=bool))

        eff = obs.create_group("effector")
        eff.create_dataset("position", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("position_norm", data=(np.ones((N, 1)) * 0.5))
        eff.create_dataset("type", data=np.array([b"gripper"] * N, dtype=h5py.special_dtype(vlen=bytes)))
        eff.create_dataset("timestamp", data=ts.copy())
        eff.create_dataset("stale", data=np.zeros(N, dtype=bool))

        cam = obs.create_group("camera")
        rgb = cam.create_group("rgb")
        for c in cams:
            g = rgb.create_group(c)
            d = g.create_dataset("images", (N,), dtype=h5py.special_dtype(vlen=np.dtype("uint8")))
            for i in range(N):
                d[i] = jb
            g.create_dataset("timestamp", data=ts.copy())
            g.create_dataset("stale", data=np.zeros(N, dtype=bool))
            g.create_dataset("hw_timestamp", data=ts.copy())

        hf = obs.create_group("state_hifreq")
        for k, sh in [("joints", (0, 7)), ("joint_vel", (0, 7)), ("pose", (0, 6)),
                      ("timestamp", (0,)), ("poly_ts", (0,))]:
            hf.create_dataset(k, data=np.zeros(sh, np.float64))
        hf.create_dataset("wrench", data=np.zeros((0, 6), np.float64))

        act = f.create_group("action")
        act.create_dataset("delta_ee_pose", data=np.arange(N * 6, dtype=np.float64).reshape(N, 6) * 0.05)
        act.create_dataset("gripper_cmd", data=np.ones((N, 1), np.float64))
        act.create_dataset("timestamp", data=ts.copy() + 0.001)  # action ts 略大，严格递增


# ──────────────────────────────────────────────────────────────────────────────
# 测试：字段名常量
# ──────────────────────────────────────────────────────────────────────────────

def test_state_names_count():
    """OBS_STATE_NAMES 应包含 14 个字段名。"""
    assert len(OBS_STATE_NAMES) == 14


def test_action_names_equals_state_names():
    """ACTION_NAMES 应与 OBS_STATE_NAMES 完全相同（action = next-state 语义）。"""
    assert ACTION_NAMES == OBS_STATE_NAMES


def test_state_names_content():
    """OBS_STATE_NAMES 包含 realman 规范的所有字段名（顺序精确）。"""
    expected = [
        "joint_1_rad", "joint_2_rad", "joint_3_rad", "joint_4_rad",
        "joint_5_rad", "joint_6_rad", "joint_7_rad",
        "gripper_open",
        "eef_pos_x_m", "eef_pos_y_m", "eef_pos_z_m",
        "eef_rot_euler_x_rad", "eef_rot_euler_y_rad", "eef_rot_euler_z_rad",
    ]
    assert OBS_STATE_NAMES == expected, f"OBS_STATE_NAMES 与 realman 规范不一致: {OBS_STATE_NAMES}"


def test_dim_constants():
    """STATE_DIM / ACTION_DIM 均为 14。"""
    assert STATE_DIM == 14
    assert ACTION_DIM == 14


# ──────────────────────────────────────────────────────────────────────────────
# 测试：build_feature_specs
# ──────────────────────────────────────────────────────────────────────────────

def test_feature_specs_action_keys():
    """action_hw 应包含 14 个 float 键，对应 ACTION_NAMES。"""
    a_hw, _ = build_feature_specs(cam_names=["wrist"])
    assert set(a_hw.keys()) == set(ACTION_NAMES), \
        f"action_hw 键集 != ACTION_NAMES: {set(a_hw.keys())}"
    assert all(v is float for v in a_hw.values()), "action_hw 所有值应为 float"


def test_feature_specs_obs_state_keys():
    """obs_hw 应包含 14 个 float 状态键 + 相机 shape 键。"""
    a_hw, o_hw = build_feature_specs(cam_names=["wrist"])
    for k in OBS_STATE_NAMES:
        assert k in o_hw, f"obs_hw 缺状态键 {k!r}"
        assert o_hw[k] is float, f"obs_hw[{k!r}] 应为 float，实际: {o_hw[k]}"


def test_feature_specs_cam_default_shape():
    """相机默认 shape 为 (480, 640, 3)。"""
    _, o_hw = build_feature_specs(cam_names=["wrist"])
    assert o_hw["wrist"] == (480, 640, 3)


def test_feature_specs_cam_custom_shape():
    """相机 shape 可通过 cam_hw 覆盖。"""
    _, o_hw = build_feature_specs(cam_names=["wrist"], cam_hw={"wrist": (240, 424, 3)})
    assert o_hw["wrist"] == (240, 424, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 测试：build_state_array
# ──────────────────────────────────────────────────────────────────────────────

def test_build_state_array_shape():
    """build_state_array 返回 (N, 14) float32。"""
    N = 5
    aligned = {
        "arm_joints": np.zeros((N, 7)),
        "gripper_position_norm": np.zeros((N, 1)),
        "arm_pose": np.zeros((N, 6)),
    }
    state = build_state_array(aligned)
    assert state.shape == (N, 14), f"state shape 应为 ({N},14)，实际: {state.shape}"
    assert state.dtype == np.float32, f"state dtype 应为 float32，实际: {state.dtype}"


def test_build_state_array_layout():
    """验证 14D realman 布局：joint[0:7] | gripper[7] | pos[8:11] | rot[11:14]。"""
    N = 3
    joints = np.arange(N * 7, dtype=np.float64).reshape(N, 7)  # 0..6, 7..13, ...
    gripper = np.full((N, 1), 0.75)
    pose = np.arange(N * 6, dtype=np.float64).reshape(N, 6) * 0.1

    aligned = {
        "arm_joints": joints,
        "gripper_position_norm": gripper,
        "arm_pose": pose,
    }
    state = build_state_array(aligned)

    # joint_1_rad..joint_7_rad（索引 0-6）来自 arm_joints
    np.testing.assert_allclose(state[:, 0:7], joints.astype(np.float32), atol=1e-6,
                                err_msg="state[0:7] 应为 arm_joints")

    # gripper_open（索引 7）来自 gripper_position_norm[:, 0]
    np.testing.assert_allclose(state[:, 7], np.full(N, 0.75, dtype=np.float32), atol=1e-6,
                                err_msg="state[7] 应为 gripper_position_norm[:,0]")

    # eef_pos_xyz（索引 8-10）来自 arm_pose[:, 0:3]
    np.testing.assert_allclose(state[:, 8:11], pose[:, 0:3].astype(np.float32), atol=1e-6,
                                err_msg="state[8:11] 应为 arm_pose[:,0:3]")

    # eef_rot_euler_xyz（索引 11-13）来自 arm_pose[:, 3:6]
    np.testing.assert_allclose(state[:, 11:14], pose[:, 3:6].astype(np.float32), atol=1e-6,
                                err_msg="state[11:14] 应为 arm_pose[:,3:6]")


# ──────────────────────────────────────────────────────────────────────────────
# 测试：build_action_array（next-state）
# ──────────────────────────────────────────────────────────────────────────────

def test_build_action_array_shape():
    """build_action_array 返回与 state 相同形状。"""
    state = np.random.rand(6, 14).astype(np.float32)
    action = build_action_array(state)
    assert action.shape == state.shape


def test_build_action_array_next_state():
    """action[i] == state[i+1]（i < N-1）。"""
    N = 5
    state = np.arange(N * 14, dtype=np.float32).reshape(N, 14)
    action = build_action_array(state)
    np.testing.assert_allclose(action[:-1], state[1:], atol=1e-7,
                                err_msg="action[i] 应等于 state[i+1]（next-state）")


def test_build_action_array_last_frame_copy():
    """末帧 action[N-1] == state[N-1]（复制末帧）。"""
    N = 4
    state = np.random.rand(N, 14).astype(np.float32)
    action = build_action_array(state)
    np.testing.assert_allclose(action[-1], state[-1], atol=1e-7,
                                err_msg="末帧 action 应等于末帧 state")


def test_build_action_array_single_frame():
    """单帧：action[0] == state[0]（无下一帧，复制自身）。"""
    state = np.array([[1.0] * 14], dtype=np.float32)
    action = build_action_array(state)
    np.testing.assert_allclose(action[0], state[0], atol=1e-7)


# ──────────────────────────────────────────────────────────────────────────────
# 测试：episode_to_lerobot_arrays（end-to-end per-episode 接口）
# ──────────────────────────────────────────────────────────────────────────────

def test_episode_to_lerobot_arrays_state_shape(tmp_path):
    """episode_to_lerobot_arrays 返回 state (N, 14) float32。"""
    from tools.align_offline import align_by_image_timestamp

    p = str(tmp_path / "ep.h5")
    N = 4
    _mk_v2(p, N=N, cams=["wrist"])

    aligned = align_by_image_timestamp(p)
    with h5py.File(p, "r") as h5:
        result = episode_to_lerobot_arrays(aligned, h5, cam_names=["wrist"])

    state = result["state"]
    assert state.shape == (N, 14), f"state shape 应为 ({N},14)，实际: {state.shape}"
    assert state.dtype == np.float32


def test_episode_to_lerobot_arrays_action_next_state(tmp_path):
    """验证 action = next-state 语义：action[i] == state[i+1]，末帧复制。"""
    from tools.align_offline import align_by_image_timestamp

    p = str(tmp_path / "ep.h5")
    N = 5
    _mk_v2(p, N=N, cams=["wrist"])

    aligned = align_by_image_timestamp(p)
    with h5py.File(p, "r") as h5:
        result = episode_to_lerobot_arrays(aligned, h5, cam_names=["wrist"])

    state = result["state"]
    action = result["action"]

    # action[i] == state[i+1]（i < N-1）
    np.testing.assert_allclose(action[:-1], state[1:], atol=1e-6,
                                err_msg="action[i] 应等于 state[i+1]")
    # 末帧复制
    np.testing.assert_allclose(action[-1], state[-1], atol=1e-6,
                                err_msg="末帧 action 应等于末帧 state")


def test_episode_to_lerobot_arrays_images(tmp_path):
    """验证 images 字典包含相机键，每帧为 RGB HWC ndarray。"""
    from tools.align_offline import align_by_image_timestamp

    p = str(tmp_path / "ep.h5")
    N = 3
    _mk_v2(p, N=N, cams=["wrist", "exterior"], img_hw=(8, 12))

    aligned = align_by_image_timestamp(p)
    with h5py.File(p, "r") as h5:
        result = episode_to_lerobot_arrays(aligned, h5, cam_names=["wrist", "exterior"])

    imgs = result["images"]
    assert "wrist" in imgs and "exterior" in imgs
    assert len(imgs["wrist"]) == N
    for frame in imgs["wrist"]:
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3 and frame.shape[2] == 3  # RGB HWC


def test_episode_to_lerobot_arrays_task_field(tmp_path):
    """task 字段正确传递。"""
    from tools.align_offline import align_by_image_timestamp

    p = str(tmp_path / "ep.h5")
    _mk_v2(p, N=3, cams=["wrist"])

    aligned = align_by_image_timestamp(p)
    with h5py.File(p, "r") as h5:
        result = episode_to_lerobot_arrays(aligned, h5, cam_names=["wrist"], task="pick_cube")

    assert result["task"] == "pick_cube"
    assert result["N"] == 3
