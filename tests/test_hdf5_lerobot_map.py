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


# ──────────────────────────────────────────────────────────────────────────────
# 辅助（新增）：构造带 stale 的合成 v2 hdf5，每帧图像有可辨识内容
# ──────────────────────────────────────────────────────────────────────────────

def _mk_v2_with_stale(p, N=6, stale_indices=None, img_hw=(8, 8)):
    """生成合规 franka-hdf5-v2，图像像素值 = 帧编号（0-based）以便区分。

    Args:
        p: 输出路径
        N: 帧数
        stale_indices: list of int，哪些帧 stale
        img_hw: (H, W)
    """
    import franka_hdf5_schema as S

    stale_indices = stale_indices or []
    H, W = img_hw

    ts = np.arange(N, dtype=np.float64)

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
        eff.create_dataset("position_norm", data=np.ones((N, 1)) * 0.5)
        eff.create_dataset("type", data=np.array([b"gripper"] * N,
                                                  dtype=h5py.special_dtype(vlen=bytes)))
        eff.create_dataset("timestamp", data=ts.copy())
        eff.create_dataset("stale", data=np.zeros(N, dtype=bool))

        cam = obs.create_group("camera")
        rgb = cam.create_group("rgb")
        g = rgb.create_group("wrist")
        d = g.create_dataset("images", (N,), dtype=h5py.special_dtype(vlen=np.dtype("uint8")))
        for i in range(N):
            # 每帧图像填充值 = 帧编号（0-based），以便测试精确验证来源
            img = np.full((H, W, 3), i, dtype=np.uint8)
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            d[i] = np.frombuffer(enc.tobytes(), np.uint8)
        cam_stale = np.zeros(N, dtype=bool)
        for idx in stale_indices:
            cam_stale[idx] = True
        g.create_dataset("timestamp", data=ts.copy())
        g.create_dataset("stale", data=cam_stale)
        g.create_dataset("hw_timestamp", data=ts.copy())

        hf = obs.create_group("state_hifreq")
        for k, sh in [("joints", (0, 7)), ("joint_vel", (0, 7)), ("pose", (0, 6)),
                      ("timestamp", (0,)), ("poly_ts", (0,))]:
            hf.create_dataset(k, data=np.zeros(sh, np.float64))
        hf.create_dataset("wrench", data=np.zeros((0, 6), np.float64))

        act = f.create_group("action")
        act.create_dataset("delta_ee_pose", data=np.zeros((N, 6), np.float64))
        act.create_dataset("gripper_cmd", data=np.ones((N, 1), np.float64))
        act.create_dataset("timestamp", data=ts.copy() + 0.001)


# ──────────────────────────────────────────────────────────────────────────────
# 新增测试（Codex 审查）：drop 模式图像-state 对齐
# ──────────────────────────────────────────────────────────────────────────────

class TestDropModeImageAlignment:
    """验证 drop 模式下图像来自正确的原始帧（使用 anchor_indices 索引）。"""

    def test_drop_mode_images_come_from_correct_original_frames(self, tmp_path):
        """drop 中间帧后，第 i 个输出图像应来自原始帧 anchor_indices[i]。

        构造 6 帧 hdf5，帧 2 和 4 stale。drop 后输出 4 帧，
        对应原始下标 [0, 1, 3, 5]。验证第 i 个图像的像素值 == 原始帧编号。
        """
        from tools.align_offline import align_by_image_timestamp as _align

        p = str(tmp_path / "ep.h5")
        stale_idx = [2, 4]
        _mk_v2_with_stale(p, N=6, stale_indices=stale_idx)

        aligned = _align(p, on_stale="drop")
        assert "anchor_indices" in aligned

        expected_orig_indices = [i for i in range(6) if i not in stale_idx]
        np.testing.assert_array_equal(
            aligned["anchor_indices"],
            np.array(expected_orig_indices, dtype=np.intp),
            err_msg="drop 模式 anchor_indices 与预期原始下标不符"
        )

        with h5py.File(p, "r") as h5:
            result = episode_to_lerobot_arrays(aligned, h5, cam_names=["wrist"])

        assert result["N"] == 4
        imgs = result["images"]["wrist"]
        assert len(imgs) == 4

        for out_i, orig_i in enumerate(expected_orig_indices):
            # 图像填充值 == 帧编号，均值应接近 orig_i（JPEG 高质量压缩，atol=5）
            mean_val = imgs[out_i].mean()
            assert abs(mean_val - orig_i) < 5, (
                f"输出第 {out_i} 帧图像均值 {mean_val:.1f} != 原始帧 {orig_i}（错位）"
            )

    def test_interpolate_mode_images_sequential(self, tmp_path):
        """interpolate 模式：图像按 0,1,2,...,N-1 顺序读取（anchor_indices=arange）。"""
        from tools.align_offline import align_by_image_timestamp as _align

        p = str(tmp_path / "ep.h5")
        _mk_v2_with_stale(p, N=5, stale_indices=[1])  # 有 stale 但 interpolate 不 drop

        aligned = _align(p, on_stale="interpolate")
        np.testing.assert_array_equal(aligned["anchor_indices"], np.arange(5, dtype=np.intp))

        with h5py.File(p, "r") as h5:
            result = episode_to_lerobot_arrays(aligned, h5, cam_names=["wrist"])

        imgs = result["images"]["wrist"]
        assert len(imgs) == 5
        for i in range(5):
            mean_val = imgs[i].mean()
            assert abs(mean_val - i) < 5, (
                f"interpolate 模式第 {i} 帧图像均值 {mean_val:.1f} != {i}"
            )

    def test_drop_mode_state_count_matches_images(self, tmp_path):
        """drop 后 state 帧数 == images 帧数 == anchor_indices 长度。"""
        from tools.align_offline import align_by_image_timestamp as _align

        p = str(tmp_path / "ep.h5")
        _mk_v2_with_stale(p, N=8, stale_indices=[1, 3, 6])

        aligned = _align(p, on_stale="drop")
        with h5py.File(p, "r") as h5:
            result = episode_to_lerobot_arrays(aligned, h5, cam_names=["wrist"])

        N_out = result["N"]
        assert N_out == 5  # 8 - 3 stale
        assert result["state"].shape[0] == N_out
        assert result["action"].shape[0] == N_out
        assert len(result["images"]["wrist"]) == N_out
        assert len(aligned["anchor_indices"]) == N_out

    def test_anchor_indices_out_of_range_raises(self, tmp_path):
        """anchor_indices 超出 hdf5 图像帧数时应抛出 ValueError（带相机名）。"""
        from tools.align_offline import align_by_image_timestamp as _align

        p = str(tmp_path / "ep.h5")
        _mk_v2_with_stale(p, N=4)

        aligned = _align(p, on_stale="interpolate")
        # 人为篡改 anchor_indices 使其超出范围
        aligned["anchor_indices"] = np.array([0, 1, 2, 100], dtype=np.intp)

        with h5py.File(p, "r") as h5:
            with pytest.raises(ValueError, match="wrist"):
                episode_to_lerobot_arrays(aligned, h5, cam_names=["wrist"])


# ──────────────────────────────────────────────────────────────────────────────
# 新增测试（Codex 审查）：N=0 空 episode
# ──────────────────────────────────────────────────────────────────────────────

class TestEmptyEpisode:
    """验证 N=0 空 episode 边界处理。"""

    def test_build_action_array_n0_returns_empty(self):
        """N=0 时 build_action_array 返回 (0, 14) 空数组，不越界。"""
        state = np.empty((0, 14), dtype=np.float32)
        action = build_action_array(state)
        assert action.shape == (0, 14), f"N=0 时 action shape 应为 (0,14)，实际: {action.shape}"
        assert action.dtype == np.float32

    def test_build_state_array_n0_returns_empty(self):
        """N=0 时 build_state_array 返回 (0, 14) 空数组。"""
        aligned = {
            "arm_joints": np.empty((0, 7)),
            "gripper_position_norm": np.empty((0, 1)),
            "arm_pose": np.empty((0, 6)),
        }
        state = build_state_array(aligned)
        assert state.shape == (0, 14), f"N=0 时 state shape 应为 (0,14)，实际: {state.shape}"


# ──────────────────────────────────────────────────────────────────────────────
# 新增测试（Codex 审查）：损坏 jpeg 报错
# ──────────────────────────────────────────────────────────────────────────────

class TestCorruptJpeg:
    """验证 _decode 对损坏 jpeg 的处理。"""

    def test_corrupt_jpeg_raises_value_error(self, tmp_path):
        """episode_to_lerobot_arrays 遇到损坏 jpeg 时应抛出带信息的 ValueError。"""
        import franka_hdf5_schema as S
        from tools.align_offline import align_by_image_timestamp as _align

        p = str(tmp_path / "ep.h5")
        N = 3
        ts = np.arange(N, dtype=np.float64)

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
            arm.create_dataset("joints", data=np.zeros((N, 7), np.float64))
            arm.create_dataset("joint_vel", data=np.zeros((N, 7), np.float64))
            arm.create_dataset("pose", data=np.zeros((N, 6), np.float64))
            arm.create_dataset("timestamp", data=ts.copy())
            arm.create_dataset("stale", data=np.zeros(N, dtype=bool))

            eff = obs.create_group("effector")
            eff.create_dataset("position", data=np.zeros((N, 1), np.float64))
            eff.create_dataset("position_norm", data=np.ones((N, 1)) * 0.5)
            eff.create_dataset("type", data=np.array([b"gripper"] * N,
                                                      dtype=h5py.special_dtype(vlen=bytes)))
            eff.create_dataset("timestamp", data=ts.copy())
            eff.create_dataset("stale", data=np.zeros(N, dtype=bool))

            cam = obs.create_group("camera")
            rgb = cam.create_group("rgb")
            g = rgb.create_group("wrist")
            d = g.create_dataset("images", (N,), dtype=h5py.special_dtype(vlen=np.dtype("uint8")))
            img = np.zeros((8, 8, 3), np.uint8)
            ok, enc = cv2.imencode(".jpg", img)
            d[0] = np.frombuffer(enc.tobytes(), np.uint8)
            d[1] = np.array([0xDE, 0xAD, 0xBE, 0xEF], dtype=np.uint8)  # 损坏
            d[2] = np.frombuffer(enc.tobytes(), np.uint8)
            g.create_dataset("timestamp", data=ts.copy())
            g.create_dataset("stale", data=np.zeros(N, dtype=bool))
            g.create_dataset("hw_timestamp", data=ts.copy())

            hf = obs.create_group("state_hifreq")
            for k, sh in [("joints", (0, 7)), ("joint_vel", (0, 7)), ("pose", (0, 6)),
                          ("timestamp", (0,)), ("poly_ts", (0,))]:
                hf.create_dataset(k, data=np.zeros(sh, np.float64))
            hf.create_dataset("wrench", data=np.zeros((0, 6), np.float64))

            act = f.create_group("action")
            act.create_dataset("delta_ee_pose", data=np.zeros((N, 6), np.float64))
            act.create_dataset("gripper_cmd", data=np.ones((N, 1), np.float64))
            act.create_dataset("timestamp", data=ts.copy() + 0.001)

        aligned = _align(p, on_stale="interpolate")
        with h5py.File(p, "r") as h5:
            with pytest.raises(ValueError, match="None|损坏|imdecode"):
                episode_to_lerobot_arrays(aligned, h5, cam_names=["wrist"])


# ──────────────────────────────────────────────────────────────────────────────
# 新增测试（Codex 审查）：build_state_array shape 校验
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildStateArrayValidation:
    """验证 build_state_array 的轻量 shape 校验。"""

    def test_length_mismatch_raises(self):
        """arm_joints 和 gripper_position_norm 长度不一致时抛 ValueError。"""
        aligned = {
            "arm_joints": np.zeros((5, 7)),
            "gripper_position_norm": np.zeros((4, 1)),  # 长度不符
            "arm_pose": np.zeros((5, 6)),
        }
        with pytest.raises(ValueError, match="不一致|mismatch"):
            build_state_array(aligned)

    def test_gripper_wrong_ndim_raises(self):
        """gripper_position_norm 非 (N,1) 二维时抛 ValueError。"""
        aligned = {
            "arm_joints": np.zeros((5, 7)),
            "gripper_position_norm": np.zeros((5,)),  # 1D 而非 (N,1)
            "arm_pose": np.zeros((5, 6)),
        }
        with pytest.raises(ValueError, match="gripper_position_norm"):
            build_state_array(aligned)


# ──────────────────────────────────────────────────────────────────────────────
# Task 7 TDD：转换器对缺 effector/hw_timestamp 字段 ep 发出 warning
# ──────────────────────────────────────────────────────────────────────────────

def _make_minimal_v2_episode(path, N, with_hw_ts=False, hw_ts_all_nan=False):
    """创建最小完整合规 franka-hdf5-v2 episode（含图像），供转换器 subprocess 测试使用。

    比 test_align_offline 中同名函数多了 infos/schema_version、effector/type、
    camera images 等字段，以通过 schema validate_episode 校验。

    Args:
        path: 输出文件路径
        N: 帧数
        with_hw_ts: 是否写入 observations/effector/hw_timestamp
        hw_ts_all_nan: with_hw_ts=True 时是否将 hw_timestamp 全设为 NaN
    """
    import franka_hdf5_schema as _S

    H, W = 64, 64  # SVT-AV1 最小分辨率 64x64
    img = np.zeros((H, W, 3), np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    jb = np.frombuffer(enc.tobytes(), np.uint8)
    ts = np.arange(N, dtype=np.float64)  # 0..N-1 严格递增

    with h5py.File(str(path), "w") as f:
        # infos
        inf = f.create_group("infos")
        inf.create_dataset("schema_version", data=np.bytes_(_S.SCHEMA_VERSION))
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

        # arm
        arm = obs.create_group("arm")
        arm.create_dataset("joints", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("joint_vel", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("pose", data=np.zeros((N, 6), np.float64))
        arm.create_dataset("timestamp", data=ts.copy())
        arm.create_dataset("stale", data=np.zeros(N, dtype=bool))

        # effector
        eff = obs.create_group("effector")
        eff.create_dataset("position", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("position_norm", data=np.ones((N, 1), np.float64) * 0.5)
        eff.create_dataset("type", data=np.array([b"gripper"] * N,
                                                  dtype=h5py.special_dtype(vlen=bytes)))
        eff.create_dataset("timestamp", data=ts.copy())
        eff.create_dataset("stale", data=np.zeros(N, dtype=bool))
        if with_hw_ts:
            hw = np.full(N, np.nan) if hw_ts_all_nan else (ts - 800.0)
            eff.create_dataset("hw_timestamp", data=hw.astype(np.float64))

        # camera/rgb/wrist（含图像）
        cam = obs.create_group("camera")
        rgb = cam.create_group("rgb")
        g = rgb.create_group("wrist")
        d = g.create_dataset("images", (N,), dtype=h5py.special_dtype(vlen=np.dtype("uint8")))
        for i in range(N):
            d[i] = jb
        g.create_dataset("timestamp", data=ts.copy())
        g.create_dataset("stale", data=np.zeros(N, dtype=bool))
        g.create_dataset("hw_timestamp", data=ts.copy())  # camera hw_ts schema 必须字段

        # state_hifreq（M=0 合规占位）
        hf = obs.create_group("state_hifreq")
        for k, sh in [("joints", (0, 7)), ("joint_vel", (0, 7)), ("pose", (0, 6)),
                      ("timestamp", (0,)), ("poly_ts", (0,))]:
            hf.create_dataset(k, data=np.zeros(sh, np.float64))
        hf.create_dataset("wrench", data=np.zeros((0, 6), np.float64))

        # action
        act = f.create_group("action")
        act.create_dataset("delta_ee_pose", data=np.zeros((N, 6), np.float64))
        act.create_dataset("gripper_cmd", data=np.ones((N, 1), np.float64))
        act.create_dataset("timestamp", data=ts.copy() + 0.001)  # 严格递增且 > obs ts

    return path


def test_converter_v30_warns_when_hw_timestamp_missing(tmp_path):
    """v3.0 转换器遇到缺 effector/hw_timestamp 的 ep 时 log warning。"""
    ep_dir = tmp_path / "src"
    ep_dir.mkdir()
    _make_minimal_v2_episode(ep_dir / "ep0.h5", N=30, with_hw_ts=False)

    out = tmp_path / "out"
    import subprocess
    _P = "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop"
    r = subprocess.run([
        "/home/ubuntu/Desktop/jhli/envs/franka-teleop/bin/python",
        f"{_P}/scripts/tools/hdf5_to_lerobot.py",
        "--in", str(ep_dir),
        "--repo-id", "local/test",
        "--fps", "30",
        "--root", str(out),
        "--task", "test",
    ], capture_output=True, text=True)
    assert r.returncode == 0, f"转换器退出非 0: stderr={r.stderr[:500]}"
    combined = (r.stderr + r.stdout).lower()
    assert "hw_timestamp" in combined, f"应 warning 提示 hw_timestamp 缺失: {combined[:500]}"
    assert "rebuild" in combined or "退回" in (r.stderr + r.stdout), \
        f"warning 应指引 rebuild polymetis: {combined[:500]}"


def test_converter_v21_warns_when_hw_timestamp_missing(tmp_path):
    """v2.1 转换器同样要 warning。"""
    ep_dir = tmp_path / "src"
    ep_dir.mkdir()
    _make_minimal_v2_episode(ep_dir / "ep0.h5", N=30, with_hw_ts=False)

    out = tmp_path / "out"
    import subprocess
    _P = "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop"
    r = subprocess.run([
        "/home/ubuntu/Desktop/jhli/envs/franka-teleop/bin/python",
        f"{_P}/scripts/tools/hdf5_to_lerobot_v21.py",
        "--in-dir", str(ep_dir),
        "--out", str(out),
        "--fps", "30",
        "--task", "test",
        "--robot-type", "franka",
    ], capture_output=True, text=True)
    assert r.returncode == 0, f"转换器退出非 0: stderr={r.stderr[:500]}"
    combined = (r.stderr + r.stdout).lower()
    assert "hw_timestamp" in combined, f"应 warning: {combined[:500]}"
    assert "rebuild" in combined or "退回" in (r.stderr + r.stdout), \
        f"warning 应指引 rebuild polymetis: {combined[:500]}"
