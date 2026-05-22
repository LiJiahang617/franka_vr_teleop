"""tests/test_align_offline.py — Task 5 离线对齐转换器 TDD（合成数据，不碰真机）。

覆盖：
  - 纯线性轨迹位置插值误差 < 1e-6 m
  - 纯旋转轨迹 SLERP 插值单调（旋转角度单调递增）
  - arm_joints / arm_pose / gripper / action_delta_ee_pose 形状正确
  - on_stale="interpolate"：stale 帧保留，输出长度 = N_anchor
  - on_stale="keep"：同 interpolate，stale 帧保留
  - on_stale="drop"：stale 帧被丢弃，输出长度 < N_anchor
  - cam_anchor 指定不存在的相机时抛 ValueError
  - on_stale 非法值抛 ValueError
  - 所有锚帧 stale 且 drop 时抛 ValueError
  - state_hifreq 原样返回（不重采，shape 不变）
  - CLI 正常运行退出 0，生成 .npz 文件
  - 多相机场景：正确选用 cam_anchor 指定相机
"""
import os

import h5py
import numpy as np
import pytest
import franka_hdf5_schema as S

# 被测模块（conftest 已把 scripts/ 加入 sys.path）
from tools.align_offline import align_by_image_timestamp, main as cli_main


# ---------------------------------------------------------------------------
# 辅助：构造合规 v2 hdf5（合成数据）
# ---------------------------------------------------------------------------

def _write_v2_synthetic(
    path: str,
    N: int = 10,
    M: int = 0,
    cam_names=("wrist",),
    arm_ts_offset: float = 0.005,
    eff_ts_offset: float = 0.007,
    act_ts_offset: float = 0.003,
    stale_indices: dict | None = None,
) -> dict:
    """生成合规 franka-hdf5-v2 文件，带已知偏移时间戳。

    Args:
        path: 输出路径
        N: 帧数
        M: state_hifreq 帧数（M=0 合规）
        cam_names: 相机名称（第一个默认为锚相机）
        arm_ts_offset: arm 时间戳相对锚的固定偏移（秒）
        eff_ts_offset: effector 时间戳偏移
        act_ts_offset: action 时间戳偏移
        stale_indices: dict 形如 {"wrist": [2, 3]}，指定哪些帧 stale

    Returns:
        真值 dict，包含各模态的合成轨迹（用于验证插值精度）
    """
    import franka_hdf5_schema as S

    stale_indices = stale_indices or {}

    # 锚时间轴（等间距 30 Hz）
    anchor_ts = np.arange(N, dtype=np.float64) * (1.0 / 30.0) + 10.0

    # 各模态时间戳（固定偏移，使插值点落在锚轴上）
    arm_ts = anchor_ts + arm_ts_offset
    eff_ts = anchor_ts + eff_ts_offset
    act_ts = anchor_ts + act_ts_offset

    # 合成线性轨迹：joints 线性增，pose 位置线性 + 旋转线性（Euler 步进）
    joints = np.outer(np.arange(N, dtype=np.float64), np.ones(7)) * 0.01  # (N,7)
    joint_vel = np.outer(np.arange(N, dtype=np.float64), np.ones(7)) * 0.001
    # pose 前 3 列位置线性
    positions = np.outer(np.arange(N, dtype=np.float64), np.array([0.001, 0.002, 0.003]))
    # 后 3 列欧拉角线性递增（rx 从 0 到 0.3 rad）
    angles = np.outer(np.arange(N, dtype=np.float64), np.array([0.03, 0.0, 0.0]))
    pose = np.concatenate([positions, angles], axis=1).astype(np.float64)

    # effector
    gripper_pos = (np.arange(N, dtype=np.float64) * 0.004).reshape(N, 1)
    gripper_norm = gripper_pos / 0.08

    # action
    delta_ee = np.random.default_rng(42).uniform(-0.01, 0.01, (N, 6))
    gripper_cmd = np.zeros((N, 1), np.float64)

    # state_hifreq
    if M > 0:
        hifreq_ts = np.arange(M, dtype=np.float64) / 240.0 + anchor_ts[0]
        hifreq_joints = np.zeros((M, 7), np.float64)
        hifreq_jv = np.zeros((M, 7), np.float64)
        hifreq_pose = np.zeros((M, 6), np.float64)
    else:
        hifreq_ts = np.array([], np.float64)
        hifreq_joints = np.zeros((0, 7), np.float64)
        hifreq_jv = np.zeros((0, 7), np.float64)
        hifreq_pose = np.zeros((0, 6), np.float64)

    _VLEN = h5py.special_dtype(vlen=np.dtype("uint8"))
    with h5py.File(path, "w") as f:
        # infos
        inf = f.create_group("infos")
        inf.create_dataset("schema_version", data=np.bytes_(S.SCHEMA_VERSION))
        ti = inf.create_group("task_info")
        ti.create_dataset("task_name", data=np.bytes_("test"))
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
        arm.create_dataset("joints", data=joints)
        arm.create_dataset("joint_vel", data=joint_vel)
        arm.create_dataset("pose", data=pose)
        arm.create_dataset("timestamp", data=arm_ts)
        arm.create_dataset("stale", data=np.zeros(N, dtype=bool))

        # effector
        eff = obs.create_group("effector")
        eff.create_dataset("position", data=gripper_pos)
        eff.create_dataset("position_norm", data=gripper_norm)
        eff.create_dataset("type",
                           data=np.array([b"gripper"] * N, dtype=h5py.special_dtype(vlen=bytes)))
        eff.create_dataset("timestamp", data=eff_ts)
        eff.create_dataset("stale", data=np.zeros(N, dtype=bool))

        # camera
        cam_g = obs.create_group("camera")
        rgb_g = cam_g.create_group("rgb")
        for i_cn, cn in enumerate(cam_names):
            cg = rgb_g.create_group(cn)
            imgs = cg.create_dataset("images", (N,), dtype=_VLEN)
            dummy_jpeg = bytes([0xFF, 0xD8, 0xFF, 0xD9])
            for i in range(N):
                imgs[i] = np.frombuffer(dummy_jpeg, np.uint8)
            ts_cam = anchor_ts + 0.001 * (i_cn + 1)
            cg.create_dataset("timestamp", data=ts_cam)
            # stale
            cam_stale = np.zeros(N, dtype=bool)
            if cn in stale_indices:
                for idx in stale_indices[cn]:
                    cam_stale[idx] = True
            cg.create_dataset("stale", data=cam_stale)
            cg.create_dataset("hw_timestamp", data=ts_cam * 1000.0)

        # state_hifreq
        hf = obs.create_group("state_hifreq")
        hf.create_dataset("joints", data=hifreq_joints)
        hf.create_dataset("joint_vel", data=hifreq_jv)
        hf.create_dataset("pose", data=hifreq_pose)
        hf.create_dataset("timestamp", data=hifreq_ts)
        hf.create_dataset("poly_ts", data=hifreq_ts.copy())
        hf.create_dataset("wrench", data=np.zeros((M if M else 0, 6), np.float64))

        # action
        act = f.create_group("action")
        act.create_dataset("delta_ee_pose", data=delta_ee)
        act.create_dataset("gripper_cmd", data=gripper_cmd)
        act.create_dataset("timestamp", data=act_ts)

    # 验证写出的文件合规（S 在模块顶层已导入）
    violations = S.validate_episode(path)
    assert violations == [], f"合成 hdf5 不合规：{violations}"

    return {
        "anchor_ts": anchor_ts,
        "joints": joints,
        "joint_vel": joint_vel,
        "pose": pose,
        "gripper_pos": gripper_pos,
        "gripper_norm": gripper_norm,
        "delta_ee": delta_ee,
        "gripper_cmd": gripper_cmd,
        "hifreq_joints": hifreq_joints,
        "hifreq_jv": hifreq_jv,
        "hifreq_pose": hifreq_pose,
        "hifreq_ts": hifreq_ts,
        "arm_ts": arm_ts,
        "eff_ts": eff_ts,
        "act_ts": act_ts,
    }


# ---------------------------------------------------------------------------
# 测试：基础对齐精度
# ---------------------------------------------------------------------------

class TestAlignAccuracy:
    """验证插值误差在预期范围内。"""

    def test_joints_linear_interp_exact(self, tmp_path):
        """线性轨迹：arm_joints 在线性插值后误差应极小（< 1e-12）。

        锚时间轴来自相机 ts（= anchor_ts + 0.001），arm_ts = anchor_ts + 0.005。
        两者都是线性时间轴，插值精度等于机器精度。
        """
        path = str(tmp_path / "ep.h5")
        truth = _write_v2_synthetic(path, N=10, arm_ts_offset=0.005)

        aligned = align_by_image_timestamp(path, on_stale="interpolate")

        # 验证 shape
        assert aligned["arm_joints"].shape == (10, 7)

        # 实际锚时间轴 = 相机 ts = truth["anchor_ts"] + 0.001（_write_v2_synthetic 中第一个相机偏移）
        actual_anchor_ts = aligned["anchor_ts"]
        arm_ts = truth["arm_ts"]
        for d in range(7):
            expected = np.interp(actual_anchor_ts, arm_ts, truth["joints"][:, d])
            np.testing.assert_allclose(
                aligned["arm_joints"][:, d], expected, atol=1e-12,
                err_msg=f"arm_joints 列 {d} 插值偏差"
            )

    def test_gripper_position_interp_within_1um(self, tmp_path):
        """线性夹爪轨迹：位置插值误差 < 1e-6 m。"""
        path = str(tmp_path / "ep.h5")
        truth = _write_v2_synthetic(path, N=10, eff_ts_offset=0.007)

        aligned = align_by_image_timestamp(path, on_stale="interpolate")

        # 用实际锚时间轴（来自相机 ts）计算期望值
        actual_anchor_ts = aligned["anchor_ts"]
        eff_ts = truth["eff_ts"]
        expected_pos = np.interp(actual_anchor_ts, eff_ts, truth["gripper_pos"][:, 0])
        np.testing.assert_allclose(
            aligned["gripper_position"][:, 0], expected_pos, atol=1e-12,
            err_msg="gripper_position 插值误差超出机器精度"
        )

    def test_ee_position_interp_exact(self, tmp_path):
        """EE 位置（arm_pose 前 3 列）线性插值精度达到机器精度。"""
        path = str(tmp_path / "ep.h5")
        truth = _write_v2_synthetic(path, N=10, arm_ts_offset=0.005)

        aligned = align_by_image_timestamp(path, on_stale="interpolate")

        # 用实际锚时间轴（来自相机 ts）计算期望值
        actual_anchor_ts = aligned["anchor_ts"]
        arm_ts = truth["arm_ts"]
        for d in range(3):
            expected = np.interp(actual_anchor_ts, arm_ts, truth["pose"][:, d])
            np.testing.assert_allclose(
                aligned["arm_pose"][:, d], expected, atol=1e-12,
                err_msg=f"arm_pose 位置列 {d} 插值误差超出机器精度"
            )

    def test_slerp_rotation_monotone(self, tmp_path):
        """SLERP 插值的旋转角度应单调递增（rx 线性递增场景）。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=10, arm_ts_offset=0.005)

        aligned = align_by_image_timestamp(path, on_stale="interpolate")

        # rx 列（index 3）是单调递增（合成数据 angles 是线性增）
        rx = aligned["arm_pose"][:, 3]
        # 应单调递增（diff >= 0）
        diffs = np.diff(rx)
        assert np.all(diffs >= -1e-12), (
            f"SLERP 后 rx 不单调递增：最小 diff={diffs.min():.2e}"
        )

    def test_action_delta_ee_interp_shape(self, tmp_path):
        """action_delta_ee_pose 插值结果形状正确。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=10, act_ts_offset=0.003)

        aligned = align_by_image_timestamp(path, on_stale="interpolate")

        assert aligned["action_delta_ee_pose"].shape == (10, 6)
        assert aligned["gripper_cmd"].shape == (10, 1)


# ---------------------------------------------------------------------------
# 测试：on_stale 策略
# ---------------------------------------------------------------------------

class TestOnStaleStrategies:
    """验证三种 stale 策略的正确性。"""

    def test_interpolate_keeps_stale_frames(self, tmp_path):
        """on_stale='interpolate'：stale 帧仍保留，输出长度 = N_anchor。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=10, stale_indices={"wrist": [2, 5]})

        aligned = align_by_image_timestamp(path, on_stale="interpolate")

        assert len(aligned["anchor_ts"]) == 10
        # stale 标记保留
        assert aligned["anchor_stale"][2] is True or bool(aligned["anchor_stale"][2])
        assert aligned["anchor_stale"][5] is True or bool(aligned["anchor_stale"][5])
        # 非 stale 帧不受影响
        assert not aligned["anchor_stale"][0]

    def test_keep_same_as_interpolate(self, tmp_path):
        """on_stale='keep'：行为同 interpolate，stale 帧保留。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=10, stale_indices={"wrist": [3]})

        aligned_interp = align_by_image_timestamp(path, on_stale="interpolate")
        aligned_keep = align_by_image_timestamp(path, on_stale="keep")

        np.testing.assert_array_equal(aligned_interp["anchor_ts"], aligned_keep["anchor_ts"])
        np.testing.assert_array_equal(aligned_interp["arm_joints"], aligned_keep["arm_joints"])
        np.testing.assert_array_equal(aligned_interp["anchor_stale"], aligned_keep["anchor_stale"])

    def test_drop_removes_stale_frames(self, tmp_path):
        """on_stale='drop'：stale 帧被丢弃，输出长度 < N_anchor。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=10, stale_indices={"wrist": [2, 5, 8]})

        aligned = align_by_image_timestamp(path, on_stale="drop")

        # 3 个 stale 帧被丢弃
        assert len(aligned["anchor_ts"]) == 7
        # 输出的 stale 全为 False
        assert not np.any(aligned["anchor_stale"])

    def test_drop_no_stale_frames_returns_full(self, tmp_path):
        """on_stale='drop'：没有 stale 帧时，输出长度 = N_anchor。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=10)

        aligned = align_by_image_timestamp(path, on_stale="drop")

        assert len(aligned["anchor_ts"]) == 10

    def test_drop_all_stale_raises(self, tmp_path):
        """on_stale='drop' 且所有帧都是 stale 时，应抛出 ValueError。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=5, stale_indices={"wrist": [0, 1, 2, 3, 4]})

        with pytest.raises(ValueError, match="空"):
            align_by_image_timestamp(path, on_stale="drop")

    def test_invalid_on_stale_raises(self, tmp_path):
        """非法 on_stale 值应抛出 ValueError。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=5)

        with pytest.raises(ValueError, match="on_stale"):
            align_by_image_timestamp(path, on_stale="invalid_mode")


# ---------------------------------------------------------------------------
# 测试：cam_anchor 参数
# ---------------------------------------------------------------------------

class TestCamAnchor:
    """验证 cam_anchor 参数正确选择锚相机。"""

    def test_default_anchor_is_first_camera(self, tmp_path):
        """不指定 cam_anchor 时，默认取字典序第一个相机。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=8, cam_names=("exterior", "wrist"))

        # 默认锚 = "exterior"（字典序第一）
        aligned_default = align_by_image_timestamp(path)
        aligned_explicit = align_by_image_timestamp(path, cam_anchor="exterior")

        np.testing.assert_array_equal(aligned_default["anchor_ts"], aligned_explicit["anchor_ts"])

    def test_explicit_anchor_uses_that_camera(self, tmp_path):
        """指定 cam_anchor='wrist' 时，锚时间轴来自 wrist 相机。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=8, cam_names=("exterior", "wrist"))

        aligned_ext = align_by_image_timestamp(path, cam_anchor="exterior")
        aligned_wrist = align_by_image_timestamp(path, cam_anchor="wrist")

        # 不同相机时间戳略有差异（见 _write_v2_synthetic 中 0.001 * (i_cn + 1) 偏移）
        assert not np.array_equal(aligned_ext["anchor_ts"], aligned_wrist["anchor_ts"])

    def test_invalid_cam_anchor_raises(self, tmp_path):
        """不存在的 cam_anchor 应抛出 ValueError。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=5)

        with pytest.raises(ValueError, match="不存在"):
            align_by_image_timestamp(path, cam_anchor="nonexistent_cam")


# ---------------------------------------------------------------------------
# 测试：state_hifreq 原样返回
# ---------------------------------------------------------------------------

class TestStateHifreq:
    """state_hifreq 数据原样返回，不重采。"""

    def test_state_hifreq_passthrough_with_data(self, tmp_path):
        """有 state_hifreq 数据时，原样返回不变。"""
        path = str(tmp_path / "ep.h5")
        truth = _write_v2_synthetic(path, N=10, M=30)

        aligned = align_by_image_timestamp(path)

        assert aligned["state_hifreq_joints"].shape == (30, 7)
        assert aligned["state_hifreq_timestamp"].shape == (30,)
        np.testing.assert_array_equal(aligned["state_hifreq_joints"], truth["hifreq_joints"])

    def test_state_hifreq_empty_m0(self, tmp_path):
        """M=0 时 state_hifreq 为空数组。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=10, M=0)

        aligned = align_by_image_timestamp(path)

        assert aligned["state_hifreq_joints"].shape == (0, 7)
        assert aligned["state_hifreq_timestamp"].shape == (0,)


# ---------------------------------------------------------------------------
# 测试：输出 dict 完整性
# ---------------------------------------------------------------------------

class TestOutputKeys:
    """验证返回 dict 包含所有预期键，且形状正确。"""

    def test_all_keys_present(self, tmp_path):
        """返回 dict 必须包含所有规定键。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=8)

        aligned = align_by_image_timestamp(path)

        expected_keys = {
            "anchor_ts", "anchor_stale",
            "arm_joints", "arm_joint_vel", "arm_pose",
            "gripper_position", "gripper_position_norm", "gripper_cmd",
            "action_delta_ee_pose",
            "state_hifreq_joints", "state_hifreq_joint_vel",
            "state_hifreq_pose", "state_hifreq_timestamp",
        }
        missing = expected_keys - set(aligned.keys())
        assert not missing, f"缺失键：{missing}"

    def test_shapes_consistent(self, tmp_path):
        """各对齐输出的第 0 维与 anchor_ts 长度一致。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=12)

        aligned = align_by_image_timestamp(path)

        N_out = len(aligned["anchor_ts"])
        per_frame_keys = [
            "arm_joints", "arm_joint_vel", "arm_pose",
            "gripper_position", "gripper_position_norm", "gripper_cmd",
            "action_delta_ee_pose", "anchor_stale",
        ]
        for k in per_frame_keys:
            assert aligned[k].shape[0] == N_out, (
                f"{k}.shape[0]={aligned[k].shape[0]} != N_out={N_out}"
            )


# ---------------------------------------------------------------------------
# 测试：CLI
# ---------------------------------------------------------------------------

class TestCLI:
    """CLI run 退出 0，生成 .npz 输出文件。"""

    def test_cli_runs_and_produces_npz(self, tmp_path):
        """CLI 正常运行后生成 .npz，内含 anchor_ts 键。"""
        h5_path = str(tmp_path / "ep.h5")
        npz_path = str(tmp_path / "aligned.npz")
        _write_v2_synthetic(h5_path, N=10)

        # 调用 main()，不应抛异常
        cli_main(["--in", h5_path, "--out", npz_path, "--on-stale", "interpolate"])

        assert os.path.exists(npz_path), ".npz 文件未生成"
        data = np.load(npz_path)
        assert "anchor_ts" in data.files, ".npz 缺少 anchor_ts"
        assert len(data["anchor_ts"]) == 10

    def test_cli_drop_stale(self, tmp_path):
        """CLI --on-stale drop 正确丢弃 stale 帧。"""
        h5_path = str(tmp_path / "ep.h5")
        npz_path = str(tmp_path / "aligned.npz")
        _write_v2_synthetic(h5_path, N=10, stale_indices={"wrist": [1, 4]})

        cli_main(["--in", h5_path, "--out", npz_path, "--on-stale", "drop"])

        data = np.load(npz_path)
        assert len(data["anchor_ts"]) == 8  # 10 - 2 stale

    def test_cli_cam_anchor_flag(self, tmp_path):
        """CLI --cam-anchor 指定不同相机时输出时间轴不同。"""
        h5_path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(h5_path, N=8, cam_names=("exterior", "wrist"))

        npz_ext = str(tmp_path / "aligned_ext.npz")
        npz_wrist = str(tmp_path / "aligned_wrist.npz")

        cli_main(["--in", h5_path, "--out", npz_ext, "--cam-anchor", "exterior"])
        cli_main(["--in", h5_path, "--out", npz_wrist, "--cam-anchor", "wrist"])

        ext_ts = np.load(npz_ext)["anchor_ts"]
        wrist_ts = np.load(npz_wrist)["anchor_ts"]
        assert not np.array_equal(ext_ts, wrist_ts)
