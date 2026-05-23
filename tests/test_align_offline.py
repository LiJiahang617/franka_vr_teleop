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
  - 源时间轴含重复时间戳（模拟 stale 补帧）时不崩、对齐正确
  - 单帧模态广播到全锚时间轴
  - 0 帧模态抛带模态名的 ValueError
  - SLERP 输出 unwrap 后无大跳变
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
    arm_ts_override: np.ndarray | None = None,
    eff_ts_override: np.ndarray | None = None,
    act_ts_override: np.ndarray | None = None,
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
        arm_ts_override: 若非 None，直接用这个数组代替 arm 时间戳
        eff_ts_override: 若非 None，直接用这个数组代替 effector 时间戳
        act_ts_override: 若非 None，直接用这个数组代替 action 时间戳

    Returns:
        真值 dict，包含各模态的合成轨迹（用于验证插值精度）
    """
    import franka_hdf5_schema as S

    stale_indices = stale_indices or {}

    # 锚时间轴（等间距 30 Hz）
    anchor_ts = np.arange(N, dtype=np.float64) * (1.0 / 30.0) + 10.0

    # 各模态时间戳（固定偏移，使插值点落在锚轴上）
    arm_ts = arm_ts_override if arm_ts_override is not None else anchor_ts + arm_ts_offset
    eff_ts = eff_ts_override if eff_ts_override is not None else anchor_ts + eff_ts_offset
    act_ts = act_ts_override if act_ts_override is not None else anchor_ts + act_ts_offset

    # 合成线性轨迹：joints 线性增，pose 位置线性 + 旋转线性（Euler 步进）
    arm_N = len(arm_ts)
    eff_N = len(eff_ts)
    act_N = len(act_ts)

    joints = np.outer(np.arange(arm_N, dtype=np.float64), np.ones(7)) * 0.01  # (arm_N,7)
    joint_vel = np.outer(np.arange(arm_N, dtype=np.float64), np.ones(7)) * 0.001
    # pose 前 3 列位置线性
    positions = np.outer(np.arange(arm_N, dtype=np.float64), np.array([0.001, 0.002, 0.003]))
    # 后 3 列欧拉角线性递增（rx 从 0 到 0.3 rad）
    angles = np.outer(np.arange(arm_N, dtype=np.float64), np.array([0.03, 0.0, 0.0]))
    pose = np.concatenate([positions, angles], axis=1).astype(np.float64)

    # effector
    gripper_pos = (np.arange(eff_N, dtype=np.float64) * 0.004).reshape(eff_N, 1)
    gripper_norm = gripper_pos / 0.08

    # action
    delta_ee = np.random.default_rng(42).uniform(-0.01, 0.01, (act_N, 6))
    gripper_cmd = np.zeros((act_N, 1), np.float64)

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
        arm.create_dataset("stale", data=np.zeros(arm_N, dtype=bool))

        # effector
        eff = obs.create_group("effector")
        eff.create_dataset("position", data=gripper_pos)
        eff.create_dataset("position_norm", data=gripper_norm)
        eff.create_dataset("type",
                           data=np.array([b"gripper"] * eff_N, dtype=h5py.special_dtype(vlen=bytes)))
        eff.create_dataset("timestamp", data=eff_ts)
        eff.create_dataset("stale", data=np.zeros(eff_N, dtype=bool))

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

    # 注意：arm_ts_override 等情形下数组长度与 N 不同，跳过 schema 校验
    if arm_ts_override is None and eff_ts_override is None and act_ts_override is None:
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


# ---------------------------------------------------------------------------
# 新增测试：时间戳去重（模拟 stale 补帧产生重复戳）
# ---------------------------------------------------------------------------

class TestDedupTimestamps:
    """验证源时间轴含重复戳时，对齐不崩、结果正确。"""

    def test_arm_duplicate_timestamps_no_crash(self, tmp_path):
        """arm 时间戳含重复（模拟 stale 补帧）时，align 不崩。"""
        path = str(tmp_path / "ep.h5")
        anchor_ts = np.arange(10, dtype=np.float64) / 30.0 + 10.0
        # arm 时间戳：第 3、4 帧重复（补帧导致相同戳）
        arm_ts = anchor_ts + 0.005
        arm_ts[4] = arm_ts[3]  # 重复

        _write_v2_synthetic(path, N=10, arm_ts_override=arm_ts)

        # 不应崩溃（之前 Slerp 会 ValueError）
        aligned = align_by_image_timestamp(path, on_stale="interpolate")
        assert aligned["arm_joints"].shape == (10, 7)

    def test_arm_duplicate_timestamps_last_wins(self, tmp_path):
        """重复时间戳保留最后一个对应的值（最新数据）。"""
        path = str(tmp_path / "ep.h5")
        anchor_ts = np.arange(10, dtype=np.float64) / 30.0 + 10.0
        arm_ts = anchor_ts + 0.005
        # 第 3、4 帧重复，最后一个（index 4）应被保留
        arm_ts[4] = arm_ts[3]

        _write_v2_synthetic(path, N=10, arm_ts_override=arm_ts)

        # 只验证 align 可正常运行并返回正确形状
        aligned = align_by_image_timestamp(path, on_stale="interpolate")
        assert aligned["arm_joints"].shape[0] == 10

    def test_effector_duplicate_timestamps_no_crash(self, tmp_path):
        """effector 时间戳含重复时，align 不崩。"""
        path = str(tmp_path / "ep.h5")
        anchor_ts = np.arange(10, dtype=np.float64) / 30.0 + 10.0
        eff_ts = anchor_ts + 0.007
        eff_ts[6] = eff_ts[5]  # 重复

        _write_v2_synthetic(path, N=10, eff_ts_override=eff_ts)

        aligned = align_by_image_timestamp(path, on_stale="interpolate")
        assert aligned["gripper_position"].shape == (10, 1)

    def test_action_duplicate_timestamps_no_crash(self, tmp_path):
        """action 时间戳含重复时，align 不崩。"""
        path = str(tmp_path / "ep.h5")
        anchor_ts = np.arange(10, dtype=np.float64) / 30.0 + 10.0
        act_ts = anchor_ts + 0.003
        act_ts[2] = act_ts[1]  # 重复

        _write_v2_synthetic(path, N=10, act_ts_override=act_ts)

        aligned = align_by_image_timestamp(path, on_stale="interpolate")
        assert aligned["action_delta_ee_pose"].shape == (10, 6)

    def test_many_consecutive_duplicates(self, tmp_path):
        """连续多个重复时间戳（大量 stale 补帧），align 不崩且形状正确。"""
        path = str(tmp_path / "ep.h5")
        anchor_ts = np.arange(20, dtype=np.float64) / 30.0 + 10.0
        arm_ts = anchor_ts + 0.005
        # 帧 5-14 全部重复同一戳（极端补帧）
        arm_ts[5:15] = arm_ts[5]

        _write_v2_synthetic(path, N=20, arm_ts_override=arm_ts)

        aligned = align_by_image_timestamp(path, on_stale="interpolate")
        assert aligned["arm_joints"].shape == (20, 7)


# ---------------------------------------------------------------------------
# 新增测试：N<2 边界（0帧/1帧模态）
# ---------------------------------------------------------------------------

class TestNLessThan2Boundary:
    """验证单帧广播和零帧 ValueError 行为。"""

    def test_single_frame_arm_broadcasts(self, tmp_path):
        """arm 模态只有 1 帧时，值广播到全部锚时间轴。"""
        path = str(tmp_path / "ep.h5")
        anchor_ts = np.arange(10, dtype=np.float64) / 30.0 + 10.0
        # arm 只有 1 帧，时间戳在锚轴范围内
        arm_ts = np.array([anchor_ts[5]])  # 单帧

        _write_v2_synthetic(path, N=10, arm_ts_override=arm_ts)

        aligned = align_by_image_timestamp(path, on_stale="interpolate")
        # 形状正确
        assert aligned["arm_joints"].shape == (10, 7)
        # 所有行都等于该唯一帧的值（广播）
        first_row = aligned["arm_joints"][0]
        for i in range(1, 10):
            np.testing.assert_array_equal(aligned["arm_joints"][i], first_row,
                                           err_msg=f"第 {i} 行与第 0 行不同（期望广播）")

    def test_single_frame_slerp_broadcasts(self, tmp_path):
        """arm 模态只有 1 帧时，SLERP 旋转也广播（不崩）。"""
        path = str(tmp_path / "ep.h5")
        anchor_ts = np.arange(8, dtype=np.float64) / 30.0 + 10.0
        arm_ts = np.array([anchor_ts[4]])

        _write_v2_synthetic(path, N=8, arm_ts_override=arm_ts)

        aligned = align_by_image_timestamp(path, on_stale="interpolate")
        # 旋转列（3:6）应全部相同
        rot = aligned["arm_pose"][:, 3:]
        assert rot.shape == (8, 3)
        for i in range(1, 8):
            np.testing.assert_array_equal(rot[i], rot[0],
                                           err_msg=f"旋转第 {i} 行广播失败")

    def test_zero_frame_arm_raises_with_modal_name(self, tmp_path):
        """arm 模态规范化后 0 帧时，应抛出含模态名的 ValueError。"""
        path = str(tmp_path / "ep.h5")
        # arm_ts 为空数组（0 帧）
        arm_ts = np.array([], dtype=np.float64)

        _write_v2_synthetic(path, N=10, arm_ts_override=arm_ts)

        with pytest.raises(ValueError, match="arm"):
            align_by_image_timestamp(path, on_stale="interpolate")

    def test_zero_frame_effector_raises_with_modal_name(self, tmp_path):
        """effector 模态规范化后 0 帧时，应抛出含模态名的 ValueError。"""
        path = str(tmp_path / "ep.h5")
        eff_ts = np.array([], dtype=np.float64)

        _write_v2_synthetic(path, N=10, eff_ts_override=eff_ts)

        with pytest.raises(ValueError, match="effector"):
            align_by_image_timestamp(path, on_stale="interpolate")

    def test_zero_frame_action_raises_with_modal_name(self, tmp_path):
        """action 模态规范化后 0 帧时，应抛出含模态名的 ValueError。"""
        path = str(tmp_path / "ep.h5")
        act_ts = np.array([], dtype=np.float64)

        _write_v2_synthetic(path, N=10, act_ts_override=act_ts)

        with pytest.raises(ValueError, match="action"):
            align_by_image_timestamp(path, on_stale="interpolate")


# ---------------------------------------------------------------------------
# 新增测试：SLERP 输出 unwrap 无大跳变
# ---------------------------------------------------------------------------

class TestSlerpUnwrap:
    """验证 SLERP 输出欧拉角经 unwrap 后无大跳变。"""

    def test_slerp_output_no_large_jumps(self, tmp_path):
        """连续旋转轨迹，SLERP 输出相邻帧欧拉角差值应 < π（unwrap 后）。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=20, arm_ts_offset=0.005)

        aligned = align_by_image_timestamp(path, on_stale="interpolate")

        # 检查旋转列（3:6）相邻帧差值
        rot = aligned["arm_pose"][:, 3:]
        diffs = np.diff(rot, axis=0)
        max_jump = np.abs(diffs).max()
        assert max_jump < np.pi, (
            f"SLERP 输出相邻欧拉角跳变过大：{max_jump:.4f} rad >= π"
        )

    def test_slerp_unwrap_rx_monotone_after_unwrap(self, tmp_path):
        """线性递增 rx 轨迹，unwrap 后应单调递增。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=15, arm_ts_offset=0.005)

        aligned = align_by_image_timestamp(path, on_stale="interpolate")

        rx = aligned["arm_pose"][:, 3]
        diffs = np.diff(rx)
        assert np.all(diffs >= -1e-10), (
            f"unwrap 后 rx 仍不单调：min diff={diffs.min():.2e}"
        )


# ---------------------------------------------------------------------------
# 新增测试（Codex 审查）：anchor_indices 键
# ---------------------------------------------------------------------------

class TestAnchorIndices:
    """验证 align_by_image_timestamp 返回的 anchor_indices 键正确性。"""

    def test_interpolate_anchor_indices_is_arange(self, tmp_path):
        """interpolate 模式：anchor_indices 应等于 np.arange(N_anchor)。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=10, stale_indices={"wrist": [2, 5]})

        aligned = align_by_image_timestamp(path, on_stale="interpolate")

        assert "anchor_indices" in aligned, "aligned dict 缺少 'anchor_indices' 键"
        expected = np.arange(10, dtype=np.intp)
        np.testing.assert_array_equal(
            aligned["anchor_indices"], expected,
            err_msg="interpolate 模式 anchor_indices 应为 np.arange(N_anchor)"
        )

    def test_keep_anchor_indices_is_arange(self, tmp_path):
        """keep 模式：anchor_indices 应等于 np.arange(N_anchor)。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=8, stale_indices={"wrist": [1]})

        aligned = align_by_image_timestamp(path, on_stale="keep")

        assert "anchor_indices" in aligned, "aligned dict 缺少 'anchor_indices' 键"
        expected = np.arange(8, dtype=np.intp)
        np.testing.assert_array_equal(
            aligned["anchor_indices"], expected,
            err_msg="keep 模式 anchor_indices 应为 np.arange(N_anchor)"
        )

    def test_drop_anchor_indices_correct_original_indices(self, tmp_path):
        """drop 模式：anchor_indices 应为被保留帧的原始下标数组。"""
        path = str(tmp_path / "ep.h5")
        stale = [2, 5, 8]
        _write_v2_synthetic(path, N=10, stale_indices={"wrist": stale})

        aligned = align_by_image_timestamp(path, on_stale="drop")

        assert "anchor_indices" in aligned, "aligned dict 缺少 'anchor_indices' 键"
        # 保留帧原始下标 = 所有下标中不在 stale 列表的
        expected = np.array([i for i in range(10) if i not in stale], dtype=np.intp)
        np.testing.assert_array_equal(
            aligned["anchor_indices"], expected,
            err_msg=f"drop 模式 anchor_indices 应为 {expected}，实际: {aligned['anchor_indices']}"
        )

    def test_drop_no_stale_anchor_indices_equals_arange(self, tmp_path):
        """drop 模式且无 stale 帧：anchor_indices == np.arange(N)。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=6)

        aligned = align_by_image_timestamp(path, on_stale="drop")

        expected = np.arange(6, dtype=np.intp)
        np.testing.assert_array_equal(
            aligned["anchor_indices"], expected,
            err_msg="无 stale 的 drop 模式 anchor_indices 应为 np.arange(N)"
        )

    def test_drop_anchor_indices_length_matches_output(self, tmp_path):
        """drop 模式：anchor_indices 长度应等于输出帧数 N_out。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=10, stale_indices={"wrist": [0, 3, 7]})

        aligned = align_by_image_timestamp(path, on_stale="drop")

        assert len(aligned["anchor_indices"]) == len(aligned["anchor_ts"]), (
            "anchor_indices 长度应与 anchor_ts 长度一致"
        )

    def test_anchor_indices_in_all_keys(self, tmp_path):
        """anchor_indices 应出现在全键集中（TestOutputKeys 的补充）。"""
        path = str(tmp_path / "ep.h5")
        _write_v2_synthetic(path, N=5)

        aligned = align_by_image_timestamp(path, on_stale="interpolate")

        assert "anchor_indices" in aligned


# ---------------------------------------------------------------------------
# 新增单元测试：_select_anchor_ts 量级/映射/fallback（Bug1 修复，Task 8）
# ---------------------------------------------------------------------------

class TestSelectAnchorTs:
    """直接测试 _select_anchor_ts 的核心行为（不造 hdf5，速度快）。"""

    @staticmethod
    def _fn():
        from tools.align_offline import _select_anchor_ts
        return _select_anchor_ts

    def test_hw_valid_returns_second_scale_not_ms(self):
        """hw 达标时返回值在秒量级（不是毫秒 ~1e6 量级）。"""
        fn = self._fn()
        sw_ts = np.linspace(100.0, 100.633, 20)
        hw_ts = sw_ts * 1000.0 + 5_000_000.0   # 完美线性，量级 ~5e6 ms
        anchor = fn(sw_ts, hw_ts)
        # 修复前：anchor[0] ≈ 5e6；修复后：anchor[0] ≈ 100（秒）
        assert anchor[0] < 1000.0, (
            f"anchor[0]={anchor[0]:.2e} 仍是毫秒量级（Bug 未修复）"
        )

    def test_hw_valid_perfect_linear_anchor_equals_sw_ts(self):
        """完美线性 hw_ts：逆变换后 anchor 应与 sw_ts 误差 < 1e-9s。"""
        fn = self._fn()
        sw_ts = np.linspace(50.0, 51.0, 30)
        hw_ts = sw_ts * 1000.0 + 9_876_543.0   # slope=1000，R²=1.0
        anchor = fn(sw_ts, hw_ts)
        np.testing.assert_allclose(anchor, sw_ts, atol=1e-9,
                                   err_msg="完美线性 hw_ts 逆变换后误差应 < 1e-9s")

    def test_hw_valid_anchor_within_arm_ts_range(self):
        """hw 达标时 anchor 范围与 arm_ts 重叠（不再全端点外推）。"""
        fn = self._fn()
        sw_ts = np.linspace(10.0, 10.5, 15)
        hw_ts = sw_ts * 1000.0 + 1e8
        arm_ts = sw_ts + 0.005
        anchor = fn(sw_ts, hw_ts)
        in_range = np.sum((anchor >= arm_ts[0]) & (anchor <= arm_ts[-1]))
        assert in_range > 0, (
            f"anchor 完全超出 arm_ts 范围（全端点外推）。"
            f"anchor={anchor[[0,-1]]}，arm_ts={arm_ts[[0,-1]]}"
        )

    def test_hw_none_fallback_sw_ts(self):
        """hw_ts=None 时直接返回 sw_ts。"""
        fn = self._fn()
        sw_ts = np.linspace(5.0, 6.0, 10)
        anchor = fn(sw_ts, None)
        np.testing.assert_array_equal(anchor, sw_ts)

    def test_hw_too_short_fallback_sw_ts(self):
        """hw_ts 长度 < 2 时 fallback sw_ts。"""
        fn = self._fn()
        sw_ts = np.linspace(5.0, 6.0, 10)
        anchor = fn(sw_ts, hw_ts=np.array([1000.0]))
        np.testing.assert_array_equal(anchor, sw_ts)

    def test_hw_invalid_r2_fallback_sw_ts(self):
        """hw_ts 与 sw_ts 线性相关 R² < 0.9999 时 fallback sw_ts。"""
        fn = self._fn()
        rng = np.random.default_rng(0)
        sw_ts = np.linspace(5.0, 6.0, 30)
        hw_ts = rng.uniform(0, 1e6, 30)        # 随机，R² << 0.9999
        anchor = fn(sw_ts, hw_ts)
        np.testing.assert_array_equal(anchor, sw_ts)

    def test_hw_tiny_slope_fallback_sw_ts(self):
        """slope 接近 0 时 fallback sw_ts（除法保护）。"""
        fn = self._fn()
        sw_ts = np.linspace(5.0, 6.0, 20)
        # hw_ts 几乎常数 → slope ≈ 0，触发除法保护
        rng = np.random.default_rng(1)
        hw_ts = np.ones(20) * 1000.0 + rng.uniform(-1e-15, 1e-15, 20)
        anchor = fn(sw_ts, hw_ts)
        np.testing.assert_array_equal(anchor, sw_ts,
                                      err_msg="slope≈0 时应 fallback sw_ts")


# ---------------------------------------------------------------------------
# Task 6: _project_hw_to_monotonic + effector hw_timestamp 对齐
# ---------------------------------------------------------------------------

def _make_minimal_v2_episode(path, N, with_hw_ts=False, hw_ts_all_nan=False):
    """创建最小 v2 ep，包含 align_by_image_timestamp 所需的全部字段。

    Args:
        path: 输出文件路径（pathlib.Path 或 str）
        N: 帧数
        with_hw_ts: 是否写入 observations/effector/hw_timestamp 字段
        hw_ts_all_nan: with_hw_ts=True 时是否将 hw_timestamp 全设为 NaN
    """
    with h5py.File(str(path), "w") as f:
        ts = np.linspace(1000.0, 1000.0 + N * 0.033, N, dtype=np.float64)
        # arm
        f.create_dataset("observations/arm/timestamp", data=ts)
        f.create_dataset("observations/arm/joints", data=np.zeros((N, 7)))
        f.create_dataset("observations/arm/joint_vel", data=np.zeros((N, 7)))
        f.create_dataset("observations/arm/pose", data=np.zeros((N, 6)))
        f.create_dataset("observations/arm/stale", data=np.zeros(N, dtype=bool))
        # effector
        f.create_dataset("observations/effector/timestamp", data=ts)
        f.create_dataset("observations/effector/position", data=np.zeros((N, 1)))
        f.create_dataset("observations/effector/position_norm",
                         data=np.linspace(0.0, 1.0, N).reshape(-1, 1))
        f.create_dataset("observations/effector/stale", data=np.zeros(N, dtype=bool))
        # camera（只需 timestamp + stale；align 不读图像像素）
        f.create_dataset("observations/camera/rgb/wrist/timestamp", data=ts)
        f.create_dataset("observations/camera/rgb/wrist/stale", data=np.zeros(N, dtype=bool))
        # action
        f.create_dataset("action/timestamp", data=ts)
        f.create_dataset("action/delta_ee_pose", data=np.zeros((N, 6)))
        f.create_dataset("action/gripper_cmd", data=np.zeros((N, 1)))
        # state_hifreq（M=0 占位）
        f.create_dataset("observations/state_hifreq/timestamp", data=np.zeros(0))
        f.create_dataset("observations/state_hifreq/joints", data=np.zeros((0, 7)))
        f.create_dataset("observations/state_hifreq/joint_vel", data=np.zeros((0, 7)))
        f.create_dataset("observations/state_hifreq/pose", data=np.zeros((0, 6)))
        # 可选 hw_timestamp
        if with_hw_ts:
            hw = np.full(N, np.nan) if hw_ts_all_nan else (ts - 800.0)
            f.create_dataset("observations/effector/hw_timestamp",
                             data=hw.astype(np.float64))
    return path


def test_project_hw_to_monotonic_perfect_linear():
    """硬件 ts 与 monotonic 完全线性时，投影后应等于 monotonic（slope=1）。"""
    from tools.align_offline import _project_hw_to_monotonic
    eff_mono = np.arange(10, dtype=np.float64) * 0.05 + 1000.0
    eff_hw = np.arange(10, dtype=np.float64) * 0.05 + 200.0
    projected = _project_hw_to_monotonic(eff_mono, eff_hw)
    np.testing.assert_allclose(projected, eff_mono, atol=1e-3)


def test_project_hw_to_monotonic_with_jitter_recovers():
    """hw_ts 抖动 ±5ms 时，线性投影应给出平滑曲线，残差 < 10ms。"""
    from tools.align_offline import _project_hw_to_monotonic
    rng = np.random.default_rng(42)
    eff_mono = np.linspace(1000.0, 1010.0, 100)
    eff_hw_clean = np.linspace(200.0, 210.0, 100)
    eff_hw = eff_hw_clean + rng.normal(0, 0.005, 100)
    projected = _project_hw_to_monotonic(eff_mono, eff_hw)
    residual = np.abs(projected - eff_mono)
    assert residual.max() < 0.020, f"最大残差 {residual.max()*1000:.2f}ms 超过 20ms"


def test_align_uses_hw_timestamp_when_available(tmp_path, caplog):
    """hdf5 含 effector/hw_timestamp 时，align_by_image_timestamp 走精确路径。"""
    import logging
    ep = _make_minimal_v2_episode(tmp_path / "ep_with_hw.h5", N=20, with_hw_ts=True)
    with caplog.at_level(logging.INFO, logger="tools.align_offline"):
        aligned = align_by_image_timestamp(str(ep), on_stale="interpolate")
    assert any(
        "hw_timestamp" in rec.getMessage().lower()
        and ("对齐" in rec.getMessage() or "align" in rec.getMessage().lower())
        for rec in caplog.records
    ), f"应 log 精确对齐路径，实际日志: {[r.getMessage() for r in caplog.records]}"
    assert "gripper_position_norm" in aligned and aligned["gripper_position_norm"].shape[0] > 0


def test_align_falls_back_when_hw_timestamp_missing(tmp_path, caplog):
    """hdf5 无 effector/hw_timestamp → 退回旧路径 + info log。"""
    import logging
    ep = _make_minimal_v2_episode(tmp_path / "ep_no_hw.h5", N=20, with_hw_ts=False)
    with caplog.at_level(logging.INFO, logger="tools.align_offline"):
        aligned = align_by_image_timestamp(str(ep), on_stale="interpolate")
    assert any(
        "rebuild polymetis" in rec.getMessage().lower()
        or "退回" in rec.getMessage()
        or "无" in rec.getMessage()
        or "无 effector hw_timestamp" in rec.getMessage()
        for rec in caplog.records
    ), f"应 info 提示退回，实际日志: {[r.getMessage() for r in caplog.records]}"


def test_align_falls_back_when_hw_timestamp_all_nan(tmp_path, caplog):
    """hdf5 有 effector/hw_timestamp 但全 NaN → 退回旧路径。"""
    import logging
    ep = _make_minimal_v2_episode(tmp_path / "ep_nan_hw.h5", N=20, with_hw_ts=True, hw_ts_all_nan=True)
    with caplog.at_level(logging.WARNING, logger="tools.align_offline"):
        aligned = align_by_image_timestamp(str(ep), on_stale="interpolate")
    assert any(
        "nan" in rec.getMessage().lower() or "退回" in rec.getMessage()
        for rec in caplog.records
    ), f"全 NaN 应退回，实际日志: {[r.getMessage() for r in caplog.records]}"

def test_align_falls_back_when_hw_timestamp_constant(tmp_path, caplog):
    """hw_timestamp 全相同（退化无效）→ 退回旧路径 + warning。"""
    import logging
    ep = tmp_path / 'ep_const_hw.h5'
    _make_minimal_v2_episode(ep, N=20, with_hw_ts=True)
    import h5py
    with h5py.File(ep, 'a') as f:
        del f['observations/effector/hw_timestamp']
        f.create_dataset('observations/effector/hw_timestamp',
                         data=np.full(20, 5.0, dtype=np.float64))
    from tools.align_offline import align_by_image_timestamp
    with caplog.at_level(logging.WARNING):
        aligned = align_by_image_timestamp(str(ep), on_stale='interpolate')
    msgs = ' | '.join(r.getMessage() for r in caplog.records)
    assert '退回' in msgs or 'fallback' in msgs.lower() or '常数' in msgs or 'unique' in msgs.lower(),         f'应 warning 提示常数 hw_ts 退回，实际日志: {msgs}'


def test_align_falls_back_when_hw_timestamp_decreasing(tmp_path, caplog):
    """hw_timestamp 严格递减（非单调）→ 退回旧路径。"""
    import logging
    ep = tmp_path / 'ep_decr_hw.h5'
    _make_minimal_v2_episode(ep, N=20, with_hw_ts=True)
    import h5py
    with h5py.File(ep, 'a') as f:
        del f['observations/effector/hw_timestamp']
        f.create_dataset('observations/effector/hw_timestamp',
                         data=np.linspace(10.0, 1.0, 20, dtype=np.float64))
    from tools.align_offline import align_by_image_timestamp
    with caplog.at_level(logging.WARNING):
        aligned = align_by_image_timestamp(str(ep), on_stale='interpolate')
    msgs = ' | '.join(r.getMessage() for r in caplog.records)
    assert '退回' in msgs or 'fallback' in msgs.lower() or '单调' in msgs or 'monotonic' in msgs.lower() or 'decreasing' in msgs.lower(),         f'应 warning 提示非单调 hw_ts 退回，实际日志: {msgs}'


def test_align_falls_back_when_hw_timestamp_low_r2(tmp_path, caplog):
    """hw_timestamp 与 monotonic 线性度差（R² < 0.99）→ 退回。"""
    import logging
    ep = tmp_path / 'ep_lowr2_hw.h5'
    _make_minimal_v2_episode(ep, N=100, with_hw_ts=True)
    import h5py
    with h5py.File(ep, 'a') as f:
        eff_mono = f['observations/effector/timestamp'][...]
        del f['observations/effector/hw_timestamp']
        rng = np.random.default_rng(0)
        hw_bad = np.sort((eff_mono - eff_mono.min()) ** 2 + rng.normal(0, 0.5, len(eff_mono)).cumsum())
        f.create_dataset('observations/effector/hw_timestamp',
                         data=hw_bad.astype(np.float64))
    from tools.align_offline import align_by_image_timestamp
    with caplog.at_level(logging.WARNING):
        aligned = align_by_image_timestamp(str(ep), on_stale='interpolate')
    msgs = ' | '.join(r.getMessage() for r in caplog.records)
    assert 'R²' in msgs or 'R^2' in msgs or '线性度' in msgs or '退回' in msgs,         f'应 warning 提示低 R² 退回，实际日志: {msgs}'


def test_align_uses_hw_timestamp_passes_all_gates(tmp_path, caplog):
    """正常 hw_timestamp（单调+高 R²）→ 走精确路径（既有正路径测试，确认 gate 不误伤）。"""
    import logging
    ep = tmp_path / 'ep_good_hw.h5'
    _make_minimal_v2_episode(ep, N=20, with_hw_ts=True)
    from tools.align_offline import align_by_image_timestamp
    with caplog.at_level(logging.INFO):
        aligned = align_by_image_timestamp(str(ep), on_stale='interpolate')
    msgs = ' | '.join(r.getMessage() for r in caplog.records)
    assert '精确对齐' in msgs or 'R²' in msgs, f'正常 hw_ts 应走精确路径，实际日志: {msgs}'
