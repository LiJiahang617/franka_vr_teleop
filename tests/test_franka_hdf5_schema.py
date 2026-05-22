"""franka-hdf5-v2 schema 基础校验测试（原 v1 测试更新为 v2）。

原 v1 测试套件于 PhaseD-T2 升级为 v2：
  - 去除 observations/timestamp (N,1) 共戳相关测试
  - 加入每模态独立 ts + stale + hw_timestamp 相关基础覆盖
  - 保留 shape/dtype/calibration/state_hifreq 等核心校验
"""
import h5py
import numpy as np
import pytest
import franka_hdf5_schema as S


def _write_conformant(path, N=5, M=40):
    """生成完整合规 franka-hdf5-v2 文件（含单相机 wrist）。"""
    ts_arm = np.arange(N, dtype=np.float64) * 0.033 + 1.0
    ts_eff = ts_arm + 0.001
    ts_cam = ts_arm + 0.0005
    ts_act = ts_arm + 0.002

    with h5py.File(path, "w") as f:
        infos = f.create_group("infos")
        infos.create_dataset("schema_version", data=np.bytes_(S.SCHEMA_VERSION))
        ti = infos.create_group("task_info")
        ti.create_dataset("task_name", data=np.bytes_("pick"))
        ti.create_dataset("collection_frequency", data=np.array([30.0, 29.7], np.float64))
        ti.create_dataset("total_frames", data=np.int64(N))
        ti.create_dataset("robot", data=np.bytes_("franka_panda"))
        infos.create_group("camera_params")
        cal = infos.create_group("calibration")
        cal.create_dataset("oc2base_R", data=np.eye(3, dtype=np.float64))
        cal.create_dataset("quality", data=np.bytes_("{}"))
        cal.create_dataset("vr_source", data=np.bytes_("unity"))

        obs = f.create_group("observations")

        # arm 模态（独立 ts + stale）
        arm = obs.create_group("arm")
        arm.create_dataset("joints", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("joint_vel", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("pose", data=np.zeros((N, 6), np.float64))
        arm.create_dataset("timestamp", data=ts_arm)
        arm.create_dataset("stale", data=np.zeros(N, dtype=bool))

        # effector 模态（独立 ts + stale）
        eff = obs.create_group("effector")
        eff.create_dataset("position", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("position_norm", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("type", data=np.array([b"gripper"] * N,
                           dtype=h5py.special_dtype(vlen=bytes)))
        eff.create_dataset("timestamp", data=ts_eff)
        eff.create_dataset("stale", data=np.zeros(N, dtype=bool))

        # camera（单相机 wrist，独立 ts + stale + hw_timestamp）
        cam_g = obs.create_group("camera")
        rgb_g = cam_g.create_group("rgb")
        _VLEN = h5py.special_dtype(vlen=np.dtype("uint8"))
        wrist = rgb_g.create_group("wrist")
        imgs = wrist.create_dataset("images", (N,), dtype=_VLEN)
        dummy = bytes([0xFF, 0xD8, 0xFF, 0xD9])
        for i in range(N):
            imgs[i] = np.frombuffer(dummy, np.uint8)
        wrist.create_dataset("timestamp", data=ts_cam)
        wrist.create_dataset("stale", data=np.zeros(N, dtype=bool))
        wrist.create_dataset("hw_timestamp", data=ts_cam * 1000.0)

        # state_hifreq（含 wrench 占位）
        hf = obs.create_group("state_hifreq")
        hf.create_dataset("joints", data=np.zeros((M, 7), np.float64))
        hf.create_dataset("joint_vel", data=np.zeros((M, 7), np.float64))
        hf.create_dataset("pose", data=np.zeros((M, 6), np.float64))
        hf.create_dataset("timestamp", data=np.arange(M, dtype=np.float64))
        hf.create_dataset("poly_ts", data=np.arange(M, dtype=np.float64))
        hf.create_dataset("wrench", data=np.zeros((M, 6), np.float64))

        # action 模态（独立 ts）
        act = f.create_group("action")
        act.create_dataset("delta_ee_pose", data=np.zeros((N, 6), np.float64))
        act.create_dataset("gripper_cmd", data=np.zeros((N, 1), np.float64))
        act.create_dataset("timestamp", data=ts_act)


def test_conformant_passes(tmp_path):
    """完整合规 v2 episode 校验通过。"""
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    assert S.validate_episode(p) == []


def test_missing_schema_version(tmp_path):
    """缺少 schema_version 必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    with h5py.File(p, "a") as f:
        del f["infos/schema_version"]
    v = S.validate_episode(p)
    assert any("schema_version" in x for x in v)


def test_wrong_joint_shape(tmp_path):
    """arm/joints shape 错误（5,6 而非 5,7）必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    with h5py.File(p, "a") as f:
        del f["observations/arm/joints"]
        f["observations/arm"].create_dataset("joints", data=np.zeros((5, 6), np.float64))
    v = S.validate_episode(p)
    assert any("observations/arm/joints" in x and "shape" in x for x in v)


def test_n_misaligned_action(tmp_path):
    """action/delta_ee_pose 帧数与 N 不符必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    with h5py.File(p, "a") as f:
        del f["action/delta_ee_pose"]
        f["action"].create_dataset("delta_ee_pose", data=np.zeros((4, 6), np.float64))
    v = S.validate_episode(p)
    assert any("action/delta_ee_pose" in x for x in v)


def test_missing_calibration_R(tmp_path):
    """缺少 oc2base_R 必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    with h5py.File(p, "a") as f:
        del f["infos/calibration/oc2base_R"]
    v = S.validate_episode(p)
    assert any("oc2base_R" in x for x in v)


def test_hifreq_independent_length_ok(tmp_path):
    """M != N 时 state_hifreq 仍然通过。"""
    p = str(tmp_path / "ep.h5")
    _write_conformant(p, N=5, M=123)
    assert S.validate_episode(p) == []


def test_dtype_only_violation_caught(tmp_path):
    """arm/pose dtype float32 而非 float64 必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    with h5py.File(p, "a") as f:
        d = f["observations/arm/pose"][...]
        del f["observations/arm/pose"]
        f["observations/arm"].create_dataset("pose", data=d.astype(np.float32))
    v = S.validate_episode(p)
    assert any("observations/arm/pose" in x and "float64" in x for x in v)


def test_hifreq_internal_length_mismatch_caught(tmp_path):
    """state_hifreq 内部各字段长度不一致必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_conformant(p, N=5, M=40)
    with h5py.File(p, "a") as f:
        del f["observations/state_hifreq/joint_vel"]
        f["observations/state_hifreq"].create_dataset(
            "joint_vel", data=np.zeros((39, 7), np.float64))  # M-1，不一致
    v = S.validate_episode(p)
    assert any("observations/state_hifreq/joint_vel" in x for x in v)


def test_missing_arm_stale(tmp_path):
    """arm 缺少 stale 字段必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    with h5py.File(p, "a") as f:
        del f["observations/arm/stale"]
    v = S.validate_episode(p)
    assert any("observations/arm/stale" in x for x in v)


def test_missing_camera_hw_timestamp(tmp_path):
    """camera/{cn} 缺少 hw_timestamp 必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    with h5py.File(p, "a") as f:
        del f["observations/camera/rgb/wrist/hw_timestamp"]
    v = S.validate_episode(p)
    assert any("hw_timestamp" in x for x in v)
