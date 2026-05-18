import h5py
import numpy as np
import pytest
import franka_hdf5_schema as S


def _write_conformant(path, N=5, M=40):
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
        obs.create_dataset("timestamp", data=np.arange(N, dtype=np.float64).reshape(N, 1) + 1.0)
        arm = obs.create_group("arm")
        arm.create_dataset("joints", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("joint_vel", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("pose", data=np.zeros((N, 6), np.float64))
        arm.create_dataset("timestamp", data=np.arange(N, dtype=np.float64))
        eff = obs.create_group("effector")
        eff.create_dataset("position", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("position_norm", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("type", data=np.array([b"gripper"] * N, dtype=h5py.special_dtype(vlen=bytes)))
        eff.create_dataset("timestamp", data=np.arange(N, dtype=np.float64))
        obs.create_group("camera")
        hf = obs.create_group("state_hifreq")
        hf.create_dataset("joints", data=np.zeros((M, 7), np.float64))
        hf.create_dataset("joint_vel", data=np.zeros((M, 7), np.float64))
        hf.create_dataset("pose", data=np.zeros((M, 6), np.float64))
        hf.create_dataset("timestamp", data=np.arange(M, dtype=np.float64))
        hf.create_dataset("poly_ts", data=np.arange(M, dtype=np.float64))
        act = f.create_group("action")
        act.create_dataset("delta_ee_pose", data=np.zeros((N, 6), np.float64))
        act.create_dataset("gripper_cmd", data=np.zeros((N, 1), np.float64))
        act.create_dataset("timestamp", data=np.arange(N, dtype=np.float64))


def test_conformant_passes(tmp_path):
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    assert S.validate_episode(p) == []


def test_missing_schema_version(tmp_path):
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    with h5py.File(p, "a") as f:
        del f["infos/schema_version"]
    v = S.validate_episode(p)
    assert any("schema_version" in x for x in v)


def test_wrong_joint_shape(tmp_path):
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    with h5py.File(p, "a") as f:
        del f["observations/arm/joints"]
        f["observations/arm"].create_dataset("joints", data=np.zeros((5, 6), np.float64))
    v = S.validate_episode(p)
    assert any("observations/arm/joints" in x and "shape" in x for x in v)


def test_n_misaligned_action(tmp_path):
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    with h5py.File(p, "a") as f:
        del f["action/delta_ee_pose"]
        f["action"].create_dataset("delta_ee_pose", data=np.zeros((4, 6), np.float64))
    v = S.validate_episode(p)
    assert any("action/delta_ee_pose" in x for x in v)


def test_non_monotonic_master_ts(tmp_path):
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    with h5py.File(p, "a") as f:
        ts = f["observations/timestamp"][...]
        ts[2] = ts[1]
        f["observations/timestamp"][...] = ts
    v = S.validate_episode(p)
    assert any("单调" in x for x in v)


def test_missing_calibration_R(tmp_path):
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    with h5py.File(p, "a") as f:
        del f["infos/calibration/oc2base_R"]
    v = S.validate_episode(p)
    assert any("oc2base_R" in x for x in v)


def test_hifreq_independent_length_ok(tmp_path):
    p = str(tmp_path / "ep.h5")
    _write_conformant(p, N=5, M=123)
    assert S.validate_episode(p) == []



def test_master_timestamp_must_be_2d_n1(tmp_path):
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    with h5py.File(p, "a") as f:
        ts = f["observations/timestamp"][...].reshape(-1)  # 退化成 1D (N,)
        del f["observations/timestamp"]
        f["observations"].create_dataset("timestamp", data=ts)
    v = S.validate_episode(p)
    assert any("observations/timestamp" in x and "shape" in x for x in v)


def test_dtype_only_violation_caught(tmp_path):
    p = str(tmp_path / "ep.h5")
    _write_conformant(p)
    with h5py.File(p, "a") as f:
        d = f["observations/arm/pose"][...]
        del f["observations/arm/pose"]
        f["observations/arm"].create_dataset("pose", data=d.astype(np.float32))
    v = S.validate_episode(p)
    assert any("observations/arm/pose" in x and "float64" in x for x in v)


def test_hifreq_internal_length_mismatch_caught(tmp_path):
    p = str(tmp_path / "ep.h5")
    _write_conformant(p, N=5, M=40)
    with h5py.File(p, "a") as f:
        del f["observations/state_hifreq/joint_vel"]
        f["observations/state_hifreq"].create_dataset(
            "joint_vel", data=np.zeros((39, 7), np.float64))  # M-1, 不一致
    v = S.validate_episode(p)
    assert any("observations/state_hifreq/joint_vel" in x for x in v)