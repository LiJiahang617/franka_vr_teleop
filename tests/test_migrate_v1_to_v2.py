"""测试 franka-hdf5 v1→v2 离线迁移工具。

合成一个符合 v1 格式的 HDF5 文件，调用 migrate()，
验证输出 v2 文件满足 validate_episode() 并逐字段检查关键转换规则。
"""
import sys
import os

import h5py
import numpy as np
import pytest

# 确保 scripts/ 和仓库根在 path 里
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_scripts = os.path.join(_repo, "scripts")
for _d in (_repo, _scripts):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import franka_hdf5_schema as S
from tools.migrate_v1_to_v2 import migrate

_VLEN_BYTES = h5py.special_dtype(vlen=np.dtype("uint8"))


def _write_v1_episode(path, N=8, M=3, cam_names=("wrist", "exterior")):
    """合成一个符合 franka-hdf5-v1 格式的 episode 文件。

    v1 特征：
    - infos/schema_version = b"franka-hdf5-v1"
    - observations/timestamp (N,1) 共用戳
    - arm/effector/camera 各有 timestamp (N,) 副本（内容相同）
    - 无 stale 字段、无 hw_timestamp 字段
    - state_hifreq 无 wrench 字段
    """
    shared_ts = np.arange(N, dtype=np.float64) * 0.033 + 10.0  # (N,) 从 10s 起

    with h5py.File(path, "w") as f:
        # infos
        infos = f.create_group("infos")
        infos.create_dataset("schema_version", data=np.bytes_("franka-hdf5-v1"))
        ti = infos.create_group("task_info")
        ti.create_dataset("task_name", data=np.bytes_("pick_v1"))
        ti.create_dataset("collection_frequency", data=np.array([30.0, 29.5], np.float64))
        ti.create_dataset("total_frames", data=np.int64(N))
        ti.create_dataset("robot", data=np.bytes_("franka_panda"))
        infos.create_group("camera_params")
        cal = infos.create_group("calibration")
        cal.create_dataset("oc2base_R", data=np.eye(3, dtype=np.float64))
        cal.create_dataset("quality", data=np.bytes_("{}"))
        cal.create_dataset("vr_source", data=np.bytes_("unityvr"))

        # observations
        obs = f.create_group("observations")
        # v1 共用时间戳 (N,1)
        obs.create_dataset("timestamp", data=shared_ts.reshape(N, 1))

        # arm（无 stale）
        arm = obs.create_group("arm")
        arm.create_dataset("joints", data=np.random.randn(N, 7).astype(np.float64))
        arm.create_dataset("joint_vel", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("pose", data=np.random.randn(N, 6).astype(np.float64))
        arm.create_dataset("timestamp", data=shared_ts.copy())  # (N,) 副本

        # effector（无 stale）
        eff = obs.create_group("effector")
        eff.create_dataset("position", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("position_norm", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("type", data=np.array([b"gripper"] * N,
                           dtype=h5py.special_dtype(vlen=bytes)))
        eff.create_dataset("timestamp", data=shared_ts.copy())  # (N,) 副本

        # camera（无 stale，无 hw_timestamp）
        cam = obs.create_group("camera")
        rgb = cam.create_group("rgb")
        dummy_jpg = bytes([0xFF, 0xD8, 0xFF, 0xD9])  # 最小合法 JPEG 标记
        for cn in cam_names:
            g = rgb.create_group(cn)
            imgs = g.create_dataset("images", (N,), dtype=_VLEN_BYTES)
            for i in range(N):
                imgs[i] = np.frombuffer(dummy_jpg, np.uint8)
            g.create_dataset("timestamp", data=shared_ts.copy())  # (N,) 副本

        # state_hifreq（无 wrench）
        hf = obs.create_group("state_hifreq")
        hf.create_dataset("joints", data=np.zeros((M, 7), np.float64))
        hf.create_dataset("joint_vel", data=np.zeros((M, 7), np.float64))
        hf.create_dataset("pose", data=np.zeros((M, 6), np.float64))
        hf_ts = np.arange(M, dtype=np.float64) * 0.005 + 10.0
        hf.create_dataset("timestamp", data=hf_ts)
        hf.create_dataset("poly_ts", data=hf_ts.copy())

        # action
        act = f.create_group("action")
        act.create_dataset("delta_ee_pose", data=np.zeros((N, 6), np.float64))
        act.create_dataset("gripper_cmd", data=np.zeros((N, 1), np.float64))
        act.create_dataset("timestamp", data=shared_ts.copy())


def _write_v1_episode_no_subts(path, N=5):
    """合成 v1 episode，arm/effector 没有自己的 timestamp（极端情况）。"""
    shared_ts = np.arange(N, dtype=np.float64) * 0.033 + 20.0

    with h5py.File(path, "w") as f:
        infos = f.create_group("infos")
        infos.create_dataset("schema_version", data=np.bytes_("franka-hdf5-v1"))
        ti = infos.create_group("task_info")
        ti.create_dataset("task_name", data=np.bytes_("pick_v1_nosubts"))
        ti.create_dataset("collection_frequency", data=np.array([30.0, 29.0], np.float64))
        ti.create_dataset("total_frames", data=np.int64(N))
        ti.create_dataset("robot", data=np.bytes_("franka_panda"))
        infos.create_group("camera_params")
        cal = infos.create_group("calibration")
        cal.create_dataset("oc2base_R", data=np.eye(3, dtype=np.float64))
        cal.create_dataset("quality", data=np.bytes_("{}"))
        cal.create_dataset("vr_source", data=np.bytes_("unityvr"))

        obs = f.create_group("observations")
        obs.create_dataset("timestamp", data=shared_ts.reshape(N, 1))

        arm = obs.create_group("arm")
        arm.create_dataset("joints", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("joint_vel", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("pose", data=np.zeros((N, 6), np.float64))
        # 不写 arm/timestamp

        eff = obs.create_group("effector")
        eff.create_dataset("position", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("position_norm", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("type", data=np.array([b"gripper"] * N,
                           dtype=h5py.special_dtype(vlen=bytes)))
        # 不写 effector/timestamp

        cam = obs.create_group("camera")
        rgb = cam.create_group("rgb")
        g = rgb.create_group("wrist")
        dummy_jpg = bytes([0xFF, 0xD8, 0xFF, 0xD9])
        imgs = g.create_dataset("images", (N,), dtype=_VLEN_BYTES)
        for i in range(N):
            imgs[i] = np.frombuffer(dummy_jpg, np.uint8)
        # 不写 camera/timestamp

        hf = obs.create_group("state_hifreq")
        hf.create_dataset("joints", data=np.zeros((0, 7), np.float64))
        hf.create_dataset("joint_vel", data=np.zeros((0, 7), np.float64))
        hf.create_dataset("pose", data=np.zeros((0, 6), np.float64))
        hf.create_dataset("timestamp", data=np.zeros((0,), np.float64))
        hf.create_dataset("poly_ts", data=np.zeros((0,), np.float64))

        act = f.create_group("action")
        act.create_dataset("delta_ee_pose", data=np.zeros((N, 6), np.float64))
        act.create_dataset("gripper_cmd", data=np.zeros((N, 1), np.float64))
        act.create_dataset("timestamp", data=shared_ts.copy())


# ============================================================
# 测试：标准 v1 迁移
# ============================================================

class TestMigrateV1ToV2:

    def test_validate_passes(self, tmp_path):
        """迁移后 validate_episode 通过（无 violations）。"""
        v1 = str(tmp_path / "v1.h5")
        v2 = str(tmp_path / "v2.h5")
        _write_v1_episode(v1)
        migrate(v1, v2)
        assert S.validate_episode(v2) == []

    def test_schema_version_updated(self, tmp_path):
        """输出文件 schema_version == franka-hdf5-v2。"""
        v1 = str(tmp_path / "v1.h5")
        v2 = str(tmp_path / "v2.h5")
        _write_v1_episode(v1)
        migrate(v1, v2)
        with h5py.File(v2, "r") as f:
            sv_ds = f["infos/schema_version"]
            raw = sv_ds[()] if sv_ds.shape == () else sv_ds[0]
            sv = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
        assert sv == S.SCHEMA_VERSION

    def test_stale_all_false(self, tmp_path):
        """迁移后 arm/effector/camera 各 stale 字段全为 False。"""
        v1 = str(tmp_path / "v1.h5")
        v2 = str(tmp_path / "v2.h5")
        _write_v1_episode(v1, N=8, cam_names=("wrist", "exterior"))
        migrate(v1, v2)
        with h5py.File(v2, "r") as f:
            assert not np.any(f["observations/arm/stale"][...])
            assert not np.any(f["observations/effector/stale"][...])
            for cn in ["wrist", "exterior"]:
                assert not np.any(f[f"observations/camera/rgb/{cn}/stale"][...])

    def test_hw_timestamp_equals_sw_timestamp(self, tmp_path):
        """camera hw_timestamp 应等于对应的软件戳（v1 无真硬件戳）。"""
        v1 = str(tmp_path / "v1.h5")
        v2 = str(tmp_path / "v2.h5")
        _write_v1_episode(v1, N=8, cam_names=("wrist",))
        migrate(v1, v2)
        with h5py.File(v2, "r") as f:
            sw_ts = f["observations/camera/rgb/wrist/timestamp"][...]
            hw_ts = f["observations/camera/rgb/wrist/hw_timestamp"][...]
        np.testing.assert_array_equal(hw_ts, sw_ts)

    def test_state_hifreq_wrench_shape(self, tmp_path):
        """state_hifreq/wrench 存在且 shape=(M,6)。"""
        N, M = 6, 10
        v1 = str(tmp_path / "v1.h5")
        v2 = str(tmp_path / "v2.h5")
        _write_v1_episode(v1, N=N, M=M)
        migrate(v1, v2)
        with h5py.File(v2, "r") as f:
            wrench = f["observations/state_hifreq/wrench"]
            assert wrench.shape == (M, 6)
            assert wrench.dtype == np.float64
            assert np.all(wrench[...] == 0.0)

    def test_shared_ts_removed(self, tmp_path):
        """v2 输出不含 observations/timestamp（共用戳已删除）。"""
        v1 = str(tmp_path / "v1.h5")
        v2 = str(tmp_path / "v2.h5")
        _write_v1_episode(v1)
        migrate(v1, v2)
        with h5py.File(v2, "r") as f:
            assert "observations/timestamp" not in f

    def test_timestamps_derived_from_shared(self, tmp_path):
        """arm/effector/camera 各 timestamp 与 v1 中的副本或共用戳一致。"""
        N = 6
        v1 = str(tmp_path / "v1.h5")
        v2 = str(tmp_path / "v2.h5")
        _write_v1_episode(v1, N=N, cam_names=("wrist",))
        # 读取 v1 共用戳
        with h5py.File(v1, "r") as f:
            v1_shared = np.asarray(f["observations/timestamp"]).reshape(-1)
        migrate(v1, v2)
        with h5py.File(v2, "r") as f:
            arm_ts = f["observations/arm/timestamp"][...]
            eff_ts = f["observations/effector/timestamp"][...]
            cam_ts = f["observations/camera/rgb/wrist/timestamp"][...]
        np.testing.assert_array_equal(arm_ts, v1_shared)
        np.testing.assert_array_equal(eff_ts, v1_shared)
        np.testing.assert_array_equal(cam_ts, v1_shared)

    def test_joint_data_preserved(self, tmp_path):
        """迁移后 arm/joints 数值与 v1 一致。"""
        v1 = str(tmp_path / "v1.h5")
        v2 = str(tmp_path / "v2.h5")
        _write_v1_episode(v1, N=5)
        with h5py.File(v1, "r") as f:
            orig_joints = f["observations/arm/joints"][...]
        migrate(v1, v2)
        with h5py.File(v2, "r") as f:
            new_joints = f["observations/arm/joints"][...]
        np.testing.assert_array_equal(new_joints, orig_joints)

    def test_two_cameras_migrated(self, tmp_path):
        """双相机 episode 迁移后两个相机子组均通过校验。"""
        v1 = str(tmp_path / "v1.h5")
        v2 = str(tmp_path / "v2.h5")
        _write_v1_episode(v1, N=4, cam_names=("wrist", "exterior"))
        migrate(v1, v2)
        assert S.validate_episode(v2) == []
        with h5py.File(v2, "r") as f:
            cams = sorted(f["observations/camera/rgb"].keys())
        assert cams == ["exterior", "wrist"]

    def test_metadata_preserved(self, tmp_path):
        """task_name / oc2base_R 等元信息在迁移后保留。"""
        v1 = str(tmp_path / "v1.h5")
        v2 = str(tmp_path / "v2.h5")
        _write_v1_episode(v1)
        migrate(v1, v2)
        with h5py.File(v2, "r") as f:
            raw = f["infos/task_info/task_name"][()]
            tn = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
            R = f["infos/calibration/oc2base_R"][...]
        assert tn == "pick_v1"
        np.testing.assert_array_almost_equal(R, np.eye(3))


# ============================================================
# 测试：边界/异常情况
# ============================================================

class TestMigrateEdgeCases:

    def test_rejects_non_v1_file(self, tmp_path):
        """已经是 v2 的文件应被拒绝迁移（raise ValueError）。"""
        # 构造一个 v2 文件（简单写 schema_version = franka-hdf5-v2 即可）
        v2_src = str(tmp_path / "already_v2.h5")
        with h5py.File(v2_src, "w") as f:
            infos = f.create_group("infos")
            infos.create_dataset("schema_version", data=np.bytes_("franka-hdf5-v2"))
        v2_out = str(tmp_path / "out.h5")
        with pytest.raises(ValueError, match="franka-hdf5-v1"):
            migrate(v2_src, v2_out)

    def test_state_hifreq_m_zero(self, tmp_path):
        """M=0 的 state_hifreq 迁移后 wrench shape=(0,6) 合规。"""
        v1 = str(tmp_path / "v1.h5")
        v2 = str(tmp_path / "v2.h5")
        _write_v1_episode(v1, N=4, M=0)
        migrate(v1, v2)
        assert S.validate_episode(v2) == []
        with h5py.File(v2, "r") as f:
            assert f["observations/state_hifreq/wrench"].shape == (0, 6)

    def test_fallback_from_shared_ts_when_subts_missing(self, tmp_path):
        """v1 arm/effector/camera 缺子 timestamp 时，从共用戳回退并通过校验。"""
        v1 = str(tmp_path / "v1.h5")
        v2 = str(tmp_path / "v2.h5")
        _write_v1_episode_no_subts(v1, N=5)
        migrate(v1, v2)
        assert S.validate_episode(v2) == []

    def test_action_data_preserved(self, tmp_path):
        """delta_ee_pose / gripper_cmd 数值在迁移后保持不变。"""
        v1 = str(tmp_path / "v1.h5")
        v2 = str(tmp_path / "v2.h5")
        _write_v1_episode(v1, N=5)
        with h5py.File(v1, "r") as f:
            orig_delta = f["action/delta_ee_pose"][...]
            orig_cmd = f["action/gripper_cmd"][...]
        migrate(v1, v2)
        with h5py.File(v2, "r") as f:
            new_delta = f["action/delta_ee_pose"][...]
            new_cmd = f["action/gripper_cmd"][...]
        np.testing.assert_array_equal(new_delta, orig_delta)
        np.testing.assert_array_equal(new_cmd, orig_cmd)
