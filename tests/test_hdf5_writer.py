import sys, numpy as np, h5py, pytest
sys.path.insert(0, "/home/ubuntu/Desktop/jhli")
sys.path.insert(0, "/home/ubuntu/Desktop/jhli/franka_vr_teleop/scripts")
import franka_hdf5_schema as S
from core.hdf5_writer import HDF5EpisodeWriter


def _frame(i):
    """生成 v2 格式测试帧（含各模态独立时间戳 + stale 字段）。"""
    ts = 1.0 + i * 0.033
    return dict(
        ts=ts,
        # v2 独立时间戳（各模态略微偏移，模拟真实采集）
        arm_ts=ts + 0.001,
        effector_ts=ts + 0.001,
        cam_ts={"wrist": ts + 0.003},
        cam_hw_ts={"wrist": ts + 0.003},  # Task 8 实填真硬件戳；此处软件戳占位
        arm_stale=False,
        effector_stale=False,
        cam_stale={"wrist": False},
        joints=np.zeros(7), joint_vel=np.zeros(7),
        ee_pose=np.array([0.1, 0, 0.3, 0, 0, 0], float),
        gripper_m=0.04, gripper_norm=0.5, gripper_cmd=0.0,
        delta_ee_pose=np.zeros(6),
        cams={"wrist": np.zeros((4,), np.uint8)},   # 已编码 jpeg 字节占位
    )


def test_writer_produces_conformant_episode(tmp_path):
    p = str(tmp_path / "ep0001.h5")
    w = HDF5EpisodeWriter(p, task_name="pick", target_fps=30.0,
                          oc2base_R=np.eye(3), quality={"angle_err_deg": 1.0},
                          vr_source="unity", cam_names=["wrist"])
    for i in range(5):
        w.add(_frame(i))
    w.close()
    assert S.validate_episode(p) == []
    with h5py.File(p, "r") as f:
        assert f["observations/arm/joints"].shape == (5, 7)
        assert f["action/delta_ee_pose"].shape == (5, 6)
        assert f["observations/effector/position"].shape == (5, 1)
        # v2: 各模态独立时间戳
        assert f["observations/arm/timestamp"].shape == (5,)
        assert f["observations/arm/stale"].shape == (5,)
        assert f["observations/camera/rgb/wrist/hw_timestamp"].shape == (5,)
        assert f["observations/camera/rgb/wrist/stale"].shape == (5,)
        # state_hifreq 占位 M=0，wrench 字段存在
        assert f["observations/state_hifreq/joints"].shape == (0, 7)
        assert f["observations/state_hifreq/wrench"].shape == (0, 6)
        assert f["infos/calibration/oc2base_R"].shape == (3, 3)
        # v2: 无共用 observations/timestamp
        assert "observations/timestamp" not in f


def test_writer_rejects_empty_episode(tmp_path):
    p = str(tmp_path / "ep_empty.h5")
    w = HDF5EpisodeWriter(p, task_name="x", target_fps=30.0, oc2base_R=np.eye(3),
                          quality={}, vr_source="unity", cam_names=["wrist"])
    with pytest.raises(ValueError):
        w.close()  # 0 帧应拒绝


def _frame_with_hw_ts(i, effector_hw_ts=None):
    """生成带可选 effector_hw_ts 的测试帧（复用 _frame 结构）。"""
    base = _frame(i)
    base["effector_hw_ts"] = effector_hw_ts
    return base


def _write_ep(tmp_path, name, frames):
    """用 write_episode 落盘辅助（统一关键字参数）。"""
    from core.hdf5_writer import write_episode
    out = tmp_path / name
    write_episode(
        str(out),
        frames,
        task_name="test",
        target_fps=30.0,
        oc2base_R=np.eye(3),
        quality={},
        vr_source="unity",
        cam_names=["wrist"],
    )
    return out


def test_write_episode_hw_timestamp_all_none_skips_dataset(tmp_path):
    """全部帧 effector_hw_ts=None → 不写 observations/effector/hw_timestamp 数据集。"""
    frames = [_frame_with_hw_ts(i, effector_hw_ts=None) for i in range(5)]
    out = _write_ep(tmp_path, "ep_no_hw_ts.h5", frames)
    with h5py.File(out, "r") as f:
        assert "observations/effector/hw_timestamp" not in f, (
            "全 None 不应写 hw_timestamp 数据集"
        )


def test_write_episode_hw_timestamp_all_present_writes_array(tmp_path):
    """全部帧 effector_hw_ts 为有效 float → 写整列。"""
    frames = [_frame_with_hw_ts(i, effector_hw_ts=100.0 + i * 0.033) for i in range(5)]
    out = _write_ep(tmp_path, "ep_with_hw_ts.h5", frames)
    with h5py.File(out, "r") as f:
        assert "observations/effector/hw_timestamp" in f
        arr = f["observations/effector/hw_timestamp"][...]
        assert arr.shape == (5,)
        assert arr.dtype == np.float64
        np.testing.assert_allclose(arr, [100.0, 100.033, 100.066, 100.099, 100.132])


def test_write_episode_hw_timestamp_partial_none_fills_nan(tmp_path):
    """部分帧 None → 写整列，None→NaN。"""
    frames = [
        _frame_with_hw_ts(0, effector_hw_ts=100.0),
        _frame_with_hw_ts(1, effector_hw_ts=None),
        _frame_with_hw_ts(2, effector_hw_ts=100.066),
    ]
    out = _write_ep(tmp_path, "ep_partial_hw_ts.h5", frames)
    with h5py.File(out, "r") as f:
        arr = f["observations/effector/hw_timestamp"][...]
        assert arr.shape == (3,)
        assert arr[0] == 100.0
        assert np.isnan(arr[1])
        assert arr[2] == 100.066
