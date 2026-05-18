import sys, numpy as np, h5py, pytest
sys.path.insert(0, "/home/ubuntu/Desktop/jhli")
sys.path.insert(0, "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop/scripts")
import franka_hdf5_schema as S
from core.hdf5_writer import HDF5EpisodeWriter


def _frame(i):
    return dict(
        ts=1.0 + i * 0.033,
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
        # 240Hz 占位: 合法空数组
        assert f["observations/state_hifreq/joints"].shape == (0, 7)
        assert f["infos/calibration/oc2base_R"].shape == (3, 3)


def test_writer_rejects_empty_episode(tmp_path):
    p = str(tmp_path / "ep_empty.h5")
    w = HDF5EpisodeWriter(p, task_name="x", target_fps=30.0, oc2base_R=np.eye(3),
                          quality={}, vr_source="unity", cam_names=["wrist"])
    with pytest.raises(ValueError):
        w.close()  # 0 帧应拒绝
