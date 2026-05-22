import copy, importlib.util, os, sys
import numpy as np, h5py

sys.path.insert(0, "/home/ubuntu/Desktop/jhli")
sys.path.insert(0, "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop/scripts")
import franka_hdf5_schema as S
from core.hdf5_writer import HDF5EpisodeWriter, write_episode

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_a = importlib.util.spec_from_file_location(
    "async_saver", os.path.join(_P, "scripts/core/async_saver.py"))
asv = importlib.util.module_from_spec(_a); _a.loader.exec_module(asv)


def _frame(i):
    """生成 v2 格式测试帧（含各模态独立时间戳 + stale 字段）。"""
    ts = 1.0 + i * 0.033
    return dict(
        ts=ts,
        arm_ts=ts + 0.001,
        effector_ts=ts + 0.001,
        cam_ts={"wrist": ts + 0.003},
        cam_hw_ts={"wrist": ts + 0.003},
        arm_stale=False,
        effector_stale=False,
        cam_stale={"wrist": False},
        joints=np.zeros(7), joint_vel=np.zeros(7),
        ee_pose=np.array([0.1, 0, 0.3, 0, 0, 0], float),
        gripper_m=0.04, gripper_norm=0.5, gripper_cmd=0.0,
        delta_ee_pose=np.zeros(6),
        cams={"wrist": np.zeros((4,), np.uint8)},
    )


def test_write_episode_module_fn_conformant(tmp_path):
    p = str(tmp_path / "ep.h5")
    frames = [_frame(i) for i in range(5)]
    write_episode(p, frames, task_name="pick", target_fps=30.0,
                  oc2base_R=np.eye(3), quality={}, vr_source="unity",
                  cam_names=["wrist"])
    assert S.validate_episode(p) == []


def test_async_saver_writes_conformant_episode(tmp_path):
    p = str(tmp_path / "epA.h5")
    frames = [_frame(i) for i in range(4)]

    def sink(path, payload):
        write_episode(path, payload["frames"], **payload["meta"])

    with asv.AsyncEpisodeSaver(sink=sink, maxsize=5) as s:
        s.submit(p, {"frames": copy.deepcopy(frames),
                     "meta": dict(task_name="x", target_fps=30.0,
                                  oc2base_R=np.eye(3), quality={},
                                  vr_source="unity", cam_names=["wrist"])})
    assert S.validate_episode(p) == []
    with h5py.File(p, "r") as f:
        assert f["observations/arm/joints"].shape == (4, 7)
        # v2 字段验证
        assert f["observations/arm/stale"].shape == (4,)
        assert f["observations/camera/rgb/wrist/hw_timestamp"].shape == (4,)


def test_deepcopy_snapshot_isolates_from_buffer_reuse(tmp_path):
    # deepcopy 后清空/复用原 buffer，不得污染已提交快照
    p = str(tmp_path / "epIso.h5")
    buf = [_frame(i) for i in range(3)]
    snap = copy.deepcopy(buf)
    buf.clear(); buf.append(_frame(99))      # 复用 buffer
    write_episode(p, snap, task_name="x", target_fps=30.0, oc2base_R=np.eye(3),
                  quality={}, vr_source="unity", cam_names=["wrist"])
    with h5py.File(p, "r") as f:
        assert f["observations/arm/joints"].shape == (3, 7)   # 仍是快照的 3 帧


def test_close_still_synchronous_behaviour(tmp_path):
    # 既有同步 close 行为零变化（空 episode 仍抛 ValueError）
    import pytest
    w = HDF5EpisodeWriter(str(tmp_path / "e.h5"), task_name="x",
                          target_fps=30.0, oc2base_R=np.eye(3), quality={},
                          vr_source="u", cam_names=["wrist"])
    with pytest.raises(ValueError):
        w.close()


def test_state_hifreq_block_written_correctly(tmp_path):
    """state_hifreq_block 指定时，写出 M 行数据而非 M=0 占位。"""
    p = str(tmp_path / "ep_hf.h5")
    frames = [_frame(i) for i in range(3)]
    M = 10
    block = {
        "joints": np.zeros((M, 7), np.float64),
        "joint_vel": np.zeros((M, 7), np.float64),
        "pose": np.zeros((M, 6), np.float64),
        "timestamp": np.linspace(1.0, 1.1, M),
        "poly_ts": np.linspace(1.0, 1.1, M),
        "wrench": np.zeros((M, 6), np.float64),
    }
    write_episode(p, frames, task_name="x", target_fps=30.0, oc2base_R=np.eye(3),
                  quality={}, vr_source="u", cam_names=["wrist"],
                  state_hifreq_block=block)
    assert S.validate_episode(p) == []
    with h5py.File(p, "r") as f:
        assert f["observations/state_hifreq/joints"].shape == (M, 7)
        assert f["observations/state_hifreq/wrench"].shape == (M, 6)
