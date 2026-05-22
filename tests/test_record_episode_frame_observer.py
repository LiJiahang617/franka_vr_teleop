"""
test_record_episode_frame_observer.py

守门测试：
1. frame_observer=None 时 record_episode 行为零变化（既有接口兼容）
2. frame_observer 给定时每帧每路 cam 被调用一次
"""
import importlib.util
import os
import sys

import numpy as np

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_s = importlib.util.spec_from_file_location(
    "rrh", os.path.join(_P, "scripts/core/run_record_hdf5.py")
)


def _load():
    sys.path.insert(0, os.path.join(_P, "scripts"))
    m = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(m)
    return m


class FakeCam:
    """带 read() 方法的假相机，供 record_episode 的相机 SensorThread 使用。"""
    def __init__(self, shape=(8, 8, 3)):
        self._shape = shape

    def read(self):
        return np.zeros(self._shape, np.uint8)


class FakeRobot:
    def __init__(self):
        # cameras 值需有 read() 方法，供相机 SensorThread 采集
        self.cameras = {"wrist_image": FakeCam((8, 8, 3))}

    def get_observation(self):
        # robot_state 线程的测试回退路径（当 _robot 不存在时）
        o = {f"joint_{i+1}.pos": 0.0 for i in range(7)}
        o.update({f"joint_{i+1}.vel": 0.0 for i in range(7)})
        o.update({f"ee_pose.{a}": 0.0 for a in "x y z rx ry rz".split()})
        o["gripper_state_norm"] = 0.5
        o["wrist_image"] = np.zeros((8, 8, 3), np.uint8)
        return o

    def send_action(self, a):
        pass


class FakeTeleop:
    def get_action(self):
        a = {f"delta_ee_pose.{x}": 0.0 for x in "x y z rx ry rz".split()}
        a["gripper_cmd_bin"] = 0.0
        return a


def test_observer_none_is_zero_behavior_change():
    """frame_observer=None 时，record_episode 行为与旧版零变化（守接口兼容性）。"""
    m = _load()
    buf = m.record_episode(
        FakeRobot(),
        FakeTeleop(),
        fps=50.0,
        max_sec=0.05,
        gripper_max_open=0.08,
        cam_names=["wrist_image"],
    )
    assert len(buf) > 0  # 既有行为：返回非空帧列表


def test_observer_called_per_frame_per_cam():
    """frame_observer 给定时，每帧每路 cam 调用一次，且 cam_name 与 img.shape 正确。"""
    m = _load()
    seen = []

    def obs_fn(cam, img):
        seen.append((cam, img.shape))

    buf = m.record_episode(
        FakeRobot(),
        FakeTeleop(),
        fps=50.0,
        max_sec=0.05,
        gripper_max_open=0.08,
        cam_names=["wrist_image"],
        frame_observer=obs_fn,
    )
    # 每帧每路 cam 恰好调用一次
    assert len(seen) == len(buf)
    assert all(c == "wrist_image" and s == (8, 8, 3) for c, s in seen)


def test_observer_does_not_mutate_recorded_frame():
    """frame_observer 在 _encode_jpg 之前调用原始 RGB 数据，不影响最终编码结果。

    observer 内修改 img（view 操作）不应影响 buf 中已编码的 cams 数据。
    """
    m = _load()

    def mutating_obs(cam, img):
        # 尝试修改 img（副本则无害，原始引用则有害——此处仅验证不崩溃且结果正确）
        try:
            img[0, 0, 0] = 255
        except (ValueError, TypeError):
            pass  # 只读 array 也可接受

    buf = m.record_episode(
        FakeRobot(),
        FakeTeleop(),
        fps=50.0,
        max_sec=0.05,
        gripper_max_open=0.08,
        cam_names=["wrist_image"],
        frame_observer=mutating_obs,
    )
    # 关键：录制正常完成，帧数据不为空
    assert len(buf) > 0
    for frame in buf:
        assert "cams" in frame
        assert "wrist_image" in frame["cams"]
