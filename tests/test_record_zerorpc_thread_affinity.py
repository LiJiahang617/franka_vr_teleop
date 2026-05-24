"""真机 zerorpc gevent thread-affinity 回归测试（2026-05-23）。

根因复现：zerorpc Client 基于 gevent，Hub 是 thread-local 绑 OS 线程；
Phase D 引入的 SensorThread/HistoryCollectorThread 从 daemon thread 调
robot._robot.* 会触发 gevent Waiter.switch 跨线程断言。

回归：record_episode 在 `robot._robot` 存在时（真机路径）必须只在主线程
访问 zerorpc client；不得起后台线程调度其 RPC 方法。

测试构造一个 ThreadAffinityClient：记录每次调用的 thread id；任何
非主线程的调用都使断言失败。
"""
import importlib.util
import os
import sys
import threading

import numpy as np

_P = "/home/ubuntu/Desktop/jhli/franka_vr_teleop"
sys.path.insert(0, _P)
sys.path.insert(0, os.path.join(_P, "scripts"))

_rrh_spec = importlib.util.spec_from_file_location(
    "rrh", os.path.join(_P, "scripts/core/run_record_hdf5.py")
)


def _load_rrh():
    m = importlib.util.module_from_spec(_rrh_spec)
    _rrh_spec.loader.exec_module(m)
    return m


class FakeCam:
    def read(self):
        return np.zeros((8, 8, 3), np.uint8)


class FakeTeleop:
    def get_action(self):
        return {
            "delta_ee_pose.x": 0.0, "delta_ee_pose.y": 0.0, "delta_ee_pose.z": 0.0,
            "delta_ee_pose.rx": 0.0, "delta_ee_pose.ry": 0.0, "delta_ee_pose.rz": 0.0,
            "gripper_cmd_bin": 0.0,
        }


def test_real_robot_zerorpc_only_called_from_main_thread():
    """真机路径下，robot._robot.* 必须只在主线程被调，
    后台线程访问会污染 gevent 状态导致 record 崩溃。
    """
    main_tid = threading.get_ident()
    offender = {"tid": None, "method": None}

    class ThreadAffinityClient:
        """仿 zerorpc client：跨线程调任意 RPC 都会被记录为污染。"""
        def _check_main(self, method):
            tid = threading.get_ident()
            if tid != main_tid and offender["tid"] is None:
                offender["tid"] = tid
                offender["method"] = method
            return tid

        def robot_get_joint_positions(self):
            self._check_main("robot_get_joint_positions")
            return [0.0] * 7

        def robot_get_joint_velocities(self):
            self._check_main("robot_get_joint_velocities")
            return [0.0] * 7

        def robot_get_ee_pose(self):
            self._check_main("robot_get_ee_pose")
            return [0.0] * 6

        def gripper_get_state(self):
            self._check_main("gripper_get_state")
            return {"width": 0.04}

    class RealLikeRobot:
        def __init__(self):
            self._robot = ThreadAffinityClient()   # 触发真机路径
            self.cameras = {"wrist_image": FakeCam()}
            self.config = type("C", (), {"gripper_max_open": 0.08})()

        def send_action(self, a):
            # 真机 send_action 内部也会调 zerorpc，但这里用 client 直接调一次
            # 模拟同线程调用（不应被记为 offender）
            self._robot.robot_get_joint_positions()

    m = _load_rrh()
    robot = RealLikeRobot()
    teleop = FakeTeleop()

    # hifreq_rate=240：真机路径下应被强制关；不应起 HistoryCollectorThread
    buf, block = m.record_episode(
        robot, teleop, fps=30.0, max_sec=0.15,
        gripper_max_open=0.08, cam_names=["wrist_image"],
        hifreq_rate=240.0,
    )

    assert offender["tid"] is None, (
        f"zerorpc client 被非主线程访问（tid={offender['tid']}，"
        f"method={offender['method']}）—— 真机 gevent thread-affinity 会崩"
    )
    assert block is None, "真机路径下 state_hifreq 应被强制关，block 必须 None"
    assert len(buf) > 0, "应录到至少一帧"
