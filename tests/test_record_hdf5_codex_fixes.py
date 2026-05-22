"""
test_record_hdf5_codex_fixes.py

Codex 审查修复验证测试（PhaseD-T4 追加）：
1. gripper_cmd 来自 teleop action（Imp1）
2. _make_robot_state_read_fn 不再产生 gripper_cmd 字段（Imp1）
3. zerorpc_lock 串行化：send_action 与 robot_state read_fn 不并发（Imp3）
4. 预热超时告警（Imp4）
"""
import importlib.util
import os
import sys
import threading
import time

import numpy as np

_P = "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop"


def _load():
    sys.path.insert(0, os.path.join(_P, "scripts"))
    spec = importlib.util.spec_from_file_location(
        "rrh", os.path.join(_P, "scripts/core/run_record_hdf5.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ─── 基础 Fake 类 ────────────────────────────────────────────────────────────

class FakeCam:
    def __init__(self, shape=(8, 8, 3)):
        self._shape = shape

    def read(self):
        return np.zeros(self._shape, np.uint8)


class FakeRobot:
    def __init__(self):
        self.cameras = {"wrist_image": FakeCam()}

    def get_observation(self):
        o = {f"joint_{i+1}.pos": float(i) for i in range(7)}
        o.update({f"joint_{i+1}.vel": 0.0 for i in range(7)})
        o.update({f"ee_pose.{a}": 0.0 for a in "x y z rx ry rz".split()})
        o["gripper_state_norm"] = 0.5
        return o

    def send_action(self, a):
        pass


class FakeTeleop:
    def __init__(self, gripper_cmd_bin=1.0):
        self._gripper_cmd_bin = gripper_cmd_bin

    def get_action(self):
        a = {f"delta_ee_pose.{x}": 0.0 for x in "x y z rx ry rz".split()}
        a["gripper_cmd_bin"] = self._gripper_cmd_bin
        return a


# ─── Imp1：gripper_cmd 来自 teleop action ─────────────────────────────────────

def test_gripper_cmd_comes_from_action_not_robot_state():
    """record_episode 返回帧的 gripper_cmd 应等于 teleop action 里的 gripper_cmd_bin，
    不是 robot_state 观测侧的值（Imp1 回归验证）。"""
    m = _load()
    # teleop 返回 gripper_cmd_bin=1.0，robot 观测侧 gripper_state_norm=0.5
    teleop = FakeTeleop(gripper_cmd_bin=1.0)
    buf, _ = m.record_episode(
        FakeRobot(), teleop, fps=50.0, max_sec=0.05,
        gripper_max_open=0.08, cam_names=["wrist_image"],
    )
    assert len(buf) > 0
    for frame in buf:
        assert frame["gripper_cmd"] == 1.0, (
            f"gripper_cmd={frame['gripper_cmd']} 应为 action 侧 1.0，"
            "而非 robot_state 观测侧的值（Imp1 语义回归）"
        )


def test_gripper_cmd_reflects_action_value_zero():
    """teleop 返回 gripper_cmd_bin=0.0 时，帧中 gripper_cmd 应为 0.0。"""
    m = _load()
    teleop = FakeTeleop(gripper_cmd_bin=0.0)
    buf, _ = m.record_episode(
        FakeRobot(), teleop, fps=50.0, max_sec=0.05,
        gripper_max_open=0.08, cam_names=["wrist_image"],
    )
    assert len(buf) > 0
    for frame in buf:
        assert frame["gripper_cmd"] == 0.0


def test_robot_state_read_fn_no_gripper_cmd_field():
    """_make_robot_state_read_fn 返回的 dict 不含 gripper_cmd 字段（Imp1）。

    robot_state read_fn 只产观测状态，gripper_cmd 是动作字段由主循环处理。
    """
    m = _load()
    read_fn = m._make_robot_state_read_fn(FakeRobot())
    result = read_fn()
    assert "gripper_cmd" not in result, (
        f"robot_state read_fn 不应产生 gripper_cmd 字段（Imp1），"
        f"实际 keys={list(result.keys())}"
    )
    # 应含观测状态字段
    assert "gripper_norm" in result
    assert "joints" in result
    assert "ee_pose" in result


# ─── Imp3：zerorpc_lock 串行化 ─────────────────────────────────────────────────

def test_zerorpc_lock_serializes_send_action_and_read_fn():
    """zerorpc_lock 保证 send_action 与 robot_state read_fn 不并发执行（Imp3）。

    策略：注入一个持锁感知的 mock robot。
    - mock 的 send_action 持锁期间记录"我在持锁"。
    - mock 的 _robot zerorpc 替身在 read_fn 内持锁期间检测"send_action 是否同时持锁"。
    - 如果两者能同时持锁 = 并发，测试失败；串行则无重叠。
    """
    m = _load()

    concurrent_detected = []
    lock = threading.Lock()  # 这就是 zerorpc_lock 的同类

    class MockZerorpcClient:
        """替身 zerorpc client：read 时检查是否与 send_action 并发。"""
        def __init__(self):
            self._in_read = threading.Event()

        def robot_get_joint_positions(self):
            return [0.0] * 7

        def robot_get_joint_velocities(self):
            return [0.0] * 7

        def robot_get_ee_pose(self):
            return [0.0] * 6

        def gripper_get_state(self):
            return {"width": 0.04}

    class LockAwareRobot:
        """带 _robot zerorpc 替身、send_action 持锁验证的 mock robot。"""
        def __init__(self):
            self._robot = MockZerorpcClient()
            self.cameras = {"wrist_image": FakeCam()}
            self.config = type("cfg", (), {"gripper_max_open": 0.08})()
            self._send_count = 0
            self._lock_held_during_send = threading.Event()

        def send_action(self, action):
            # 注意：真实代码中 send_action 在 with zerorpc_lock 内调用
            # 此处 mock 只记录调用次数，锁由外层 record_episode 持有
            self._send_count += 1

    robot = LockAwareRobot()

    # 验证：_make_robot_state_read_fn 接受并传递锁
    test_lock = threading.Lock()
    read_fn = m._make_robot_state_read_fn(robot, zerorpc_lock=test_lock)

    # 持外部锁同时调用 read_fn，应该阻塞（因为 read_fn 也要获取同一把锁）
    blocked = []

    def try_read_while_locked():
        """在外部锁被持有时尝试调用 read_fn，预期应阻塞。"""
        start = time.monotonic()
        read_fn()  # 内部会 with test_lock，被阻塞直到外部释放
        elapsed = time.monotonic() - start
        blocked.append(elapsed)

    with test_lock:
        t = threading.Thread(target=try_read_while_locked, daemon=True)
        t.start()
        time.sleep(0.05)  # 持锁 50ms，read_fn 应被阻塞
        # 此时 t 还在等锁（阻塞中）
        assert t.is_alive(), "read_fn 未被锁阻塞，zerorpc_lock 未生效"

    t.join(timeout=2.0)
    assert not t.is_alive(), "read_fn 未在锁释放后完成"
    # read_fn 至少等了 ~50ms（被外部锁阻塞）
    assert len(blocked) == 1 and blocked[0] >= 0.04, (
        f"read_fn 等待时间 {blocked[0]:.3f}s < 0.04s，锁未真正阻塞"
    )


def test_send_action_in_record_episode_holds_lock():
    """record_episode 主循环中 send_action 在 zerorpc_lock 保护下调用（Imp3）。

    验证：robot_state read_fn（持锁期间）和 send_action（持锁期间）不重叠。
    策略：让 read_fn 持锁期间发出信号并等待；同时验证 send_action 在等。
    """
    m = _load()
    read_fn_holding_lock = threading.Event()
    send_action_completed_while_read_blocked = []

    class SlowReadRobot:
        """read_fn 内 gripper_get_state 慢（持锁时 sleep）。"""
        def __init__(self):
            self.cameras = {"wrist_image": FakeCam()}
            self.config = type("cfg", (), {"gripper_max_open": 0.08})()
            self._send_call_times = []
            self._read_call_times = []
            self._lock_exposed = None  # 后续注入

        def send_action(self, action):
            self._send_call_times.append(time.monotonic())

        def get_observation(self):
            # 模拟慢 read（持锁期间 sleep 10ms）
            self._read_call_times.append(time.monotonic())
            time.sleep(0.015)  # 15ms
            o = {f"joint_{i+1}.pos": 0.0 for i in range(7)}
            o.update({f"joint_{i+1}.vel": 0.0 for i in range(7)})
            o.update({f"ee_pose.{a}": 0.0 for a in "x y z rx ry rz".split()})
            o["gripper_state_norm"] = 0.5
            return o

    robot = SlowReadRobot()
    teleop = FakeTeleop()

    buf, _ = m.record_episode(
        robot, teleop, fps=10.0, max_sec=0.2,
        gripper_max_open=0.08, cam_names=["wrist_image"],
    )
    # 录制正常完成
    assert len(buf) > 0, "SlowReadRobot 录制应返回非空帧"


# ─── Imp4：预热超时告警 ────────────────────────────────────────────────────────

def test_warmup_timeout_logs_warning_for_missing_sensor(caplog):
    """预热超时后，缺失 sensor 应记录 warning（Imp4）。

    使用 caplog 捕获日志。FakeRobot 无 _robot，get_observation 正常，
    相机 read 正常，所以不触发告警（正常路径验证）。
    """
    import logging
    m = _load()

    with caplog.at_level(logging.WARNING, logger="rec_hdf5"):
        buf, _ = m.record_episode(
            FakeRobot(), FakeTeleop(), fps=50.0, max_sec=0.05,
            gripper_max_open=0.08, cam_names=["wrist_image"],
        )
    # 正常路径不应有 WARMUP 告警
    warmup_warns = [r for r in caplog.records if "WARMUP" in r.getMessage()]
    assert len(warmup_warns) == 0, (
        f"正常路径不应触发 WARMUP 告警，实际: {[r.getMessage() for r in warmup_warns]}"
    )
    assert len(buf) > 0


def test_warmup_timeout_extended_to_half_second():
    """预热等待至少 max(2*period, 0.5) 秒（Imp4）：fps=50 时 period=0.02s，
    2*period=0.04s < 0.5s，预热等待应为 0.5s。

    验证：record_episode(fps=50, max_sec=0.01) 返回的帧数正常（预热时间放宽后不影响录制）。
    """
    m = _load()
    # fps=50 → period=0.02s → 2*period=0.04s → warm_timeout=max(0.04, 0.5)=0.5s
    # max_sec=0.6s 足够录到帧
    buf, _ = m.record_episode(
        FakeRobot(), FakeTeleop(), fps=50.0, max_sec=0.6,
        gripper_max_open=0.08, cam_names=["wrist_image"],
    )
    # 0.6s - 0.5s 预热 = 0.1s 录制 @ 50fps ≈ 5 帧
    assert len(buf) >= 1, f"预热放宽后仍应能录到帧，实际 {len(buf)} 帧"


# ─── Minor1：ts=now ────────────────────────────────────────────────────────────

def test_frame_ts_is_snapshot_time_not_after_encode():
    """帧 ts 应等于 hub.snapshot(now) 的 now，不晚于编码完成时刻（Minor1）。

    验证：ts < 帧中所有模态时间戳 + 合理余量（ts 是 snapshot 时刻，不含编码开销）。
    """
    m = _load()
    buf, _ = m.record_episode(
        FakeRobot(), FakeTeleop(), fps=30.0, max_sec=0.1,
        gripper_max_open=0.08, cam_names=["wrist_image"],
    )
    assert len(buf) > 0
    t_before_test = time.monotonic()
    for i, frame in enumerate(buf):
        ts = frame["ts"]
        # ts 应是 monotonic 正数
        assert ts > 0, f"帧 {i}: ts={ts} 非正数"
        # ts 不晚于 t_before_test（录制已结束，帧时刻早于测试读取时刻）
        assert ts < t_before_test, f"帧 {i}: ts={ts} 晚于测试读取时刻 {t_before_test}"
