"""
RecorderController 单元测试。

验证：
- events dict 写入语义与终端键盘逐字等价
- 命令队列（start/home 走队列，不直接调机器人）
- status_snapshot 返回可 JSON 序列化的 dict
- 并发 increment_episode_count 线程安全
"""
import importlib.util
import os
import threading

import pytest

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_s = importlib.util.spec_from_file_location(
    "recorder_controller", os.path.join(_P, "scripts/ui/recorder_controller.py")
)
rc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rc)

_us = importlib.util.spec_from_file_location(
    "ui_state", os.path.join(_P, "scripts/ui/state.py")
)
ust = importlib.util.module_from_spec(_us)
_us.loader.exec_module(ust)


def _make():
    """构造默认 events dict 和对应的 RecorderController。"""
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    c = rc.RecorderController(events=events)
    return c, events


def test_save_writes_exit_early_equiv_keep():
    """save_episode → exit_early=True，等价键盘 → 键（keep）。"""
    c, ev = _make()
    c.save_episode()
    assert ev["exit_early"] is True
    assert ev["rerecord_episode"] is False
    assert ev["stop_recording"] is False


def test_discard_writes_rerecord_equiv_discard():
    """discard_episode → rerecord_episode=True + exit_early=True，等价键盘 ← 键。"""
    c, ev = _make()
    c.discard_episode()
    assert ev["rerecord_episode"] is True
    # 与键盘 ← 等价：lerobot 模式 rerecord 同时置 exit_early=True 提前结束 ep
    assert ev["exit_early"] is True


def test_stop_writes_stop_recording_equiv_esc():
    """stop_recording → stop_recording=True + exit_early=True，等价键盘 Esc 键。"""
    c, ev = _make()
    c.stop_recording()
    assert ev["stop_recording"] is True
    assert ev["exit_early"] is True


def test_start_enqueues_start_command():
    """start_recording 把 'start' 入队，不直接调机器人（守坑 7）。"""
    c, _ = _make()
    c.start_recording()
    cmd = c._cmd_q.get_nowait()
    assert cmd == "start"


def test_home_enqueues_home_command_not_direct_call():
    """
    守坑 7：UI 不直接调 robot.reset (zerorpc)，走命令队列由录制器主循环串行执行。
    """
    c, _ = _make()
    c.go_home()
    assert c._cmd_q.get_nowait() == "home"


def test_status_snapshot_returns_jsonable_dict():
    """status_snapshot 返回含必要字段的纯 dict，可直接 JSON 序列化。"""
    c, _ = _make()
    snap = c.status_snapshot()
    assert isinstance(snap, dict)
    for k in ("state", "episode_count", "fps", "log_tail"):
        assert k in snap


def test_controller_lock_protects_concurrent_episode_count():
    """increment_episode_count 在并发写入时线程安全，不出数据竞争。"""
    c, _ = _make()

    def bump():
        for _ in range(1000):
            c.increment_episode_count()

    ts = [threading.Thread(target=bump) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert c.status_snapshot()["episode_count"] == 4000
"""
Task 2 修复新增测试（追加到 test_ui_recorder_controller.py 末尾）。

验证：
- start_recording() 在队列满时返回 False，未满时返回 True
- go_home() 在队列满时返回 False，未满时返回 True
"""
import importlib.util
import os
import queue

_P = "/home/ubuntu/Desktop/jhli/franka_vr_teleop"
_s = importlib.util.spec_from_file_location(
    "recorder_controller", os.path.join(_P, "scripts/ui/recorder_controller.py")
)
rc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rc)


def _make():
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    c = rc.RecorderController(events=events)
    return c, events


# ==================== 返回值语义 ====================

def test_start_recording_returns_true_on_success():
    """start_recording() 队列未满时返回 True。"""
    c, _ = _make()
    result = c.start_recording()
    assert result is True


def test_start_recording_returns_false_on_full_queue():
    """start_recording() 队列满时返回 False（不抛异常）。"""
    c, _ = _make()
    # 替换为 maxsize=1 的满队列
    c._cmd_q = queue.Queue(maxsize=1)
    c._cmd_q.put_nowait("dummy")
    result = c.start_recording()
    assert result is False


def test_go_home_returns_true_on_success():
    """go_home() 队列未满时返回 True。"""
    c, _ = _make()
    result = c.go_home()
    assert result is True


def test_go_home_returns_false_on_full_queue():
    """go_home() 队列满时返回 False（不抛异常）。"""
    c, _ = _make()
    c._cmd_q = queue.Queue(maxsize=1)
    c._cmd_q.put_nowait("dummy")
    result = c.go_home()
    assert result is False
