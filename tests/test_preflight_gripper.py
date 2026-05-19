import importlib.util, os
import pytest
_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_s = importlib.util.spec_from_file_location(
    "preflight", os.path.join(_P, "scripts/core/preflight.py"))
pf = importlib.util.module_from_spec(_s); _s.loader.exec_module(pf)


class FakeGripper:
    """可控 width/is_moving/prev_ok 序列的假 zerorpc client。"""
    def __init__(self, width_targets_to_meas, error_code=0, prev_ok=True):
        self._map = width_targets_to_meas      # {target: settled_width}
        self._w = 0.04
        self._err = error_code
        self._prev_ok = prev_ok
        self._moving_left = 0
    def gripper_initialize(self): pass
    def gripper_get_state(self):
        if self._moving_left > 0:
            self._moving_left -= 1
            moving = True
        else:
            moving = False
        return {"width": self._w, "is_moving": moving, "is_grasped": False,
                "prev_command_successful": self._prev_ok, "error_code": self._err}
    def gripper_goto(self, width, speed, force, ei=-1.0, eo=-1.0, blocking=True):
        self._w = self._map.get(round(width, 4), width)
        self._moving_left = 2                  # 模拟 2 次轮询后 settle


def test_span_ok_true_when_real_travel():
    assert pf.gripper_goto_span_ok([0.0001, 0.0700, 0.0400]) is True   # 跨度 .07>.02


def test_span_ok_false_when_stuck_homing_lost():
    # 丢 homing: goto 不同目标 width 都不动(假阴性陷阱的真阳性场景)
    assert pf.gripper_goto_span_ok([0.0400, 0.0400, 0.0400]) is False  # 跨度 0


def test_health_verdict_proc_dead_blocks():
    v = pf.gripper_health_verdict(state={"error_code": 0}, proc_alive=False,
                                  connected=True)
    assert v.ok is False and "子进程" in v.reason


def test_health_verdict_not_connected_blocks():
    v = pf.gripper_health_verdict(state={"error_code": 0}, proc_alive=True,
                                  connected=False)
    assert v.ok is False and "Connected" in v.reason


def test_health_verdict_error_code_blocks():
    v = pf.gripper_health_verdict(state={"error_code": 7}, proc_alive=True,
                                  connected=True)
    assert v.ok is False


def test_run_gripper_preflight_pass_path():
    g = FakeGripper({0.0: 0.0001, 0.07: 0.0700, 0.04: 0.0400})
    res = pf.run_gripper_preflight(
        client=g, proc_probe=lambda: True, log_probe=lambda: True,
        targets=(0.0, 0.07, 0.04), settle_timeout=0.5, poll=0.0)
    assert res.ok is True

def test_run_gripper_preflight_fail_gives_actionable_reason():
    g = FakeGripper({0.0: 0.04, 0.07: 0.04, 0.04: 0.04})   # 丢 homing
    res = pf.run_gripper_preflight(
        client=g, proc_probe=lambda: True, log_probe=lambda: True,
        targets=(0.0, 0.07, 0.04), settle_timeout=0.5, poll=0.0)
    assert res.ok is False
    assert "Desk Homing" in res.reason or "homing" in res.reason.lower()
