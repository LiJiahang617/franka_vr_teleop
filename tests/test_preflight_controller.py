"""controller preflight 测试：3 个场景。"""
import importlib.util
import os

import pytest

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_s = importlib.util.spec_from_file_location(
    "preflight", os.path.join(_P, "scripts/core/preflight.py")
)
pf = importlib.util.module_from_spec(_s)
_s.loader.exec_module(pf)


class _FakeClientOK:
    """start 成功，ee_pose 立即返回有效 6D 列表。"""

    def robot_start_cartesian_impedance_control(self, Kx, Kxd):
        self.started = (Kx, Kxd)

    def robot_get_ee_pose(self):
        return [0.4, 0.0, 0.3, 0.0, 1.5, 0.0]  # 有效 6D float


class _FakeClientStartFails:
    """robot_start_cartesian_impedance_control 抛异常（polymetis 不可达）。"""

    def robot_start_cartesian_impedance_control(self, Kx, Kxd):
        raise RuntimeError("polymetis manager 不可达")

    def robot_get_ee_pose(self):
        raise RuntimeError("不应进入此路径")


class _FakeClientEEPoseInvalid:
    """start 成功，但 ee_pose 永远返回形状错误的数据（非 6D）。"""

    def robot_start_cartesian_impedance_control(self, Kx, Kxd):
        pass

    def robot_get_ee_pose(self):
        return [0.0]  # 只有 1 元素，不是 6D


def test_controller_preflight_ok_with_valid_ee_pose():
    """正常路径：start 成功 + ee_pose 立即返回有效 6D，Verdict.ok=True。"""
    res = pf.run_controller_preflight(client=_FakeClientOK(), settle_timeout=1.0, poll=0.01)
    assert res.ok is True
    assert "就绪" in res.reason or "ready" in res.reason.lower()


def test_controller_preflight_fails_when_start_raises():
    """start 抛异常 → Verdict.ok=False + reason 含 polymetis/50051 指引。"""
    res = pf.run_controller_preflight(client=_FakeClientStartFails(), settle_timeout=0.5, poll=0.01)
    assert res.ok is False
    assert "polymetis" in res.reason.lower() or "50051" in res.reason
    assert "controller" in res.reason.lower() or "impedance" in res.reason.lower()


def test_controller_preflight_fails_when_ee_pose_never_valid():
    """ee_pose 长期返回形状错误 → settle 超时 → Verdict.ok=False。"""
    res = pf.run_controller_preflight(client=_FakeClientEEPoseInvalid(), settle_timeout=0.3, poll=0.05)
    assert res.ok is False
    assert "未返回有效" in res.reason or "register" in res.reason.lower()
