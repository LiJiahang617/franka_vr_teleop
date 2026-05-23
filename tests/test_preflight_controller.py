"""controller preflight 测试 — 用 starter 注入避免真 subprocess。

测试策略：
- 通过 starter= 参数注入 mock 函数，完全绕开 subprocess 调用（hermetic）
- FakeClient 仅实现 robot_get_ee_pose，不依赖 zerorpc 或 polymetis
"""
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
    """ee_pose 立即返回有效 6D 列表。"""

    def robot_get_ee_pose(self):
        return [0.4, 0.0, 0.3, 0.0, 1.5, 0.0]


class _FakeClientEEPoseInvalid:
    """ee_pose 永远返回形状错误的数据（非 6D）。"""

    def robot_get_ee_pose(self):
        return [0.0]  # 只有 1 元素，不是 6D


def test_controller_preflight_ok_with_injected_starter():
    """starter 注入成功 + ee_pose valid → Verdict.ok=True，不触发真 subprocess。"""
    called = {"count": 0, "Kx": None, "Kxd": None}

    def starter(Kx, Kxd):
        called["count"] += 1
        called["Kx"] = Kx
        called["Kxd"] = Kxd

    res = pf.run_controller_preflight(
        client=_FakeClientOK(), settle_timeout=1.0, poll=0.01, starter=starter,
    )
    assert res.ok is True
    assert "就绪" in res.reason or "ready" in res.reason.lower()
    assert called["count"] == 1
    assert called["Kx"] == [100.0, 100.0, 100.0, 40.0, 40.0, 40.0]
    assert called["Kxd"] == [1.0, 1.0, 1.0, 0.2, 0.2, 0.2]


def test_controller_preflight_fails_when_starter_raises():
    """starter 抛 Exception → Verdict.ok=False + reason 含失败信息，不触发真 subprocess。"""
    def starter(Kx, Kxd):
        raise RuntimeError("simulated polymetis unreachable")

    res = pf.run_controller_preflight(
        client=_FakeClientOK(), settle_timeout=0.5, poll=0.01, starter=starter,
    )
    assert res.ok is False
    assert "simulated polymetis unreachable" in res.reason or "polymetis" in res.reason.lower()


def test_controller_preflight_fails_when_ee_pose_never_valid():
    """starter OK 但 ee_pose 始终错形 → settle 超时 → Verdict.ok=False。"""
    def starter(Kx, Kxd):
        pass  # 不做任何事，ee_pose 端不会变 valid

    res = pf.run_controller_preflight(
        client=_FakeClientEEPoseInvalid(), settle_timeout=0.3, poll=0.05, starter=starter,
    )
    assert res.ok is False
    assert "未返回有效" in res.reason or "register" in res.reason.lower()


def test_controller_preflight_uses_custom_Kx_Kxd():
    """传 Kx/Kxd 参数应透传到 starter。"""
    received = {}

    def starter(Kx, Kxd):
        received["Kx"] = Kx
        received["Kxd"] = Kxd

    pf.run_controller_preflight(
        client=_FakeClientOK(),
        Kx=[1, 2, 3, 4, 5, 6],
        Kxd=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        settle_timeout=0.5,
        poll=0.01,
        starter=starter,
    )
    assert received["Kx"] == [1, 2, 3, 4, 5, 6]
    assert received["Kxd"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
