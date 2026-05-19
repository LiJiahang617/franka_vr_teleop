import importlib.util, os, copy
import numpy as np
import pytest

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 指向远端仓库（本文件 scp 到远端后 __file__ 变为远端路径，_P 自动正确）
_s = importlib.util.spec_from_file_location(
    "rrh", os.path.join(_P, "scripts/core/run_record_hdf5.py"))


def _load():
    import sys; sys.path.insert(0, os.path.join(_P, "scripts"))
    m = importlib.util.module_from_spec(_s); _s.loader.exec_module(m); return m


class FakeRobot:
    def __init__(self): self.cameras = {"wrist_image": object()}; self.resets = 0
    def get_observation(self):
        o = {f"joint_{i+1}.pos": 0.0 for i in range(7)}
        o.update({f"joint_{i+1}.vel": 0.0 for i in range(7)})
        o.update({f"ee_pose.{a}": 0.0 for a in "x y z rx ry rz".split()})
        o["gripper_state_norm"] = 0.5
        o["wrist_image"] = np.zeros((4, 4, 3), np.uint8)
        return o
    def send_action(self, a): pass
    def reset(self): self.resets += 1


class FakeTeleop:
    def get_action(self):
        a = {f"delta_ee_pose.{x}": 0.0 for x in "x y z rx ry rz".split()}
        a["gripper_cmd_bin"] = 0.0; return a


class FakeSaver:
    def __init__(self): self.submitted = []
    def submit(self, p, d): self.submitted.append((p, d))


def test_reset_called_between_episodes_not_after_last(tmp_path):
    m = _load(); r = FakeRobot()
    m.run_episodes(r, FakeTeleop(), FakeSaver(),
                   fps=50.0, episode_sec=0.02, gripper_max_open=0.08,
                   cam_names=["wrist_image"], out_dir=str(tmp_path),
                   task_name="t", oc2base_R=np.eye(3), vr_source="u",
                   episodes=3, decide=lambda i: "keep",
                   reset_fn=r.reset, reset_wait=0.0)
    assert r.resets == 2          # 3 条 ep -> 条间 reset 2 次, 最后一条后不 reset


def test_reset_skipped_when_disabled(tmp_path):
    m = _load(); r = FakeRobot()
    m.run_episodes(r, FakeTeleop(), FakeSaver(),
                   fps=50.0, episode_sec=0.02, gripper_max_open=0.08,
                   cam_names=["wrist_image"], out_dir=str(tmp_path),
                   task_name="t", oc2base_R=np.eye(3), vr_source="u",
                   episodes=2, decide=lambda i: "keep",
                   reset_fn=None, reset_wait=0.0)   # 关闭 = 不传 reset_fn
    assert r.resets == 0


def test_reset_not_called_after_stop(tmp_path):
    m = _load(); r = FakeRobot()
    seq = iter(["keep", "stop"])
    m.run_episodes(r, FakeTeleop(), FakeSaver(),
                   fps=50.0, episode_sec=0.02, gripper_max_open=0.08,
                   cam_names=["wrist_image"], out_dir=str(tmp_path),
                   task_name="t", oc2base_R=np.eye(3), vr_source="u",
                   episodes=5, decide=lambda i: next(seq),
                   reset_fn=r.reset, reset_wait=0.0)
    assert r.resets == 1          # ep0 keep 后 reset 1 次, ep1 stop 后不 reset


# ── 以下为 parse_reset_config 纯函数单测（Phase B T5 review-fix） ──

def _get_parse_fn():
    """离线加载 run_record_hdf5 模块并取 parse_reset_config 函数（不触硬件）。"""
    m = _load()
    return m.parse_reset_config


def test_parse_reset_bool_true_variants():
    """缺省/True/字符串 true 变体 → rbe 均为 True。"""
    prc = _get_parse_fn()
    # 缺省（空 dict）→ True, 1.0
    rbe, rw = prc({})
    assert rbe is True
    assert rw == 1.0
    # 显式 bool True
    rbe, _ = prc({"reset_between_episodes": True})
    assert rbe is True
    # 字符串 true 变体
    for v in ("true", "True", "TRUE", "1", "yes", "on"):
        rbe, _ = prc({"reset_between_episodes": v})
        assert rbe is True, f"期望 True，got False，输入={v!r}"


def test_parse_reset_bool_false_variants():
    """False/字符串 false 变体 → rbe 均为 False（防 bool('false')==True 高危误判）。"""
    prc = _get_parse_fn()
    # 显式 bool False
    rbe, _ = prc({"reset_between_episodes": False})
    assert rbe is False
    # 字符串 false 变体（核心修复验证）
    for v in ("false", "False", "FALSE", "off", "0", "no"):
        rbe, _ = prc({"reset_between_episodes": v})
        assert rbe is False, f"期望 False，got True（bool(str) 误判！），输入={v!r}"


def test_parse_reset_bool_invalid_raises():
    """非法 reset_between_episodes 值 → 抛 ValueError 并含 reset_between_episodes 字样。"""
    prc = _get_parse_fn()
    for v in ("maybe", 123, []):
        with pytest.raises(ValueError, match="reset_between_episodes"):
            prc({"reset_between_episodes": v})


def test_parse_reset_wait_valid():
    """合法 reset_wait 值的解析验证。"""
    prc = _get_parse_fn()
    # 缺省 → 1.0
    _, rw = prc({})
    assert rw == 1.0
    # 0 → 0.0
    _, rw = prc({"reset_wait": 0})
    assert rw == 0.0
    # 2.5
    _, rw = prc({"reset_wait": 2.5})
    assert rw == 2.5
    # None → 1.0（缺省语义）
    _, rw = prc({"reset_wait": None})
    assert rw == 1.0


def test_parse_reset_wait_invalid_raises():
    """非法 reset_wait → 抛 ValueError 并含 reset_wait 字样。"""
    prc = _get_parse_fn()
    import math
    for v in (-1, math.nan, math.inf, "abc"):
        with pytest.raises(ValueError, match="reset_wait"):
            prc({"reset_wait": v})
