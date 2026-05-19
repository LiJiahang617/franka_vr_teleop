import importlib.util, os, time
import pytest

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_s = importlib.util.spec_from_file_location(
    "preflight", os.path.join(_P, "scripts/core/preflight.py"))
pf = importlib.util.module_from_spec(_s); _s.loader.exec_module(pf)


class FakeGripper:
    """可控 width/is_moving/prev_ok 序列的假 zerorpc client。"""
    def __init__(self, width_targets_to_meas, error_code=0, prev_ok=True, extra_fields=None):
        self._map = width_targets_to_meas      # {target: settled_width}
        self._w = 0.04
        self._err = error_code
        self._prev_ok = prev_ok
        self._moving_left = 0
        self._extra_fields = extra_fields or {}  # 用于测试缺字段场景（传 None 移除字段）
        self.calls = []                          # 记录 RPC 调用名列表（用于断言无 RPC）

    def gripper_initialize(self):
        self.calls.append("gripper_initialize")

    def gripper_get_state(self):
        self.calls.append("gripper_get_state")
        if self._moving_left > 0:
            self._moving_left -= 1
            moving = True
        else:
            moving = False
        state = {"width": self._w, "is_moving": moving, "is_grasped": False,
                 "prev_command_successful": self._prev_ok, "error_code": self._err}
        # 处理 extra_fields：None 值表示删除该字段，其他值表示覆盖
        for k, v in self._extra_fields.items():
            if v is None:
                state.pop(k, None)
            else:
                state[k] = v
        return state

    def gripper_goto(self, width, speed, force, ei=-1.0, eo=-1.0, blocking=True):
        self.calls.append(f"gripper_goto:{width}")
        self._w = self._map.get(round(width, 4), width)
        self._moving_left = 2                  # 模拟 2 次轮询后 settle


# ---------------------------------------------------------------------------
# 原有 7 个测试（保留不变）
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 新增测试（锁 Imp#1-4 加固）
# ---------------------------------------------------------------------------

def test_proc_fail_returns_before_any_client_rpc():
    """Imp#1：proc_probe=False → 立即返回 Verdict(False)，不触发任何 client RPC。"""
    g = FakeGripper({0.0: 0.0001, 0.07: 0.0700, 0.04: 0.0400})
    res = pf.run_gripper_preflight(
        client=g, proc_probe=lambda: False, log_probe=lambda: True,
        targets=(0.0, 0.07, 0.04), settle_timeout=0.5, poll=0.0)
    assert res.ok is False
    assert "子进程" in res.reason
    assert g.calls == [], f"期望无 client RPC，实际调用了: {g.calls}"


def test_connected_fail_returns_before_any_client_rpc():
    """Imp#1：log_probe=False → 立即返回 Verdict(False)，不触发任何 client RPC。"""
    g = FakeGripper({0.0: 0.0001, 0.07: 0.0700, 0.04: 0.0400})
    res = pf.run_gripper_preflight(
        client=g, proc_probe=lambda: True, log_probe=lambda: False,
        targets=(0.0, 0.07, 0.04), settle_timeout=0.5, poll=0.0)
    assert res.ok is False
    assert "Connected" in res.reason
    assert g.calls == [], f"期望无 client RPC，实际调用了: {g.calls}"


def test_targets_too_few_config_error():
    """Imp#4：targets < 2 → 配置错误 Verdict(False)，无真机 RPC（proc/connected 先通过）。"""
    for bad_targets in [(), (0.04,)]:
        g = FakeGripper({})
        res = pf.run_gripper_preflight(
            client=g, proc_probe=lambda: True, log_probe=lambda: True,
            targets=bad_targets, settle_timeout=0.5, poll=0.0)
        assert res.ok is False, f"targets={bad_targets} 应返回 False"
        assert "targets 需≥2" in res.reason, f"错误信息应含 'targets 需≥2'，实际: {res.reason}"
        # 应在 RPC 之前拦截（gripper_initialize/gripper_get_state 不应被调用）
        assert "gripper_initialize" not in g.calls, f"targets<2 不应发起 RPC，calls={g.calls}"


def test_settle_timeout_returns_fail():
    """Imp#2：is_moving 恒 True → settle 超时返回 Verdict(False) 含可行动文案，不假通过。"""
    class AlwaysMovingGripper:
        """is_moving 永远返回 True 的 fake client。"""
        def __init__(self):
            self.calls = []
        def gripper_initialize(self):
            self.calls.append("gripper_initialize")
        def gripper_get_state(self):
            self.calls.append("gripper_get_state")
            return {"width": 0.04, "is_moving": True, "error_code": 0}
        def gripper_goto(self, width, speed, force, ei=-1.0, eo=-1.0, blocking=True):
            self.calls.append(f"gripper_goto:{width}")

    g = AlwaysMovingGripper()
    res = pf.run_gripper_preflight(
        client=g, proc_probe=lambda: True, log_probe=lambda: True,
        targets=(0.0, 0.07), settle_timeout=0.05, poll=0.0)
    assert res.ok is False
    assert "超时" in res.reason and "未稳定" in res.reason, f"超时错误文案不符: {res.reason}"


def test_missing_state_field_error_code_actionable():
    """Imp#3：get_state 缺 error_code → Verdict(False) 含可行动文案，不抛 KeyError。"""
    g = FakeGripper({0.0: 0.0001, 0.07: 0.07}, extra_fields={"error_code": None})
    res = pf.run_gripper_preflight(
        client=g, proc_probe=lambda: True, log_probe=lambda: True,
        targets=(0.0, 0.07), settle_timeout=0.5, poll=0.0)
    assert res.ok is False
    assert "缺字段" in res.reason or "error_code" in res.reason
    # 验证不抛 KeyError（若抛则 pytest 会报 ERRORS 而非 FAILED）


def test_missing_state_field_is_moving_actionable():
    """Imp#3：get_state 缺 is_moving → Verdict(False) 含可行动文案，不静默当 False。"""
    class MissingIsMovingGripper:
        """初始 get_state 正常，goto 后 get_state 缺 is_moving 字段。"""
        def __init__(self):
            self._first_call = True  # 第一次 get_state（初始门）正常返回
            self.calls = []
        def gripper_initialize(self):
            self.calls.append("gripper_initialize")
        def gripper_get_state(self):
            self.calls.append("gripper_get_state")
            if self._first_call:
                self._first_call = False
                # 初始 get_state 字段完整，通过 gripper_state_fields_ok
                return {"width": 0.04, "is_moving": False, "error_code": 0}
            # settle 轮询时缺 is_moving
            return {"width": 0.04, "error_code": 0}
        def gripper_goto(self, width, speed, force, ei=-1.0, eo=-1.0, blocking=True):
            self.calls.append(f"gripper_goto:{width}")

    g = MissingIsMovingGripper()
    res = pf.run_gripper_preflight(
        client=g, proc_probe=lambda: True, log_probe=lambda: True,
        targets=(0.0, 0.07), settle_timeout=0.5, poll=0.0)
    assert res.ok is False
    assert "is_moving" in res.reason, f"应提示缺 is_moving，实际: {res.reason}"


def test_missing_state_field_width_actionable():
    """Imp#3：settle 后 get_state 缺 width → Verdict(False) 含可行动文案，不抛 KeyError。"""
    class MissingWidthAfterSettleGripper:
        """settle 完成后（is_moving=False）取 width 时返回缺 width 的 dict。"""
        def __init__(self):
            self._goto_count = 0
            self._poll_count = 0
            self.calls = []
        def gripper_initialize(self):
            self.calls.append("gripper_initialize")
        def gripper_get_state(self):
            self.calls.append("gripper_get_state")
            if self._goto_count == 0:
                # 初始 get_state（gripper_state_fields_ok 校验）
                return {"width": 0.04, "is_moving": False, "error_code": 0}
            # goto 后：先返回 is_moving=False（settle 立即完成），再返回缺 width
            self._poll_count += 1
            if self._poll_count == 1:
                return {"is_moving": False, "error_code": 0}   # settle 轮询，无 width 但有 is_moving
            # 第二次调用（取最终 width）
            return {"is_moving": False, "error_code": 0}       # 缺 width
        def gripper_goto(self, width, speed, force, ei=-1.0, eo=-1.0, blocking=True):
            self.calls.append(f"gripper_goto:{width}")
            self._goto_count += 1
            self._poll_count = 0

    g = MissingWidthAfterSettleGripper()
    res = pf.run_gripper_preflight(
        client=g, proc_probe=lambda: True, log_probe=lambda: True,
        targets=(0.0, 0.07), settle_timeout=0.5, poll=0.0)
    assert res.ok is False
    assert "width" in res.reason, f"应提示缺 width，实际: {res.reason}"


# ---------------------------------------------------------------------------
# 新增：gripper_state_fields_ok 纯函数单测
# ---------------------------------------------------------------------------

def test_gripper_state_fields_ok_all_present():
    v = pf.gripper_state_fields_ok({"error_code": 0, "is_moving": False, "width": 0.04})
    assert v.ok is True


def test_gripper_state_fields_ok_missing_field():
    for missing_key in ("error_code", "is_moving", "width"):
        state = {"error_code": 0, "is_moving": False, "width": 0.04}
        del state[missing_key]
        v = pf.gripper_state_fields_ok(state)
        assert v.ok is False, f"缺 {missing_key} 应返回 False"
        assert "缺字段" in v.reason


def test_gripper_state_fields_ok_error_code_nonzero():
    v = pf.gripper_state_fields_ok({"error_code": 5, "is_moving": False, "width": 0.04})
    assert v.ok is False and "error_code" in v.reason
