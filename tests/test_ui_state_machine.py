"""
UI 状态机纯逻辑测试。
无 Flask 依赖，测试 UIState 枚举、StateMachine 合法/非法转移、线程安全 snapshot。
"""
import importlib.util, os

# 动态加载 state 模块，避免包导入依赖
_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_s = importlib.util.spec_from_file_location(
    "ui_state", os.path.join(_P, "scripts/ui/state.py"))
ust = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ust)


def test_initial_state_is_initializing():
    """初始状态必须是 INITIALIZING。"""
    sm = ust.StateMachine()
    assert sm.state == ust.UIState.INITIALIZING


def test_legal_transitions_match_spec_3_4():
    """spec §3.4: initializing→waiting→recording→confirming→saving→ready→waiting 全链路合法转移。"""
    sm = ust.StateMachine()
    sm.transition(ust.UIState.WAITING)
    sm.transition(ust.UIState.RECORDING)
    sm.transition(ust.UIState.CONFIRMING)
    sm.transition(ust.UIState.SAVING)
    sm.transition(ust.UIState.READY)
    sm.transition(ust.UIState.WAITING)
    assert sm.state == ust.UIState.WAITING


def test_illegal_transition_raises():
    """非法转移（跳过 waiting 直接到 recording）必须抛 IllegalTransition。"""
    import pytest
    sm = ust.StateMachine()  # initializing
    with pytest.raises(ust.IllegalTransition):
        sm.transition(ust.UIState.RECORDING)  # 跳过 waiting，非法


def test_state_is_thread_safe_snapshot():
    """snapshot() 返回含 state 字符串的纯 dict，调用方不持锁。"""
    sm = ust.StateMachine()
    snap = sm.snapshot()
    assert isinstance(snap, dict) and snap["state"] == "initializing"
