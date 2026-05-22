"""
UI 状态机模型。

纯逻辑，无 Flask 依赖。
状态流转：initializing → waiting → recording → confirming → saving → ready → (waiting 循环)
               confirming → waiting  （丢弃当前 episode，直接回 waiting）

红线：
- 所有状态访问通过 threading.RLock 保护（zerorpc 单线程之外 UI 线程可能并发读）
- 非法转移必须 fail-loud 抛 IllegalTransition（spec §3.4）
"""
import threading
from enum import Enum


class UIState(str, Enum):
    """UI 状态枚举，值为小写字符串，便于 JSON 序列化。"""
    INITIALIZING = "initializing"
    WAITING = "waiting"
    RECORDING = "recording"
    CONFIRMING = "confirming"
    SAVING = "saving"
    READY = "ready"


class IllegalTransition(RuntimeError):
    """非法状态转移异常。"""


# 合法转移表：每个状态可到达的目标状态集合
_LEGAL: dict[UIState, set[UIState]] = {
    UIState.INITIALIZING: {UIState.WAITING},
    UIState.WAITING:      {UIState.RECORDING},
    UIState.RECORDING:    {UIState.CONFIRMING},
    UIState.CONFIRMING:   {UIState.SAVING, UIState.WAITING},   # WAITING 对应丢弃分支
    UIState.SAVING:       {UIState.READY},
    UIState.READY:        {UIState.WAITING},
}


class StateMachine:
    """线程安全 UI 状态机。

    使用 threading.RLock 保护内部状态，snapshot() 返回纯 dict，
    调用方无需持锁。
    """

    def __init__(self) -> None:
        self._state: UIState = UIState.INITIALIZING
        self._lock: threading.RLock = threading.RLock()

    @property
    def state(self) -> UIState:
        """当前状态（线程安全读）。"""
        with self._lock:
            return self._state

    def transition(self, new: UIState) -> None:
        """转移到新状态。非法转移抛 IllegalTransition。"""
        with self._lock:
            if new not in _LEGAL.get(self._state, set()):
                raise IllegalTransition(
                    f"非法转移: {self._state.value} → {new.value}"
                )
            self._state = new

    def snapshot(self) -> dict:
        """返回当前状态的纯字典快照，调用方不持锁。"""
        with self._lock:
            return {"state": self._state.value}
