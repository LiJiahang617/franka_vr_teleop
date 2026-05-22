"""
RecorderController：UI 与录制器之间的桥梁。

职责：
- 持有 events dict（供 EpisodeDecider 消费，语义与终端键盘逐字等价）
- 持有命令队列（start/home 走队列，由 Task 5 录制器主循环串行消费；守坑 7 不直接调 zerorpc）
- 持有状态机（Task 1 的 StateMachine）
- 线程安全：episode_count / latest_frames / log_tail 均在 _lock 下操作
- 录制器线程在 Task 5 接入，本 Task 用 stub 占位
"""
import collections
import importlib.util
import os
import queue
import threading
from typing import Optional

import numpy as np

# 从同包 state 模块加载 StateMachine（保持与 Task 1 一致的加载方式）
_UI_DIR = os.path.dirname(os.path.abspath(__file__))
_state_spec = importlib.util.spec_from_file_location(
    "ui_state", os.path.join(_UI_DIR, "state.py")
)
_state_mod = importlib.util.module_from_spec(_state_spec)
_state_spec.loader.exec_module(_state_mod)
StateMachine = _state_mod.StateMachine

# events dict 必须含有的三个键（与 EpisodeDecider 字段逐字等价）
_REQUIRED_KEYS = frozenset({"exit_early", "rerecord_episode", "stop_recording"})


class RecorderController:
    """UI 录制控制器。

    methods 对应按钮语义，写入 events dict 与终端键盘逐字等价：
    - save_episode    → exit_early=True             （= → 键 keep）
    - discard_episode → rerecord_episode=True + exit_early=True  （= ← 键 discard）
    - stop_recording  → stop_recording=True + exit_early=True    （= Esc 键 stop）
    - start_recording → 命令队列入 "start"（不直接调机器人），成功返回 True，队列满返回 False
    - go_home         → 命令队列入 "home"（不直接调机器人），成功返回 True，队列满返回 False

    Attributes:
        _events: 与 EpisodeDecider 共享的 events dict。
        _sm: 状态机实例。
        _cmd_q: 命令队列（Task 5 录制器主循环消费）。
        _lock: RLock，保护 _ep_count / _latest_frames / _log_tail。
        _ep_count: 已完成 episode 计数。
        _fps: 录制帧率（UI 状态显示用）。
        _log_tail: 近 200 条日志（环形缓冲）。
        _latest_frames: cam_name → ndarray RGB（Task 5 填充，Task 3 读取）。
        _recorder_thread: 录制器线程（Task 5 接入，当前为 None）。
    """

    def __init__(self, events: dict, *, fps: float = 30.0) -> None:
        """初始化 RecorderController。

        Args:
            events: 与 EpisodeDecider 共享的 events dict，
                    必须含 exit_early / rerecord_episode / stop_recording 三键。
                    缺键则 fail-loud KeyError（与 EpisodeDecider 同款校验）。
            fps: 录制帧率，仅用于 status_snapshot 展示，默认 30.0。
        """
        # 校验必含三键，缺键 fail-loud
        missing = _REQUIRED_KEYS - events.keys()
        if missing:
            raise KeyError(f"events dict 缺少必要字段: {missing}")

        self._events: dict = events
        self._sm: StateMachine = StateMachine()
        self._cmd_q: queue.Queue = queue.Queue(maxsize=64)
        self._lock: threading.RLock = threading.RLock()
        self._ep_count: int = 0
        self._fps: float = fps
        self._log_tail: collections.deque = collections.deque(maxlen=200)
        # cam_name → ndarray RGB；Task 5 由录制器主循环 hook 写入
        self._latest_frames: dict = {}
        # Task 5 接入录制器线程，当前 stub 为 None
        self._recorder_thread: Optional[threading.Thread] = None

    # ---------- 按钮动作（写 events dict，等价键盘输入） ----------

    def save_episode(self) -> None:
        """保存当前 episode（等价键盘 → 键 keep）。

        写入 exit_early=True，EpisodeDecider 判定为 keep 并提交保存。
        """
        self._events["exit_early"] = True
        self._log("UI save_episode: exit_early=True (等价键盘 → keep)")

    def discard_episode(self) -> None:
        """丢弃当前 episode（等价键盘 ← 键 discard）。

        写入 rerecord_episode=True + exit_early=True，EpisodeDecider 判定为 discard。
        lerobot 模式下 rerecord 需同时置 exit_early=True 提前结束当前 ep。
        """
        self._events["rerecord_episode"] = True
        self._events["exit_early"] = True
        self._log("UI discard_episode: rerecord_episode=True + exit_early=True (等价键盘 ← discard)")

    def stop_recording(self) -> None:
        """停止整个录制会话（等价键盘 Esc 键 stop）。

        写入 stop_recording=True + exit_early=True，EpisodeDecider 判定为 stop→break 循环。
        """
        self._events["stop_recording"] = True
        self._events["exit_early"] = True
        self._log("UI stop_recording: stop_recording=True + exit_early=True (等价键盘 Esc stop)")

    # ---------- 命令队列动作（不直接调机器人，守坑 7） ----------

    def start_recording(self) -> bool:
        """请求开始录制（写命令队列 'start'，Task 5 录制器线程消费）。

        Returns:
            True：命令成功入队；False：队列已满，命令未入队。
        """
        try:
            self._cmd_q.put_nowait("start")
            self._log("UI start_recording: 命令入队 'start'")
            return True
        except queue.Full:
            self._log("UI start_recording: 命令队列已满，丢弃 'start'")
            return False

    def go_home(self) -> bool:
        """请求机械臂回 Home（写命令队列 'home'，Task 5 录制器线程消费）。

        不直接调 robot.reset (zerorpc)，守坑 7（避免与采集线程并发）。

        Returns:
            True：命令成功入队；False：队列已满，命令未入队。
        """
        try:
            self._cmd_q.put_nowait("home")
            self._log("UI go_home: 命令入队 'home'")
            return True
        except queue.Full:
            self._log("UI go_home: 命令队列已满，丢弃 'home'")
            return False

    # ---------- 状态/帧缓存 ----------

    def increment_episode_count(self) -> None:
        """episode 完成时递增计数（线程安全）。"""
        with self._lock:
            self._ep_count += 1

    def status_snapshot(self) -> dict:
        """返回当前状态快照（线程安全，纯 dict，可直接 JSON 序列化）。

        Returns:
            含 state / episode_count / fps / log_tail 的纯字典。
        """
        with self._lock:
            return {
                "state": self._sm.state.value,
                "episode_count": self._ep_count,
                "fps": self._fps,
                "log_tail": list(self._log_tail),
            }

    def update_latest_frame(self, cam_name: str, rgb: "np.ndarray") -> None:
        """更新相机最新帧缓存（Task 5 录制器主循环每帧 hook 调用）。

        Args:
            cam_name: 相机名称（如 'wrist_image'）。
            rgb: HxWx3 RGB uint8 ndarray，会被 copy 以防调用方后续修改原数组。
        """
        with self._lock:
            self._latest_frames[cam_name] = rgb.copy()

    def get_latest_frame(self, cam_name: str) -> "Optional[np.ndarray]":
        """获取相机最新帧（Task 3 预览路由调用）。

        Returns:
            HxWx3 RGB uint8 ndarray 副本，无帧时返回 None。
        """
        with self._lock:
            arr = self._latest_frames.get(cam_name)
            return arr.copy() if arr is not None else None

    # ---------- 内部辅助 ----------

    def _log(self, msg: str) -> None:
        """写入环形日志缓冲（线程安全）。"""
        with self._lock:
            self._log_tail.append(msg)
