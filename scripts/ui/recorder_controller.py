"""
RecorderController：UI 与录制器之间的桥梁。

职责：
- 持有 events dict（供 EpisodeDecider 消费，语义与终端键盘逐字等价）
- 持有命令队列（start/home 走队列，由 Task 5 录制器主循环串行消费；守坑 7 不直接调 zerorpc）
- 持有状态机（Task 1 的 StateMachine）
- 线程安全：episode_count / latest_frames / log_tail / frame_count / duration_sec 均在 _lock 下操作
- 录制器线程在 Task 5 接入（attach_record_args + start + _record_main）
"""
import collections
import importlib.util
import logging
import os
import queue
import threading
from typing import Optional

log = logging.getLogger("recorder_controller")

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
        _lock: RLock，保护 _ep_count / _latest_frames / _log_tail / _frame_count / _duration_sec。
        _ep_count: 已完成 episode 计数。
        _fps: 录制帧率（UI 状态显示用）。
        _log_tail: 近 200 条日志（环形缓冲）。
        _latest_frames: cam_name → ndarray RGB（Task 5 填充，Task 3 读取）。
        _recorder_thread: 录制器线程（Task 5 接入）。
        _record_args: attach_record_args 保存的录制参数字典。
        _should_stop: 后台线程退出标志。
        _frame_count: 当前 episode 已录帧数（status_snapshot 暴露）。
        _duration_sec: 当前 episode 已录时长（秒）（status_snapshot 暴露）。
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
        # Task 5 接入录制器线程
        self._recorder_thread: Optional[threading.Thread] = None
        self._record_args: Optional[dict] = None
        self._should_stop: bool = False
        # 当前 episode 进度（status_snapshot 暴露给 Task 4 前端）
        self._frame_count: int = 0
        self._duration_sec: float = 0.0

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
        同时置 _should_stop=True 通知后台线程命令循环退出。
        """
        self._events["stop_recording"] = True
        self._events["exit_early"] = True
        self._should_stop = True
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
            含 state / episode_count / fps / log_tail / frame_count / duration_sec 的纯字典。
            frame_count: 当前 episode 已录帧数（无录制时为 0）。
            duration_sec: 当前 episode 已录时长秒（无录制时为 0.0）。
        """
        with self._lock:
            return {
                "state": self._sm.state.value,
                "episode_count": self._ep_count,
                "fps": self._fps,
                "log_tail": list(self._log_tail),
                "frame_count": self._frame_count,
                "duration_sec": self._duration_sec,
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

    # ---------- Task 5：录制参数装载 + 后台线程 ----------

    def attach_record_args(self, *, robot, teleop, saver, run_episodes_fn=None,
                           fps, episode_sec, gripper_max_open, cam_names,
                           out_dir, task_name, oc2base_R, vr_source,
                           episodes, reset_fn=None, reset_wait=0.0) -> None:
        """装载录制参数（start() 前调用，后台线程消费）。

        Args:
            robot: Franka 实例（或 Fake）。
            teleop: teleop 实例（或 Fake）。
            saver: 存盘器实例（AsyncEpisodeSaver 或 Fake）。
            run_episodes_fn: 可注入的 run_episodes 实现（None=使用真 run_episodes）。
                            测试中注入 FakeRunner，真机走真 run_episodes。
            fps: 录制帧率。
            episode_sec: 每条 episode 最长时间（秒）。
            gripper_max_open: 夹爪最大开度（米）。
            cam_names: 相机名列表。
            out_dir: 输出目录。
            task_name: 任务名称。
            oc2base_R: 3x3 标定旋转矩阵。
            vr_source: VR 来源标识。
            episodes: 录制 episode 总数。
            reset_fn: 可选 Callable，episode 间调用回 home（守坑 7 由后台线程串行调）。
            reset_wait: reset 后等待时间（秒）。
        """
        self._record_args = dict(
            robot=robot,
            teleop=teleop,
            saver=saver,
            run_episodes_fn=run_episodes_fn,
            fps=fps,
            episode_sec=episode_sec,
            gripper_max_open=gripper_max_open,
            cam_names=cam_names,
            out_dir=out_dir,
            task_name=task_name,
            oc2base_R=oc2base_R,
            vr_source=vr_source,
            episodes=episodes,
            reset_fn=reset_fn,
            reset_wait=reset_wait,
        )

    def start(self) -> None:
        """启动后台录制线程（_record_main 消费命令队列）。

        attach_record_args 须在 start() 前调用。
        线程为 daemon，进程退出时自动清理。
        """
        self._should_stop = False
        self._recorder_thread = threading.Thread(
            target=self._record_main, daemon=True, name="recorder-main"
        )
        self._recorder_thread.start()

    def wait_until_done(self, timeout: float = 5.0) -> None:
        """等待后台线程退出（join with timeout，无僵尸线程）。

        Args:
            timeout: join 超时秒数，超时后仅打警告，不抛异常。
        """
        t = self._recorder_thread
        if t is None:
            return
        t.join(timeout)
        if t.is_alive():
            log.warning("[RecorderController] 后台录制线程 join 超时，仍在运行")
        self._recorder_thread = None

    def update_recording_progress(self, *, frame_count: int, duration_sec: float) -> None:
        """更新当前 episode 录制进度（后台线程 frame_observer 调用，线程安全）。

        Args:
            frame_count: 当前 episode 已录帧数。
            duration_sec: 当前 episode 已录时长（秒）。
        """
        with self._lock:
            self._frame_count = frame_count
            self._duration_sec = duration_sec

    def reset_recording_progress(self) -> None:
        """重置 episode 进度计数（每条 ep 开始前调用，线程安全）。"""
        with self._lock:
            self._frame_count = 0
            self._duration_sec = 0.0

    def _make_stop_flag(self):
        """返回传给 run_episodes 的 stop_flag callable。

        组合 exit_early / stop_recording（events dict）和 _should_stop（UI 全局退出），
        与 EpisodeDecider.episode_stop_flag() 等价 + 额外 UI 全局兜底。
        """
        events = self._events

        def _flag():
            return bool(
                events.get("exit_early")
                or events.get("stop_recording")
                or self._should_stop
            )

        return _flag

    def _record_main(self) -> None:
        """后台录制线程主循环：消费命令队列，串行执行 start/home 命令。

        - 'start'：调用 run_episodes_fn（真机或 Fake），传入 frame_observer hook。
        - 'home'：串行调用 reset_fn（守坑 7：UI 线程不直调 zerorpc）。
        - 其他命令：丢弃并记日志。
        - _should_stop=True 时退出循环。
        """
        while not self._should_stop:
            try:
                cmd = self._cmd_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if cmd == "start":
                self._handle_start_cmd()
            elif cmd == "home":
                self._handle_home_cmd()
            else:
                log.warning(f"[RecorderController] 未知命令: {cmd!r}，忽略")

    def _handle_start_cmd(self) -> None:
        """处理 'start' 命令：调用 run_episodes_fn 执行录制。"""
        if self._record_args is None:
            log.error("[RecorderController] attach_record_args 未调用，无法 start")
            return

        args = self._record_args
        run_fn = args["run_episodes_fn"]

        # frame_observer：每帧每路 cam 写入 latest_frames 缓存（Task 3 预览路由读取）
        def _frame_obs(cam_name: str, img: "np.ndarray") -> None:
            self.update_latest_frame(cam_name, img)

        try:
            run_fn(
                args["robot"],
                args["teleop"],
                args["saver"],
                fps=args["fps"],
                episode_sec=args["episode_sec"],
                gripper_max_open=args["gripper_max_open"],
                cam_names=args["cam_names"],
                out_dir=args["out_dir"],
                task_name=args["task_name"],
                oc2base_R=args["oc2base_R"],
                vr_source=args["vr_source"],
                episodes=args["episodes"],
                decide=lambda ep: "keep",  # UI 模式：由按钮事件控制，decide 默认 keep
                reset_fn=args["reset_fn"],
                reset_wait=args["reset_wait"],
                stop_flag=self._make_stop_flag(),
                frame_observer=_frame_obs,
            )
        except Exception as exc:
            log.exception(f"[RecorderController] run_episodes_fn 异常: {exc}")

    def _handle_home_cmd(self) -> None:
        """处理 'home' 命令：串行调用 reset_fn（守坑 7，不在 UI 线程直调 zerorpc）。"""
        if self._record_args is None:
            log.error("[RecorderController] attach_record_args 未调用，无法 home")
            return

        reset_fn = self._record_args.get("reset_fn")
        if reset_fn is not None:
            try:
                reset_fn()
            except Exception as exc:
                log.exception(f"[RecorderController] reset_fn 异常: {exc}")
        else:
            log.warning("[RecorderController] 'home' 命令：reset_fn 未配置，跳过")

    # ---------- 内部辅助 ----------

    def _log(self, msg: str) -> None:
        """写入环形日志缓冲（线程安全）。"""
        with self._lock:
            self._log_tail.append(msg)
