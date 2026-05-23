"""
RecorderController：UI 与录制器之间的桥梁。

职责：
- 持有 events dict（供 EpisodeDecider 消费，语义与终端键盘逐字等价）
- 持有命令队列（start/home 走队列，由 Task 5 录制器主循环串行消费；守坑 7 不直调 zerorpc）
- 持有状态机（Task 1 的 StateMachine）
- 线程安全：episode_count / latest_frames / log_tail / frame_count / duration_sec 均在 _lock 下操作
- 录制器线程在 Task 5 接入（attach_record_args + start + _record_main）
"""
import collections
import importlib.util
import logging
import os
import queue
import sys
import threading
import time
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

# 动态加载 EpisodeDecider（与 UI 包分离，避免循环依赖）
_SCRIPTS_DIR = os.path.dirname(_UI_DIR)
_ep_key_spec = importlib.util.spec_from_file_location(
    "episode_keyboard",
    os.path.join(_SCRIPTS_DIR, "core", "episode_keyboard.py"),
)
_ep_key_mod = importlib.util.module_from_spec(_ep_key_spec)
_ep_key_spec.loader.exec_module(_ep_key_mod)
EpisodeDecider = _ep_key_mod.EpisodeDecider

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
        _should_stop: 后台线程退出标志（仅 _record_main 命令循环用）。
        _frame_count: 当前 episode 已录帧数（status_snapshot 暴露）。
        _duration_sec: 当前 episode 已录时长（秒）（status_snapshot 暴露）。
        _first_cam: cam_names[0]，用于 frame_observer 判断「新帧」（缺陷 2 修复）。
        _record_t0: 每条 episode 录制起点时间戳（monotonic）。
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
        # _should_stop 仅用于 _record_main 命令消费循环的退出
        self._should_stop: bool = False
        # 当前 episode 进度（status_snapshot 暴露给 Task 4 前端）
        self._frame_count: int = 0
        self._duration_sec: float = 0.0
        # 缺陷 2：记录第一路相机名和录制起点（attach_record_args / _handle_start_cmd 填充）
        self._first_cam: Optional[str] = None
        self._record_t0: float = 0.0
        # Preview sampler：waiting/standby 状态下持续 read cam 填 _latest_frames
        # 让 UI 预览在录制开始前就能显示（否则 /api/preview/<cam> 永远 404）
        self._preview_thread: Optional[threading.Thread] = None
        self._preview_stop: Optional[threading.Event] = None
        self._preview_paused: bool = False  # recording 时置 True，让 frame_observer 接管

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

    # ---------- Preview sampler：standby 时 UI 也能看相机 ----------

    def start_preview_sampler(self, cameras: dict, fps: float = 10.0) -> None:
        """启动 preview sampler 线程：standby 时持续 read 各路 cam 填 _latest_frames。

        UI 设计原本只在 RECORDING 时通过 frame_observer 写 _latest_frames，
        waiting/initializing 状态下 /api/preview/<cam> 永远 404。本线程兜底：
        - standby 时（_preview_paused=False）：10Hz read cam 填缓存
        - recording 时（_preview_paused=True）：跳过 read，让 frame_observer 接管

        Args:
            cameras: dict[cam_name, camera_obj]，cam.read() 返 ndarray 或 (ndarray, hw_ts)
            fps: preview 采样率，默认 10Hz（够流畅，不抢真 record 30Hz）
        """
        if self._preview_thread is not None:
            return  # 已启动
        self._preview_stop = threading.Event()

        def _loop():
            period = 1.0 / fps
            while not self._preview_stop.is_set():
                if not self._preview_paused:
                    for cn, cam in cameras.items():
                        try:
                            img = cam.read()
                            # RealsenseHwWrapper.read() 返 (rgb, hw_ts) tuple
                            if isinstance(img, tuple) and len(img) == 2:
                                img = img[0]
                            if img is not None and hasattr(img, "shape"):
                                self.update_latest_frame(cn, img)
                        except Exception as e:
                            log.warning(
                                f"[preview_sampler] {cn} read 失败: {type(e).__name__}: {e}"
                            )
                self._preview_stop.wait(period)

        self._preview_thread = threading.Thread(
            target=_loop, name="PreviewSampler", daemon=True
        )
        self._preview_thread.start()
        log.info(f"[preview_sampler] 启动，{len(cameras)} 路相机 @{fps}Hz")

    def stop_preview_sampler(self, timeout: float = 2.0) -> None:
        """停止 preview sampler 线程（cleanup 时调用）。"""
        if self._preview_thread is None:
            return
        if self._preview_stop is not None:
            self._preview_stop.set()
        self._preview_thread.join(timeout=timeout)
        if self._preview_thread.is_alive():
            log.warning(f"[preview_sampler] 在 {timeout}s 内未停止")
        self._preview_thread = None

    def pause_preview_sampler(self) -> None:
        """录制开始时调：暂停 preview sampler，让 record_episode 的 frame_observer 接管。"""
        self._preview_paused = True

    def resume_preview_sampler(self) -> None:
        """录制结束/丢弃后调：恢复 preview sampler，standby 时 UI 仍能看预览。"""
        self._preview_paused = False

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
        # 缺陷 2：记录第一路相机名，用于 frame_observer 判断「新帧」
        self._first_cam = cam_names[0] if cam_names else None

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
        """处理 'start' 命令：调用 run_episodes_fn 执行录制。

        缺陷 1 修复：UI 模式复用 EpisodeDecider，与终端键盘模式逐字等价。
          - EpisodeDecider(self._events) 驱动 decide 闭包
          - stop_flag = dec.episode_stop_flag()（由 EpisodeDecider 提供）
          - stop 不 reset（保留全局停止标志让 run_episodes 跳出循环）
        缺陷 2 修复：frame_observer 接线 frame_count / duration_sec。
          - 仅当 cam == self._first_cam 时计为新帧，避免多路相机重复计数
          - 录制起点 _record_t0 在首帧到达时记录
          - decide 闭包在每条 ep 决策后 reset 进度计数器
        """
        if self._record_args is None:
            log.error("[RecorderController] attach_record_args 未调用，无法 start")
            return

        args = self._record_args
        run_fn = args["run_episodes_fn"]

        # 缺陷 1 修复：构造 EpisodeDecider，与终端键盘模式写法逐字等价
        dec = EpisodeDecider(self._events)

        # 缺陷 2 修复：帧计数状态（闭包共享）
        frame_state = {"count": 0, "t0": None}

        # 缺陷 2 修复：frame_observer 接线 frame_count / duration_sec
        first_cam = self._first_cam

        def _frame_obs(cam: str, img: "np.ndarray") -> None:
            # 每帧更新预览缓存（所有相机）
            self.update_latest_frame(cam, img)
            # 仅 first_cam 计为新帧，避免多路相机重复计数
            if cam == first_cam:
                now = time.monotonic()
                if frame_state["t0"] is None:
                    # 首帧到达时记录录制起点
                    frame_state["t0"] = now
                frame_state["count"] += 1
                elapsed = now - frame_state["t0"]
                self.update_recording_progress(
                    frame_count=frame_state["count"],
                    duration_sec=elapsed,
                )

        # 缺陷 1 修复：decide 闭包复用 EpisodeDecider，与 run_record_hdf5.main 写法等价
        def decide(ep):
            action = dec.decide_after_episode()
            # stop 不 reset：stop_recording 是全局停止标志，保留让 run_episodes 跳出循环
            if action in ("keep", "discard"):
                dec.reset_episode_flags()
            # 缺陷 2 修复：每条 ep 决策后重置进度计数，下一条 ep 从 0 计
            self.reset_recording_progress()
            frame_state["count"] = 0
            frame_state["t0"] = None
            return action

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
                decide=decide,
                reset_fn=args["reset_fn"],
                reset_wait=args["reset_wait"],
                # 缺陷 1 修复：stop_flag 由 EpisodeDecider 提供（不再用 _make_stop_flag）
                stop_flag=dec.episode_stop_flag(),
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
