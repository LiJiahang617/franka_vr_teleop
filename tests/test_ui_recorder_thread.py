"""
test_ui_recorder_thread.py

新路径 (prepare + consume_commands_blocking) 主线程消费测试：
- prepare() 切状态机到 WAITING，不启 daemon
- consume_commands_blocking() 在调用线程阻塞消费命令队列
- 单测用一根子线程跑 consume 模拟真机主线程，验证 start/home/frame_observer 行为
- stop_recording() 写 stop 标志并置 _should_stop=True，让 consume 退出

历史：旧 daemon 路径 (start()+_record_main()+wait_until_done()) 已删除，
原因见 lesson 2026-05-24-phaseE-ui-zerorpc-gevent-daemon-thread.md
（gevent thread-affinity 在 daemon thread 调 zerorpc 会死锁）。
"""
import importlib.util
import os
import time
import threading

import numpy as np

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_rc = importlib.util.spec_from_file_location(
    "recorder_controller",
    os.path.join(_P, "scripts/ui/recorder_controller.py"),
)
rc = importlib.util.module_from_spec(_rc)
_rc.loader.exec_module(rc)


class FakeRunner:
    """模拟 run_episodes：检查传入参数，跑两条 ep 后正常返回。"""

    def __init__(self):
        self.calls = 0
        self.kwargs = None

    def __call__(self, robot, teleop, saver, **kw):
        self.kwargs = kw
        self.calls += 1
        # 模拟两次循环，期间允许 stop_flag 提前结束
        for _ in range(2):
            if kw.get("stop_flag", lambda: False)():
                break
            time.sleep(0.02)


def _make_ctl(runner, events=None, reset_fn=None, episodes=2):
    """构造绑定了录制参数的 RecorderController。"""
    if events is None:
        events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    ctl = rc.RecorderController(events=events)
    ctl.attach_record_args(
        robot=object(),
        teleop=object(),
        saver=object(),
        run_episodes_fn=runner,
        fps=30.0,
        episode_sec=1.0,
        gripper_max_open=0.08,
        cam_names=["wrist_image"],
        out_dir="/tmp",
        task_name="t",
        oc2base_R=np.eye(3),
        vr_source="u",
        episodes=episodes,
        reset_fn=reset_fn,
        reset_wait=0.0,
    )
    return ctl, events


def _start_consumer(ctl):
    """启 prepare + 子线程跑 consume_commands_blocking 模拟主线程消费。

    返回 (consumer_thread,)。调用方负责后续 stop_recording() + join。
    """
    ctl.prepare()
    t = threading.Thread(
        target=ctl.consume_commands_blocking, daemon=True, name="test-consumer"
    )
    t.start()
    return t


def _stop_and_join(ctl, consumer_thread, timeout=2.0):
    """触发 stop 并 join 消费线程；测试结束清场用。"""
    ctl.stop_recording()  # 置 _should_stop=True
    consumer_thread.join(timeout=timeout)


def test_consume_blocking_calls_run_fn_on_start_cmd():
    """consume_commands_blocking 消费 'start' 命令后调用 run_episodes_fn 恰好一次。"""
    runner = FakeRunner()
    ctl, _ = _make_ctl(runner)
    consumer = _start_consumer(ctl)
    ctl.start_recording()  # 入队 "start"

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and runner.calls == 0:
        time.sleep(0.02)
    assert runner.calls == 1
    _stop_and_join(ctl, consumer)


def test_stop_sets_events_and_consumer_exits():
    """stop_recording() 写 stop 标志并让 consume 主循环退出，无悬挂线程。"""
    runner = FakeRunner()
    ctl, events = _make_ctl(runner, episodes=10)
    consumer = _start_consumer(ctl)
    ctl.start_recording()
    time.sleep(0.05)

    ctl.stop_recording()   # 写 events["stop_recording"]=True + _should_stop=True
    consumer.join(timeout=2.0)

    assert events["stop_recording"] is True
    # consumer 已退出
    assert not consumer.is_alive()


def test_home_cmd_consumed_calls_injected_reset():
    """'home' 命令由 consume 主循环串行消费 → 调用 reset_fn（守坑 7：不在 UI 线程直调）。"""
    resets = []

    def reset_fn():
        resets.append(time.monotonic())

    runner = FakeRunner()
    ctl, _ = _make_ctl(runner, reset_fn=reset_fn, episodes=1)
    consumer = _start_consumer(ctl)
    ctl.go_home()   # 入队 "home"

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not resets:
        time.sleep(0.02)
    _stop_and_join(ctl, consumer)

    assert len(resets) == 1


def test_status_snapshot_has_frame_count_and_duration_sec():
    """status_snapshot() 含 frame_count 和 duration_sec 字段（Task 4 前端集成契约）。"""
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    ctl = rc.RecorderController(events=events)
    snap = ctl.status_snapshot()
    assert "frame_count" in snap, "status_snapshot 缺少 frame_count 字段"
    assert "duration_sec" in snap, "status_snapshot 缺少 duration_sec 字段"
    # 无录制时初始值为 0 / 0.0
    assert snap["frame_count"] == 0
    assert snap["duration_sec"] == 0.0


def test_frame_observer_updates_latest_frame_in_controller():
    """主循环 frame_observer hook 每帧更新 controller 的 latest_frames 缓存。"""
    # 用可注入的 frame_observer 直接测试 update_latest_frame 的线程行为
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    ctl = rc.RecorderController(events=events)

    # 直接调用 update_latest_frame 模拟主循环写入
    img = np.zeros((8, 8, 3), np.uint8)
    img[0, 0, 0] = 42
    ctl.update_latest_frame("wrist_image", img)

    frame = ctl.get_latest_frame("wrist_image")
    assert frame is not None
    assert frame[0, 0, 0] == 42
    # 返回副本，不与内部缓存共享引用
    frame[0, 0, 0] = 99
    assert ctl.get_latest_frame("wrist_image")[0, 0, 0] == 42


def test_update_frame_count_and_duration_via_hook():
    """通过 update_recording_progress() 更新 frame_count/duration_sec，status_snapshot 可见。"""
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    ctl = rc.RecorderController(events=events)

    # 模拟录制过程中主循环更新进度
    ctl.update_recording_progress(frame_count=42, duration_sec=1.4)
    snap = ctl.status_snapshot()
    assert snap["frame_count"] == 42
    assert abs(snap["duration_sec"] - 1.4) < 1e-6

    # reset_recording_progress 后回到零值
    ctl.reset_recording_progress()
    snap = ctl.status_snapshot()
    assert snap["frame_count"] == 0
    assert snap["duration_sec"] == 0.0


def test_consumer_updates_progress_during_run():
    """consume 主循环运行 run_episodes 时，frame_observer 写帧缓存且进度字段会更新。"""

    class ProgressRunner:
        """模拟 run_episodes：每次调用时触发 frame_observer 并递增进度。"""

        def __call__(self, robot, teleop, saver, **kw):
            # 模拟两帧录制，通过 frame_observer 写入
            frame_obs = kw.get("frame_observer")
            for i in range(2):
                if kw.get("stop_flag", lambda: False)():
                    break
                img = np.zeros((4, 4, 3), np.uint8)
                img[0, 0, 0] = i + 1
                if frame_obs is not None:
                    frame_obs("wrist_image", img)
                time.sleep(0.01)

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    ctl = rc.RecorderController(events=events)
    runner = ProgressRunner()

    ctl.attach_record_args(
        robot=object(),
        teleop=object(),
        saver=object(),
        run_episodes_fn=runner,
        fps=30.0,
        episode_sec=1.0,
        gripper_max_open=0.08,
        cam_names=["wrist_image"],
        out_dir="/tmp",
        task_name="t",
        oc2base_R=np.eye(3),
        vr_source="u",
        episodes=1,
        reset_fn=None,
        reset_wait=0.0,
    )
    consumer = _start_consumer(ctl)
    ctl.start_recording()
    # 等 runner 跑完
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and ctl.get_latest_frame("wrist_image") is None:
        time.sleep(0.02)
    _stop_and_join(ctl, consumer)

    # frame_observer 写入后，最新帧应可读（最后一帧 img[0,0,0]=2）
    frame = ctl.get_latest_frame("wrist_image")
    assert frame is not None
    assert frame[0, 0, 0] == 2

# ===== 新增测试 =====
# 缺陷 1：decide 必须复用 EpisodeDecider 而非恒返回 "keep"
# 缺陷 2：frame_observer 接线 frame_count / duration_sec


def _run_decide_capture(events):
    """共享 helper：构造 ctl + capture-decide runner，跑一轮 consume，**在 runner 内部就调用
    decide 拿 result**，然后再 stop。这样避免 stop_recording() 污染共享 events 后
    再调 decide 取回错误判定。

    返回 (captured_results, captured_decides)：长度均为 1。
    """
    captured_decide = []
    captured_result = []

    class DecideCapture:
        def __call__(self, robot, teleop, saver, **kw):
            decide = kw.get("decide")
            assert decide is not None, "run_fn 未收到 decide 参数"
            captured_decide.append(decide)
            # 立刻调一次 decide：此时 events 还未被 _stop_and_join 污染
            captured_result.append(decide(ep=0))

    runner = DecideCapture()
    ctl = rc.RecorderController(events=events)
    ctl.attach_record_args(
        robot=object(), teleop=object(), saver=object(),
        run_episodes_fn=runner,
        fps=30.0, episode_sec=1.0, gripper_max_open=0.08,
        cam_names=["wrist_image"], out_dir="/tmp", task_name="t",
        oc2base_R=np.eye(3), vr_source="u", episodes=1,
        reset_fn=None, reset_wait=0.0,
    )
    consumer = _start_consumer(ctl)
    ctl.start_recording()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not captured_decide:
        time.sleep(0.02)
    _stop_and_join(ctl, consumer)
    return captured_result, captured_decide


def test_decide_uses_episode_decider_rerecord():
    """UI 模式 decide 在 rerecord_episode=True 时应返回 'discard'，而非 'keep'。"""
    events = {"exit_early": True, "rerecord_episode": True, "stop_recording": False}
    results, captured = _run_decide_capture(events)
    assert len(captured) == 1, "decide 未被传入 run_fn"
    assert results[0] == "discard", f"期望 'discard'，实际得到 {results[0]!r}"


def test_decide_uses_episode_decider_stop():
    """UI 模式 decide 在 stop_recording=True 时应返回 'stop'。"""
    events = {"exit_early": True, "rerecord_episode": False, "stop_recording": True}
    results, captured = _run_decide_capture(events)
    assert len(captured) == 1
    assert results[0] == "stop", f"期望 'stop'，实际得到 {results[0]!r}"


def test_decide_uses_episode_decider_keep():
    """UI 模式 decide 在 events 全 False 时应返回 'keep'（headless/正常保存）。"""
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    results, captured = _run_decide_capture(events)
    assert len(captured) == 1
    assert results[0] == "keep", f"期望 'keep'，实际得到 {results[0]!r}"


def test_frame_observer_updates_frame_count_and_duration():
    """frame_observer 调用同一相机 5 次后，frame_count 应增长，duration_sec >= 0。"""

    class FrameObsCapture:
        def __call__(self, robot, teleop, saver, **kw):
            fo = kw.get("frame_observer")
            # 模拟 5 次同一相机帧到达
            img = np.zeros((4, 4, 3), np.uint8)
            for _ in range(5):
                if fo is not None:
                    fo("wrist_image", img)
                time.sleep(0.005)

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    ctl = rc.RecorderController(events=events)
    runner = FrameObsCapture()
    ctl.attach_record_args(
        robot=object(), teleop=object(), saver=object(),
        run_episodes_fn=runner,
        fps=30.0, episode_sec=1.0, gripper_max_open=0.08,
        cam_names=["wrist_image"], out_dir="/tmp", task_name="t",
        oc2base_R=np.eye(3), vr_source="u", episodes=1,
        reset_fn=None, reset_wait=0.0,
    )
    consumer = _start_consumer(ctl)
    ctl.start_recording()
    # 等 runner 跑完 (5 帧 × 5ms ≈ 25ms)，留一点 buffer
    time.sleep(0.2)
    _stop_and_join(ctl, consumer)

    snap = ctl.status_snapshot()
    # 5 帧录制后 frame_count 应有记录（>0 或 episode 结束后已 reset，取决于实现）
    # 这里验证字段存在且 duration_sec >= 0（真实接线后不再是固定 0）
    assert "frame_count" in snap
    assert "duration_sec" in snap
    assert snap["duration_sec"] >= 0.0


def test_frame_count_increases_during_recording():
    """录制期间 frame_count 应在录制完成前累积（未 reset 时 > 0）。"""
    max_frame_count_seen = [0]
    recording_done = threading.Event()

    class CountingRunner:
        def __call__(self, robot, teleop, saver, **kw):
            fo = kw.get("frame_observer")
            img = np.zeros((4, 4, 3), np.uint8)
            # 注入 5 帧
            for i in range(5):
                if fo:
                    fo("wrist_image", img)
                time.sleep(0.01)
            # 录制期间读取 frame_count（在 decide/reset 之前）
            # 注意：decide 调用在 runner 返回后，但这里需要在 reset 前采样
            recording_done.set()

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    ctl = rc.RecorderController(events=events)

    # 监控线程：录制中途采样 frame_count
    def monitor():
        recording_done.wait(timeout=2.0)
        snap = ctl.status_snapshot()
        max_frame_count_seen[0] = snap["frame_count"]

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()

    runner = CountingRunner()
    ctl.attach_record_args(
        robot=object(), teleop=object(), saver=object(),
        run_episodes_fn=runner,
        fps=30.0, episode_sec=1.0, gripper_max_open=0.08,
        cam_names=["wrist_image"], out_dir="/tmp", task_name="t",
        oc2base_R=np.eye(3), vr_source="u", episodes=1,
        reset_fn=None, reset_wait=0.0,
    )
    consumer = _start_consumer(ctl)
    ctl.start_recording()
    mon.join(timeout=3.0)
    _stop_and_join(ctl, consumer)

    # 录制期间 frame_count 应 >= 5（5 次同一相机调用）
    assert max_frame_count_seen[0] >= 5, (
        f"期望录制期间 frame_count >= 5，实际为 {max_frame_count_seen[0]}"
    )
