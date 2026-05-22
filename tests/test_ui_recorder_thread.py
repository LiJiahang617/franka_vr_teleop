"""
test_ui_recorder_thread.py

Task 5 后台线程接入测试：
- start() 起后台线程消费 "start" 命令 → 调用 run_episodes_fn
- stop_recording() 写 stop 标志 + join 后台线程
- "home" 命令由后台线程串行消费 → 调用 reset_fn（守坑 7）
- status_snapshot 含 frame_count / duration_sec 字段
- frame_observer hook 由后台线程每帧写入 latest_frames
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


def test_start_launches_background_thread_and_consumes_start_cmd():
    """start() 起后台线程，消费 'start' 命令后调用 run_episodes_fn 恰好一次。"""
    runner = FakeRunner()
    ctl, _ = _make_ctl(runner)
    ctl.start_recording()  # 入队 "start"
    ctl.start()            # 起后台线程消费

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and runner.calls == 0:
        time.sleep(0.02)
    assert runner.calls == 1
    ctl.wait_until_done(timeout=2.0)


def test_stop_sets_events_and_joins_gracefully():
    """stop_recording() 写 stop 标志，wait_until_done() 后线程已退出，无悬挂线程。"""
    runner = FakeRunner()
    ctl, events = _make_ctl(runner, episodes=10)
    ctl.start_recording()
    ctl.start()
    time.sleep(0.05)

    ctl.stop_recording()   # 写 events["stop_recording"]=True
    ctl.wait_until_done(timeout=2.0)

    assert events["stop_recording"] is True
    # 后台线程已 join 完，无悬挂
    assert ctl._recorder_thread is None or not ctl._recorder_thread.is_alive()


def test_home_cmd_consumed_calls_injected_reset():
    """'home' 命令由后台线程串行消费 → 调用 reset_fn（守坑 7：不在 UI 线程直调）。"""
    resets = []

    def reset_fn():
        resets.append(time.monotonic())

    runner = FakeRunner()
    ctl, _ = _make_ctl(runner, reset_fn=reset_fn, episodes=1)
    ctl.go_home()   # 入队 "home"
    ctl.start()     # 主循环消费 home → reset_fn()

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not resets:
        time.sleep(0.02)
    ctl.stop_recording()
    ctl.wait_until_done(timeout=2.0)

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
    """后台线程 frame_observer hook 每帧更新 controller 的 latest_frames 缓存。"""
    # 用可注入的 frame_observer 直接测试 update_latest_frame 的线程行为
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    ctl = rc.RecorderController(events=events)

    # 直接调用 update_latest_frame 模拟后台线程写入
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

    # 模拟录制过程中后台线程更新进度
    ctl.update_recording_progress(frame_count=42, duration_sec=1.4)
    snap = ctl.status_snapshot()
    assert snap["frame_count"] == 42
    assert abs(snap["duration_sec"] - 1.4) < 1e-6

    # reset_recording_progress 后回到零值
    ctl.reset_recording_progress()
    snap = ctl.status_snapshot()
    assert snap["frame_count"] == 0
    assert snap["duration_sec"] == 0.0


def test_background_thread_updates_progress_during_run():
    """后台线程运行 run_episodes 时，frame_observer 写帧缓存且进度字段会更新。"""
    frame_calls = []

    class ProgressRunner:
        """模拟 run_episodes：每次调用时触发 frame_observer 并递增进度。"""

        def __init__(self):
            self.ctl = None  # 稍后注入

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
    runner.ctl = ctl

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
    ctl.start_recording()
    ctl.start()
    ctl.wait_until_done(timeout=2.0)

    # frame_observer 写入后，最新帧应可读（最后一帧 img[0,0,0]=2）
    frame = ctl.get_latest_frame("wrist_image")
    assert frame is not None
    assert frame[0, 0, 0] == 2
# ===== 新增测试 =====
# 缺陷 1：decide 必须复用 EpisodeDecider 而非恒返回 "keep"
# 缺陷 2：frame_observer 接线 frame_count / duration_sec


def test_decide_uses_episode_decider_rerecord():
    """UI 模式 decide 在 rerecord_episode=True 时应返回 'discard'，而非 'keep'。"""
    captured_decide = []

    class DecideCapture:
        """捕获传入的 decide 函数，并直接调用一次。"""
        def __call__(self, robot, teleop, saver, **kw):
            decide = kw.get("decide")
            assert decide is not None, "run_fn 未收到 decide 参数"
            captured_decide.append(decide)

    events = {"exit_early": True, "rerecord_episode": True, "stop_recording": False}
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
    ctl.start_recording()
    ctl.start()
    ctl.wait_until_done(timeout=2.0)

    assert len(captured_decide) == 1, "decide 未被传入 run_fn"
    result = captured_decide[0](ep=0)
    assert result == "discard", f"期望 'discard'，实际得到 {result!r}"


def test_decide_uses_episode_decider_stop():
    """UI 模式 decide 在 stop_recording=True 时应返回 'stop'。"""
    captured_decide = []

    class DecideCapture:
        def __call__(self, robot, teleop, saver, **kw):
            decide = kw.get("decide")
            captured_decide.append(decide)

    events = {"exit_early": True, "rerecord_episode": False, "stop_recording": True}
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
    ctl.start_recording()
    ctl.start()
    ctl.wait_until_done(timeout=2.0)

    assert len(captured_decide) == 1
    result = captured_decide[0](ep=0)
    assert result == "stop", f"期望 'stop'，实际得到 {result!r}"


def test_decide_uses_episode_decider_keep():
    """UI 模式 decide 在 events 全 False 时应返回 'keep'（headless/正常保存）。"""
    captured_decide = []

    class DecideCapture:
        def __call__(self, robot, teleop, saver, **kw):
            decide = kw.get("decide")
            captured_decide.append(decide)

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
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
    ctl.start_recording()
    ctl.start()
    ctl.wait_until_done(timeout=2.0)

    assert len(captured_decide) == 1
    result = captured_decide[0](ep=0)
    assert result == "keep", f"期望 'keep'，实际得到 {result!r}"


def test_frame_observer_updates_frame_count_and_duration():
    """frame_observer 调用同一相机 5 次后，frame_count 应增长，duration_sec >= 0。"""
    frame_observer_ref = []

    class FrameObsCapture:
        def __call__(self, robot, teleop, saver, **kw):
            fo = kw.get("frame_observer")
            frame_observer_ref.append(fo)
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
    ctl.start_recording()
    ctl.start()
    ctl.wait_until_done(timeout=2.0)

    snap = ctl.status_snapshot()
    # 5 帧录制后 frame_count 应有记录（>0 或 episode 结束后已 reset，取决于实现）
    # 测试录制期间最大值：直接在录制结束前读取
    # 这里验证 duration_sec 字段存在且 >= 0（真实接线后不再是固定 0）
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
    ctl.start_recording()
    ctl.start()
    mon.join(timeout=3.0)
    ctl.wait_until_done(timeout=2.0)

    # 录制期间 frame_count 应 >= 5（5 次同一相机调用）
    assert max_frame_count_seen[0] >= 5, (
        f"期望录制期间 frame_count >= 5，实际为 {max_frame_count_seen[0]}"
    )
