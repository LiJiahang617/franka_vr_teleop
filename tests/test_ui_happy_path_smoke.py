"""
UI happy path smoke: start → 状态轮询 → save → 状态机过渡 → stop → 优雅退出。

mock FakeRunner 跑两条 ep，全程 test_client；不起真 Flask 服务器，不触真机。
验证 UI 关键链路：routes → controller → events dict → stop_flag → fake run_episodes。
"""
import importlib.util
import os
import time

import numpy as np

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 加载 control_panel（含 build_app）
_cp_spec = importlib.util.spec_from_file_location(
    "control_panel", os.path.join(_P, "scripts/ui/control_panel.py")
)
_cp = importlib.util.module_from_spec(_cp_spec)
_cp_spec.loader.exec_module(_cp)

# 加载 recorder_controller
_rc_spec = importlib.util.spec_from_file_location(
    "recorder_controller", os.path.join(_P, "scripts/ui/recorder_controller.py")
)
_rc = importlib.util.module_from_spec(_rc_spec)
_rc_spec.loader.exec_module(_rc)


class FakeRunner:
    """模拟 run_episodes：跑两条 ep，支持 stop_flag 提前退出。"""

    def __init__(self):
        self.calls = 0
        self.kwargs = {}

    def __call__(self, robot, teleop, saver, **kw):
        self.calls += 1
        self.kwargs = kw
        for _ in range(2):
            if kw.get("stop_flag", lambda: False)():
                break
            time.sleep(0.01)


def _make_setup():
    """构造 events + controller + app + test_client，attach_record_args 完毕。"""
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    runner = FakeRunner()
    ctl = _rc.RecorderController(events=events)
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
        task_name="smoke",
        oc2base_R=np.eye(3),
        vr_source="u",
        episodes=2,
        reset_fn=None,
        reset_wait=0.0,
    )
    app = _cp.build_app(controller=ctl)
    client = app.test_client()
    return client, ctl, events, runner


def test_happy_path_start_to_stop_all_routes_survive():
    """
    完整 happy path smoke：
    1. GET / → 200（控制面板 HTML 渲染）
    2. GET /api/status → 200，JSON 含 'state'
    3. POST /api/start → 200，ok=True，cmd_q 收到 'start'
    4. POST /api/save → 200，events[exit_early]=True
    5. POST /api/discard → 200，events[rerecord_episode]=True
    6. GET /api/preview/wrist_image → 返回合法 JSON（无帧时返回 404 或 200 含 null）
    7. POST /api/stop → 200，events[stop_recording]=True
    8. Cache-Control 头在每条响应中均存在
    9. 后台线程可 start → wait_until_done 完成（无阻塞）
    """
    client, ctl, events, runner = _make_setup()

    # 1. GET / → HTML 渲染
    r = client.get("/")
    assert r.status_code == 200, f"GET / 失败: {r.status_code}"
    body = r.data
    assert b"<html" in body or b"<!DOCTYPE" in body.lower() or b"button" in body, (
        "GET / 响应不含 HTML 标志"
    )

    # 2. GET /api/status → JSON 含 state
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.get_json()
    assert "state" in data, f"status 缺 state 字段: {data}"

    # 3. POST /api/start → ok=True，命令入队
    r = client.post("/api/start")
    assert r.status_code == 200
    j = r.get_json()
    assert j.get("ok") is True, f"/api/start 返回 ok!=True: {j}"
    assert ctl._cmd_q.get_nowait() == "start"

    # 4. POST /api/save → events[exit_early]=True（等价键盘 → keep）
    r = client.post("/api/save")
    assert r.status_code == 200
    assert events["exit_early"] is True, "save 未写入 exit_early"

    # 5. POST /api/discard → events[rerecord_episode]=True（等价键盘 ← discard）
    events["exit_early"] = False  # 重置
    r = client.post("/api/discard")
    assert r.status_code == 200
    assert events["rerecord_episode"] is True, "discard 未写入 rerecord_episode"
    assert events["exit_early"] is True, "discard 未写入 exit_early"

    # 6. GET /api/preview/wrist_image → 无帧时 404 或 200 含 null data
    r = client.get("/api/preview/wrist_image")
    assert r.status_code in (200, 404), f"preview 状态码意外: {r.status_code}"
    if r.status_code == 200:
        j = r.get_json()
        assert "data" in j, f"preview 响应缺 data 字段: {j}"

    # 7. POST /api/stop → events[stop_recording]=True（等价键盘 Esc stop）
    events["exit_early"] = False  # 重置
    r = client.post("/api/stop")
    assert r.status_code == 200
    assert events["stop_recording"] is True, "stop 未写入 stop_recording"
    assert events["exit_early"] is True, "stop 未写入 exit_early"

    # 8. Cache-Control 头在所有已测路由响应中均出现
    for path, method in [
        ("/", "GET"),
        ("/api/status", "GET"),
        ("/api/start", "POST"),
        ("/api/save", "POST"),
    ]:
        if method == "GET":
            resp = client.get(path)
        else:
            resp = client.post(path)
        cc = resp.headers.get("Cache-Control", "")
        assert "no-store" in cc, (
            f"{method} {path} 响应缺 Cache-Control no-store（红线）: {cc!r}"
        )

    # 9. 后台线程 start → run_episodes_fn 被调用 → wait_until_done 不阻塞
    events2 = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    runner2 = FakeRunner()
    ctl2 = _rc.RecorderController(events=events2)
    ctl2.attach_record_args(
        robot=object(),
        teleop=object(),
        saver=object(),
        run_episodes_fn=runner2,
        fps=30.0,
        episode_sec=0.05,
        gripper_max_open=0.08,
        cam_names=["wrist_image"],
        out_dir="/tmp",
        task_name="smoke2",
        oc2base_R=np.eye(3),
        vr_source="u",
        episodes=1,
        reset_fn=None,
        reset_wait=0.0,
    )
    ctl2.start_recording()  # 入队 "start"
    ctl2.start()            # 起后台线程

    # 等 FakeRunner 跑完（episodes=1，sleep 0.05s×2=约 0.1s）
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and runner2.calls == 0:
        time.sleep(0.02)
    assert runner2.calls == 1, f"run_episodes_fn 未被调用，calls={runner2.calls}"

    # 通知后台命令循环退出（_should_stop=True），再 join
    ctl2.stop_recording()
    thread_ref = ctl2._recorder_thread  # join 前先拿引用
    ctl2.wait_until_done(timeout=3.0)
    # thread_ref.is_alive() should be False（已正常退出）
    assert thread_ref is None or not thread_ref.is_alive(), (
        "wait_until_done 后后台线程仍在运行"
    )
