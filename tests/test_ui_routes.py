"""
Flask 路由集成测试（无真服务器，使用 app.test_client() 离线测试）。

验证：
- 6 条路由正确 wire 到 RecorderController 方法
- events dict 写入语义与终端键盘逐字等价
- 命令队列走法（start/home）
- 所有响应含 Cache-Control: no-cache, no-store, must-revalidate
- HTTP 动词限制（POST-only / GET-only）
- POST /api/payload-calib 占位路由（修订 B）
"""
import importlib.util
import os

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_cp = importlib.util.spec_from_file_location(
    "control_panel", os.path.join(_P, "scripts/ui/control_panel.py")
)
cp = importlib.util.module_from_spec(_cp)
_cp.loader.exec_module(cp)

_rc = importlib.util.spec_from_file_location(
    "recorder_controller", os.path.join(_P, "scripts/ui/recorder_controller.py")
)
rc = importlib.util.module_from_spec(_rc)
_rc.loader.exec_module(rc)


def _client():
    """构造 test_client + 真实 RecorderController（events dict 可直接断言）。"""
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    ctl = rc.RecorderController(events=events)
    app = cp.build_app(controller=ctl)
    return app.test_client(), ctl, events


def test_api_start_enqueues_start_cmd():
    """POST /api/start 返回 200 + ok=True，命令队列收到 'start'。"""
    c, ctl, _ = _client()
    r = c.post("/api/start")
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert ctl._cmd_q.get_nowait() == "start"


def test_api_save_sets_exit_early_true():
    """POST /api/save 返回 200，events['exit_early']=True（等价键盘 →）。"""
    c, _, ev = _client()
    r = c.post("/api/save")
    assert r.status_code == 200
    assert ev["exit_early"] is True


def test_api_discard_sets_rerecord_and_exit_early():
    """POST /api/discard 返回 200，events rerecord+exit_early 均为 True（等价键盘 ←）。"""
    c, _, ev = _client()
    r = c.post("/api/discard")
    assert r.status_code == 200
    assert ev["rerecord_episode"] is True and ev["exit_early"] is True


def test_api_stop_sets_stop_recording():
    """POST /api/stop 返回 200，events['stop_recording']=True（等价键盘 Esc）。"""
    c, _, ev = _client()
    r = c.post("/api/stop")
    assert r.status_code == 200
    assert ev["stop_recording"] is True


def test_api_home_enqueues_home_cmd_not_direct():
    """POST /api/home 返回 200，命令队列收到 'home'（不直接调机器人，守坑 7）。"""
    c, ctl, _ = _client()
    r = c.post("/api/home")
    assert r.status_code == 200
    assert ctl._cmd_q.get_nowait() == "home"


def test_api_status_returns_json_snapshot():
    """GET /api/status 返回 200 且含必要字段的 JSON。"""
    c, _, _ = _client()
    r = c.get("/api/status")
    assert r.status_code == 200
    j = r.get_json()
    for k in ("state", "episode_count", "fps", "log_tail"):
        assert k in j


def test_all_api_responses_have_no_cache_headers():
    """所有 API 端点响应均含 Cache-Control: no-cache, no-store, must-revalidate（红线）。"""
    c, _, _ = _client()
    for url, method in [
        ("/api/start", "post"),
        ("/api/save", "post"),
        ("/api/discard", "post"),
        ("/api/stop", "post"),
        ("/api/home", "post"),
        ("/api/status", "get"),
        ("/api/payload-calib", "post"),
    ]:
        r = getattr(c, method)(url)
        cc = r.headers.get("Cache-Control", "")
        assert "no-cache" in cc and "no-store" in cc and "must-revalidate" in cc, url


def test_api_only_accepts_correct_verb():
    """路由动词限制：POST-only 路由拒绝 GET，GET-only 路由拒绝 POST（返回 405）。"""
    c, _, _ = _client()
    # /api/start 仅 POST
    assert c.get("/api/start").status_code == 405
    # /api/status 仅 GET
    assert c.post("/api/status").status_code == 405


# --- 修订 B：payload-calib 占位路由 ---

def test_api_payload_calib_returns_200():
    """POST /api/payload-calib 返回 200（占位路由可达）。"""
    c, _, _ = _client()
    r = c.post("/api/payload-calib")
    assert r.status_code == 200


def test_api_payload_calib_supported_false():
    """POST /api/payload-calib JSON 含 supported=False（明确标识为扩展位）。"""
    c, _, _ = _client()
    r = c.post("/api/payload-calib")
    j = r.get_json()
    assert j["supported"] is False


def test_api_payload_calib_guidance_nonempty():
    """POST /api/payload-calib JSON 含非空字符串 guidance（用户引导文案）。"""
    c, _, _ = _client()
    r = c.post("/api/payload-calib")
    j = r.get_json()
    assert isinstance(j.get("guidance"), str) and len(j["guidance"]) > 0


def test_api_payload_calib_no_side_effects():
    """POST /api/payload-calib 不写 events dict，不写命令队列（无副作用）。"""
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    ctl = rc.RecorderController(events=events)
    app = cp.build_app(controller=ctl)
    client = app.test_client()
    client.post("/api/payload-calib")
    # events dict 全部保持 False
    assert events == {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    # 命令队列为空
    assert ctl._cmd_q.empty()
