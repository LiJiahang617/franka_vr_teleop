"""
Flask app 骨架烟测。
用 app.test_client() 离线测，不起服务器不占端口。
验证：build_app 工厂函数、after_request cache 头红线、/api/ping 路由。
"""
import importlib.util, os

# 动态加载 control_panel 模块
_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_s = importlib.util.spec_from_file_location(
    "control_panel", os.path.join(_P, "scripts/ui/control_panel.py"))
cp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cp)


def test_build_app_returns_flask_instance():
    """build_app 必须返回 Flask 实例（有 test_client 属性）。"""
    app = cp.build_app(controller=None)
    assert app is not None
    assert hasattr(app, "test_client")


def test_after_request_adds_no_cache_headers_red_line():
    """红线：所有响应必须含 no-cache/no-store/must-revalidate + Pragma + Expires。"""
    app = cp.build_app(controller=None)
    c = app.test_client()
    r = c.get("/api/ping")
    cc = r.headers.get("Cache-Control", "")
    # spec §3.4 红线 + lesson 2026-05-04-flask-no-cache: 三段全到位
    assert "no-cache" in cc and "no-store" in cc and "must-revalidate" in cc
    assert r.headers.get("Pragma") == "no-cache"
    assert r.headers.get("Expires") == "0"


def test_ping_route_smoke():
    """/api/ping 烟测：200 OK，返回 {"ok": True}。"""
    app = cp.build_app(controller=None)
    r = app.test_client().get("/api/ping")
    assert r.status_code == 200
    assert r.get_json() == {"ok": True}
