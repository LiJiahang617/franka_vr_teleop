"""
Task 4：HTML/JS 模板断言测试（TDD 先写失败，后写实现）。

验证：
- GET / 返回 200 且包含必要按钮 ID
- 双相机 img 槽位存在
- 约 30Hz 的 setInterval 轮询
- fetch 含 cache:'no-store' 双保险
- 响应 Cache-Control 头完整
- 不含被砍掉的按钮（外骨骼/高跟随等）
- 修订 A：页面含玻璃态 CSS 标志（backdrop-filter）
- 修订 B：含 btn-payload 负载标定按钮
"""
import importlib.util
import os
import re

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
    """构造带真实 RecorderController 的 test_client。"""
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    ctl = rc.RecorderController(events=events)
    return cp.build_app(controller=ctl).test_client()


def test_root_returns_200():
    """GET / 返回 200。"""
    r = _client().get("/")
    assert r.status_code == 200


def test_root_content_type_html():
    """GET / 返回 Content-Type: text/html。"""
    r = _client().get("/")
    assert "text/html" in r.content_type


def test_root_renders_html_with_required_buttons():
    """GET / 含 5 个必要控制按钮 ID。"""
    html = _client().get("/").data.decode("utf-8")
    for btn in ("btn-start", "btn-save", "btn-discard", "btn-home", "btn-stop"):
        assert f'id="{btn}"' in html, f"缺少按钮 {btn}"


def test_root_template_has_two_camera_img_slots():
    """GET / 含双相机 img 槽位。"""
    html = _client().get("/").data.decode("utf-8")
    assert 'id="cam-wrist_image"' in html, "缺少 wrist 相机槽位"
    assert 'id="cam-exterior_image"' in html, "缺少 exterior 相机槽位"


def test_root_has_status_polling_script_around_30hz():
    """setInterval 间隔在 25~50ms（约 30Hz）。"""
    html = _client().get("/").data.decode("utf-8")
    # 查找所有 setInterval 调用的间隔参数
    matches = re.findall(r"setInterval\s*\([^,]+,\s*(\d+)\s*\)", html)
    assert matches, "未找到 setInterval"
    # 至少有一个在 25~50ms 范围内
    intervals = [int(m) for m in matches]
    assert any(25 <= v <= 50 for v in intervals), f"setInterval 间隔不在 25~50ms: {intervals}"


def test_root_uses_no_store_fetch_double_safety():
    """fetch 调用含 cache:'no-store' 或 cache:\"no-store\"（双保险红线）。"""
    html = _client().get("/").data.decode("utf-8")
    assert "cache: 'no-store'" in html or 'cache: "no-store"' in html, (
        "未找到 fetch cache:no-store 双保险"
    )


def test_root_response_has_no_cache_headers():
    """GET / 响应含完整 Cache-Control no-cache 头（after_request 红线）。"""
    r = _client().get("/")
    cc = r.headers.get("Cache-Control", "")
    assert "no-cache" in cc, "缺少 no-cache"
    assert "no-store" in cc, "缺少 no-store"
    assert "must-revalidate" in cc, "缺少 must-revalidate"


def test_root_does_not_contain_deprecated_buttons():
    """spec §3.4：不含外骨骼/高跟随等已砍按钮。"""
    html = _client().get("/").data.decode("utf-8").lower()
    for banned in ("外骨骼", "exoskeleton", "高跟随", "high_follow"):
        assert banned not in html, f"发现应砍掉的按钮关键词: {banned}"


# --- 修订 A：玻璃态深色视觉风格 ---

def test_root_has_glassmorphism_backdrop_filter():
    """修订 A：CSS 含 backdrop-filter（玻璃态卡片标志）。"""
    html = _client().get("/").data.decode("utf-8")
    assert "backdrop-filter" in html, "缺少 backdrop-filter（玻璃态 CSS 标志）"


def test_root_has_dark_gradient_background():
    """修订 A：CSS 含页面背景渐变色。"""
    html = _client().get("/").data.decode("utf-8")
    # 背景渐变含 135deg 或 0f1d34（深色背景色）
    assert "135deg" in html or "#0f1d34" in html or "0f1d34" in html, (
        "缺少深色背景渐变"
    )


# --- 修订 B：负载标定按钮 ---

def test_root_has_payload_calib_button():
    """修订 B：含 btn-payload 负载标定按钮。"""
    html = _client().get("/").data.decode("utf-8")
    assert 'id="btn-payload"' in html, "缺少 btn-payload 负载标定按钮"


def test_root_payload_button_calls_payload_calib_api():
    """修订 B：btn-payload 关联 /api/payload-calib 调用。"""
    html = _client().get("/").data.decode("utf-8")
    assert "/api/payload-calib" in html, "btn-payload 未关联 /api/payload-calib"
