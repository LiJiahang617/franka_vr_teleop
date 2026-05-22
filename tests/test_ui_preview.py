"""
Task 3：相机预览编码 + /api/preview/<cam> 路由测试。

测试逻辑：
- 验证 encode_preview_jpeg 缩放到 ≤320×240、返回合法 jpeg bytes
- 验证 encode_preview_base64 返回 data-url 格式
- 验证 RGB→BGR 通道序（复用 _encode_jpg 约定，lesson 2026-05-19-rgb-bgr）
- 验证 /api/preview/<cam> 路由：有帧返回 200+JSON data_url，无帧返回 404
- 验证响应 Cache-Control 头（after_request 覆盖，红线）
"""
import importlib.util
import os
import base64

import cv2
import numpy as np

# ---------- 动态加载模块（与其他 UI 测试保持一致的加载方式） ----------

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_P, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pv = _load("preview", "scripts/ui/preview.py")
cp = _load("control_panel", "scripts/ui/control_panel.py")
rc = _load("recorder_controller", "scripts/ui/recorder_controller.py")


# ---------- 辅助函数 ----------

def _make_rgb(h=480, w=640):
    """生成纯黄色 RGB 测试图像（R=240, G=210, B=30）。"""
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (240, 210, 30)  # 黄色：R 远大于 B
    return rgb


def _client_with_frame(cam, rgb):
    """创建已预置帧的 Flask test client。"""
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    ctl = rc.RecorderController(events=events)
    ctl.update_latest_frame(cam, rgb)
    return cp.build_app(controller=ctl).test_client()


# ---------- 测试：encode_preview_jpeg ----------

def test_encode_preview_jpeg_respects_max_size_and_returns_bytes():
    """大图缩放后 jpeg bytes 合法且尺寸 ≤ 320×240。"""
    rgb = _make_rgb(720, 1280)
    b = pv.encode_preview_jpeg(rgb, max_w=320, max_h=240, quality=60)
    assert isinstance(b, (bytes, bytearray)) and len(b) > 0
    # 反解校验 jpeg 合规 + 尺寸
    arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None
    assert arr.shape[0] <= 240 and arr.shape[1] <= 320


def test_encode_preview_jpeg_small_image_not_upscaled():
    """小图（100×80）不放大，原样编码后 ≤ 320×240。"""
    rgb = _make_rgb(80, 100)
    b = pv.encode_preview_jpeg(rgb, max_w=320, max_h=240, quality=60)
    arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None
    assert arr.shape[0] <= 80 and arr.shape[1] <= 100


# ---------- 测试：encode_preview_base64 ----------

def test_encode_preview_base64_data_url_format():
    """data-url 前缀正确，base64 payload 可解码为合法 jpeg。"""
    rgb = _make_rgb()
    s = pv.encode_preview_base64(rgb)
    assert s.startswith("data:image/jpeg;base64,")
    payload = s.split(",", 1)[1]
    raw = base64.b64decode(payload)
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None


def test_encode_preserves_yellow_rb_channel_order_reuses_encode_jpg_convention():
    """复用 _encode_jpg 同款 RGB→BGR→imencode 通道序（lesson 2026-05-19-rgb-bgr）。

    黄色图像 R=240 >> B=30：
    - 正确顺序（RGB→BGR）：imencode 存 B<G<R，imdecode(IMREAD_COLOR)→BGR，
      mean_r（BGR 第 2 通道）>> mean_b（BGR 第 0 通道）
    - 反色（误用 BGR→RGB）：R↔B 对调，mean_b >> mean_r → 测试失败
    """
    rgb = _make_rgb()
    b = pv.encode_preview_jpeg(rgb)
    bgr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    mean_b, mean_g, mean_r = bgr.reshape(-1, 3).mean(axis=0)
    assert mean_r > mean_b + 50, (
        f"通道序异常（可能反色）：mean_r={mean_r:.1f}, mean_b={mean_b:.1f}"
    )


# ---------- 测试：/api/preview/<cam> 路由 ----------

def test_api_preview_returns_base64_jpeg_for_known_cam():
    """有帧时返回 200，JSON 含 cam 和合法 data_url，jpeg 尺寸 ≤ 320。"""
    c = _client_with_frame("wrist_image", _make_rgb())
    r = c.get("/api/preview/wrist_image")
    assert r.status_code == 200
    j = r.get_json()
    assert j["cam"] == "wrist_image"
    assert j["data_url"].startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(j["data_url"].split(",", 1)[1])
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None and arr.shape[1] <= 320


def test_api_preview_404_when_no_frame_yet():
    """无帧时返回 404（cam 名已知但尚未有帧写入）。"""
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    ctl = rc.RecorderController(events=events)
    c = cp.build_app(controller=ctl).test_client()
    r = c.get("/api/preview/wrist_image")
    assert r.status_code == 404


def test_api_preview_503_when_controller_none():
    """controller=None 时返回 503（复用 Task 2 模式）。"""
    c = cp.build_app(controller=None).test_client()
    r = c.get("/api/preview/wrist_image")
    assert r.status_code == 503


def test_api_preview_response_has_no_cache_headers():
    """预览响应必须含 Cache-Control: no-cache, no-store（after_request 红线）。"""
    c = _client_with_frame("wrist_image", _make_rgb())
    r = c.get("/api/preview/wrist_image")
    cc = r.headers.get("Cache-Control", "")
    assert "no-cache" in cc and "no-store" in cc
