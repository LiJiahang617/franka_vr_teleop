"""§11.2 图像色彩通道序预检 TDD。

测试 image_color_verdict / run_color_preflight 纯函数，
复用 hdf5_lerobot_map._decode 约定（与录制管道同款 round-trip）。
"""
import importlib.util
import os

import cv2
import numpy as np

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_s = importlib.util.spec_from_file_location(
    "preflight", os.path.join(_P, "scripts/core/preflight.py"))
pf = importlib.util.module_from_spec(_s)
_s.loader.exec_module(pf)

_h = importlib.util.spec_from_file_location(
    "hdf5_lerobot_map", os.path.join(_P, "scripts/tools/hdf5_lerobot_map.py"))
hlm = importlib.util.module_from_spec(_h)
_h.loader.exec_module(hlm)


def _enc(rgb: np.ndarray) -> bytes:
    """与录制管道同款编码（RGB→BGR→JPEG bytes）。"""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return buf.tobytes()


def test_color_verdict_yellow_ok():
    """正常暖色帧（黄色，R≫B）应通过色彩预检（不误报）。"""
    rgb = np.zeros((32, 32, 3), np.uint8)
    rgb[:] = (240, 210, 30)   # 黄色：R=240,G=210,B=30，R≫B
    v = pf.image_color_verdict([hlm._decode(_enc(rgb))])
    assert v.ok is True


def test_color_verdict_detects_rb_swapped():
    """模拟 RGB/BGR 反转：黄色帧被错存为青蓝（R低B高），应被检出。"""
    rgb = np.zeros((32, 32, 3), np.uint8)
    rgb[:] = (240, 210, 30)   # 黄色
    # 人为反通道，模拟整体 RGB/BGR 反（R低B高，无暖色）
    swapped = hlm._decode(_enc(rgb))[..., ::-1].copy()
    v = pf.image_color_verdict([swapped])
    assert v.ok is False
    assert "色彩" in v.reason or "RGB" in v.reason


def test_run_color_preflight_uses_decode_convention():
    """run_color_preflight 用 _decode 约定解码，正常帧应通过。"""
    rgb = np.zeros((16, 16, 3), np.uint8)
    rgb[:] = (240, 210, 30)   # 黄色
    res = pf.run_color_preflight(
        decode_fn=hlm._decode,
        encoded_frames=[_enc(rgb)],
    )
    assert res.ok is True
