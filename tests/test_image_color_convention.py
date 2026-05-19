"""回归: 相机 RGB 图像经录制编码 + hdf5_lerobot_map._decode 解码后色彩须保真。

根因(2026-05-19): _encode_jpg 旧实现 cv2.imencode 直接吃相机 RGB(ColorMode.RGB),
但 cv2.imencode 按 OpenCV 惯例当 BGR; 下游 _decode 又 imdecode(BGR)+BGR2RGB
→ 净多一次 R↔B 互换, 黄香蕉变青蓝。修复: _encode_jpg 编码前 RGB→BGR。
"""
import importlib.util
import os

import cv2
import numpy as np

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_s = importlib.util.spec_from_file_location(
    "hdf5_lerobot_map", os.path.join(_P, "scripts/tools/hdf5_lerobot_map.py"))
hlm = importlib.util.module_from_spec(_s)
_s.loader.exec_module(hlm)


def _encode_like_recorder(rgb):
    # 复刻修复后 _encode_jpg 的编码惯例(相机 RGB → RGB2BGR → imencode)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return buf.tobytes()


def test_color_roundtrip_yellow_stays_yellow():
    # 纯黄 RGB patch (R 高, G 中, B 低); bug 会变青蓝(R 低, B 高)
    rgb = np.zeros((32, 32, 3), np.uint8)
    rgb[:] = (240, 210, 30)  # R,G,B
    out = hlm._decode(_encode_like_recorder(rgb))  # 真实下游解码端
    r, g, b = out[..., 0].mean(), out[..., 1].mean(), out[..., 2].mean()
    # jpeg 有损, 用宽松判据但方向必须对: 黄 = R≫B
    assert r > 150 and b < 90 and r - b > 100, f"色彩反了? R={r:.0f} G={g:.0f} B={b:.0f}"


def test_encode_jpg_source_converts_rgb2bgr_before_imencode():
    # 源码守卫: 防有人删掉 RGB->BGR 又把通道弄反
    src = open(os.path.join(_P, "scripts/core/run_record_hdf5.py")).read()
    i_def = src.find("def _encode_jpg")
    seg = src[i_def:i_def + 600]
    i_cvt = seg.find("COLOR_RGB2BGR")
    i_enc = seg.find('cv2.imencode(".jpg"')
    assert i_cvt != -1, "_encode_jpg 缺 RGB→BGR 转换"
    assert i_enc != -1, "_encode_jpg 未找到 imencode"
    assert i_cvt < i_enc, "RGB→BGR 必须在 imencode 之前"
