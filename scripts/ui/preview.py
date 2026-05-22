"""
相机帧预览编码模块。

复用 _encode_jpg 同款通道序：cvtColor(RGB→BGR) → imencode(.jpg)
（lesson 2026-05-19-rgb-bgr：浏览器 img 标签解 jpeg 期望 BGR 存储，
若用反序编码则 R↔B 互换，浏览器显示反色）
"""
import base64

import cv2
import numpy as np


def encode_preview_jpeg(
    rgb: np.ndarray,
    max_w: int = 320,
    max_h: int = 240,
    quality: int = 60,
) -> bytes:
    """将 RGB 帧等比缩小（≤max_w×max_h）并编码为 jpeg bytes。

    仅缩小不放大：若原图已小于上限则不做 resize。
    通道序复用 _encode_jpg：cvtColor(RGB→BGR) 后 imencode
    （lesson 2026-05-19-rgb-bgr）。

    Args:
        rgb: HxWx3 uint8 RGB ndarray。
        max_w: 输出宽度上限（像素），默认 320。
        max_h: 输出高度上限（像素），默认 240。
        quality: jpeg 压缩质量 1-100，默认 60。

    Returns:
        jpeg 编码的 bytes。

    Raises:
        ValueError: rgb 形状不合法时。
        RuntimeError: imencode 失败时。
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb 必须是 HxWx3 ndarray，实际 shape={rgb.shape}")

    h, w = rgb.shape[:2]
    # 等比缩放：仅缩小不放大（scale ≤ 1.0）
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        rgb_small = rgb

    # 复用 _encode_jpg 同款通道序：RGB→BGR 后 imencode
    bgr = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode 失败")
    return buf.tobytes()


def encode_preview_base64(rgb: np.ndarray, **kw) -> str:
    """将 RGB 帧编码为 data-url 格式的 base64 jpeg 字符串。

    Args:
        rgb: HxWx3 uint8 RGB ndarray。
        **kw: 透传给 encode_preview_jpeg（max_w, max_h, quality）。

    Returns:
        形如 "data:image/jpeg;base64,<payload>" 的字符串。
    """
    b = encode_preview_jpeg(rgb, **kw)
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")
