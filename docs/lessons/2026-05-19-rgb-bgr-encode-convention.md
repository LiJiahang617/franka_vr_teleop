# RGB/BGR 编码惯例不一致致录制图像反色（黄变青）

日期：2026-05-19　发现：用户目检数据（香蕉黄变青蓝）

## 症状
hdf5/lerobot 录制数据图像 R↔B 互换：黄(255,210,30)→青(30,210,255)。
fps/joint_vel/schema/几何均正常，单纯颜色通道反。

## 根因
- 相机 `color_mode=ColorMode.RGB` → `cam.read()` 出 **RGB** ndarray。
- `run_record_hdf5.py:_encode_jpg` 旧版 `cv2.imencode(".jpg", img)` 直接编码——
  但 cv2.imencode 按 OpenCV 惯例**默认输入 BGR**，把 RGB 当 BGR 写进 jpeg。
- 下游 `hdf5_lerobot_map._decode`：`cv2.imdecode(...)`(cv2 当 BGR) +
  `cvtColor(BGR2RGB)`——假设 jpeg 是 BGR，于是**净多一次 R↔B 互换**。
- 单次互换 ⇒ 黄↔青（绿不变）。编码端与解码端对 jpeg 通道序的假设不一致。

## 修复
`_encode_jpg` 在 `cv2.imencode` 前 `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)`，
使存盘 jpeg 为规范 BGR jpeg；下游 imdecode(BGR)+BGR2RGB 即正确。
commit 56930d8 + TDD `tests/test_image_color_convention.py`（端到端色彩不变量用
真实 `_decode` + 源码守卫）。

## 教训（与 2026-02-13-camera-double-rotation 同族）
- **"数据看着有内容" ≠ 正确**。schema/fps/非全黑校验抓不到通道序错；离线
  inspect 脚本只验 mean/shape 也漏。需**直接验证色彩语义**（已知色物体/通道判据）。
- 跨边界传图像必须显式锁定并校验通道序（谁出 RGB、谁要 BGR），勿靠惯例默认。
- 用户目检抓到 = 自动校验缺这一环 → 并入数据预检门（spec §11.2）。

## 影响
此前**所有 hdf5 录制（含 Phase A 验收 ep0000_1779196921.h5）图像 R/B 反**；
几何/fps/joint_vel/夹爪不受影响。既有数据可一次性 R↔B 重编码挽救或重录。
