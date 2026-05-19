# 夹爪预检门正解判据 vs §10.5B 假阴性

> 录制: 2026-05-20  
> 相关: Phase B T6/T7, spec §11.2, §10.5(B)

## 背景

Franka 夹爪录制前须做"width 真变"预检，防止丢 homing/僵死进入录制。
最初直觉判据（以下简称**错误判据**）：goto 后固定 0.5s，读 width，比前后相邻差 > 0.01 → 真变。

## 错误判据为何假阴性

丢 homing 场景下，夹爪 goto 不响应但 is_moving 会 pulse 短暂 True，固定 0.5s
可能命中"已 False 但 width 没变"的窗口，相邻样本差≈0 → 误判"正常"。
结果：开录后才发现夹爪全程不动，录了无效数据。

## §10.5(B) 正解判据

1. **子进程强存活**：`pgrep -f franka_hand_client`（**非** 端口 `:50052 LISTEN`）  
   - 端口 LISTEN 不足：外壳进程可崩而端口暂留，zerorpc 看似通但 franka_hand_client 子进程已死。
2. **连接就绪**：`_gripper_live.log` 出现 `Connected.`（LISTEN 后还需 ~6-7s）  
   - 不得在"LISTEN 到 Connected."窗口内提前验。
3. **width 真变**：多目标 goto → `settle`（轮询 `is_moving→False` 或超时 8s，**非固定 0.5s**）
   → 记 settled width → `span = max-min > 0.02`（**非相邻样本差 > 0.01**）。  
   - span 整体跨度：[0.0001, 0.07, 0.04] → span=0.0699 > 0.02 → 正常。  
     [0.04, 0.04, 0.04] → span=0 → 丢 homing 被拦。
4. **陈旧日志风险**：`_gripper_live.log` 必须在重启脚本中截断（`: > _gripper_live.log`），
   否则上轮残留 `Connected.` 致假阳性（log_probe 永远返 True）。

## 色彩弱判据的已知局限

`image_color_verdict` 用统计法（B-R 均值差 > 60 且无暖色像素比 < 5%）做 RGB/BGR 粗筛，
**宁漏勿误报正常画面**（验收硬要求）。局限：纯蓝场景可能漏报；内容真的偏蓝时可能假阳性。
如需更强判据：引入"已知色卡帧"或调阈值（留 `color_preflight: false` cfg 开关关闭）。

## 代码位置

- `scripts/core/preflight.py`：`gripper_goto_span_ok`, `run_gripper_preflight`, `image_color_verdict`, `run_color_preflight`
- `tests/test_preflight_gripper.py`：span/proc/connected/error/pass/fail 8 tests
- `tests/test_preflight_color.py`：色彩 round-trip 3 tests
- 关联: `docs/lessons/2026-05-19-rgb-bgr-encode-convention.md`（编码约定）
