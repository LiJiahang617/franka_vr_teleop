# Phase D 同步重构：关键经验总结

**Task**: PhaseD-T9（收官）  
**日期**: 2026-05-23  
**范围**: Tasks 2-9，schema v1→v2 bump + 多模态时间戳 + 5→3 线程修正 + hw_timestamp + state_hifreq 240Hz + 离线对齐 + 转换器 realman 对齐 + 迁移工具

---

## 经验 1：schema bump 必须同步全部消费方

**背景**：`franka-hdf5-v1` → `franka-hdf5-v2` 改动了 N 的来源（observations/timestamp → action/timestamp）、新增 stale/hw_timestamp/wrench，共 11 个测试文件依赖 schema。

**教训**：
- bump 前必须用 `grep -rn` 实证当前所有消费方，不能凭印象列清单（计划写于旧版本，实际消费方比预期多）。
- v2 `validate_episode` 收紧后，旧 writer 写的文件立即全部报 violation，Task 2-6 之间有预期的 fail 窗口期——提前写清楚"这些 fail 是 schema bump 预期，不是 bug"，避免后续 agent 误判。
- `observations/timestamp` 遗留字段在 v2 validator 里选择"忽略而非报错"——这样 Task 4 改 writer 前，v1 写法的文件不会因为有多余字段而报双重 violation。

---

## 经验 2：zerorpc 单线程 vs 多模态多线程——5→3 线程修正

**背景**：原计划"5 线程：arm/effector/wrist_cam/exterior_cam/state_hifreq"，后发现 zerorpc client 非线程安全——arm/effector/state_hifreq 都走同一个 zerorpc client（端口 4242），三线程并发调用会冲突。

**修正**：
- **3 个 SensorThread**：`robot_state` 线程（单线程内串行调 arm+effector+state_hifreq），`wrist_cam` 线程，`exterior_cam` 线程。
- `robot_state` 线程因 zerorpc 往返约 0.4ms，串行 3 次 = ~1.2ms，远低于 33ms 录制节拍，不构成瓶颈。
- **核心原则**：zerorpc client 绑定到单线程，绝不在多线程中并发调用同一 client。后续扩展如需多 client 并发，必须先确认是否各自有独立 socket。

---

## 经验 3：时间戳规范化——hw_ts 必须映射回 monotonic 域

**背景**：RealSense 的 `hw_timestamp`（`get_timestamp()`）在 `global_time_enabled=1` 时返回**毫秒**级绝对时间（约 1e6 ms = 1000s 量级），而 arm/effector 的 `time.monotonic()` 戳是**秒**级相对时间（约 1e3 s 量级）。

**Bug**：Task 8 初始实现在 `_select_anchor_ts` 中直接返回毫秒原值用作插值锚，与 arm_ts/eff_ts（秒域）量级不符，`np.interp` 全端点外推，对齐完全错位。

**修复**：`hw_ts` 通过线性回归（`slope`, `intercept`，R²>0.9999 达标）逆变换回 monotonic 秒域：
```python
sw_c_hat = (hw_c - intercept) / slope  # 将毫秒硬件戳映射回秒域软件时间
```
映射后 hw_ts 保留低抖动特性，同时与 arm/eff/act 同基准，插值正确。

**结论**：跨域时间戳（GPS/hardware timer/monotonic）做对齐时，**必须先统一到同一量纲和基准**，再做插值；量纲不符是隐性 bug，不报错但结果全错。

---

## 经验 4：state_hifreq 240Hz——zerorpc 锁与 overrun 计数

**背景**：state_hifreq 线程按 240Hz（4.17ms 间隔）轮询 zerorpc，与录制主线程的 robot_state 线程共用同一 zerorpc client。

**解决**：引入 `zerorpc_lock`（`threading.Lock()`），robot_state 线程和 state_hifreq 线程均在锁保护下调用 zerorpc，保证串行。zerorpc 单次往返 ~0.4ms，state_hifreq 线程以 240Hz 调 3 个 RPC = ~1.2ms 占用，与录制线程交织后实测零 overrun。

**工程规范**：凡新增 zerorpc 线程，必须先确认其 client 是否与已有线程共用——若是，加 Lock；若否（独立端口/独立 client），才可独立线程无锁并发。

---

## 经验 5：视频编码 vlen uint8 的 buffer copy 陷阱

**Bug**：`realsense_hw_wrapper.read()` 用 `np.asanyarray(color_frame)` 返回的是视图，pipeline 读下一帧时底层 buffer 被复用，导致已写入帧队列的 ndarray 内容被脏写（后一帧的图像覆盖前一帧）。

**修复**：改 `np.array(np.asanyarray(color_frame))` 强制 copy，使返回 ndarray 独立。

**结论**：任何从 C 扩展库（RealSense SDK / OpenCV）拿到的 ndarray，如需持久化（入队列/写 HDF5），必须确认是 copy 而非 view。`arr.flags.owndata` 为 False 时必须主动 `arr.copy()`。

---

## 经验 6：合成 TDD 验证离线对齐

**背景**：`align_offline.py` 的 SLERP/线性插值逻辑不能用真机数据验证，必须合成精确数据集做单测。

**陷阱**：
- SLERP 输入时间戳有重复值时 `scipy.spatial.transform.Slerp` 抛 ValueError（"t values must be in strictly increasing order"）——解决：插值前对 anchor_ts 去重并重排锚数组。
- `interpolate` 模式下 `drop` 实现：先按图像帧 anchor_indices 切片，anchor_indices 必须精确对应图像帧，否则帧序错位（T6 修复 bug）。
- 欧拉角 `unwrap` 保证 SLERP 前后旋转不跨 2π 跳点，但欧拉角→旋转矩阵→Slerp→欧拉角的往返在奇异点附近仍有差异，测试误差阈值应 `< 1e-6 m`（位置）而非 `< 1e-9`。

---

## 经验 7：转换器输出对齐 realman——next-state action 语义

**背景**：Task 6 要求 Franka 转换器输出与 realman 数据集完全对齐——`action[i] = state[i+1]`（next-state 语义），14D 布局（7 joint + 1 gripper + 6 EEF pose）。

**关键决策**：
- hdf5 层保留 `action/delta_ee_pose` 原始增量（录制语义不变）。
- 转换层**不消费**原始增量，重新从 state 序列构造 action（`action[i] = state[i+1]`，末帧复制）。
- 与 realman 数据集验证：`mean|action[i] - state[i+1]| == 0.0`，逐帧逐维精确匹配。

**结论**：录制格式（hdf5 忠实记录采集信息）与训练格式（lerobot 对标 realman 语义）之间需要明确的语义转换层——不能期望录制格式直接等于训练期望。

---

## 经验 8：迁移工具的 v1 schema 反向工程

**背景**：v1 时代已有 arm/effector/camera 各自的 `timestamp(N,)` 副本（都是从共用戳复制），v2 保留了这个结构并增加 stale/hw_timestamp/wrench。

**迁移原则**（v1→v2）：
1. `observations/timestamp(N,1)` 共用戳 → 不复制到 v2（v2 无此路径）
2. 各模态 timestamp 从 v1 的副本直接用；若缺失则从共用戳回退
3. stale 全 False（v1 无缺帧概念）
4. `hw_timestamp = timestamp`（v1 无真硬件戳，用软件戳占位）
5. `wrench = zeros(M,6)`（Phase F 实填，M=0 时 shape=(0,6)）

**复用 validate_episode 做自检**：迁移函数末尾直接调 `validate_episode`，失败即 raise RuntimeError——这把 v2 格式契约变成迁移工具的隐式测试，保证输出合规。

---

## 关键数字

| 阶段 | passed | failed | error |
|---|---|---|---|
| Phase C 基线（Task 2 前） | 397 | 0 | 0 |
| Task 2 后（schema bump，预期 fail 窗口） | 370 | 8 | 50 |
| Task 4 收口（writer 适配 v2） | 增加（T4 新增） | 0 | 0 |
| Task 6 收口（全消费方 schema bump 收口） | 增加（T6 新增） | 0 | 0 |
| Phase D 收官（Task 9，全回归） | **602** | **0** | **0** |

---

*见 `docs/lessons/2026-05-22-schema-v1-to-v2-bump.md`（v2 bump 消费方清单）；`docs/phaseD-v2-acceptance.md`（真机验收 7 条，DEFERRED）。*
