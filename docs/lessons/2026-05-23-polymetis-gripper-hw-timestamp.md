# polymetis 丢弃 libfranka 硬件时戳 → 用 cmake 宏捡回精确对齐

**日期**：2026-05-23
**前置背景**：参 [2026-05-23-polymetis-gripper-width-feedback-lag.md](2026-05-23-polymetis-gripper-width-feedback-lag.md)（实测发现 effector width 反馈滞后 ~100ms）

## 问题

Phase D 真机录制后用 rerun 看夹爪状态，图像中夹爪开始张开后 130-150ms 才在 `observation.state.gripper_norm` 曲线上反映。`align_offline` 用 polymetis 接收时刻作 effector_ts，无法消除 push 周期内的 0-200ms 相位抖动。

## 调研

libfranka 10Hz gripper push 是 firmware 限制（franka_ros / franka_ros2 / polymetis / frankapy 同样受限）。UMI / Delay-Aware Diffusion Policy 等学术界方案要么换视觉测 width，要么模型层吸收延迟，对我们单 VR 场景成本高。

**发现可利用资源**：`franka::GripperState.time` 字段（硬件 push 物理时戳）被 polymetis `franka_hand_client.cpp:50` 用 `setTimestampToNow` 覆盖丢弃。捡回来能消除 polymetis 接收时刻的相位误差。

## 方案

**polymetis 端**：
- `franka_hand_client.cpp`：cmake 宏 `ENABLE_GRIPPER_HW_TIMESTAMP`（默认 ON）控制是否用 `franka_gripper_state.time` 替代 `setTimestampToNow`
- `franka_interface_server.py`：`gripper_get_state` 返回 dict 加 `timestamp` 字段

**本项目端**：
- 源数据 schema v2 内增**可选**字段 `observations/effector/hw_timestamp`（不 bump 版本号）
- `hdf5_writer.write_episode` 写该字段（None→NaN，全 None→不写）
- `record_episode` 主循环透传 `effector_hw_ts`
- `align_offline._project_hw_to_monotonic` 用线性回归把硬件戳轴映射到 monotonic 域
- 转换器对缺字段 ep 输出 warning 提示 rebuild polymetis

## Commit 链

### polymetis fork (`Shenzhaolong1330/fairo-franka`)
- `ad44bbce` cpp + CMakeLists（cmake option + std::round + nanos 溢出保护）
- `5c74a40d` server.py 转发 timestamp 字段

### lerobot_franka_teleop
- `370a532` schema v2 加可选字段 + validate
- `ec869ac` hdf5_writer 写可选 hw_timestamp
- `c135bf3` record_episode 透传 effector_hw_ts
- `e270377` align_offline `_project_hw_to_monotonic` + 三分支
- `6a300a8` 转换器 warning 提示
- `9c0d5ae` spike 脚本

## 真机验证

spike 实测（300 次 poll @ 5ms）：
- R² = 0.999833（> 0.99 判据）
- slope = 49.50（hw_ts 时钟与 wall-clock 不同频率，但强线性 → polyfit 处理）
- 残差 stddev ≈ **22.68 ms (wall-clock 域)**

这意味着 `_project_hw_to_monotonic` 投影后 effector 时间戳精度从 polymetis 接收时刻的 ±100ms 降到 ~22ms 级抖动。push 周期 200ms 仍是 firmware 极限，不可破。

## 教训

1. **上游库丢弃硬件元数据时要警觉**——polymetis 用 `setTimestampToNow` 覆盖了 libfranka 已有的硬件 push 时戳。这是个未利用的现成资源。
2. **schema 加可选字段是平滑演进的好范式**——OPTIONAL_FIELDS_V2 不 bump 版本，旧数据集仍 PASS，新数据带额外字段。validate 容忍缺失。
3. **跨时间轴同步用线性回归 + R²**——hw_ts 不必与 wall-clock 同频（slope ≠ 1 是预期），polyfit 处理任意 slope；判据看 R² 而非 slope ≈ 1。这与 Phase D Task 8 cam hw_ts 是同模式。
4. **spike PASS 判据要 match 实际下游使用方式**——T8 初次 spike 误判 slope=50 为 fail，实际上 align_offline 用 polyfit 不在乎 slope。判据应该按下游约束写，不是按理想化的 slope=1 写。
5. **cmake option 是可逆性的保险**——`-DENABLE_GRIPPER_HW_TIMESTAMP=OFF` 一键退回，调试时无需 git revert。

## 后续工作

- bump fork 推到 lijiahang0617/fairo-franka（用户后续 GitHub 操作）
- README 加 rebuild polymetis 指引
- 用户真机录制新 ep 验证 rerun 显示精度提升从 130ms 降到 ~22ms 范围
