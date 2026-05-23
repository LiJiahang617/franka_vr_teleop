# Phase E UI 与 zerorpc gevent thread-affinity 架构冲突

**日期**：2026-05-24
**状态**：UI 模式不可用，回退键盘模式（yaml `record.ui.enabled: false`）

## 现象

跑 `scripts/core/run_record_hdf5_ui.py`：
- 浏览器打开 → preview 双相机 OK（preview_sampler 主线程 read cam）
- 点「开始录制」→ controller 状态切 RECORDING ✓
- 但 `frame_count=0` 永远不涨；`teleop.get_action()` 内部第一次调 zerorpc 就卡死

## 根因（与 Phase D Sub2 commit f81fbe9 同根因）

`RecorderController.start()` 启 **daemon thread** 跑 `_record_main` → 在 daemon thread 内调 `run_episodes` → `record_episode` → 主循环里：
- `teleop.get_action()` → `UnityVRRobot._measured_joints()` → `self._client.robot_get_joint_positions()` (zerorpc)
- `robot.send_action(action)` → `_send_action_cartesian` → `robot._robot.robot_update_desired_ee_pose(...)` (zerorpc)

两个 zerorpc client `self._client` / `robot._robot` 在**主线程**（`build_robot_and_teleop`）创建，绑主线程 gevent Hub。daemon thread 调它们 → gevent thread-affinity 跨线程死锁。

调试证据（DBG log）：
- `tick 1 A start` ✓
- `reader.get_transformations_and_buttons` 0.0ms（不走 zerorpc，OK）
- `measured_joints took 0.0ms`（强制返 zeros 后）✓
- `tick 1 B got_action` ✓
- **`tick 1 C send_action_done` 永远没出**（卡 send_action 的 zerorpc 调用）

## 为什么键盘模式不卡

`scripts/core/run_record_hdf5.py main()` 是主线程跑 `run_episodes`：
- record_episode 主循环跑在主线程
- 所有 zerorpc 调用都在主线程（与 client 创建同线程）
- 不触发 gevent thread-affinity

## 为什么 Phase D Sub2 修过类似问题

Phase D Task 4 把 robot_state 读放后台 SensorThread，触发同问题。commit f81fbe9 修法：真机路径下不开 robot_state SensorThread，主线程 inline read。

但 Phase E 整个 run_episodes 都在 daemon thread 跑——**架构级问题**，不是单点修。

## 修复方向（DEFERRED）

三个候选：
1. **Flask 移到子线程，主线程跑 run_episodes**（最干净，但需重构 controller.start + Flask app.run 顺序）
2. **每次 daemon thread 启动时重建 zerorpc clients**（在 daemon thread 内创建，Hub 绑 daemon thread）—— 复杂，要改 UnityVRRobot + Franka 两处
3. **使用基于 thread-safe IPC 的非 zerorpc 通信**（如 grpc.aio）—— 需要替换 zerorpc，影响范围大

## 当前应对

- yaml `record.ui.enabled: false` 回退键盘模式
- 用户继续用 `run_record_hdf5.py` 键盘录制（`→` 保存 / `←` 丢弃 / `Esc` 停止）
- UI 模式留代码不删，等专项重构

## 教训

1. **zerorpc + gevent + 多线程是雷区**——任何把 zerorpc 调用搬到非主线程的设计都会触发 thread-affinity。Phase D Sub2 修过一次，Phase E 没考虑就又踩。
2. **架构级测试缺失**：Phase E 离线 8 Task 都 PASS（FakeRobot 不触发 gevent），真机才暴露。需要在 Phase E 测试套加 zerorpc gevent mock 检测跨线程访问（参考 commit f81fbe9 的 ThreadAffinityClient）。
3. **preview_sampler 真机用 cam.read() 与 record SensorThread 不冲突**（本次实测）——librealsense pipeline 内部 thread-safe，多 thread cam.read() 各自拿不同 frame。但 zerorpc 不是。
