# polymetis franka_hand width 反馈链路滞后 ~100ms（vs 主臂 1kHz 实时）

**日期**：2026-05-23
**触发场景**：用户用 rerun 看 Phase D 录制数据（`local/franka_phaseD_acceptance`，30Hz），观察到图像中夹爪开始张开后 130-150ms 才在 `observation.state.gripper_norm` 曲线上看到上升趋势——他抓着的胡萝卜已经下落了，state 还没动。

## 实测数据（spike）

### 1. 单次开夹爪 cmd → width 真实变化延迟

发 `gripper_goto(0.08, blocking=False)` 后，200Hz poll `gripper_get_state`：

- `is_moving=True` 在 **0.8ms** 内立即置位（grpc 标志位很快）
- **但 width 字段 103.2ms 后才第一次更新**——期间卡在 `0.05127m`，然后**突变** -9.892mm（一次跳 9.9mm，非平滑）
- 之后又是若干帧不变，再突变——width 字段是「事件触发」式刷新，不是连续值

### 2. 各字段反馈链路新鲜度对比（300 次 poll @ ~200Hz）

| 字段 | unique values / 300 | 最大连续重复段 | 链路 |
|---|---|---|---|
| `joint_positions` | 300/300 (100%) | 1 帧 = 5ms | polymetis 1kHz 实时控制流 |
| `joint_velocities` | 300/300 (100%) | 1 帧 = 5ms | 同上 |
| `ee_pose` | 300/300 (100%) | 1 帧 = 5ms | 同上 |
| **`gripper_width`** | **2/300 (0.7%)** | **131 帧 = 655ms** | franka_hand 非实时事件推送 |

RPC 本身都是 ~1ms，所以延迟不在网络层。

## 根因

libfranka 提供两条状态流，频率/语义完全不同：

| 通道 | 频率 | polymetis 拿到 | zerorpc 转发 | 实测端到端滞后 |
|---|---|---|---|---|
| `franka::RobotState` (主臂) | **1 kHz** | `run_server` 实时控制循环 1kHz 读 | `robot_get_*` | **< 5ms** |
| `franka::GripperState` (夹爪) | 事件式 (~10-30Hz 不连续) | `franka_hand_client.cpp::getGripperState` 调 `gripper_->readOnce()` 阻塞返回最后一次硬件推送 | `gripper_get_state` | **~100-150ms + 字段更新不连续** |

注：`franka_hand_client.cpp:87` 还有个上游 bug `int period = 1.0 / GRIPPER_HZ;` 把 `0.0333` 截断成 `0`，所以 `clock_nanosleep` 永远立即返回——但实际节拍由 `readOnce` 阻塞主导，所以**不是这个 bug 造成的滞后**；改了类型也不会改善 width 反馈延迟。真正原因是 franka SDK 把 gripper 设计成事件式状态推送（非实时控制对象）。

## 这不是 record_episode 同步代码的 bug

- `record_episode` 主循环：发 `send_action` → 主线程 inline read `robot_state_fn`（含 `gripper_get_state`） → 写 `effector_ts = monotonic_now`
- `align_offline` 用图像锚 ts + `np.interp(cam_ts, effector_ts, gripper_norm)` 对齐
- 时间戳本身没错——错的是 **width 物理生效时刻 vs polymetis 读到 width 时刻** 差了 ~100ms（硬件链路 inherent）
- 写到 dataset 的 `effector_ts` 是「polymetis 读到」时刻，不是「物理生效」时刻
- 所以 rerun 按 timestamp 配对显示时，cam 帧（30-50ms 旧）和 state 帧（~100ms 旧）的物理时刻差异显现为「state 滞后图像 ~80-100ms」

## 选项与建议

**A. 不修，承认特性**（推荐）
- 模仿学习训练时用 `action.gripper_cmd`（指令侧时戳准确）作主信号
- `observation.state.gripper_norm` 作辅助，知道它有 ~100ms 系统性滞后
- 模型对全 episode 一致的常数延迟鲁棒（不是 jitter），影响有限

**B. 数据层常数补偿**
- `align_offline.py` 给 `eff_ts` 减 ~100ms 偏移再插值
- 缺陷：100ms 是平均，实际有 jitter（width 字段一次跳一大段说明更新非连续），强行减常数引入新对齐误差

**C. 硬件层修**（不推荐）
- 改 `franka_hand_client.cpp` 提高更新频率、用后台线程异步刷而非阻塞 readOnce
- 改 C++ + 重 build polymetis + 重启所有服务
- 风险大，franka SDK 上游可能限制（事件式推送是 firmware 行为，client 改不了）

## 操作教训

1. **不同字段反馈链路差异巨大**：franka 主臂是 1kHz 硬实时（libfranka 控制对象），夹爪是事件式状态机（libfranka 非控制对象）。Phase D Task 1 spike 测了 polymetis 主臂 240Hz 吞吐，**未测夹爪反馈链路新鲜度**——下次设计 schema 或同步层时要对每个字段单独测，不能假设统一行为。
2. **`is_moving` ≠ width 已变**：preflight 已经被这个特性坑过一次（lesson 2026-05-23-preflight-zerorpc-async-goto-bug）——`gripper_goto(blocking=True)` 异步返回，`is_moving` 先变 True、width 才滞后更新。同根因。
3. **大段重复值不一定是 bug**：300 次 poll 看到 width 大段重复要警觉，但**先排除「字段本身就是事件式更新」**再怀疑代码缓存/锁等问题。
4. **rerun 显示对齐 ≠ 物理时刻对齐**：dataset 的 timestamp 字段是「采样写入时刻」，不一定等于「物理生效时刻」。诊断同步问题要追到链路最远端（硬件↔传感器↔上报↔transport↔我们的代码）。

## 相关 lesson 链接

- [2026-05-23-preflight-zerorpc-async-goto-bug.md](2026-05-23-preflight-zerorpc-async-goto-bug.md) —— 同根因（`blocking=True` 不等于 width 已更新）
- [2026-05-23-zerorpc-gevent-thread-affinity.md](2026-05-23-zerorpc-gevent-thread-affinity.md) —— Phase D 多线程 zerorpc 的相关坑

## 相关文件（仅记录，不做修改）

- `scripts/core/run_record_hdf5.py::record_episode` —— 同步链路
- `scripts/tools/align_offline.py` —— 离线对齐
- `/home/ubuntu/Desktop/jhli/fairo-franka/polymetis/polymetis/src/clients/franka_panda_client/franka_hand_client.cpp` —— polymetis 夹爪 client 上游
- `/home/ubuntu/Desktop/jhli/fairo-franka/polymetis/polymetis/include/polymetis/clients/franka_hand_client.hpp:10` —— `#define GRIPPER_HZ 30`（实际由 `readOnce` 阻塞节拍主导）

## 验证 spike 脚本

`/tmp/spike_gripper_latency.py` + `/tmp/spike_state_freshness.py`（远端）：复现实测；改 width 阈值或 poll 数可重测。
