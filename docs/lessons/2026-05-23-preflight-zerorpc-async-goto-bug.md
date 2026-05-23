# Preflight 假阴性"Desk Homing"——zerorpc 异步 goto 时序 bug

**日期**：2026-05-23
**触发场景**：Phase D 真机验收阶段 1，运行 `python scripts/core/run_record_hdf5.py --config ...`，preflight 报 `夹爪 goto 后 width 未真变（行程跨度 0.0000 < 0.020）→ 疑似丢 homing`，但用户已在 Franka Desk 做过 Homing，且夹爪实际能动。

## 症状

- preflight 三次 `gripper_goto(0.0)`、`gripper_goto(0.07)`、`gripper_goto(0.04)` 完成后，记录到的 settled width 三个全等于 `0.04`（初始 width），span = 0。
- 错误信息引导用户去 Franka Desk 做 Homing，但 Homing 已做过 → 用户陷入循环重 Homing 仍报同错。
- 手工调脚本 `python -c "client.gripper_goto(0.07, ...)"` 单独跑可以动，证明硬件正常。

## 根因

`scripts/core/preflight.py::run_gripper_preflight` 的 settle 逻辑：

```python
client.gripper_goto(target, ..., blocking=True)  # zerorpc 调用
while True:
    s = client.gripper_get_state()
    if not s["is_moving"]:
        break
    ...
```

**问题**：zerorpc 的 `gripper_goto(blocking=True)` 名义上是阻塞的，**但实际是异步返回**——`gripper_goto` RPC 调用立即返回，命令排队到 `franka_hand_client` C++ 进程后还有 ~0.1-0.3s 启动延迟，期间：
- `is_moving` 仍是 `False`（命令还没到硬件层）
- `width` 仍是命令前的旧值

→ settle 循环首次 `get_state` 立即看到 `is_moving=False` → break → 读到旧 width → 三次 goto 记录三次旧 width → span=0 → 假阴性"丢 homing"。

## 修复

`scripts/core/preflight.py::run_gripper_preflight` 改成**两阶段 settle**：

```python
# Phase 1：等 is_moving 变 True（最多 1.5s）— 等命令真正下发到硬件
moving_start_deadline = t0 + 1.5
while time.monotonic() < moving_start_deadline:
    s = client.gripper_get_state()
    if s["is_moving"]:
        break
    time.sleep(poll)

# Phase 2：等 is_moving 变 False（最多 settle_timeout）— 等动作完成
while True:
    s = client.gripper_get_state()
    if not s["is_moving"]:
        break
    if time.monotonic() - t0 >= settle_timeout:
        return Verdict(False, "超时...")
    time.sleep(poll)
```

回归测试见 `tests/test_preflight_gripper.py::test_preflight_handles_zerorpc_async_goto_delay`：用 `AsyncGoto` 假 client 模拟 `gripper_goto` 后 0~0.2s `is_moving=False`、0.2~0.5s `is_moving=True`、0.5s 后 `is_moving=False` 且 width 更新。旧代码 FAIL（假报 Desk Homing），新代码 PASS。

## 教训

1. **`blocking=True` 标志不等于真阻塞**——zerorpc 的 RPC 语义独立于 server 端实现，特别是 server 端转发到子进程（`franka_hand_client`）时，RPC 返回时机和实际硬件状态有 100-300ms 间隔。
2. **settle 逻辑首次 poll 不能信任**——多线程/多进程系统中，命令"发出"与"开始执行"之间有窗口，settle 检测必须先观测到"开始"（is_moving=True）再等"结束"（is_moving=False）。
3. **预检的错误信息会误导用户**——"疑似丢 homing"信息让用户反复 Homing，浪费时间且引发对硬件的怀疑。预检失败的根因分类要细分硬件 vs 软件时序。
4. **真机才暴露的 bug**：单元测试用 `FakeGripper` 同步更新 width/is_moving，覆盖不到 zerorpc 异步时序。回归测试必须显式模拟"goto 返回后 N ms 内硬件未响应"才能锁住此类 bug。

## 相关 commit

- 修复：`scripts/core/preflight.py`（两阶段 settle）+ `tests/test_preflight_gripper.py`（回归测试）
- 提交于 2026-05-23
