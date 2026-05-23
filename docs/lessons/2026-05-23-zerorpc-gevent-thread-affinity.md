# Phase D 多线程采集与 zerorpc gevent thread-affinity 冲突

**日期**：2026-05-23
**触发场景**：Phase D 真机验收阶段 1，运行 `run_record_hdf5.py` 录制 5s episode，preflight 通过后进入 RECORDING 状态立即崩，报：
```
[WARMUP] sensor 'robot_state' 预热超时未采到首帧，将降级为占位/stale
AssertionError: Can only use Waiter.switch method from the Hub greenlet
2026-05-23T02:25:47Z <callback at 0x... args=()> failed with AssertionError
  File "scripts/core/run_record_hdf5.py", line 438, in record_episode
    with zerorpc_lock:
```

## 症状

- 录制启动后 0.5s 内崩溃，warmup 显示 robot_state 模态从未采到首帧（线程 read 全失败）。
- 主线程持 `zerorpc_lock` 调 `robot.send_action(action)` 时炸出 gevent 异常。
- 单元测试 602 passed 全绿（用 FakeRobot 走 `get_observation` 回退路径，不触发 gevent）。

## 根因

**zerorpc Client 基于 gevent**，而 gevent 的 Hub（事件循环）是 **thread-local，绑定首次访问的 OS 线程**。后续从其他 OS 线程访问同一 zerorpc client 会触发 gevent 内部 `Waiter.switch` 跨线程断言，立即破坏整个 Hub 状态——之后**任何线程**（包括 Hub 所属的原主线程）调该 client 都会崩。

Phase D Task 4/7 引入了两条后台线程：
- `SensorThread("robot_state")` @ fps：daemon thread 调 `robot._robot.robot_get_*`
- `HistoryCollectorThread("state_hifreq")` @ 240Hz：daemon thread 调同样的 RPC

两者用 `threading.Lock` 串行化，但 **lock 只能串行调用，不能改变调用所在的 OS 线程**——只要 daemon thread 触碰 client，gevent 状态就被污染。

Phase D spike（Task 1）只测了单线程 240Hz 吞吐（0.4ms 往返 / Branch A），**未测多线程访问**，所以这个 thread-affinity 约束没被发现。

附加 bug：`scripts/core/run_record_hdf5.py:629` 硬编码 `hifreq_rate=240.0`，无视 yaml 中 `state_hifreq.enabled: false`——配置项形同虚设。

## 修复

`scripts/core/run_record_hdf5.py::record_episode` 增加 **zerorpc 真机判别**（`getattr(robot, "_robot", None) is not None`）：

1. **真机路径**（`robot._robot` 是 zerorpc Client 实例）：
   - 不创建 `robot_state` SensorThread，主循环每 tick 主线程 inline 调 `robot_state_fn()`
   - 强制 `hifreq_rate = 0.0`，不创建 `state_hifreq` HistoryCollectorThread
   - 相机 SensorThread 保留（cameras 走 librealsense direct，不经 gevent，OS 线程安全）
   - inline read 失败时降级占位+stale=True（与 SensorThread 的容错语义一致），不挂主循环
2. **测试路径**（FakeRobot 无 `_robot`）：保留原 SensorThread + HistoryCollectorThread 行为，既有 602 测试全绿。

回归测试见 `tests/test_record_zerorpc_thread_affinity.py::test_real_robot_zerorpc_only_called_from_main_thread`：用 `ThreadAffinityClient` 记录每次 RPC 的 thread id，断言所有调用都在主线程。旧代码触发 `robot_get_joint_positions` 被非主线程调用（FAIL）；新代码全部主线程（PASS）。

## 教训

1. **「非线程安全」≠「加锁就行」**——具体看库的并发模型。基于 gevent 的库（zerorpc/eventlet/gunicorn-async 等）通常是「thread-local Hub」语义：lock 串行化是必要但不充分，**还必须保证所有访问都在同一 OS 线程**。
2. **spike 必须覆盖目标使用模式**——单线程吞吐测试不能代替多线程访问测试。Task 1 spike 应该在 daemon thread 里调 zerorpc 一次才能暴露这个约束。
3. **FakeRobot 测试覆盖不到 gevent 约束**——因为 fake 不走 zerorpc。真机相关并发约束必须有显式的 thread-affinity 回归测试（构造仿 gevent 行为的 mock，断言访问线程），否则会重复掉进同一坑。
4. **配置项硬编码是隐性 bug**——`hifreq_rate=240.0` 硬写在调用现场，绕开了 yaml `state_hifreq.enabled` 开关；用户完全无法通过配置关掉。任何配置项都应该「读到底」，否则就是死代码。
5. **Phase D Task 7 的 240Hz state_hifreq 目标仍然成立**，但实现路径需要换：
   - 选项 A：dispatcher 线程（专用 OS 线程独占 zerorpc client + Hub，其他线程通过 queue 提交 callable）
   - 选项 B：每线程独立创建 zerorpc.Client（验证 gevent.hub.get_hub 在多线程下的 thread-local 行为）
   - 选项 C：把 polymetis 状态读改用非 gevent 的协议（gRPC direct / shared memory）

   当前 commit 用方案 0（关掉 240Hz）让真机可用；正式实现留给后续单独的 Phase D 续集。

## 相关 commit

- 修复：`scripts/core/run_record_hdf5.py`（真机/测试运行时分支 + inline robot_state read）+ `tests/test_record_zerorpc_thread_affinity.py`（thread-affinity 回归测试）
- 提交于 2026-05-23

## 验证

- 单元测试：`tests/ 603 passed`（+1 thread-affinity 回归，原 602 零退化）
- 真机 spike：连真 zerorpc 4242，FakeTeleop 跑 `record_episode(max_sec=2, fps=30)` 三轮，每轮 60 帧、29.88 fps、`arm_stale=False`、`block=None`（state_hifreq 真机强制关），无 gevent 异常。
