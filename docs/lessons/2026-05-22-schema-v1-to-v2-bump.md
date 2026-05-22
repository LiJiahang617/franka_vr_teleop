# franka-hdf5 v1→v2 bump 全消费方协调清单

**Task**: PhaseD-Task2  
**日期**: 2026-05-22

## 变更摘要

`franka_hdf5_schema.py` 从 `franka-hdf5-v1` 升级为 `franka-hdf5-v2`：

| 变更 | v1 | v2 |
|---|---|---|
| SCHEMA_VERSION | `franka-hdf5-v1` | `franka-hdf5-v2` |
| 共用时间戳 | `observations/timestamp(N,1) float64` | **删除**（由 `action/timestamp` 确定 N） |
| arm 模态 | timestamp(N,) | timestamp(N,) + **stale(N,) bool** |
| effector 模态 | timestamp(N,) | timestamp(N,) + **stale(N,) bool** |
| camera/{cn} | timestamp(N,) | timestamp(N,) + **stale(N,) bool** + **hw_timestamp(N,) float64** |
| action 模态 | timestamp(N,) | 同 v1（无变化） |
| state_hifreq | joints/joint_vel/pose/timestamp/poly_ts | 同 v1 + **wrench(M,6) float64**（占位） |
| 可扩展字段 | 无 | depth/tactile validate-if-present |
| N 的定义来源 | observations/timestamp.shape[0] | **action/timestamp.shape[0]** |

## 已完成（Task 2）

- [x] `franka_hdf5_schema.py` → v2，`validate_episode` 同步
- [x] `tests/test_franka_hdf5_schema_v2.py` 新建（31 passed）
- [x] `tests/test_franka_hdf5_schema.py` 更新为 v2（10 passed）

## 下游消费方完整清单（grep 实证）

以下文件直接或间接依赖 `franka_hdf5_schema` / `validate_episode` / hdf5 group 路径，
**在 Task 2 bump 后会因 schema 版本不符而 fail，需在 Task 4-6 逐步适配收口**：

### 直接 import schema 的文件

| 文件 | 用法 | Task 适配 |
|---|---|---|
| `scripts/core/hdf5_writer.py` | `S = load_franka_hdf5_schema()`，写 v1 格式，`validate_episode` fail-loud | Task 4 |
| `scripts/tools/hdf5_to_lerobot.py` | `S.validate_episode(ep)` 预校验，读 `observations/timestamp` | Task 6 |
| `scripts/tools/hdf5_to_lerobot_v21.py` | `_schema.validate_episode(h5_path)` 预校验，读 camera/rgb 路径 | Task 6 |
| `scripts/tools/hdf5_lerobot_map.py` | 读 `observations/camera/rgb/{c}/images` 等路径（无直接 schema import） | Task 6 |
| `scripts/core/schema_loader.py` | 加载器，不需改动 | - |

### 依赖 schema 的测试文件（当前 fail / error 清单）

| 测试文件 | fail 原因 | 属于 schema bump 预期？ | Task 适配 |
|---|---|---|---|
| `tests/test_hdf5_writer.py` | writer 写 v1 格式 → `validate_episode` 返回 violations → RuntimeError | **是** | Task 4 |
| `tests/test_hdf5_writer_async.py` | 同上（3个case） | **是** | Task 4 |
| `tests/test_v21_cli.py` | 合成数据用 `S.SCHEMA_VERSION` 写 v1 → 转换器预校验全部跳过 → `无有效 episode` | **是** | Task 6 |
| `tests/test_v21_structure_diff.py` | 同上（fixture 共享，全部 ERROR） | **是** | Task 6 |
| `tests/test_v21_parquet.py` | 合成 v1 数据，转换器预校验跳过（当前未显示在 fail，可能隐含） | 检查确认中 | Task 6 |
| `tests/test_v21_video.py` | 同上 | 检查确认中 | Task 6 |
| `tests/test_v21_meta.py` | 同上 | 检查确认中 | Task 6 |
| `tests/test_hdf5_lerobot_map.py` | 合成 v1 数据调 `validate_episode` 等 | **是** | Task 6 |
| `tests/test_record_hdf5_cli.py` | 调 hdf5_writer 写 v1 | **是** | Task 4 |
| `tests/test_record_loop_async.py` | 同上 | **是** | Task 4 |

### 当前全回归数字（Task 2 完成时）

- **370 passed**（基线 397 + schema_v2 新增 30 - v1 消费方 fail 57）
- **8 failed**：test_hdf5_writer(1) + test_hdf5_writer_async(3) + test_v21_cli(4) — 全部属于 schema bump 预期
- **50 errors**：test_v21_cli(32 module fixture ERROR) + test_v21_structure_diff(18 module fixture ERROR) — 全部属于 schema bump 预期
- **0 unrelated 真 bug**

## 坑记录

1. **N 的定义来源**：v1 用 `observations/timestamp.shape[0]`，v2 改为 `action/timestamp.shape[0]`。
   消费方 `hdf5_to_lerobot.py` 第 77 行还在读 `observations/timestamp.shape[0]`，Task 6 需修复。

2. **v2 validator 对 `observations/timestamp` 遗留字段的处理**：忽略（不报 violation）。
   这样即使有工具写入了该字段也不会被误判为不合规，Task 4 writer 改完后自然不再写该字段。

3. **state_hifreq/wrench** 是新增必填字段（M=0 时 shape=(0,6) 合规）。
   hdf5_writer 当前不写 wrench，Task 4 需补上。

4. **可扩展字段 validate-if-present**：depth/tactile 存在时才校验，缺失不报错。
   这样前期 writer 不写也不影响校验通过（Phase F 实填时无需再 bump schema）。
