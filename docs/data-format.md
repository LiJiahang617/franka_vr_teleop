# 数据格式说明

> `franka-hdf5-v2` schema 完整契约、`validate_episode` 校验内容、hdf5→LeRobot 转换流程与差异。

## 目录

- [1. franka-hdf5-v2 概览](#1-franka-hdf5-v2-概览)
- [2. HDF5 group / dataset 完整 schema](#2-hdf5-group--dataset-完整-schema)
- [3. validate_episode 校验内容](#3-validate_episode-校验内容)
- [4. State / Action 维度约定](#4-state--action-维度约定)
- [5. hdf5 → LeRobot 转换流程](#5-hdf5--lerobot-转换流程)
- [6. v3.0 与 v2.1 的差异](#6-v30-与-v21-的差异)
- [7. v1→v2 迁移工具](#7-v1v2-迁移工具)

---

## 1. franka-hdf5-v2 概览

`franka-hdf5-v2` 是录制阶段的**冻结中间格式**（Phase D 升级自 v1），定义在仓库根 `franka_hdf5_schema.py`。

- `SCHEMA_VERSION = "franka-hdf5-v2"`
- `JOINT_DOF = 7`，`EE_DIM = 6`（`[x, y, z, rx, ry, rz]`），`GRIPPER_MAX_M = 0.08`
- 一个 `.h5` 文件 = 一条 episode。
- 写出方：`scripts/core/hdf5_writer.py::write_episode`；校验方：`validate_episode`；消费方：`scripts/tools/hdf5_to_lerobot{,_v21}.py`。

> 改 schema 必须 bump `SCHEMA_VERSION`，并同步 writer / validator / 两个转换器 / 对应 tests。

约定：除图像 dataset 外，**所有数值 dataset 均为 `float64`**；时间戳来自 `time.monotonic()`（arm/effector）或 RealSense 硬件戳回归映射到 monotonic 秒域（camera hw_timestamp）。

### v2 相对 v1 的主要变更

| 变更点 | v1 | v2 |
|---|---|---|
| 共用时间戳 | `observations/timestamp(N,1)` | **删除**，N 改由 `action/timestamp.shape[0]` 确定 |
| arm/effector/camera 各模态 | timestamp(N,) 副本 | timestamp(N,) + **stale(N,) bool**（独立） |
| camera/{cn} | 仅 timestamp | + **hw_timestamp(N,) float64**（RealSense 硬件戳，映射至 monotonic 秒域） |
| state_hifreq | joints/joint_vel/pose/timestamp/poly_ts | + **wrench(M,6) float64**（Phase F 实填，M=0 合规） |
| 可扩展字段 | 无 | depth/tactile **validate-if-present** |
| state_hifreq 实填 | M=0 占位 | Phase D 240Hz 实填（spike-a 分支 A） |

---

## 2. HDF5 group / dataset 完整 schema

设 `N` = 帧数（由 `action/timestamp.shape[0]` 决定），`M` = 高频状态采样数（Phase D 实填，≈240×episode_sec）。

### infos/ —— 元信息

| 路径 | shape | dtype | 含义 |
|---|---|---|---|
| `infos/schema_version` | 标量 | bytes | 固定 `"franka-hdf5-v2"` |
| `infos/task_info/task_name` | 标量 | bytes | 任务名称 |
| `infos/task_info/collection_frequency` | (2,) | float64 | `[target_fps, 实测平均 fps]` |
| `infos/task_info/total_frames` | 标量 | int64 | 帧数 N |
| `infos/task_info/robot` | 标量 | bytes | 固定 `"franka_panda"` |
| `infos/camera_params` | （空 group） | — | 占位 |
| `infos/calibration/oc2base_R` | (3, 3) | float64 | oc→base 标定旋转矩阵 |
| `infos/calibration/quality` | 标量 | bytes | JSON 序列化的质量参数 dict |
| `infos/calibration/vr_source` | 标量 | bytes | VR 来源标识（如 `unityvr`） |

### observations/ —— 观测

#### arm 模态（独立时间戳）

| 路径 | shape | dtype | 含义 |
|---|---|---|---|
| `observations/arm/joints` | (N, 7) | float64 | 7 关节位置（rad） |
| `observations/arm/joint_vel` | (N, 7) | float64 | 7 关节速度 |
| `observations/arm/pose` | (N, 6) | float64 | 末端位姿 `[x,y,z,rx,ry,rz]`（base 系） |
| `observations/arm/timestamp` | (N,) | float64 | arm 模态独立软件戳（`time.monotonic()`）；非递减 |
| `observations/arm/stale` | (N,) | bool | 该帧 arm 数据是否为上帧补填（True = 陈旧） |

#### effector 模态（独立时间戳）

| 路径 | shape | dtype | 含义 |
|---|---|---|---|
| `observations/effector/position` | (N, 1) | float64 | 夹爪开度（米） |
| `observations/effector/position_norm` | (N, 1) | float64 | 夹爪开度归一化 `[0,1]` |
| `observations/effector/type` | (N,) | vlen bytes | 每帧固定 `b"gripper"` |
| `observations/effector/timestamp` | (N,) | float64 | effector 模态独立软件戳；非递减 |
| `observations/effector/stale` | (N,) | bool | 该帧 effector 是否为上帧补填 |

#### camera 模态（每相机独立时间戳 + 硬件戳）

| 路径 | shape | dtype | 含义 |
|---|---|---|---|
| `observations/camera/rgb/{cam}/images` | (N,) | vlen uint8 | 每帧 1 个 JPEG 编码字节串（RGB→BGR→imencode） |
| `observations/camera/rgb/{cam}/timestamp` | (N,) | float64 | 相机软件戳（`time.monotonic()`）；非递减 |
| `observations/camera/rgb/{cam}/stale` | (N,) | bool | 该帧图像是否为上帧补填 |
| `observations/camera/rgb/{cam}/hw_timestamp` | (N,) | float64 | RealSense 硬件戳（global_time 域经线性回归映射到 monotonic 秒域）；Phase D spike-b A 实填 |

> 相机分组 `{cam}`：当前为 `wrist_image` 与 `exterior_image`（或 `wrist`/`exterior`）。
> `hw_timestamp` 映射：RealSense `frame.get_timestamp()` 返回毫秒级绝对时间，经启动时线性回归换算为 monotonic 秒域后写入。

#### state_hifreq 模态（240Hz 独立长度 M）

| 路径 | shape | dtype | 含义 |
|---|---|---|---|
| `observations/state_hifreq/joints` | (M, 7) | float64 | 高频关节位置 |
| `observations/state_hifreq/joint_vel` | (M, 7) | float64 | 高频关节速度 |
| `observations/state_hifreq/pose` | (M, 6) | float64 | 高频末端位姿 |
| `observations/state_hifreq/timestamp` | (M,) | float64 | 高频采样时间戳（monotonic，严格递增） |
| `observations/state_hifreq/poly_ts` | (M,) | float64 | polymetis 侧时间戳 |
| `observations/state_hifreq/wrench` | (M, 6) | float64 | 力/力矩（Phase F 实填，M=0 时合规） |

> M=0 时所有 state_hifreq dataset shape 均为 `(0, *)` 或 `(0,)`，合规。

#### 可扩展字段（validate-if-present）

depth/tactile 字段存在时才校验，缺失不报错（Phase F 实填）。

### action/ —— 动作

| 路径 | shape | dtype | 含义 |
|---|---|---|---|
| `action/delta_ee_pose` | (N, 6) | float64 | 末端位姿增量 `[dx,dy,dz,drx,dry,drz]`（base 系） |
| `action/gripper_cmd` | (N, 1) | float64 | 夹爪指令 |
| `action/timestamp` | (N,) | float64 | 动作时间戳（monotonic，严格递增，**定义 N**） |

---

## 3. validate_episode 校验内容

`validate_episode(path)` 返回 violations 列表（空 = 合格）。`write_episode` 写盘后立即调用，不合规抛 `RuntimeError`。校验项：

1. `infos/schema_version` 存在且 == `"franka-hdf5-v2"`。
2. 必需 group 齐全：`infos`、`infos/calibration`、`observations`、`observations/arm`、`observations/effector`、`observations/camera`、`observations/state_hifreq`、`action`。
3. `action/timestamp` 存在且为一维 float64；以其首维定 `N`。
4. 逐模态校验 **shape 精确匹配** 且 dtype 正确：
   - arm：`joints(N,7)`、`joint_vel(N,7)`、`pose(N,6)`、`timestamp(N,)` float64、`stale(N,)` bool
   - effector：`position(N,1)`、`position_norm(N,1)`、`timestamp(N,)` float64、`stale(N,)` bool、`type` 存在
   - camera/{cn}：`images(N,)` vlen uint8、`timestamp(N,)` float64、`stale(N,)` bool、`hw_timestamp(N,)` float64
   - action：`delta_ee_pose(N,6)`、`gripper_cmd(N,1)` float64
5. `observations/camera/rgb` 存在且至少含 1 个相机子组。
6. state_hifreq：`joints(M,7)`、`joint_vel(M,7)`、`pose(M,6)`、`timestamp(M,)`、`poly_ts(M,)`、`wrench(M,6)` float64（M=0 合规）。
7. `infos/calibration/oc2base_R` 存在、shape == `(3,3)`、dtype == float64。
8. 时间戳单调性：
   - `action/timestamp`：**严格递增**（N≥2）
   - arm/effector/camera `timestamp`：**非递减**（N≥2，stale 允许相等）
   - `state_hifreq/timestamp`：**严格递增**（M≥2）

---

## 4. State / Action 维度约定

转换器输出与 realman 数据集完全对齐（见 `scripts/tools/hdf5_lerobot_map.py`）。

### Observation State —— 14D（realman 布局）

```
joint_1_rad ... joint_7_rad,              # 索引 0-6   ：7 关节位置
gripper_open,                              # 索引 7     ：夹爪归一化开度
eef_pos_x_m, eef_pos_y_m, eef_pos_z_m,   # 索引 8-10  ：末端位置
eef_rot_euler_x_rad, ..., eef_rot_euler_z_rad  # 索引 11-13 ：末端姿态
```

数据来源：`observations/arm/joints(N,7)` → joint_1..7；`observations/effector/position_norm(N,1)` → gripper_open；`observations/arm/pose(N,6)` → eef_pos_xyz + eef_rot_euler_xyz。

### Action —— 14D（next-state 语义）

字段名与 `observation.state` 完全相同，数值 = next-state：
- `action[i] = observation.state[i+1]`（i < N-1）
- `action[N-1] = observation.state[N-1]`（末帧复制）

**不使用** hdf5 的 `action/delta_ee_pose`（原始增量保留在 hdf5，lerobot 转换不消费）。

依据：实测 realman 数据集 `mean|action[i]-state[i+1]|=0.0`（action 本质就是 next-state）。

---

## 5. hdf5 → LeRobot 转换流程

两个独立转换器，输出不同 LeRobot 版本：

### 5.1 hdf5_to_lerobot.py → LeRobot v3.0

```bash
python scripts/tools/hdf5_to_lerobot.py \
    --in <hdf5_dir> --repo-id local/franka_x --fps 30 \
    --root <out_dir> --task "任务描述"
```

### 5.2 hdf5_to_lerobot_v21.py → LeRobot v2.1

```bash
python scripts/tools/hdf5_to_lerobot_v21.py \
    --in-dir <hdf5_dir> --out <out_dir> \
    --fps 30 --task "任务描述" --robot-type franka
```

独立实现，不依赖任何版本 lerobot。输出与 realman 数据集逐字段对齐（14D state + 14D next-state action）。

可选的离线对齐步骤（以图像戳为锚做线性插值 + SLERP）：

```bash
python scripts/tools/align_offline.py \
    --in <ep.h5> --out <aligned_dir> --on-stale interpolate
```

---

## 6. v3.0 与 v2.1 的差异

| 维度 | LeRobot v3.0（`hdf5_to_lerobot.py`） | LeRobot v2.1（`hdf5_to_lerobot_v21.py`） |
|---|---|---|
| 产出方式 | franka2 本机 lerobot 库直转 | 独立实现，不依赖 lerobot |
| meta 文件 | `tasks.parquet` / `episodes/` / `stats.json` | `info.json` / `tasks.jsonl` / `episodes.jsonl` |
| 数据文件 | `file-NNN.parquet`（chunk 分组） | `data/chunk-000/episode_NNNNNN.parquet` |
| 视频 | `videos/{key}/chunk/` | `videos/chunk-000/observation.images.{cam}/` |
| 互通性 | **v3.0 与 v2.1 不互通** | 同左 |
| 适用管线 | franka2 本机训练 / 可视化 | RoboCOIN / realman 参考集 / GR00T |
| action 语义 | 14D next-state | 14D next-state（与 v3.0 一致） |

---

## 7. v1→v2 迁移工具

对已有 v1 数据，使用 `scripts/tools/migrate_v1_to_v2.py` 离线迁移：

```bash
python scripts/tools/migrate_v1_to_v2.py \
    --in <episode_v1.h5> \
    --out <episode_v2.h5>
```

迁移逻辑：
- `schema_version` 改 `franka-hdf5-v2`
- 共用戳 `observations/timestamp(N,1)` → 各模态独立 `timestamp(N,)`
- 新增各模态 `stale(N,) = False`（v1 无缺帧概念）
- 新增 camera `hw_timestamp = timestamp`（v1 无真硬件戳，用软件戳占位）
- 新增 `state_hifreq/wrench = zeros(M,6)`（M=0 时 shape=(0,6)）
- 迁移后自动调用 `validate_episode` 自检

---

*相关文档：[architecture.md](architecture.md)、[development-guide.md](development-guide.md)。*
