# 数据格式说明

> `franka-hdf5-v1` schema 完整契约、`validate_episode` 校验内容、hdf5→LeRobot 转换流程与差异。

## 目录

- [1. franka-hdf5-v1 概览](#1-franka-hdf5-v1-概览)
- [2. HDF5 group / dataset 完整 schema](#2-hdf5-group--dataset-完整-schema)
- [3. validate_episode 校验内容](#3-validate_episode-校验内容)
- [4. State / Action 维度约定](#4-state--action-维度约定)
- [5. hdf5 → LeRobot 转换流程](#5-hdf5--lerobot-转换流程)
- [6. v3.0 与 v2.1 的差异](#6-v30-与-v21-的差异)

---

## 1. franka-hdf5-v1 概览

`franka-hdf5-v1` 是录制阶段的**冻结中间格式**，定义在仓库根 `franka_hdf5_schema.py`。

- `SCHEMA_VERSION = "franka-hdf5-v1"`
- `JOINT_DOF = 7`，`EE_DIM = 6`（`[x, y, z, rx, ry, rz]`），`GRIPPER_MAX_M = 0.08`
- 一个 `.h5` 文件 = 一条 episode。
- 写出方：`scripts/core/hdf5_writer.py::write_episode`；校验方：`validate_episode`；消费方：`scripts/tools/hdf5_to_lerobot{,_v21}.py`。

> 改 schema 必须 bump `SCHEMA_VERSION`，并同步 writer / validator / 两个转换器 / 对应 tests。

约定：除图像 dataset 外，**所有数值 dataset 均为 `float64`**；时间戳来自 `time.monotonic()`，严格单调递增。

---

## 2. HDF5 group / dataset 完整 schema

设 `N` = 帧数，`M` = 高频状态采样数（当前为 0，Phase D 填）。

### infos/ —— 元信息

| 路径 | shape | dtype | 含义 |
|---|---|---|---|
| `infos/schema_version` | 标量 | bytes | 固定 `"franka-hdf5-v1"` |
| `infos/task_info/task_name` | 标量 | bytes | 任务名称 |
| `infos/task_info/collection_frequency` | (2,) | float64 | `[target_fps, 实测平均 fps]` |
| `infos/task_info/total_frames` | 标量 | int64 | 帧数 N |
| `infos/task_info/robot` | 标量 | bytes | 固定 `"franka_panda"` |
| `infos/camera_params` | （空 group） | — | 占位 |
| `infos/calibration/oc2base_R` | (3, 3) | float64 | oc→base 标定旋转矩阵 |
| `infos/calibration/quality` | 标量 | bytes | JSON 序列化的质量参数 dict |
| `infos/calibration/vr_source` | 标量 | bytes | VR 来源标识（= `control_mode`，如 `unityvr`） |

### observations/ —— 观测

| 路径 | shape | dtype | 含义 |
|---|---|---|---|
| `observations/timestamp` | (N, 1) | float64 | 帧时间戳（`time.monotonic()`），严格单调递增 |
| `observations/arm/joints` | (N, 7) | float64 | 7 关节位置（rad） |
| `observations/arm/joint_vel` | (N, 7) | float64 | 7 关节速度；未接通时零填 |
| `observations/arm/pose` | (N, 6) | float64 | 末端位姿 `[x,y,z,rx,ry,rz]`（base 系） |
| `observations/arm/timestamp` | (N,) | float64 | 同 `observations/timestamp` 一维副本 |
| `observations/effector/position` | (N, 1) | float64 | 夹爪开度（米）= `position_norm × gripper_max_open` |
| `observations/effector/position_norm` | (N, 1) | float64 | 夹爪开度归一化 `[0,1]` |
| `observations/effector/type` | (N,) | vlen bytes | 每帧固定 `b"gripper"` |
| `observations/effector/timestamp` | (N,) | float64 | 时间戳一维副本 |
| `observations/camera/rgb/{cam}/images` | (N,) | vlen uint8 | 每帧 1 个 JPEG 编码字节串 |
| `observations/camera/rgb/{cam}/timestamp` | (N,) | float64 | 时间戳一维副本 |
| `observations/state_hifreq/joints` | (M, 7) | float64 | 高频关节位置（当前 M=0 占位） |
| `observations/state_hifreq/joint_vel` | (M, 7) | float64 | 高频关节速度 |
| `observations/state_hifreq/pose` | (M, 6) | float64 | 高频末端位姿 |
| `observations/state_hifreq/timestamp` | (M,) | float64 | 高频采样时间戳 |
| `observations/state_hifreq/poly_ts` | (M,) | float64 | polymetis 侧时间戳 |

> 相机分组 `{cam}`：录制时取自 `robot.cameras.keys()`，当前为 `wrist_image` 与 `exterior_image`。图像存为 JPEG 编码后的变长 uint8（编码在录制循环 `_encode_jpg` 中完成：RGB→BGR→`cv2.imencode('.jpg')`）。
>
> `state_hifreq` 当前是 writer 写入的全 0 占位空数组（M=0），Phase D 高频采集接通后才填实数据。

### action/ —— 动作

| 路径 | shape | dtype | 含义 |
|---|---|---|---|
| `action/delta_ee_pose` | (N, 6) | float64 | 末端位姿增量 `[dx,dy,dz,drx,dry,drz]`（base 系） |
| `action/gripper_cmd` | (N, 1) | float64 | 夹爪指令（来自 `gripper_cmd_bin`） |
| `action/timestamp` | (N,) | float64 | 时间戳一维副本 |

---

## 3. validate_episode 校验内容

`validate_episode(path)` 返回 violations 列表（空 = 合格）。`write_episode` 写盘后立即调用，不合规抛 `RuntimeError`。校验项：

1. `infos/schema_version` 存在且 == `"franka-hdf5-v1"`。
2. 必需 group 齐全：`infos`、`infos/calibration`、`observations`、`observations/arm`、`observations/effector`、`observations/camera`、`observations/state_hifreq`、`action`。
3. `observations/timestamp` 存在（缺失直接返回）；以其首维定 `N`。
4. 逐 dataset 校验 **shape 精确匹配** 且 **dtype == float64**：
   - `observations/timestamp (N,1)`、`arm/joints (N,7)`、`arm/joint_vel (N,7)`、`arm/pose (N,6)`、`arm/timestamp (N,)`；
   - `effector/position (N,1)`、`effector/position_norm (N,1)`、`effector/timestamp (N,)`；
   - `action/delta_ee_pose (N,6)`、`action/gripper_cmd (N,1)`、`action/timestamp (N,)`。
5. `observations/effector/type` 存在。
6. `observations/state_hifreq/joints` 存在；以其首维定 `M`，校验 `state_hifreq` 各 dataset shape/dtype（`joints (M,7)`、`joint_vel (M,7)`、`pose (M,6)`、`timestamp (M,)`、`poly_ts (M,)`）。
7. `infos/calibration/oc2base_R` 存在、shape == `(3,3)`、dtype == float64。
8. `observations/timestamp` 严格单调递增（`N>=2` 时任意 `diff <= 0` 即违规）。

---

## 4. State / Action 维度约定

转换器把 hdf5 字段映射为 LeRobot 的 `observation.state` / `action` 向量（见 `scripts/tools/hdf5_lerobot_map.py`）。

### Action —— 7D

`ACTION_KEYS` 顺序（与 numpy 列序一致）：

```
delta_ee_pose.x, delta_ee_pose.y, delta_ee_pose.z,
delta_ee_pose.rx, delta_ee_pose.ry, delta_ee_pose.rz,
gripper_cmd_bin
```

= 6 维末端位姿增量 + 1 维夹爪指令。来源：hdf5 `action/delta_ee_pose` (6,) 拼 `action/gripper_cmd` (1,)。

### Observation State —— 14D

`OBS_STATE_KEYS`（native layout）顺序：

```
joint_1.pos ... joint_7.pos,           # 索引 0-6   ：7 关节位置
ee_pose.x, ee_pose.y, ee_pose.z,        # 索引 7-9   ：末端位置
ee_pose.rx, ee_pose.ry, ee_pose.rz,     # 索引 10-12 ：末端姿态
gripper_norm                            # 索引 13    ：夹爪归一化开度
```

= 7 关节 + 6 末端位姿 + 1 夹爪。来源：hdf5 `observations/arm/joints` (7,) + `observations/arm/pose` (6,) + `observations/effector/position_norm` (1,)。

> **realman layout**（`hdf5_to_lerobot_v21.py --state-layout realman`）：仅重排列序为 `joints(0-6) + gripper(7) + ee_pose(8-13)` 并改用 realman 命名（`joint_{i}_rad` / `gripper_open` / `eef_pos_*` / `eef_rot_euler_*`）。action 恒 7D 不变。

LeRobot frame dict 键：`"action"` (float32, 7,)、`"observation.state"` (float32, 14,)、`"observation.images.{cam}"` (uint8 HWC)、`"task"` (str)。

---

## 5. hdf5 → LeRobot 转换流程

两个独立转换器，输出不同 LeRobot 版本：

### 5.1 hdf5_to_lerobot.py → LeRobot v3.0

```bash
python scripts/tools/hdf5_to_lerobot.py \
    --in <hdf5_dir> --repo-id local/franka_x --fps 30 \
    --root <out_dir> --task "任务描述"
```

流程：用首个合规 episode 定 features（含实际图像 H/W）→ `LeRobotDataset.create()` → 逐 episode `validate_episode` 校验（不合规跳过并 warn）→ 逐帧 `add_frame` → `save_episode`。依赖 franka2 本机 lerobot（v3.0）。

### 5.2 hdf5_to_lerobot_v21.py → LeRobot v2.1

```bash
python scripts/tools/hdf5_to_lerobot_v21.py \
    --in-dir <hdf5_dir> --out <out_dir> \
    --fps 30 --task "任务描述" --robot-type franka --state-layout native
```

独立实现，**不依赖任何版本 lerobot**。模块结构：

- `v21_meta`：构建 `meta/info.json`、`tasks.jsonl`、`episodes.jsonl`、`episodes_stats.jsonl`。
- `v21_parquet`：episode 帧写入 `data/chunk-000/episode_NNNNNN.parquet`（state/action + 5 元列，无图像列）。
- `v21_video`：episode 视频导出（ffmpeg `libx264` `yuv420p`，同 realman 实测参数；始终用 libx264 规避 SVT-AV1 <32px 崩溃；原子写 `.mp4.tmp` → `os.replace`）。
- `convert`：串联 meta/parquet/video。`fps` 必须为整数；不合规 episode 预校验跳过、不分配输出索引；通过校验但处理失败则 fail-loud 中止；首个合规 episode 定相机名与尺寸，后续 episode 校验一致性。

---

## 6. v3.0 与 v2.1 的差异

| 维度 | LeRobot v3.0（`hdf5_to_lerobot.py`） | LeRobot v2.1（`hdf5_to_lerobot_v21.py`） |
|---|---|---|
| 产出方式 | franka2 本机 lerobot 库直转 | 独立实现，不依赖 lerobot |
| meta 文件 | `tasks.parquet` / `episodes/` / `stats.json` | `info.json` / `tasks.jsonl` / `episodes.jsonl` / `episodes_stats.jsonl` |
| 数据文件 | `file-NNN.parquet`（chunk 分组） | `data/chunk-000/episode_NNNNNN.parquet` |
| 视频 | `videos/{key}/chunk/` | `videos/chunk-000/observation.images.{cam}/` |
| 互通性 | lerobot 对 `codebase_version` 强校验，**v3.0 与 v2.1 不互通** | 同左 |
| 适用管线 | franka2 本机训练 / 可视化 | 既有 RoboCOIN `visualize_dataset` / realman 参考集 / GR00T `modality.json` |
| state layout | native（OBS_STATE_KEYS 原序） | `--state-layout` 可选 `native` / `realman` |

> 用户既有训练/可视化管线均为 v2.1。v3.0 转换器保留产 v3.0 不改；需要 v2.1 时用独立 v2.1 转换器。

---

*相关文档：[architecture.md](architecture.md)、[development-guide.md](development-guide.md)。*
