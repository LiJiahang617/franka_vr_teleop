# 系统架构与原理

> Franka 数采系统（Route B）—— Quest VR 世界系遥操 Franka，录制 `franka-hdf5-v1`，转 LeRobot 数据集。

## 目录

- [1. 系统概述](#1-系统概述)
- [2. 仓库目录结构](#2-仓库目录结构)
- [3. 三进程服务架构](#3-三进程服务架构)
- [4. 数据采集全链路数据流](#4-数据采集全链路数据流)
- [5. scripts/core 各模块详解](#5-scriptscore-各模块详解)
- [6. Route B 遥操方案](#6-route-b-遥操方案)
- [7. 数据预检门与异步保存](#7-数据预检门与异步保存)
- [8. 配置系统](#8-配置系统)
- [9. 关键设计决策](#9-关键设计决策)

---

## 1. 系统概述

本仓库（`lerobot_franka_teleop`）是 Franka Research 3 机械臂的**遥操作数据采集系统**。当前主线为 **Route B**：

> **Route B = Quest VR 世界系遥操 Franka → 录制 `franka-hdf5-v1` → 转 LeRobot 数据集**

与历史方案（主从同构关节映射 / SpaceMouse / Oculus 头显系）相比，Route B 的核心特征：

- **世界系遥操**：Quest 控制器位姿在 Unity 世界系下解算，与头显朝向无关。用户戴头显面朝机械臂 base +X 方向，长按 Meta 键重置世界系即可（±几度可接受，无需 SVD 标定）。
- **中间格式 `franka-hdf5-v1`**：录制阶段先写自定义 HDF5 schema（冻结契约），与 LeRobot 版本解耦；再由独立转换器产出 LeRobot 数据集。
- **采集与落盘解耦**：录制循环只采帧，异步后台线程串行写盘 + 校验，背压有界。

全链路：

```
数据采集（VR 遥操 + 录制）  →  franka-hdf5-v1 (.h5)  →  转换器  →  LeRobot 数据集（v3.0 或 v2.1）  →  可视化 / 训练
```

---

## 2. 仓库目录结构

```
lerobot_franka_teleop/
├── franka_hdf5_schema.py          # franka-hdf5-v1 schema 契约 + validate_episode（仓库根 loose 模块）
├── setup.py                       # 主包安装 + console_scripts 入口
├── NOTICE.md                      # 第三方代码出处声明
│
├── lerobot_robot_franka/          # 子包①：Franka 机器人侧（LeRobot Robot 接口实现）
│   └── lerobot_robot_franka/
│       ├── franka.py                       # Franka(Robot)：connect/get_observation/send_action/reset
│       ├── config_franka.py                # FrankaConfig
│       ├── franka_interface_server.py      # zerorpc 服务端（运行在 polymetis 环境，端口 4242）
│       └── franka_interface_client.py      # zerorpc 客户端（运行在 franka-teleop 环境）
│
├── lerobot_teleoperator_franka/   # 子包②：遥操作设备侧（LeRobot Teleoperator 接口实现）
│   └── lerobot_teleoperator_franka/
│       ├── unityvr_teleop.py               # UnityVR 世界系遥操作（Route B 主线）
│       ├── unityvr_robot.py                # UnityVR teleop 的 robot 适配层
│       ├── unityvr_mapping.py              # 世界系 → base 的纯 delta 映射（可单测）
│       ├── unity_vr_reader.py              # adb logcat 读取 Unity app 的 VR 位姿
│       ├── vr_align.py                     # VR↔Franka 坐标系对齐（Kabsch，纯数学）
│       └── teleop_factory.py               # create_teleop 工厂
│
├── scripts/
│   ├── core/                      # 录制 / 回放 / 训练 / 可视化核心入口
│   ├── tools/                     # hdf5→lerobot 转换器、数据集检查工具
│   ├── services/                  # 三进程服务的启动包装脚本（*.sh）
│   ├── config/                    # record_cfg_unityvr.yaml（Route B 录制配置）
│   ├── utils/                     # 数据集辅助工具
│   └── help/                      # franka-help 命令
│
├── tests/                         # pytest 测试（317 用例，纯逻辑离线可跑）
├── debug/                         # 诊断与运维脚本（服务起停巡检、VR 映射验证）
├── docs/
│   ├── lessons/                   # 踩坑教训
│   ├── architecture.md            # 本文件
│   ├── development-guide.md       # 开发说明
│   ├── data-format.md             # franka-hdf5-v1 schema 说明
│   └── README.md                  # docs 索引
└── assets/                        # 图片（说明文档用图）
```

> 三个 Python 包：主包 `lerobot_franka_teleop`（含 `scripts/`），子包 `lerobot_robot_franka`、`lerobot_teleoperator_franka`。三者均以 editable 方式安装在 venv `envs/franka-teleop` 中。

---

## 3. 三进程服务架构

Franka 实时控制由 [Polymetis](https://polymetis-docs.github.io/) 提供。Polymetis 不支持 Python 3.10，且需独立实时环境，因此采集侧（`franka-teleop` 环境）通过 **zerorpc** 与 Polymetis 侧（`polymetis-local` 环境）通信。

服务由 `scripts/services/` 下三个包装脚本启动，全部进入 conda 环境 `/home/ubuntu/Desktop/jhli/envs/polymetis-local`，工作目录 `fairo-franka/.../python/scripts`：

| 服务脚本 | 端口 | 启动的进程 | 作用 |
|---|---|---|---|
| `_run_polymetis_rw.sh` | **50051** | `launch_robot.py robot_client=franka_hardware`（readonly=false, RT） | Franka 机械臂实时控制服务（polymetis run_server）。启动后台自动起默认笛卡尔阻抗 `Kx=[100,100,100,40,40,40]`。 |
| `_run_zerorpc_iface.sh` | **4242** | `launch_server.py`（`FrankaInterfaceServer`） | zerorpc 接口层：包装 polymetis `RobotInterface` / `GripperInterface`，对采集侧暴露 RPC。 |
| `_run_gripper.sh` | **50052** | `launch_gripper.py gripper=franka_hand` | Franka Hand 夹爪服务（polymetis gripper server）。 |

> 三进程启动时序：先臂（50051）→ 再 zerorpc（4242）→ 再夹爪（50052）。

### 进程拓扑

```
┌─────────────────────────────────────────────────────────────────────┐
│  采集侧进程  (conda env: envs/franka-teleop)                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  run_record_hdf5.py                                          │   │
│  │    ├── Franka(Robot)  ── FrankaInterfaceClient ──┐           │   │
│  │    ├── UnityVRTeleop  ── UnityVRRobot ───────────┤           │   │
│  │    └── RealSense cameras（wrist / exterior）     │           │   │
│  └──────────────────────────────────────────────────┼──────────┘   │
└─────────────────────────────────────────────────────┼──────────────┘
                                       zerorpc tcp://127.0.0.1:4242
                                                       │
┌──────────────────────────────────────────────────────┼──────────────┐
│  Polymetis 侧进程  (conda env: envs/polymetis-local)  │              │
│                                                       ▼              │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  _run_zerorpc_iface.sh  →  launch_server.py                │     │
│  │     FrankaInterfaceServer  (zerorpc, :4242)                │     │
│  │       ├── RobotInterface   ───────────┐                    │     │
│  │       └── GripperInterface ───────┐   │                    │     │
│  └───────────────────────────────────┼───┼────────────────────┘     │
│                                      │   │                          │
│  ┌──────────────────────────┐  ┌─────┼───┴──────────────────────┐   │
│  │ _run_gripper.sh          │  │ _run_polymetis_rw.sh           │   │
│  │  launch_gripper.py       │  │  launch_robot.py               │   │
│  │  Franka Hand server      │  │  franka_hardware run_server    │   │
│  │  :50052                  │  │  :50051  (RT 内核, libfranka)  │   │
│  └────────────┬─────────────┘  └────────────┬───────────────────┘   │
└───────────────┼────────────────────────────┼───────────────────────┘
                │                            │
            Franka Hand              Franka Research 3 (FCI)
```

> 服务起停/巡检脚本见 `debug/`（`franka_clean_restart.sh` 一键有序重起、`franka_cleanup.sh` 清净残留等）。三进程互相依赖时序：先臂（50051）→ 再 zerorpc（4242）→ 再夹爪（50052）。

---

## 4. 数据采集全链路数据流

```
┌─────────────┐   adb logcat    ┌──────────────────┐
│ Quest + VR  │ ──────────────▶ │ UnityVRReader     │  解析 RIGHT_POSE（左手系）
│ Unity app   │                 │  to_transform()   │  S=diag(1,1,-1) → 右手系 4x4
└─────────────┘                 └────────┬─────────┘
                                         │ cur_T / prev_T
                                         ▼
                              ┌────────────────────────┐
                              │ unityvr_mapping         │  compute_delta_action
                              │  位置: _POS_MAP @ Δp     │  → delta_ee_pose (6,) base 系
                              │  旋转: R_cal @ d_rot_oc  │
                              └────────┬───────────────┘
                                       │ action: delta_ee_pose.* + gripper_cmd_bin
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  record_episode()  录制循环  @ fps (默认 30Hz)                          │
│    每 tick:                                                            │
│      action = teleop.get_action()       ── VR delta + 夹爪指令          │
│      robot.send_action(action)          ── zerorpc → Franka 执行        │
│      obs    = robot.get_observation()   ── joints/joint_vel/ee_pose/    │
│                                            gripper + RealSense 图像     │
│      图像 _encode_jpg()  RGB→BGR→imencode(.jpg) → uint8 bytes           │
│      拼 frame dict 入 buf                                              │
└──────────────────────────────┬───────────────────────────────────────┘
                                │ 录完一条 episode
                                ▼
                   decide(ep)  ── 键盘: → keep / ← discard / Esc stop
                                │ keep
                                ▼
                   copy.deepcopy(payload)  ── frames + meta 整体隔离
                                │
                                ▼
                   AsyncEpisodeSaver.submit()  ── 入队即返回（非阻塞）
                                │
                ┌───────────────┴────────────────┐
                ▼  后台单线程 _save_loop          │  录制循环继续下一条
        write_episode()  → 写 .h5 (franka-hdf5-v1)
        validate_episode()  → schema 自检，不合规抛错
                │
                ▼
        _hdf5_episodes/epNNNN_<ts>.h5
                │
                ▼  转换器（离线，二选一）
   ┌────────────────────────┐   ┌──────────────────────────────┐
   │ hdf5_to_lerobot.py      │   │ hdf5_to_lerobot_v21.py        │
   │  → LeRobot v3.0 数据集  │   │  → LeRobot v2.1 数据集        │
   │  （franka2 本机 lerobot）│   │  （独立实现，对标 realman）   │
   └────────────────────────┘   └──────────────────────────────┘
                │                            │
                ▼                            ▼
        run_visualize / 训练          RoboCOIN / realman 管线
```

---

## 5. scripts/core 各模块详解

| 模块 | 作用 |
|---|---|
| `run_record_hdf5.py` | **录制主入口（终端键盘模式）**。解析 cfg → 构造 robot/teleop → 预检门 → 键盘监听 → `run_episodes` 录制循环 → `AsyncEpisodeSaver` 异步落盘，写 `franka-hdf5-v1`。 |
| `record_params.py` | 录制超参纯函数（可单测、无硬件依赖）：`resolve_record_fps`（fps 单一来源）、`extract_joint_vel`、`realsense_fps`（float→int）、`parse_reset_config`（严格 bool 解析防 yaml 引号陷阱）、`resolve_record_overrides`（CLI None 仅覆盖）。 |
| `hdf5_writer.py` | `write_episode` 模块级落盘函数 + `HDF5EpisodeWriter` 类。按 `franka-hdf5-v1` 缓冲帧、整体写盘、末尾 `validate_episode` 自检。同步/异步路径共用同一实现。 |
| `async_saver.py` | `AsyncEpisodeSaver`：有界队列 + 后台单线程。`submit()` 入队即返回；队列满抛 `QueueFullError`（不静默丢）；`close()` 无超时阻塞等排空（数据零丢优先）。 |
| `preflight.py` | §11.2 数据预检门：夹爪健康/homing 预检 + 图像色彩通道序预检。纯判据函数离线可测，硬件 IO 由注入的 probe 回调封装。 |
| `episode_keyboard.py` | `EpisodeDecider`：把 lerobot 键盘 events 翻译为 keep/discard/stop 决策。headless 时安全降级为计时保存。 |
| `schema_loader.py` | 加载仓库根 loose 模块 `franka_hdf5_schema.py` 的单一入口，复用 `sys.modules` 缓存。 |
| `paths.py` | 集中可配路径/端口常量（可被环境变量覆盖），整合期硬编码路径的单一真值。 |
| `run_visualize.py` | 数据集可视化入口（`franka-visualize`）。 |
| `run_replay.py` | 数据集回放入口（`franka-replay`）。 |
| `reset_robot.py` | 机械臂复位入口（`franka-reset`）。 |

---

## 6. Route B 遥操方案

### 6.1 unity_vr_reader.py — VR 位姿读取

通过 `adb logcat` 抓取运行在 Quest 上的 Realman 世界系 Unity app 输出的 `VRDeviceData: RIGHT_POSE` 日志行，正则解析出右手控制器位置 `t`、四元数 `r` 及按钮状态（grip / A / B / trigger）。接口与 `OculusReader.get_transformations_and_buttons()` 一致，可 drop-in 替换。`to_transform()` 将 Unity 左手系经 `S=diag(1,1,-1)` 转为右手系 4x4 位姿。

### 6.2 unityvr_mapping.py — 世界系→base 纯映射

`compute_delta_action(cur_T, prev_T, R_cal, pose_scaler, channel_signs, *, pos_axis_gain, rot_axis_gain)` 每 tick 算出 `delta_ee_pose (6,)`：

- **位置（极矢量）**：走固定矩阵 `_POS_MAP @ Δp`。`_POS_MAP = [[0,0,-1],[-1,0,0],[0,1,0]]`，与 realman `vr_utils.RIGHT_POSITION_MATRIX` 双重印证。
- **旋转（赝矢量）**：走 `R_cal @ d_rot_oc` 刚体换基（`rotvec(R_cal·ΔR·R_calᵀ) = R_cal @ rotvec(ΔR)`）。
- 位置与旋转张量性不同，**分用不同矩阵**。

每轴增益（§11.3）：`pos_axis_gain` / `rot_axis_gain` 叠加在 `pose_scaler` 之上，**仅缩放灵敏度、不改方向/手性**（与映射方向正交，红线，见 lesson `phaseC-axis-gain-orthogonal-to-mapping`）。默认全 1 时逐字等价历史两标量行为。

### 6.3 vr_align.py — 坐标系对齐

纯数学核心：`solve_rotation`（Kabsch SVD 解旋转，强制 det=+1 真旋转）、`validate_rotation`（正交性 + det 校验）、`gesture_pair_quality`（手势对质量评估）。

> 当前会话流程：戴头显面朝 base +X，长按 Meta 重置世界系即可（±几度可接受），`R_cal` 实际为固定坐标映射（非 SVD 标定）；`vr_align` 的 Kabsch 求解能力保留备用。

### 6.4 unityvr_teleop.py / unityvr_robot.py

`UnityVRTeleop`（镜像 `OculusTeleop`，头显朝向无关）通过 `UnityVRRobot` 适配层封装 reader + mapping，`action_features` 暴露 `delta_ee_pose.{x,y,z,rx,ry,rz}` + `joint_{1..7}.pos` + `gripper_cmd_bin`。仅当 RG（右 grip）按下时才记录运动。

---

## 7. 数据预检门与异步保存

### 7.1 预检门（preflight.py）

录制入口在 `robot.connect()` 之后、录制循环之前运行预检，任一不过即 `sys.exit(2)`——把"中途静默失败"变成"启动期可行动报错"（开录前约 10s 拦截）。

- **夹爪预检**（`use_gripper=True` 时）：
  - 进程强存活：`pgrep -f franka_hand_client`（**非**端口 `:50052` LISTEN）。
  - 连接就绪：`_gripper_live.log` 出现 `Connected.`。
  - width 真变：`gripper_goto` 后轮询 `is_moving` 直到 settle，量多目标 `span = max - min > 0.02 m`（用整体跨度，禁相邻差假阴性判据）。
- **色彩预检**（`color_preflight: true` 时）：采首帧编码→解码，判 RealSense BGR/RGB 是否误判（防黄变青反色）。

### 7.2 异步保存（async_saver.py）

`AsyncEpisodeSaver` = 有界队列（默认 `maxsize=5`）+ 后台单线程：

- `submit(path, payload)`：`put_nowait` O(1) 非阻塞，队列满抛 `QueueFullError`（不静默丢，符合快速失败背压）。
- 后台 `_save_loop` 串行调 sink = `write_episode` + `validate_episode`。
- `close()`/`__exit__`：**无超时阻塞** join 排空——必须等队列全部落盘才返回，保证零数据丢失（数据安全优先于活性）。
- payload 必须由调用方 `deepcopy`（在 buffer 复用前），本类不拷贝。

---

## 8. 配置系统

录制配置：`scripts/config/record_cfg_unityvr.yaml`（Route B 用），字段分组：

- `record`：`repo_id` / `fps`（录制频率单一来源）/ `reset_between_episodes` / `control_mode: unityvr` / `out_dir`（hdf5 输出目录）/ `color_preflight`。
- `record.depth` / `record.state_hifreq`：占位至 Phase D，当前不采集。
- `record.teleop.unityvr_config`：`pose_scaler` / `channel_signs` / `oc2base_path` / `robot_port: 4242` / `pos_axis_gain` / `rot_axis_gain`。
- `record.robot`：`ip`（NUC，`127.0.0.1`）/ `use_gripper` / `gripper_max_open`（franka hand 0.0801m）/ `execute_mode: ee_pose`。
- `record.task`：`description` / `num_episodes`。
- `record.time`：`episode_time_sec` / `reset_time_sec`。
- `record.cameras`：`wrist_cam_serial` / `exterior_cam_serial` / `width` / `height`（RealSense 支持 424/640 × 240/360）。

> 字段为单一真值；CLI 参数（`--fps` / `--episodes` / `--episode-sec` / `--out-dir` / `--task-name`）仅临时覆盖，且严格 `is None` 判断（禁 `cli or cfg` 的 falsy 误判）。`record_cfg.yaml` 为历史 LeRobot 直写链路所用。

详细字段见 `development-guide.md` 与 yaml 内注释。

---

## 9. 关键设计决策

| 决策 | 理由 |
|---|---|
| **中间格式 `franka-hdf5-v1`** | 录制与 LeRobot 版本解耦。`franka_hdf5_schema.py` 为冻结契约，改 schema 必须 bump `SCHEMA_VERSION` 并同步 writer/validator/转换器。 |
| **采集与落盘解耦 + 异步队列** | 录制循环只采帧不写盘，避免 IO 抖动破坏帧率；后台单线程串行落盘保证写顺序。 |
| **预检门前置** | 夹爪丢 homing / 色彩反色等问题在开录前约 10s 拦截，拒绝"录完才发现报废"。 |
| **fps 单一来源** | `resolve_record_fps` 统一相机 fps / 循环节拍 / hdf5 `target_fps`，杜绝三处不一致。 |
| **位置/旋转分用不同映射矩阵** | 极矢量与赝矢量张量性不同；旋转走刚体换基，位置走固定坐标映射。 |
| **每轴增益与映射方向正交** | 增益层只缩放灵敏度，绝不改方向/手性（红线）。 |
| **v3.0 / v2.1 双转换器并存** | franka2 本机 lerobot 为 v3.0；既有训练/可视化管线为 v2.1，二者不互通。v3.0 用本机 lerobot 直转，v2.1 由独立实现转换器产出。 |
| **三进程 + zerorpc** | Polymetis 需 Python<3.10 与独立实时环境，采集侧用 Python 3.10，故经 zerorpc 跨环境通信。 |
| **schema 用 loose 模块 + schema_loader** | 整合期 `franka_hdf5_schema.py` 仍是仓库根散装模块，多方消费，`schema_loader` 集中加载、复用 `sys.modules` 单实例。 |

---



---

## 10. 数采 Web UI

Phase E 新增 `scripts/ui/` 包，提供基于 Flask 的浏览器控制面板，替代终端键盘模式。

### 架构

**Flask 单进程**：主入口 `run_record_hdf5_ui.py` 在同一进程内启动 Flask HTTP 服务器（`threaded=True`，端口默认 5055）和后台录制线程。不引入 ROS service，不额外进程。

```
浏览器 (http://franka2:5055)
       ↕ HTTP (5 按钮 + /api/status 轮询 + /api/preview/*)
Flask routes (control_panel.py)
       ↕ 方法调用
RecorderController (recorder_controller.py)
  ├── events dict  ← 写入 exit_early / rerecord_episode / stop_recording
  ├── 命令队列 (queue.Queue)  ← "start" / "home"
  └── 后台录制线程  → run_episodes_fn → AsyncEpisodeSaver → HDF5
```

### 核心模块

| 模块 | 作用 |
|---|---|
| `scripts/ui/state.py` | `UIState` 枚举 + `StateMachine`（线程安全 RLock 保护，非法转移 fail-loud） |
| `scripts/ui/recorder_controller.py` | `RecorderController`：持有 events dict / 命令队列 / 状态机 / 最新帧缓存；桥接 Flask 路由与录制器后台线程 |
| `scripts/ui/control_panel.py` | `build_app(controller)`：Flask app 工厂函数，注册 6 条路由 + `@after_request` Cache-Control 钩子 |
| `scripts/ui/preview.py` | `encode_preview_jpeg`：RGB→BGR→JPEG 编码（复用 `_encode_jpg` 通道序，防反色）；`encode_preview_base64`：base64 封装供路由 JSON 返回 |
| `scripts/ui/templates/control_panel.html` | 5 按钮控制面板 + 双相机 base64 jpeg 预览 + 30Hz JS 轮询（外部文件，规避 Python 三引号 JS 换行陷阱） |
| `scripts/core/run_record_hdf5_ui.py` | UI 模式主入口：组装 robot/teleop/saver + RecorderController + Flask app，app.run(threaded=True) |

### 状态机

```
INITIALIZING → WAITING → RECORDING → CONFIRMING → SAVING → READY → (WAITING 循环)
                                          ↓（丢弃）
                                        WAITING
```

每条状态转移由 RecorderController 方法触发；非法转移抛 `IllegalTransition`（fail-loud）。

### RecorderController 桥接

- **保存/丢弃/停止**：直接写 events dict（`exit_early` / `rerecord_episode` / `stop_recording`），语义与终端键盘逐字等价，`EpisodeDecider` 消费。
- **开始/回 Home**：入命令队列，由后台录制线程串行消费（守坑 7：UI 路由禁止直调 zerorpc，防单线程并发争用）。
- **frame_observer hook**：录制循环每帧回调 `update_latest_frame(cam, rgb)`，更新共享缓存（加锁 `.copy()`）；预览路由从缓存读，不直接调 `robot.get_observation()`。

### 与终端键盘模式的关系

两者**互斥、二选一**，通过不同入口启动：

| 模式 | 入口 | 控制方式 |
|---|---|---|
| 终端键盘 | `scripts/core/run_record_hdf5.py` | `→` keep / `←` discard / Esc stop |
| Web UI | `scripts/core/run_record_hdf5_ui.py` | 浏览器 5 按钮 + 相机预览 |

`run_episodes` / `record_episode` / `AsyncEpisodeSaver` / `write_episode` 等核心链路**完全复用，零改动**。

### 关键设计约束

- **Cache-Control 红线**：`@after_request` 统一加 `no-cache, no-store, must-revalidate`（防 stale UI）。
- **HTML 模板外部文件**：规避 Python 三引号字符串内 JS `\n` 被解释为真换行导致 SyntaxError。
- **zerorpc 单线程**：UI 路由不直接调 zerorpc，all zerorpc 调用由后台录制线程串行执行。
- **接口零破坏**：`record_episode(frame_observer=None)` 默认 None = 零行为变化，既有测试全绿即证据。


*相关文档：[data-format.md](data-format.md)（schema 详解）、[development-guide.md](development-guide.md)（开发说明）、[../README.md](../README.md)（快速上手）。*
