# Franka VR Teleoperation & Data Collection

Quest VR 世界系遥操 Franka Research 3 机械臂的数据采集系统，自动录制为自定义 `franka-hdf5-v2` 中间格式，再转 LeRobot **v3.0 / v2.1** 数据集用于训练与可视化。

衍生自 [Shenzhaolong1330/lerobot_franka_teleop](https://github.com/Shenzhaolong1330/lerobot_franka_teleop)，重写为 **Route B 世界系遥操**方案：控制器位姿在 Unity 世界系下解算，与头显朝向解耦。

---

## Features

- Quest VR 世界系遥操（右手食指扳机 hold-to-enable）
- Web UI 控制（开始 / 保存 / 丢弃 / 回 Home / 负载标定 / 实时状态轮询）
- 实时双 RealSense 预览（wrist + exterior）+ frame_count + saver 队列水位
- 异步落盘 `franka-hdf5-v2`，录制循环不被 IO 阻塞，队列背压有界
- 一键转 LeRobot **v3.0**（HF 主线）或 **v2.1**（对接 RoboCOIN / Realman 训练）
- 末端负载在线辨识（17 位姿×双向逼近，输出 m / c_flange 直填 Franka Desk）
- 启动期预检门：夹爪健康检查 + 色彩通道序（BGR/RGB）校验
- 物理急停优先的安全设计：UI 不绕过 FCI，所有真机动作可随时打断
- pytest 离线测试套件 **632 用例**（纯逻辑、不需真机）

---

## Architecture

三进程服务架构（采集环境与 Polymetis 环境解耦）：

```
┌────────────────────────────────────────────────────────────────┐
│  采集侧 (venv: franka-teleop, Python 3.10)                       │
│   run_record_hdf5_ui.py                                         │
│     ├── Franka(Robot)      ── FrankaInterfaceClient ──┐         │
│     ├── UnityVRTeleop      ── adb logcat Quest        │         │
│     └── RealSense ×2 (wrist + exterior)               │         │
└────────────────────────────────────────────────────────┼───────┘
                                            zerorpc :4242
┌────────────────────────────────────────────────────────┼───────┐
│  Polymetis 侧 (conda: polymetis-local)                  ▼      │
│   FrankaInterfaceServer ────┬──── RobotInterface  :50051       │
│                             └──── GripperInterface :50052      │
└────────────────────────┬──────────────────────┬───────────────┘
                         │                      │
                  Franka Hand (USB)      Franka FR3 (FCI, RT 内核)
```

- **机械臂控制**: libfranka 0.20.x + Polymetis（笛卡尔阻抗增量 EE，DROID 思路）
- **遥操作源**: ADB logcat 读 Quest Unity app 的 RIGHT_POSE
- **坐标映射**: 位置极矢量 / 旋转赝矢量分离，固定 base↔oc 矩阵（`unityvr_mapping.py`）
- **数据落盘**: `franka-hdf5-v2` schema 冻结契约（`franka_hdf5_schema.py`），与 LeRobot 版本解耦
- **格式转换**: `scripts/tools/hdf5_to_lerobot.py`（v3.0）/ `hdf5_to_lerobot_v21.py`（v2.1）

详见 [docs/architecture.md](docs/architecture.md)。

---

## Requirements

### 硬件

- Franka Research 3 / Panda + Desk（FCI 已激活）
- Franka Hand 夹爪
- RealSense D435 ×2（手腕 + 外部第三视角）
- Meta Quest 2 / 3，USB 接电脑（ADB 调试已开启）
- 物理急停开关

### 软件

- Ubuntu 22.04 + PREEMPT_RT 实时内核（`uname -r` 应包含 `-rt`）
- libfranka 0.20.x（建议 deb 包安装）
- Python 3.10（采集侧）+ conda（Polymetis 侧）
- Polymetis（fairo-franka fork，需 build 出 `launch_robot.py` / `launch_gripper.py` / `launch_server.py`）
- LeRobot（HuggingFace 主线）
- `adb`（Android platform-tools）

---

## Quick Install

详见 [docs/QUICKSTART.md](docs/QUICKSTART.md)（5 分钟从 0 到录第一条数据）。

最小流程：

```bash
# 1. 安装本仓 (editable)
pip install -e .
pip install -e lerobot_robot_franka
pip install -e lerobot_teleoperator_franka

# 2. 拷贝并改 yaml
cp scripts/config/record_cfg_unityvr.yaml my_cfg.yaml
# 改 cameras.{wrist,exterior}.serial、home_joint_position、task.description

# 3. 启 Polymetis 三进程 (需先 export POLYMETIS_ENV / POLYMETIS_SOURCE)
bash scripts/services/_run_polymetis_rw.sh   > /tmp/polymetis.log 2>&1 &
bash scripts/services/_run_zerorpc_iface.sh  > /tmp/zerorpc.log   2>&1 &
bash scripts/services/_run_gripper.sh        > /tmp/gripper.log   2>&1 &

# 4. 启 Web UI (浏览器开 http://<host>:5055)
python scripts/core/run_record_hdf5_ui.py --config my_cfg.yaml
```

---

## Project Structure

```
lerobot_franka_teleop/
├── franka_hdf5_schema.py            # franka-hdf5-v2 schema + validate_episode
├── lerobot_robot_franka/            # 子包: Franka 机器人侧 (LeRobot Robot 接口)
│   └── lerobot_robot_franka/
│       ├── franka.py                #   Franka(Robot): connect/get_obs/send_action/reset
│       ├── franka_interface_server.py  # zerorpc 服务端 (Polymetis 环境跑)
│       └── franka_interface_client.py  # zerorpc 客户端 (采集环境跑)
├── lerobot_teleoperator_franka/     # 子包: Quest VR 遥操作 (LeRobot Teleoperator 接口)
│   └── lerobot_teleoperator_franka/
│       ├── unityvr_teleop.py        #   Route B 世界系遥操作主体
│       ├── unityvr_mapping.py       #   世界系 → base 纯 delta 映射 (可单测)
│       ├── unity_vr_reader.py       #   adb logcat 解析 Quest Unity app 流
│       └── vr_align.py              #   VR↔Franka 坐标对齐 (Kabsch)
├── scripts/
│   ├── core/                        # 录制 / 回放 / 可视化入口
│   │   ├── run_record_hdf5_ui.py    #   Web UI 录制 (主入口, 推荐)
│   │   ├── run_record_hdf5.py       #   键盘录制 (无 UI)
│   │   ├── run_replay.py            #   回放 hdf5
│   │   └── run_visualize.py         #   LeRobot 数据集 rerun 可视化
│   ├── ui/                          # Flask UI + 控制器 + 模板
│   ├── services/                    # Polymetis 三进程启动脚本 (*.sh)
│   ├── tools/                       # hdf5→lerobot 转换器 + 数据集检查工具
│   ├── config/                      # record_cfg_unityvr.yaml (Route B 录制配置)
│   └── utils/                       # 辅助工具
├── docs/
│   ├── QUICKSTART.md                # 5 分钟从 0 到第一条录制
│   ├── architecture.md              # 系统架构与原理
│   ├── data-format.md               # franka-hdf5-v2 schema + LeRobot 转换
│   ├── development-guide.md         # 开发说明 / 测试 / 扩展规范
│   └── lessons/                     # 踩坑教训沉淀
└── tests/                           # pytest 离线测试 (632 用例)
```

---

## Usage

### 录第一条数据（最小示例）

```bash
# 已启好三进程服务的前提下
python scripts/core/run_record_hdf5_ui.py --config scripts/config/record_cfg_unityvr.yaml
```

浏览器开 `http://<host>:5055`：

1. 点 **启用 VR 控制**
2. 戴 Quest，**按住右食指扳机**进入跟手模式
3. 点 **开始录制** → `state=recording` → `frame_count` 上涨
4. 操作完成 → 点 **结束并保存**（或 **丢弃** 不要这条）
5. 切回终端，转 LeRobot：

```bash
python scripts/tools/hdf5_to_lerobot_v21.py \
    --in-dir ./recordings \
    --out /tmp/dataset_v21 \
    --fps 15 --task "pick up the object and place it in the target zone"
```

更详细的步骤、参数说明、故障排除见 [docs/QUICKSTART.md](docs/QUICKSTART.md)。

---

## Documentation

| 文档 | 用途 |
|---|---|
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | 5 分钟从 0 到第一条录制 |
| [docs/architecture.md](docs/architecture.md) | 系统架构、三进程拓扑、全链路数据流 |
| [docs/data-format.md](docs/data-format.md) | `franka-hdf5-v2` schema、LeRobot 转换差异 |
| [docs/development-guide.md](docs/development-guide.md) | 测试、扩展功能、调试陷阱 |
| [docs/lessons/](docs/lessons/) | 踩坑教训（动手前必读对应条目） |

---

## License

[Apache License 2.0](LICENSE)

---

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) — Hugging Face 机器人学习框架，Robot/Teleoperator 接口契约
- [Polymetis](https://github.com/facebookresearch/fairo) — Meta / facebookresearch，Franka 实时控制后端
- [libfranka](https://github.com/frankaemika/libfranka) — Franka Emika FCI C++ 客户端
- [DROID](https://github.com/droid-dataset/droid) — 笛卡尔阻抗增量 EE 控制流思路来源
- 衍生自 [Shenzhaolong1330/lerobot_franka_teleop](https://github.com/Shenzhaolong1330/lerobot_franka_teleop) —— 在其骨架上重写 Route B 世界系遥操方案与录制管线
