# NOTICE — 项目出处与依赖

本项目（Franka Route B 世界系 VR 遥操数采）**魔改自** 上游框架
`github.com/Shenzhaolong1330/franka_vr_teleop`（其 git 历史保留作出处；
本地已移除 origin remote，仅本地版本管理，不向上游推送）。

在其 `lerobot_robot_franka` / `lerobot_teleoperator_franka` / `scripts` 骨架上，
新增/魔改：世界系 Unity 源（`unity_vr_reader.py`）、VR↔base 固定坐标映射
（`vr_align.py` + `unityvr_mapping.py`，位置极矢量/旋转赝矢量分离，矩阵经
Realman vr_utils 参考 + 真机实测双重印证）、hdf5 录制（`franka-hdf5-v1`）与
lerobot 转换。控制流为 DROID 遥操思路（zerorpc 笛卡尔阻抗增量 EE）。

## 外部运行期依赖（不入本仓，需 jhli 环境就位）
- polymetis（fairo-franka fork）+ conda `jhli/envs/polymetis-local`：臂/夹爪 FCI 三进程
- lerobot（HuggingFace）+ conda `jhli/envs/franka-teleop`：客户端/数据集
- `jhli/platform-tools/adb`：USB 读 Quest 世界系 Unity 流
- DROID（`jhli/droid`，控制流思路来源）/ Realman ROS2 teleop（VR 映射思路参考，仅本地工作站）
- 硬件：Franka FR3 + Desk(FCI/Franka Hand)、Quest 头显、RealSense 双相机
