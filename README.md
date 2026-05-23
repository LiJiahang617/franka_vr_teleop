# Franka 数采系统（Route B）

Franka Research 3 机械臂的遥操作数据采集系统：用 Quest VR 控制器在世界系下遥操 Franka，录制 `franka-hdf5-v1` 中间格式，再转换为 LeRobot 数据集用于训练与可视化。

> 详细原理见 [docs/architecture.md](docs/architecture.md)，数据格式见 [docs/data-format.md](docs/data-format.md)，开发说明见 [docs/development-guide.md](docs/development-guide.md)。

---

## 快速上手

### 1. 激活环境

采集侧使用 venv `franka-teleop`：

```bash
source /home/ubuntu/Desktop/jhli/envs/franka-teleop/bin/activate
cd /home/ubuntu/Desktop/jhli/lerobot_franka_teleop
```

### 2. 启动三个服务

Franka 实时控制由 Polymetis 提供，需先启动三进程服务（臂 50051 / zerorpc 4242 / 夹爪 50052）。在 Franka Desk 界面激活 **FCI** 后，一键有序重起：

```bash
bash debug/franka_clean_restart.sh   # 清理 → 起臂 → 验 → 起 zerorpc → 验
bash scripts/services/_run_gripper.sh &   # 再起夹爪
```

可用 `bash debug/franka_verify_zerorpc.sh` 与 `bash debug/franka_poll_gripper.sh` 验证服务就绪。

### 3. 录制一条数据

确认 `scripts/config/record_cfg_unityvr.yaml` 中的相机序列号、任务描述、episode 数等无误后：

```bash
python scripts/core/run_record_hdf5.py --config scripts/config/record_cfg_unityvr.yaml
```

录制入口会先跑预检门（夹爪健康 / 色彩通道序），通过后开始录制。键盘控制：

- `→` 结束当前 episode 并**保存**
- `←` 结束当前 episode 并**丢弃**
- `Esc` 停止录制

录制结果为 `franka-hdf5-v1` 格式的 `.h5` 文件，默认输出到 `/home/ubuntu/Desktop/jhli/_hdf5_episodes`。

#### 可选：Web UI 模式录制

先在 `scripts/config/record_cfg_unityvr.yaml` 的 `record.ui` 段设置 `enabled: true`，再启动 UI 入口：

```bash
python scripts/core/run_record_hdf5_ui.py --config scripts/config/record_cfg_unityvr.yaml
```

浏览器访问 `http://<franka2-ip>:5055` 通过按钮控制录制（保存 / 丢弃 / 停止 / 回 Home）。

> **安全提示**：`ui.host: 0.0.0.0` = 局域网可访问；仅在私网部署，不暴露公网；
> 按钮直接触发真机动作，**急停在手，随时可按**。

### 4. 转换为 LeRobot 数据集

转 **LeRobot v3.0**（franka2 本机 lerobot 直转）：

```bash
python scripts/tools/hdf5_to_lerobot.py \
    --in /home/ubuntu/Desktop/jhli/_hdf5_episodes \
    --repo-id local/franka_x --fps 30 \
    --root /home/ubuntu/Desktop/jhli/_lerobot_out \
    --task "任务描述"
```

转 **LeRobot v2.1**（对标 realman / RoboCOIN 管线）：

```bash
python scripts/tools/hdf5_to_lerobot_v21.py \
    --in-dir /home/ubuntu/Desktop/jhli/_hdf5_episodes \
    --out /home/ubuntu/Desktop/jhli/_lerobot_out_v21 \
    --fps 30 --task "任务描述" --robot-type franka --state-layout native
```

> v3.0 与 v2.1 不互通，按下游训练/可视化管线需要选择。

### 5. 可视化

```bash
franka-visualize   # 按 record_cfg_unityvr.yaml 的 visualize 段配置数据集
```

---

## 更多文档

- [docs/README.md](docs/README.md) —— 文档索引与建议阅读顺序
- [docs/architecture.md](docs/architecture.md) —— 系统架构与原理
- [docs/development-guide.md](docs/development-guide.md) —— 开发说明、测试、扩展规范
- [docs/data-format.md](docs/data-format.md) —— `franka-hdf5-v1` schema 与 LeRobot 转换
- [docs/lessons/](docs/lessons/) —— 踩坑教训

---

## 启用精确夹爪时戳（建议）

精确同步 `observation.state.gripper_norm` 与图像（消除 ~100ms 滞后）需要重 build polymetis fork：

```bash
cd /path/to/fairo-franka/polymetis/polymetis
cmake --build build --target franka_hand_client -j4   # 默认 -DENABLE_GRIPPER_HW_TIMESTAMP=ON
```

效果：`align_offline` 使用 libfranka 硬件 push 时戳精确对齐每帧 effector（时戳精度从 ±100ms 抖动降到 ~22ms 级）。不 rebuild 仍能录制/训练，只是 align 退到旧路径，会输出 warning。

详见 `docs/lessons/2026-05-23-polymetis-gripper-hw-timestamp.md`。
