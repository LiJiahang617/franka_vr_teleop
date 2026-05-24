# QUICKSTART

> 5 分钟从 0 到第一条录制。前置已读 [README.md](../README.md)（Requirements 段）。

---

## 1. 前置硬件检查

- [ ] Franka FR3 / Panda 通电，Franka Desk 已**解锁 joints** 且 **FCI 激活**（顶栏右侧绿色机器人图标）
- [ ] Franka Hand 夹爪 USB 连接（独立于 FCI）
- [ ] 2 路 RealSense D435 USB 接上，可被识别：
  ```bash
  lerobot-find-cameras realsense   # 应列出两条 serial
  ```
- [ ] Quest 头显 USB 连机器，ADB 信任：
  ```bash
  adb devices                       # 应看到 device 状态（非 unauthorized）
  ```
- [ ] **物理急停在手边**，UI 不接管急停

---

## 2. 安装与环境

```bash
# clone 后, 在 venv (推荐 Python 3.10) 内 editable 安装三个包
pip install -e .
pip install -e lerobot_robot_franka
pip install -e lerobot_teleoperator_franka

# Polymetis 侧用独立 conda 环境 (fairo-franka build 出来的)
# 把这两个 env var 指向你的本机路径:
export POLYMETIS_ENV=/path/to/your/polymetis-local         # conda env 根
export POLYMETIS_SOURCE=/path/to/your/fairo-franka         # 含 polymetis/polymetis/python
```

---

## 3. 一键启动三进程服务

Franka 实时控制由 Polymetis 提供，三进程互相依赖时序：**臂 50051 → zerorpc 4242 → 夹爪 50052**。

```bash
bash scripts/services/_run_polymetis_rw.sh   > /tmp/polymetis.log 2>&1 &
bash scripts/services/_run_zerorpc_iface.sh  > /tmp/zerorpc.log   2>&1 &
bash scripts/services/_run_gripper.sh        > /tmp/gripper.log   2>&1 &
```

等约 30 秒后验三端口都 LISTEN：

```bash
ss -ltn | grep -E '50051|4242|50052'
# 看到 3 行 LISTEN 即 OK
```

任一端口缺失：先看对应 `/tmp/*.log` 排错（最常见：FCI 未激活、libfranka 版本不匹配、conda env 未就位）。

---

## 4. 改 yaml 配置

```bash
cp scripts/config/record_cfg_unityvr.yaml my_cfg.yaml
```

关键字段（其他保持默认即可）：

| 字段 | 含义 | 示例 |
|---|---|---|
| `record.fps` | 录制帧率 (Hz)，相机/循环/写盘同源 | `15`（RealSense 友好），`30` 也可 |
| `record.out_dir` | hdf5 落盘目录 | `./recordings` |
| `record.cameras.wrist.serial` | 手腕相机 RealSense 序列号 | `lerobot-find-cameras realsense` 拿到 |
| `record.cameras.exterior.serial` | 外部相机序列号 | 同上 |
| `record.cameras.*.{width,height,fps,rotate_deg}` | 分辨率 / 帧率 / 旋转 | `640×480@15`，rotate_deg 视实际安装方向 |
| `record.robot.home_joint_position` | 回 Home 关节角 (7 维 rad) | 用 Franka Desk 教一个安全位置后拷过来 |
| `record.teleop.unityvr_config.trigger_threshold` | VR 食指扳机激活阈值 \[0..1\] | `0.85` |
| `record.teleop.unityvr_config.pose_scaler` | \[位置增益, 姿态增益\] 全局 | `[3.0, 2.0]` |
| `record.task.description` | 任务自然语言描述（写入 hdf5 / lerobot） | `"pick up the cube"` |
| `record.async_saver_maxsize` | 后台保存队列深度 | `5` |

---

## 5. 启 Web UI 录制

```bash
python scripts/core/run_record_hdf5_ui.py --config my_cfg.yaml
```

浏览器开 `http://<host>:5055`（同机访问用 `127.0.0.1`，远端用机器 IP）：

1. 点 **启用 VR 控制**（按钮变绿，后台启 adb logcat reader）
2. 戴 Quest，**按住右食指扳机**进入跟手模式（松开扳机 = 暂停跟手，机械臂不动）
3. 点 **开始录制** → 顶部 `state` 变 `recording`，`frame_count` 上涨
4. 操作完成 → 点 **结束并保存** 写 `.h5` 到 `out_dir`（或 **丢弃** 不要这条）
5. 录多条：重复 3-4。**回 Home** 按钮可在 episode 间手动回示教位

---

## 6. 转 LeRobot 数据集

**LeRobot v2.1**（对接 RoboCOIN / Realman 训练管线）：

```bash
python scripts/tools/hdf5_to_lerobot_v21.py \
    --in-dir ./recordings \
    --out /tmp/dataset_v21 \
    --fps 15 \
    --task "pick up the cube and place it in the target zone" \
    --robot-type franka --state-layout native
```

**LeRobot v3.0**（HuggingFace 主线）：

```bash
python scripts/tools/hdf5_to_lerobot.py \
    --in ./recordings \
    --repo-id local/franka_demo \
    --root /tmp/dataset_v30 \
    --fps 15 \
    --task "pick up the cube and place it in the target zone"
```

> v3.0 与 v2.1 **不互通**，按下游训练/可视化管线需要选择其一。

---

## 7. 可视化验证

```bash
python scripts/core/run_visualize.py \
    --root /tmp/dataset_v21 \
    --repo-id local/franka_demo \
    --episode-index 0 --save 1 \
    --output-dir /tmp/dataset_v21/rrd

rerun /tmp/dataset_v21/rrd/local_franka_demo_episode_0.rrd
```

Rerun 窗口里看双相机时序 + joint trace + ee_pose。

---

## 8. 故障排除

| 症状 | 原因 / 修复 |
|---|---|
| RealSense `Couldn't resolve requests` | `width/height/fps` 不是 RealSense 支持组合。640×480@15/30、1280×720@30 是安全档位 |
| UI 报 "no controller running" | 先点 **启用 VR 控制** 再点 **开始录制**；或 polymetis controller 未启，重起 `_run_polymetis_rw.sh` |
| `frame_count` 不涨 | 多半 VR 没数据：`adb logcat \| grep RIGHT_POSE` 看有无；Quest Unity app 没在前台跑也不会有 |
| `adb logcat` 累积进程 | 退 UI 时会自动清；手动残留用 `pkill -f "adb logcat"` |
| 夹爪不响应 | 端口 50052 没起；`bash scripts/services/_run_gripper.sh` 重起，看 `/tmp/gripper.log` |
| 救命：机械臂动作异常 | **按物理急停**。UI 不绕过 FCI，急停立即剥夺控制权 |

---

## 9. 末端负载标定（换工具后必做）

换末端工装（夹具、新工具）后，Polymetis 的笛卡尔阻抗会因质量参数不准而漂。流程：

1. UI 点 **负载标定** → 弹窗确认风险（机械臂会自走 17 个位姿，约 4-5 分钟）
2. 等流程跑完，日志输出末端质量 `m` (kg) 与质心 `c_flange` (3 维, m)
3. 把 `m` 与 `c_flange` 填到 **Franka Desk → End Effector → Load**
4. **重起 `_run_polymetis_rw.sh`** 让 Polymetis 重读 Franka Hand 配置

不做这一步：录制依然能跑，但增量 EE 会有静态误差（重力补偿不准）。
