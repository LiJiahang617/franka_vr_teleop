# 数据格式转换 & 可视化

录制完的 `.h5` 文件需要转换成 LeRobot 标准格式 (v2.1 推荐, v3.0 实验性) 才能
用于训练或可视化. 本文档覆盖完整 pipeline.

## 数据流

```
recordings/*.h5    -- UI 录制产物 (franka-hdf5-v2 自定义格式)
       |
       v  scripts/tools/hdf5_to_lerobot_v21.py   (推荐 v2.1)
       v  scripts/tools/hdf5_to_lerobot.py        (v3.0 实验)
       v
LeRobot 标准格式 (parquet + mp4)
       |
       v  scripts/core/run_visualize.py
       v
*.rrd    rerun 桌面打开
```

## 选哪个版本?

- **v2.1**: 较早的标准格式; 每 episode 一个 parquet + 每路 cam 一个 mp4.
  目前部分训练栈/工具仍只支持 v2.1.
- **v3.0**: LeRobot 后续设计 (所有 episode 合并到单 parquet + 单 mp4 chunk);
  **当前 lerobot 主线代码 (本项目依赖) 已用 v3.0**, 
  调 LeRobotDataset 加载, 必须给 v3.0 数据.

推荐做法:
1. 先 v2.1 (LeRobotDataset 不兼容时, 用 v2.1 mp4 直接看视频 + parquet 离线分析)
2. 再 v3.0 (跑 run_visualize 生成 rerun .rrd)

## 转换 LeRobot v2.1

v2.1 是较早的 LeRobot 标准格式; 部分老训练栈/工具仍只读 v2.1. 每个 episode 一个 parquet + 每路 cam 一个 mp4.

```bash
cd $FRANKA_TELEOP_ROOT  # 项目根
$POLYMETIS_ENV/../franka-teleop/bin/python scripts/tools/hdf5_to_lerobot_v21.py \
    --in-dir ./recordings \
    --out /tmp/dataset_v21 \
    --fps 15 \
    --task "your task description"
```

参数:
- `--in-dir`: 含 .h5 文件的目录 (UI 录制默认 `./recordings`)
- `--out`: 输出根目录
- `--fps`: 帧率 (跟 yaml `record.fps` 一致)
- `--task`: 任务描述 (写入 LeRobot tasks.jsonl)

产物:
```
/tmp/dataset_v21/
  data/chunk-000/
    episode_000000.parquet
    episode_000001.parquet
  videos/chunk-000/
    observation.images.wrist_image/
      episode_000000.mp4
    observation.images.exterior_image/
      episode_000000.mp4
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    episodes_stats.jsonl
```

## 转换 LeRobot v3.0 (run_visualize 必需)

v3.0 是 LeRobot 当前主线设计 (本项目依赖 lerobot 版本即 v3) (所有 episode 合并到单 parquet + 单 mp4 chunk).

```bash
$POLYMETIS_ENV/../franka-teleop/bin/python scripts/tools/hdf5_to_lerobot.py \
    --in ./recordings \
    --repo-id local/franka_dataset \
    --fps 15 \
    --root /tmp/dataset_v30 \
    --task "your task description"
```

## Rerun 可视化

LeRobot v2.1 / v3.0 数据都可以转 rerun `.rrd` 文件本机或远端 rerun 桌面打开.

### 1. 生成 .rrd 文件

```bash
$POLYMETIS_ENV/../franka-teleop/bin/python scripts/core/run_visualize.py \
    --root /tmp/dataset_v21 \
    --repo-id local/franka_dataset \
    --episode-index 0 \
    --save 1 \
    --output-dir /tmp/dataset_v21/rrd
```

参数:
- `--root`: LeRobot 数据集根目录 (v2.1 或 v3.0 都行)
- `--repo-id`: yaml `record.repo_id` 同源
- `--episode-index`: 要可视化的 episode 0-based 索引
- `--save 1`: 保存 .rrd 文件 (不弹窗)
- `--output-dir`: .rrd 输出目录

产物: `/tmp/dataset_v21/rrd/local_franka_dataset_episode_0.rrd`

### 2. 打开 .rrd

本机看 (推荐, native 体验):
```bash
# scp 到本机
scp <franka-machine>:/tmp/dataset_v21/rrd/local_franka_dataset_episode_0.rrd /tmp/
# 本机已装 rerun (pip install rerun-sdk)
rerun /tmp/local_franka_dataset_episode_0.rrd
```

远端打开 (需 X11 forwarding 或 VNC):
```bash
$POLYMETIS_ENV/../franka-teleop/bin/rerun /tmp/dataset_v21/rrd/local_franka_dataset_episode_0.rrd
```

### 3. 批量看多个 episode

```bash
for i in 0 1 2; do
    $POLYMETIS_ENV/../franka-teleop/bin/python scripts/core/run_visualize.py \
        --root /tmp/dataset_v21 \
        --repo-id local/franka_dataset \
        --episode-index $i \
        --save 1 \
        --output-dir /tmp/dataset_v21/rrd
done
ls /tmp/dataset_v21/rrd/*.rrd
```

## 常见问题

### `[align] effector hw_timestamp 残差过大, 退回 effector_ts`

夹爪硬件时戳偶尔卡死同值 (libfranka 设计), 残差 > 50ms 时自动 fallback 到
软件时戳. 不影响数据可用性, 见 `docs/lessons/2026-05-23-polymetis-gripper-width-feedback-lag.md`.

### 转换后 mp4 是空白 / 黑屏

检查 yaml `record.color_preflight: true`, 启动期色彩通道序预检会拦截
BGR/RGB 误用. 也可能 yaml 改了 `rotate_deg` 但视频未旋转 (lerobot v2.1
converter 已 rotate, v3.0 待验证).

### rerun 看不到图像 / 看到所有 episode 重叠

- 确认 `--episode-index <N>` 跟数据集匹配
- v3.0 数据集单 mp4 含所有 episode, rerun viewer 用 episode 时间窗切片
- 重启 rerun viewer

## 训练 (后续)

转好的 LeRobot v2.1 数据集可以直接喂训练:
```bash
cd <LEROBOT_REPO>
python -m lerobot.scripts.train \
    --dataset.repo_id=local/franka_dataset \
    --dataset.root=/tmp/dataset_v21 \
    --policy.type=act \
    --output_dir=outputs/franka_act_test
```

具体训练参数见 LeRobot 官方文档.
