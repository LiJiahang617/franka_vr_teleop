# Phase D v2 端到端真机验收清单（DEFERRED 用户现场）

> **状态**：DEFERRED——离线部分已全部就绪（602 passed），此清单待用户在 franka2 真机上逐条跑通后贴回确认。
> **前提**：三进程服务已正常运行（arm 50051 / zerorpc 4242 / 夹爪 50052）；`franka-teleop` venv 激活。

---

## 验收条件 1：重录 v2 episode

**目的**：确认新录制流程产出 franka-hdf5-v2 格式文件。

```bash
cd /home/ubuntu/Desktop/jhli/lerobot_franka_teleop
source /home/ubuntu/Desktop/jhli/envs/franka-teleop/bin/activate

python scripts/core/run_record_hdf5.py \
    scripts/config/record_cfg_unityvr.yaml \
    --episodes 1 --episode-sec 5 \
    --out-dir /tmp/phaseD_acceptance_v2
```

**期望结果**：
- 录制正常完成（无 RuntimeError / validate_episode 违规）
- `/tmp/phaseD_acceptance_v2/` 下生成 1 个 `ep0000_*.h5` 文件
- 终端打印 `write_episode` 成功，无 violations

---

## 验收条件 2：v2 schema 校验

**目的**：确认生成的 episode 符合 franka-hdf5-v2 契约。

```bash
python - << 'EOF'
import sys
sys.path.insert(0, '/home/ubuntu/Desktop/jhli/lerobot_franka_teleop')
import glob, franka_hdf5_schema as S

eps = sorted(glob.glob('/tmp/phaseD_acceptance_v2/*.h5'))
assert eps, "没有找到 .h5 文件"
ep = eps[0]
v = S.validate_episode(ep)
if v:
    print("FAIL violations:", v)
    sys.exit(1)
print(f"PASS validate_episode({ep}): no violations")
EOF
```

**期望结果**：打印 `PASS validate_episode(...): no violations`，脚本退出 0。

---

## 验收条件 3：多模态时间戳独立性检查

**目的**：确认 arm / effector / camera 的 timestamp 互相不完全相同（各自独立采集）。

```bash
python - << 'EOF'
import sys, glob
sys.path.insert(0, '/home/ubuntu/Desktop/jhli/lerobot_franka_teleop')
import h5py, numpy as np

ep = sorted(glob.glob('/tmp/phaseD_acceptance_v2/*.h5'))[0]
with h5py.File(ep, 'r') as f:
    arm_ts = f['observations/arm/timestamp'][...]
    eff_ts = f['observations/effector/timestamp'][...]
    cam_keys = sorted(f['observations/camera/rgb'].keys())
    cam_ts_0 = f[f'observations/camera/rgb/{cam_keys[0]}/timestamp'][...]

# 任意两模态 ts 数组不完全相等（独立采集时自然不同）
arm_vs_eff = np.allclose(arm_ts, eff_ts)
arm_vs_cam = np.allclose(arm_ts, cam_ts_0)
print(f"arm_ts == eff_ts: {arm_vs_eff}（应 False，独立采集）")
print(f"arm_ts == cam_ts: {arm_vs_cam}（应 False，独立采集）")
# 即使由于 v1 rollback 完全相等也不是 schema 错误，但在 v2 真机录制下应不同
EOF
```

**期望结果**：`arm_ts == eff_ts: False` 且 `arm_ts == cam_ts: False`（各模态独立采集线程打戳）。

---

## 验收条件 4：离线对齐 align_offline 跑通

**目的**：确认 align_offline 能对 v2 文件做时间插值对齐，正常退出。

```bash
source /home/ubuntu/Desktop/jhli/envs/franka-teleop/bin/activate
EP=$(ls /tmp/phaseD_acceptance_v2/*.h5 | head -1)

python scripts/tools/align_offline.py \
    --in "$EP" \
    --out /tmp/phaseD_acceptance_v2/aligned \
    --on-stale interpolate
echo "align_offline exit code: $?"
```

**期望结果**：脚本打印对齐统计信息，退出码 0；`/tmp/phaseD_acceptance_v2/aligned/` 下生成对齐后的文件。

---

## 验收条件 5：v2.1 转换 + RoboCOIN 加载

**目的**：确认 v2 episode 能通过 v2.1 转换器转成 LeRobot 数据集，并能被 RoboCOIN 管线加载。

```bash
source /home/ubuntu/Desktop/jhli/envs/franka-teleop/bin/activate

python scripts/tools/hdf5_to_lerobot_v21.py \
    --in-dir /tmp/phaseD_acceptance_v2 \
    --out /tmp/phaseD_acceptance_lerobot_v21 \
    --fps 30 \
    --task "phaseD_acceptance" \
    --robot-type franka
echo "v2.1 转换 exit code: $?"
```

然后在 RoboCOIN 环境验证（用户侧）：
```bash
# 在 RoboCOIN 训练环境验证 LeRobot 数据集可加载
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('/tmp/phaseD_acceptance_lerobot_v21')
print(f'episodes: {ds.num_episodes}, frames: {ds.num_frames}')
print('PASS v2.1 dataset loadable')
"
```

**期望结果**：转换脚本退出 0；RoboCOIN 能加载数据集并打印 episode/frame 数。

---

## 验收条件 6：v3.0 转换 + run_visualize 回看

**目的**：确认 v2 episode 能通过 v3.0 转换器并可视化。

```bash
source /home/ubuntu/Desktop/jhli/envs/franka-teleop/bin/activate

python scripts/tools/hdf5_to_lerobot.py \
    --in /tmp/phaseD_acceptance_v2 \
    --repo-id local/franka_phaseD_acceptance \
    --fps 30 \
    --root /tmp/phaseD_acceptance_lerobot_v30 \
    --task "phaseD_acceptance"
echo "v3.0 转换 exit code: $?"

python scripts/core/run_visualize.py \
    --root /tmp/phaseD_acceptance_lerobot_v30 \
    --repo-id local/franka_phaseD_acceptance
echo "run_visualize exit code: $?"
```

**期望结果**：两个命令均退出 0；`run_visualize` 生成 `.rrd` 文件（或直接弹出 rerun 可视化窗口）。

---

## 验收条件 7：录制频率不退化

**目的**：确认多线程采集（Phase D 重构后）录制 fps ≥ 29.5Hz，不因多线程开销导致帧率下降。

```bash
# 查看录制日志中的实测 fps
python - << 'EOF'
import sys, glob
sys.path.insert(0, '/home/ubuntu/Desktop/jhli/lerobot_franka_teleop')
import h5py, numpy as np

ep = sorted(glob.glob('/tmp/phaseD_acceptance_v2/*.h5'))[0]
with h5py.File(ep, 'r') as f:
    freq_arr = f['infos/task_info/collection_frequency'][...]
    N = int(f['infos/task_info/total_frames'][()])

target_fps, actual_fps = freq_arr[0], freq_arr[1]
print(f"target_fps={target_fps:.1f}, actual_fps={actual_fps:.2f}, N={N}")
if actual_fps >= 29.5:
    print("PASS fps 不退化")
else:
    print(f"WARN fps={actual_fps:.2f} < 29.5（请检查多线程采集性能）")
EOF
```

**期望结果**：`actual_fps >= 29.5`，打印 `PASS fps 不退化`。

---

## 验收完成标志

全部 7 条均通过后，请将结果贴回 agent，并在 spec 中标记 Phase D 完成。

```
✓ 条件 1: 重录 v2 episode ✓
✓ 条件 2: v2 schema 校验 ✓
✓ 条件 3: 多模态 ts 独立性 ✓（arm≠eff, arm≠cam）
✓ 条件 4: align_offline 跑通 ✓
✓ 条件 5: v2.1 转换 + RoboCOIN 加载 ✓
✓ 条件 6: v3.0 转换 + run_visualize ✓
✓ 条件 7: 录制 fps ≥ 29.5 ✓
```

---

*文档由 PhaseD-T9 生成，离线部分已全部完成（`602 passed`）。*
