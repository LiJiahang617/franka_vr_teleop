# v2.1 加载性 franka2 探测取舍 lesson

**关联**: §11.5 spec `docs/superpowers/specs/2026-05-20-lerobot-v21-converter-design.md`，
实现计划 `docs/superpowers/plans/2026-05-20-lerobot-v21-converter.md` Task 6。

---

## 一句话结论

franka2 仅有 lerobot 0.3.4 / codebase_version=v3.0，无 v2.1 lerobot；
§11.5 v2.1 转换器的"能被 v2.1 lerobot 加载"无法在 franka2 离线闭环，
离线保障靠结构对标守门 + 真机端到端，终极兼容性确认明确交用户侧（非真机，属用户环境）。

---

## 探测结果（2026-05-20）

franka2 所有 Python 环境扫描：

| 环境路径 | lerobot 版本 | CODEBASE_VERSION |
|---|---|---|
| `/home/ubuntu/Desktop/jhli/envs/franka-teleop/bin/python` | 0.3.4 | v3.0 |
| `/home/ubuntu/Desktop/jhli/envs/polymetis-local/bin/python` | 无 lerobot | — |
| `/usr/bin/python3` | 无 lerobot | — |

结论：franka2 上**无 v2.1 lerobot**，无法就地执行 `LeRobotDataset(root=...)` 加载验证。

---

## 取舍与依据

§11.5 v2.1 转换器的离线可靠性保障分两层：

1. **结构对标守门**（franka2 本地可验，已完成）  
   `tests/test_v21_structure_diff.py` 17 项测试，逐字对标 realman v2.1 参考集字段/路径/jsonl/splits/dtype/shape/计数交叉：
   - info.json 顶层键 ⊇ realman 必需键集（codebase_version/robot_type/splits/data_path/video_path/features/...）
   - features block 同构：state `fixed_size_list<float32>[14]`、action `[7]`、video block video.codec=h264/yuv420p、元列5个 shape=[1] dtype 对应
   - tasks.jsonl / episodes.jsonl / episodes_stats.jsonl 每行键集 == realman 对应
   - 路径命名正则：`data/chunk-\d{3}/episode_\d{6}.parquet`、`videos/chunk-\d{3}/observation.images.\w+/episode_\d{6}.mp4`
   - 真机 ep0000 端到端：896帧/codebase_version=v2.1/robot_type=franka/parquet 7列 fixed_size_list/h264 yuv420p 424x240（commit 链 a56ae97..3df1575）

2. **终极兼容性确认（DEFERRED，非真机，属用户环境）**  
   franka2 无 v2.1 lerobot，无法就地 LeRobotDataset 加载冒烟。  
   终极确认路径：用户在 **RoboCOIN 环境**（lerobot 0.1.0，本工作站）用 `visualize_dataset.sh` 指向转换产物，确认：
   - `LeRobotDataset` 无 codebase_version 报错
   - 首帧含 `observation.state`（14D）/ `action`（7D）/ `observation.images.exterior_image` / `observation.images.wrist_image`
   - 可视化能播放视频帧
   
   此步骤**非真机操作**，属用户环境验证，不在 franka2 离线闭环范围内，明确交用户执行。

---

## 设计决策待复核提示

§11.5 实现中有一项设计决策属"自主推进期所作"，已在 spec 顶部标注"待用户复核可推翻"：

- **保留 Route-B 原生语义，不对齐 realman 14D 绝对值**：  
  `observation.state` = [joint_1..7 rad, ee_pose xyz+rpy, gripper_norm]（与 v3.0 逐字一致）；  
  `action` = [delta_ee_pose 6D, gripper_cmd_bin 1D]（Route-B 相对 EEF 控制，非 realman 绝对关节）。  
  理由：动作空间本质不同（§10.5C），强行对齐 realman 14D 绝对需改控制器，超出转换器范畴。  
  **已留 `--state-layout realman` 旋钮**（仅重排 observation.state 列序/命名，action 恒 7D 不变）供用户调整。

---

## 关联提交

`§11.5-T1..T5` commit 链：a56ae97..3df1575（不 push）

生成时间：2026-05-20
