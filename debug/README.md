# debug/ — 诊断与运维脚本（Route B 整合期沉淀）

本会话排障/验证沉淀的可复用工具。**注意**：脚本内绝对路径已按整合
（jhli → jhli/franka_vr_teleop）做机械改写，但写于排障期、未逐一回归；
路径已按 §11.1 回归修正(2026-05-20, 指向 scripts/services/); 仍请核对服务状态。真机相关脚本仅观测/不发控制指令的标注于注释。

## 服务起停/巡检（polymetis 三进程：臂 50051 / zerorpc 4242 / 夹爪 50052）
- `franka_cleanup.sh` 按真名杀净+删锁+验空
- `franka_start_arm.sh` / `franka_poll_arm.sh` 起臂栈 + 轮询 Connected/comm-violation
- `franka_start_zerorpc.sh` / `franka_verify_zerorpc.sh` 起 4242 + 5×RPC 验
- `franka_clean_restart.sh` 一次干净有序重起（清理→臂→验→zerorpc→验）
- `franka_start_gripper.sh` / `franka_poll_gripper.sh` 起夹爪 + 判 FCI refused
- `franka_gripper_verify.sh` / `_verify2.sh` / `_diag.sh` 经 4242 实测夹爪受控
- `franka_arm_health.sh` / `franka_arm_snapshot.sh` 臂只读健康/瞬态快照
- `franka_gripper_stop_only.sh` / `franka_gripper_cleanup_check.sh` 单停夹爪

## VR→base 映射验证（只读，不连机器人）
- `franka_vr_delta_probe.py` / `_probe2.py` 旁路探针：手势→delta，客观判 det/换基一致
- `franka_rot_proof.py` 确定性证明：旧式==逆旋转、新式==换基忠实式
- `franka_rot_singularity_test.py` / `franka_empirical_axis_test.py` 旋转奇异/轴关联离线分析
- `franka_h5_inspect.py` 产出 hdf5 结构概览
- `install_fixed_mapping.py` 写入固定坐标映射 .npy（双重印证矩阵）+ 自检
