"""v1→v2 离线迁移工具：将 franka-hdf5-v1 episode 转换为 franka-hdf5-v2。

v1→v2 主要变更处理逻辑：
  1. schema_version: "franka-hdf5-v1" → "franka-hdf5-v2"
  2. 共用时间戳 observations/timestamp(N,1) → 每模态独立 timestamp(N,)
     - arm/effector/camera 的 timestamp 来自共用戳（已存在一维副本，直接复用）
     - v1 各模态已有 timestamp(N,) 副本；若缺失则从 observations/timestamp 提取
  3. 各模态新增 stale(N,) bool，全填 False（v1 无缺帧概念）
  4. camera/{cn} 新增 hw_timestamp(N,) float64 = 对应模态的 timestamp（v1 无硬件戳）
  5. state_hifreq 新增 wrench(M,6) float64 零填（Phase F 实填，M=0 时 shape=(0,6)）
  6. 删除 observations/timestamp 共用戳（v2 不要此路径）

CLI 用法：
    python scripts/tools/migrate_v1_to_v2.py --in <v1.h5> --out <v2.h5>

输入文件不修改，迁移后自动调用 validate_episode 自检。
"""
import argparse
import sys
import os

import h5py
import numpy as np

# 加载 schema_loader（仓库根在 PYTHONPATH 时直接 import）
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from core.schema_loader import load_franka_hdf5_schema

S = load_franka_hdf5_schema()
_VLEN_BYTES = h5py.special_dtype(vlen=np.dtype("uint8"))


def _read_shared_ts(f, N):
    """从 v1 文件读取共用时间戳，返回一维 (N,) float64 数组。

    v1 中共用戳存储为 (N,1)，需展平为 (N,)。
    若文件已经是非标准形状则尽力处理。
    """
    ts_ds = f.get("observations/timestamp")
    if ts_ds is None:
        # 尝试从 action/timestamp 降级
        act_ts = f.get("action/timestamp")
        if act_ts is not None:
            return np.asarray(act_ts, dtype=np.float64).reshape(-1)
        return np.arange(N, dtype=np.float64) * 0.033 + 1.0
    return np.asarray(ts_ds, dtype=np.float64).reshape(-1)


def _copy_group_meta(src_g, dst_g):
    """递归复制 group 的所有 dataset 与子 group（跳过对象 group 本身）。"""
    for key in src_g.keys():
        obj = src_g[key]
        if isinstance(obj, h5py.Dataset):
            dst_g.create_dataset(key, data=obj[...], dtype=obj.dtype)
        elif isinstance(obj, h5py.Group):
            sub = dst_g.create_group(key)
            _copy_group_meta(obj, sub)


def migrate(in_path, out_path):
    """将 franka-hdf5-v1 文件迁移到 franka-hdf5-v2，写入 out_path。

    Args:
        in_path: v1 .h5 文件路径（字符串）。
        out_path: 输出 v2 .h5 文件路径（字符串）。

    Raises:
        ValueError: 输入文件 schema_version 不是 "franka-hdf5-v1"。
        RuntimeError: 迁移后 validate_episode 自检失败。
    """
    with h5py.File(in_path, "r") as src, h5py.File(out_path, "w") as dst:
        # --- 校验输入版本 ---
        sv_ds = src.get("infos/schema_version")
        if sv_ds is None:
            raise ValueError(f"输入文件缺少 infos/schema_version，无法迁移: {in_path}")
        raw = sv_ds[()] if sv_ds.shape == () else sv_ds[0]
        sv_str = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
        if sv_str != "franka-hdf5-v1":
            raise ValueError(
                f"输入文件 schema_version={sv_str!r}，期望 'franka-hdf5-v1'，拒绝迁移。"
            )

        # --- 确定 N ---
        act_ts_ds = src.get("action/timestamp")
        if act_ts_ds is None:
            raise ValueError(f"输入文件缺少 action/timestamp，无法确定帧数: {in_path}")
        N = act_ts_ds.shape[0]

        # --- 读取 v1 共用时间戳 ---
        shared_ts = _read_shared_ts(src, N)  # (N,)

        # === 写 infos ===
        infos_src = src["infos"]
        infos_dst = dst.create_group("infos")
        # schema_version 改为 v2
        infos_dst.create_dataset("schema_version", data=np.bytes_(S.SCHEMA_VERSION))
        # 复制 task_info（全量复制）
        if "task_info" in infos_src:
            ti_dst = infos_dst.create_group("task_info")
            _copy_group_meta(infos_src["task_info"], ti_dst)
        # 复制 camera_params（空 group）
        infos_dst.create_group("camera_params")
        # 复制 calibration（oc2base_R / quality / vr_source）
        cal_dst = infos_dst.create_group("calibration")
        if "calibration" in infos_src:
            _copy_group_meta(infos_src["calibration"], cal_dst)

        # === 写 observations ===
        obs_src = src["observations"]
        obs_dst = dst.create_group("observations")

        # --- arm 模态 ---
        arm_src = obs_src["arm"]
        arm_dst = obs_dst.create_group("arm")
        arm_dst.create_dataset("joints", data=np.asarray(arm_src["joints"], np.float64))
        arm_dst.create_dataset("joint_vel", data=np.asarray(arm_src["joint_vel"], np.float64))
        arm_dst.create_dataset("pose", data=np.asarray(arm_src["pose"], np.float64))
        # v1 arm/timestamp 已是 (N,) 副本，直接复用；万一不存在则用共用戳
        if "timestamp" in arm_src:
            arm_ts = np.asarray(arm_src["timestamp"], np.float64).reshape(-1)
        else:
            arm_ts = shared_ts.copy()
        arm_dst.create_dataset("timestamp", data=arm_ts)
        # 新增 stale(N,) bool，全 False
        arm_dst.create_dataset("stale", data=np.zeros(N, dtype=bool))

        # --- effector 模态 ---
        eff_src = obs_src["effector"]
        eff_dst = obs_dst.create_group("effector")
        eff_dst.create_dataset("position", data=np.asarray(eff_src["position"], np.float64))
        eff_dst.create_dataset("position_norm", data=np.asarray(eff_src["position_norm"], np.float64))
        # effector/type：vlen bytes，逐元素复制
        if "type" in eff_src:
            type_arr = eff_src["type"][...]
            eff_dst.create_dataset("type", data=type_arr,
                                   dtype=h5py.special_dtype(vlen=bytes))
        # v1 effector/timestamp 已是 (N,) 副本
        if "timestamp" in eff_src:
            eff_ts = np.asarray(eff_src["timestamp"], np.float64).reshape(-1)
        else:
            eff_ts = shared_ts.copy()
        eff_dst.create_dataset("timestamp", data=eff_ts)
        # 新增 stale(N,) bool，全 False
        eff_dst.create_dataset("stale", data=np.zeros(N, dtype=bool))

        # --- camera 模态 ---
        cam_dst = obs_dst.create_group("camera")
        rgb_dst = cam_dst.create_group("rgb")
        rgb_src = obs_src.get("camera/rgb")
        if rgb_src is not None:
            for cn in sorted(rgb_src.keys()):
                cn_src = rgb_src[cn]
                cn_dst = rgb_dst.create_group(cn)
                # images：vlen uint8，逐帧复制
                imgs_src = cn_src["images"]
                imgs_dst = cn_dst.create_dataset("images", (N,), dtype=_VLEN_BYTES)
                for i in range(N):
                    imgs_dst[i] = imgs_src[i]
                # v1 camera/{cn}/timestamp 已是 (N,) 副本
                if "timestamp" in cn_src:
                    cam_ts = np.asarray(cn_src["timestamp"], np.float64).reshape(-1)
                else:
                    cam_ts = shared_ts.copy()
                cn_dst.create_dataset("timestamp", data=cam_ts)
                # 新增 stale(N,) bool，全 False
                cn_dst.create_dataset("stale", data=np.zeros(N, dtype=bool))
                # 新增 hw_timestamp(N,) float64 = 软件戳（v1 无硬件戳）
                cn_dst.create_dataset("hw_timestamp", data=cam_ts.copy())

        # --- state_hifreq ---
        hf_src = obs_src["state_hifreq"]
        hf_dst = obs_dst.create_group("state_hifreq")
        hj = hf_src.get("joints")
        if hj is not None:
            M = hj.shape[0]
            hf_dst.create_dataset("joints", data=np.asarray(hf_src["joints"], np.float64))
            hf_dst.create_dataset("joint_vel", data=np.asarray(hf_src["joint_vel"], np.float64))
            hf_dst.create_dataset("pose", data=np.asarray(hf_src["pose"], np.float64))
            hf_dst.create_dataset("timestamp", data=np.asarray(hf_src["timestamp"], np.float64))
            hf_dst.create_dataset("poly_ts", data=np.asarray(hf_src["poly_ts"], np.float64))
            # 新增 wrench(M,6) 零填（Phase F 实填）
            hf_dst.create_dataset("wrench", data=np.zeros((M, 6), np.float64))
        else:
            # state_hifreq 字段缺失，写空占位
            hf_dst.create_dataset("joints", data=np.zeros((0, 7), np.float64))
            hf_dst.create_dataset("joint_vel", data=np.zeros((0, 7), np.float64))
            hf_dst.create_dataset("pose", data=np.zeros((0, 6), np.float64))
            hf_dst.create_dataset("timestamp", data=np.zeros((0,), np.float64))
            hf_dst.create_dataset("poly_ts", data=np.zeros((0,), np.float64))
            hf_dst.create_dataset("wrench", data=np.zeros((0, 6), np.float64))

        # === 写 action ===
        act_src = src["action"]
        act_dst = dst.create_group("action")
        act_dst.create_dataset("delta_ee_pose",
                               data=np.asarray(act_src["delta_ee_pose"], np.float64))
        act_dst.create_dataset("gripper_cmd",
                               data=np.asarray(act_src["gripper_cmd"], np.float64))
        act_dst.create_dataset("timestamp",
                               data=np.asarray(act_src["timestamp"], np.float64).reshape(-1))

    # --- 自检 validate_episode ---
    violations = S.validate_episode(out_path)
    if violations:
        raise RuntimeError(
            f"迁移后自检失败，violations:\n" + "\n".join(f"  - {x}" for x in violations)
        )


def main():
    """CLI 入口：--in <v1.h5> --out <v2.h5>。"""
    parser = argparse.ArgumentParser(
        description="franka-hdf5-v1 → franka-hdf5-v2 离线迁移工具"
    )
    parser.add_argument("--in", dest="in_path", required=True,
                        help="输入 v1 .h5 文件路径")
    parser.add_argument("--out", dest="out_path", required=True,
                        help="输出 v2 .h5 文件路径")
    args = parser.parse_args()

    print(f"[migrate] 读取: {args.in_path}")
    migrate(args.in_path, args.out_path)
    print(f"[migrate] 迁移成功，输出: {args.out_path}（validate_episode 自检通过）")


if __name__ == "__main__":
    main()
