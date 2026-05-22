"""Franka 数采 hdf5 schema 契约 + 校验 (franka-hdf5-v2)。

v2 变更摘要（相对 v1）：
  - 删除共用时间戳 observations/timestamp(N,1)，改为每模态独立 timestamp(N,) float64
  - 每模态新增 stale(N,) bool 标志（arm/effector/camera/{cn}）
  - camera/{cn} 新增 hw_timestamp(N,) float64（RealSense 硬件戳，Task 8 实填）
  - state_hifreq 新增 wrench(M,6) float64 占位（Phase F 实填，M=0 时合规）
  - 可扩展字段 validate-if-present：depth/tactile 缺失不报错，存在则校验 shape/dtype
  - N 由 action/timestamp 定义（不再依赖 observations/timestamp）

冻结接口：Task 4（hdf5_writer）、Task 5/6（转换器）、测试并行依此。
改 schema 必须 bump SCHEMA_VERSION 并同步所有消费方。
对应 spec：docs/superpowers/specs/2026-05-18-franka-datacollection-hdf5-design.md §3
"""
import h5py
import numpy as np

SCHEMA_VERSION = "franka-hdf5-v2"
JOINT_DOF = 7
EE_DIM = 6              # [x, y, z, rx, ry, rz]
GRIPPER_MAX_M = 0.08


def _str_val(d):
    """将 HDF5 scalar/bytes dataset 读出为 str。"""
    x = d[()] if d.shape == () else d[0]
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def validate_episode(path):
    """校验一个 episode .h5 是否符合 franka-hdf5-v2。返回 violations 列表（空=合格）。

    v2 设计原则：
      - N 由 action/timestamp.shape[0] 决定（不依赖 observations/timestamp）
      - 各模态 timestamp(N,) + stale(N,) 独立存在
      - camera 每个相机组额外要求 hw_timestamp(N,) float64
      - state_hifreq 内部各字段独立长度 M（M=0 合规）
      - depth/tactile 可扩展字段：缺失不报错，存在则严格校验
    """
    v = []
    with h5py.File(path, "r") as f:
        # --- 版本标识 ---
        sv = f.get("infos/schema_version")
        if sv is None or _str_val(sv) != SCHEMA_VERSION:
            v.append(f"infos/schema_version 缺失或 != {SCHEMA_VERSION}")

        # --- 必须存在的顶层 groups ---
        for g in ["infos", "infos/calibration", "observations", "observations/arm",
                  "observations/effector", "observations/camera",
                  "observations/state_hifreq", "action"]:
            if g not in f:
                v.append(f"缺少组 {g}")

        # --- 确定 N（由 action/timestamp 定义）---
        act_ts = f.get("action/timestamp")
        if act_ts is None:
            v.append("缺 action/timestamp（无法确定 N）")
            return v
        N = act_ts.shape[0]
        if act_ts.dtype != np.float64:
            v.append(f"action/timestamp dtype {act_ts.dtype} != float64")

        def chk_f64(p, exact_shape):
            """校验 float64 dataset，shape 精确匹配。"""
            d = f.get(p)
            if d is None:
                v.append(f"缺 {p}")
                return
            if tuple(d.shape) != exact_shape:
                v.append(f"{p} shape {tuple(d.shape)} != {exact_shape}")
            if d.dtype != np.float64:
                v.append(f"{p} dtype {d.dtype} != float64")

        def chk_bool(p, n):
            """校验 bool(N,) dataset。"""
            d = f.get(p)
            if d is None:
                v.append(f"缺 {p}")
                return
            if tuple(d.shape) != (n,):
                v.append(f"{p} shape {tuple(d.shape)} != ({n},)")
            if d.dtype != bool:
                v.append(f"{p} dtype {d.dtype} != bool")

        # --- arm 模态 ---
        chk_f64("observations/arm/joints", (N, JOINT_DOF))
        chk_f64("observations/arm/joint_vel", (N, JOINT_DOF))
        chk_f64("observations/arm/pose", (N, EE_DIM))
        chk_f64("observations/arm/timestamp", (N,))
        chk_bool("observations/arm/stale", N)

        # --- effector 模态 ---
        chk_f64("observations/effector/position", (N, 1))
        chk_f64("observations/effector/position_norm", (N, 1))
        chk_f64("observations/effector/timestamp", (N,))
        chk_bool("observations/effector/stale", N)
        if "observations/effector/type" not in f:
            v.append("缺 observations/effector/type")

        # --- camera 模态（每个相机独立校验）---
        rgb_group = f.get("observations/camera/rgb")
        cam_names = sorted(rgb_group.keys()) if rgb_group is not None else []
        for cn in cam_names:
            base = f"observations/camera/rgb/{cn}"
            # images 只检查存在，不检查 shape（vlen bytes 无固定 shape）
            if f"{base}/images" not in f:
                v.append(f"缺 {base}/images")
            chk_f64(f"{base}/timestamp", (N,))
            chk_bool(f"{base}/stale", N)
            chk_f64(f"{base}/hw_timestamp", (N,))
            # 可扩展：depth（validate-if-present）
            if f"{base}/depth" in f:
                depth_ds = f[f"{base}/depth"]
                if depth_ds.shape[0] != N:
                    v.append(f"{base}/depth shape[0] {depth_ds.shape[0]} != {N}")
            if f"{base}/depth_stale" in f:
                chk_bool(f"{base}/depth_stale", N)
            if f"{base}/depth_timestamp" in f:
                chk_f64(f"{base}/depth_timestamp", (N,))

        # --- action 模态 ---
        chk_f64("action/delta_ee_pose", (N, EE_DIM))
        chk_f64("action/gripper_cmd", (N, 1))
        # action/timestamp 已在 N 确定阶段校验过 dtype，形状为 (N,) 已确认

        # --- state_hifreq（内部独立长度 M，M=0 合规）---
        hj = f.get("observations/state_hifreq/joints")
        if hj is None:
            v.append("缺 observations/state_hifreq/joints")
        else:
            M = hj.shape[0]
            chk_f64("observations/state_hifreq/joints", (M, JOINT_DOF))
            chk_f64("observations/state_hifreq/joint_vel", (M, JOINT_DOF))
            chk_f64("observations/state_hifreq/pose", (M, EE_DIM))
            chk_f64("observations/state_hifreq/timestamp", (M,))
            chk_f64("observations/state_hifreq/poly_ts", (M,))
            # wrench(M,6) 预留字段（Phase F 实填）
            if "observations/state_hifreq/wrench" not in f:
                v.append("缺 observations/state_hifreq/wrench")
            else:
                chk_f64("observations/state_hifreq/wrench", (M, 6))

        # --- 可扩展：tactile（validate-if-present）---
        tac_group = f.get("observations/tactile")
        if tac_group is not None:
            for sname in tac_group.keys():
                base = f"observations/tactile/{sname}"
                if f"{base}/values" in f:
                    ds = f[f"{base}/values"]
                    if ds.shape[0] != N:
                        v.append(f"{base}/values shape[0] {ds.shape[0]} != {N}")
                    if ds.dtype != np.float64:
                        v.append(f"{base}/values dtype {ds.dtype} != float64")
                if f"{base}/timestamp" in f:
                    chk_f64(f"{base}/timestamp", (N,))
                if f"{base}/stale" in f:
                    chk_bool(f"{base}/stale", N)

        # --- calibration ---
        cr = f.get("infos/calibration/oc2base_R")
        if cr is None or tuple(cr.shape) != (3, 3):
            v.append("infos/calibration/oc2base_R 缺失或 shape != (3,3)")
        elif cr.dtype != np.float64:
            v.append(f"infos/calibration/oc2base_R dtype {cr.dtype} != float64")

    return v
