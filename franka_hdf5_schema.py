"""Franka 数采 hdf5 schema 契约 + 校验 (franka-hdf5-v2)。

v2 变更摘要（相对 v1）：
  - 删除共用时间戳 observations/timestamp(N,1)，改为每模态独立 timestamp(N,) float64
  - 每模态新增 stale(N,) bool 标志（arm/effector/camera/{cn}）；stale 只用于传感器模态
    （标记采集缺帧/补帧），action 是遥操生成的动作、无缺帧概念，故 action 不含 stale
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


# v2 内**可选**字段（不 bump 版本号）：字段缺失视为合规；字段存在则严格校验
# observations/effector/hw_timestamp：libfranka 硬件 push 时戳（since robot start，秒）
# 由 polymetis franka_hand_client 在 ENABLE_GRIPPER_HW_TIMESTAMP=ON 时填入；
# 旧 polymetis（或 cmake 关闭该宏）录制的数据集不含该字段，validate 仍 PASS。
OPTIONAL_FIELDS_V2 = {
    "observations/effector/hw_timestamp": {
        "shape": ("N",),
        "dtype": np.float64,
    },
}


def _str_val(d):
    """将 HDF5 scalar/bytes dataset 读出为 str。

    面对坏文件（空一维、非预期类型）返回空字符串，使上层 schema_version 比对自然报 violation。
    """
    try:
        if d.shape == ():
            x = d[()]
        elif len(d.shape) == 1 and d.shape[0] == 0:
            # 空一维 dataset：无法取 d[0]，返回明显非法值
            return ""
        else:
            x = d[0]
        return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
    except Exception:
        return ""


def validate_episode(path):
    """校验一个 episode .h5 是否符合 franka-hdf5-v2。返回 violations 列表（空=合格）。

    v2 设计原则：
      - N 由 action/timestamp.shape[0] 决定（不依赖 observations/timestamp）
      - action/timestamp 必须是一维 float64；(N,1) 二维风格被拒
      - 各模态 timestamp(N,) + stale(N,) 独立存在
      - camera 每个相机组额外要求 hw_timestamp(N,) float64
      - state_hifreq 内部各字段独立长度 M（M=0 合规）
      - depth/tactile 可扩展字段：缺失不报错，存在则严格校验
      - observations/camera/rgb 组必须存在且至少含一个相机子组

    时间戳单调性契约：
      - action/timestamp：严格递增（N>=2 时）
      - arm/effector/camera 各 timestamp：非递减（N>=2 时，带 stale 允许相等不允许倒退）
      - state_hifreq/timestamp：严格递增（M>=2 时）
      - hw_timestamp：本阶段不校验单调性（Task 8 才实填）
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

        # 校验 action/timestamp 必须是一维
        if len(act_ts.shape) != 1:
            v.append(f"action/timestamp 必须是一维，实际 shape={tuple(act_ts.shape)}（v1 风格二维戳不合规）")
            return v

        # 取 N
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

        def chk_monotone_strict(p, length, label):
            """校验 timestamp 严格递增（用于 action/state_hifreq）。"""
            if length < 2:
                return
            d = f.get(p)
            if d is None:
                return  # 缺失已由 chk_f64 报告
            ts = d[...]
            if ts.ndim != 1:
                return  # shape 问题已由 chk_f64 报告
            diffs = np.diff(ts)
            if np.any(diffs <= 0):
                bad_idx = int(np.argwhere(diffs <= 0).flat[0])
                v.append(f"{p} 时间戳不严格递增（{label}，index {bad_idx}→{bad_idx+1}: "
                         f"{ts[bad_idx]:.6f}→{ts[bad_idx+1]:.6f}）")

        def chk_monotone_nondec(p, length, label):
            """校验 timestamp 非递减（用于 arm/effector/camera，允许相等因为有 stale）。"""
            if length < 2:
                return
            d = f.get(p)
            if d is None:
                return  # 缺失已由 chk_f64 报告
            ts = d[...]
            if ts.ndim != 1:
                return  # shape 问题已由 chk_f64 报告
            diffs = np.diff(ts)
            if np.any(diffs < 0):
                bad_idx = int(np.argwhere(diffs < 0).flat[0])
                v.append(f"{p} 时间戳倒退（{label}，index {bad_idx}→{bad_idx+1}: "
                         f"{ts[bad_idx]:.6f}→{ts[bad_idx+1]:.6f}）")

        # --- arm 模态 ---
        chk_f64("observations/arm/joints", (N, JOINT_DOF))
        chk_f64("observations/arm/joint_vel", (N, JOINT_DOF))
        chk_f64("observations/arm/pose", (N, EE_DIM))
        chk_f64("observations/arm/timestamp", (N,))
        chk_bool("observations/arm/stale", N)
        chk_monotone_nondec("observations/arm/timestamp", N, "arm，带 stale 允许相等")

        # --- effector 模态 ---
        chk_f64("observations/effector/position", (N, 1))
        chk_f64("observations/effector/position_norm", (N, 1))
        chk_f64("observations/effector/timestamp", (N,))
        chk_bool("observations/effector/stale", N)
        if "observations/effector/type" not in f:
            v.append("缺 observations/effector/type")
        chk_monotone_nondec("observations/effector/timestamp", N, "effector，带 stale 允许相等")

        # --- camera 模态（每个相机独立校验）---
        # rgb 组必须存在，且至少一个相机子组
        rgb_group = f.get("observations/camera/rgb")
        if rgb_group is None:
            v.append("缺 observations/camera/rgb 组（Route B 必须有相机）")
            cam_names = []
        elif not isinstance(rgb_group, h5py.Group):
            v.append("observations/camera/rgb 存在但不是 Group（文件损坏）")
            cam_names = []
        else:
            cam_names = sorted(rgb_group.keys())
            if len(cam_names) == 0:
                v.append("observations/camera/rgb 组为空（至少需要一个相机子组）")

        for cn in cam_names:
            base = f"observations/camera/rgb/{cn}"
            # images 校验：必须存在，且 shape[0] == N（vlen uint8，一维）
            imgs_ds = f.get(f"{base}/images")
            if imgs_ds is None:
                v.append(f"缺 {base}/images")
            else:
                if len(imgs_ds.shape) != 1:
                    v.append(f"{base}/images 必须是一维，实际 shape={tuple(imgs_ds.shape)}")
                elif imgs_ds.shape[0] != N:
                    v.append(f"{base}/images shape[0] {imgs_ds.shape[0]} != {N}")
            chk_f64(f"{base}/timestamp", (N,))
            chk_bool(f"{base}/stale", N)
            chk_f64(f"{base}/hw_timestamp", (N,))
            chk_monotone_nondec(f"{base}/timestamp", N, f"camera/{cn}，带 stale 允许相等")
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
        # action/timestamp 已校验一维和 dtype，单调性：严格递增（action 无 stale）
        chk_monotone_strict("action/timestamp", N, "action 无 stale 概念")

        # --- state_hifreq（内部独立长度 M，M=0 合规）---
        hj = f.get("observations/state_hifreq/joints")
        if hj is None:
            v.append("缺 observations/state_hifreq/joints")
        else:
            # 确认至少一维（防止标量抛异常）
            if len(hj.shape) < 1:
                v.append("observations/state_hifreq/joints 是标量，必须至少一维")
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
                # state_hifreq/timestamp 严格递增（240Hz 密集采样，无 stale）
                chk_monotone_strict("observations/state_hifreq/timestamp", M,
                                    "state_hifreq 无 stale 概念")

        # --- 可扩展：tactile（validate-if-present）---
        tac_group = f.get("observations/tactile")
        if tac_group is not None:
            if not isinstance(tac_group, h5py.Group):
                v.append("observations/tactile 存在但不是 Group（文件损坏）")
            else:
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

        # 可选字段：存在则严格校验 shape/dtype；不存在跳过
        for path, spec in OPTIONAL_FIELDS_V2.items():
            if path in f:
                ds = f[path]
                exp_shape = spec["shape"]
                # 解析 ("N",) 形状语义（N=帧数）
                if exp_shape == ("N",):
                    if ds.ndim != 1 or ds.shape[0] != N:
                        v.append(
                            f"{path}: 形状不合规，期望 ({N},) 实际 {ds.shape}"
                        )
                if ds.dtype != spec["dtype"]:
                    v.append(
                        f"{path}: dtype 不合规，期望 {spec['dtype']} 实际 {ds.dtype}"
                    )

    return v
