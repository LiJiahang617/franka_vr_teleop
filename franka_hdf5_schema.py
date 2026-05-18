"""Franka 数采 hdf5 schema 契约 + 校验 (franka-hdf5-v1)。

冻结接口：SP-1/SP-2/SP-3 并行依此。改 schema 必须 bump SCHEMA_VERSION 并同步三者。
对应 spec：docs/superpowers/specs/2026-05-18-franka-datacollection-hdf5-design.md §3
"""
import h5py
import numpy as np

SCHEMA_VERSION = "franka-hdf5-v1"
JOINT_DOF = 7
EE_DIM = 6              # [x, y, z, rx, ry, rz]
GRIPPER_MAX_M = 0.08


def _str_val(d):
    x = d[()] if d.shape == () else d[0]
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def validate_episode(path):
    """校验一个 episode .h5 是否符合 franka-hdf5-v1。返回 violations 列表（空=合格）。"""
    v = []
    with h5py.File(path, "r") as f:
        sv = f.get("infos/schema_version")
        if sv is None or _str_val(sv) != SCHEMA_VERSION:
            v.append(f"infos/schema_version 缺失或 != {SCHEMA_VERSION}")

        for g in ["infos", "infos/calibration", "observations", "observations/arm",
                  "observations/effector", "observations/camera",
                  "observations/state_hifreq", "action"]:
            if g not in f:
                v.append(f"缺少组 {g}")

        ts = f.get("observations/timestamp")
        if ts is None:
            v.append("缺 observations/timestamp")
            return v
        N = ts.shape[0]

        def chk(p, exact):
            d = f.get(p)
            if d is None:
                v.append(f"缺 {p}")
                return
            if tuple(d.shape) != exact:
                v.append(f"{p} shape {tuple(d.shape)} != {exact}")
            if d.dtype != np.float64:
                v.append(f"{p} dtype {d.dtype} != float64")

        chk("observations/timestamp", (N, 1))
        chk("observations/arm/joints", (N, JOINT_DOF))
        chk("observations/arm/joint_vel", (N, JOINT_DOF))
        chk("observations/arm/pose", (N, EE_DIM))
        chk("observations/arm/timestamp", (N,))
        chk("observations/effector/position", (N, 1))
        chk("observations/effector/position_norm", (N, 1))
        chk("observations/effector/timestamp", (N,))
        chk("action/delta_ee_pose", (N, EE_DIM))
        chk("action/gripper_cmd", (N, 1))
        chk("action/timestamp", (N,))

        if "observations/effector/type" not in f:
            v.append("缺 observations/effector/type")

        hj = f.get("observations/state_hifreq/joints")
        if hj is None:
            v.append("缺 observations/state_hifreq/joints")
        else:
            M = hj.shape[0]
            chk("observations/state_hifreq/joints", (M, JOINT_DOF))
            chk("observations/state_hifreq/joint_vel", (M, JOINT_DOF))
            chk("observations/state_hifreq/pose", (M, EE_DIM))
            chk("observations/state_hifreq/timestamp", (M,))
            chk("observations/state_hifreq/poly_ts", (M,))

        cr = f.get("infos/calibration/oc2base_R")
        if cr is None or tuple(cr.shape) != (3, 3):
            v.append("infos/calibration/oc2base_R 缺失或 shape != (3,3)")
        elif cr.dtype != np.float64:
            v.append(f"infos/calibration/oc2base_R dtype {cr.dtype} != float64")

        arr = np.asarray(ts).reshape(-1)
        if N >= 2 and np.any(np.diff(arr) <= 0):
            v.append("observations/timestamp 非严格单调")
    return v
