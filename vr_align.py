"""VR↔Franka 坐标系对齐：纯数学核心，无硬件依赖。"""
import datetime
import json
import os

import numpy as np


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError(f"零向量无法归一化: {v}")
    return v / n


def solve_rotation(d_oc_list, d_arm_list):
    """Kabsch: 解旋转 R，使 d_arm ≈ R @ d_oc。每对先归一化，只求旋转不求尺度。"""
    A = np.array([_unit(v) for v in d_oc_list], dtype=float)
    B = np.array([_unit(v) for v in d_arm_list], dtype=float)
    if A.shape != B.shape or A.shape[0] < 2:
        raise ValueError(f"需 >=2 对非平行向量, 得到 A{A.shape} B{B.shape}")
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R


def validate_rotation(R, ortho_tol=1e-6, det_tol=1e-3):
    """返回 (ok, ortho_err, det)，ok 表示正交且 det≈+1。"""
    R = np.asarray(R, dtype=float)
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    det = float(np.linalg.det(R))
    ok = ortho_err < ortho_tol and abs(det - 1.0) < det_tol
    return ok, ortho_err, det


def gesture_pair_quality(d_oc_list, d_arm_list):
    """2 手势质量: 返回 (oc_inter_deg, arm_inter_deg, recon_max_deg)。"""
    A = np.array([_unit(v) for v in d_oc_list], dtype=float)
    B = np.array([_unit(v) for v in d_arm_list], dtype=float)
    if A.shape[0] != 2 or B.shape[0] != 2:
        raise ValueError(f"恰需 2 对, 得到 {A.shape[0]}")
    oc_inter = float(np.degrees(np.arccos(np.clip(np.dot(A[0], A[1]), -1.0, 1.0))))
    arm_inter = float(np.degrees(np.arccos(np.clip(np.dot(B[0], B[1]), -1.0, 1.0))))
    Rm = solve_rotation(A, B)
    recon = [
        float(np.degrees(np.arccos(np.clip(np.dot(_unit(Rm @ A[i]), B[i]), -1.0, 1.0))))
        for i in range(2)
    ]
    return oc_inter, arm_inter, float(np.max(recon))


def _meta_path(npy_path):
    assert npy_path.endswith('.npy'), npy_path
    return npy_path[:-4] + '.meta.json'


def save_rotation(path, R, quality, oc_ref_rotvec):
    """存 R(.npy) + sidecar(.meta.json)。quality 为质量指标，oc_ref_rotvec 为漂移基准。"""
    np.save(path, np.asarray(R, dtype=float))
    meta = {
        'saved_at': datetime.datetime.now().isoformat(),
        'quality': {k: float(v) for k, v in dict(quality).items()},
        'oc_ref_rotvec': [float(x) for x in oc_ref_rotvec],
    }
    with open(_meta_path(path), 'w') as f:
        json.dump(meta, f, indent=2)


def load_rotation(path):
    """返回 (R, meta)，文件不存在时返回 None。"""
    if not os.path.exists(path):
        return None
    R = np.load(path)
    mp = _meta_path(path)
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    return R, meta


def resolve_mapping(loaded_R, legacy_pos_M, legacy_rot_M, legacy_pos_signs, legacy_rot_signs):
    """标定 R 存在则整体取代 legacy：pos_M=rot_M=R，signs=1。"""
    if loaded_R is not None:
        Rm = np.asarray(loaded_R, dtype=float)
        ones = np.ones(3)
        return Rm.copy(), Rm.copy(), ones.copy(), ones.copy(), 'calibrated'
    return (
        np.asarray(legacy_pos_M, float),
        np.asarray(legacy_rot_M, float),
        np.asarray(legacy_pos_signs, float),
        np.asarray(legacy_rot_signs, float),
        'legacy',
    )
