"""把固定坐标映射 R_cal 写入 .stage3_oc2arm_R.npy，替代易错的 2 手势 SVD。
矩阵双重印证: 验证过的 Realman vr_utils.RIGHT_ROTATION_MATRIX == 本会话从用户
反馈实测反推的正确 R_cal == [[0,0,1],[1,0,0],[0,1,0]] (det=+1, 正交真旋转)。
零代码改动: vr_align.load_rotation + 已修好的 compute_delta_action 照常读用。"""
import json
import shutil
import datetime
import os
import numpy as np

JHLI = "/home/ubuntu/Desktop/jhli"
NPY = f"{JHLI}/.stage3_oc2arm_R.npy"
META = f"{JHLI}/.stage3_oc2arm_R.meta.json"

# 1) 备份现有(错方向 SVD 那份), 不静默销毁
ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
for p in (NPY, META):
    if os.path.exists(p):
        shutil.copy2(p, f"{p}.bak-{ts}")
        print(f"备份 {p} -> {p}.bak-{ts}")

# 2) 写固定矩阵
R_fixed = np.array([[0., 0., 1.],
                    [1., 0., 0.],
                    [0., 1., 0.]], dtype=float)
det = float(np.linalg.det(R_fixed))
ortho = float(np.linalg.norm(R_fixed.T @ R_fixed - np.eye(3)))
assert abs(det - 1.0) < 1e-9 and ortho < 1e-9, f"非法旋转 det={det} ortho={ortho}"
np.save(NPY, R_fixed)

meta = {
    "saved_at": datetime.datetime.now().isoformat(),
    "source": "FIXED coordinate mapping (NOT SVD). 替代易错 2 手势标定;"
              " 双重印证: Realman vr_utils.RIGHT_ROTATION_MATRIX + 本会话实测反推。"
              " 会话流程: 戴头显面朝 base +X 长按 Meta 重置世界系(±几度可接受), 无需标定。",
    "fixed_mapping": True,
    "matrix": R_fixed.tolist(),
    "quality": {"oc_inter_deg": 90.0, "angle_err_deg": 0.0, "recon_max_deg": 0.0},
    "oc_ref_rotvec": [0.0, 0.0, 0.0],
}
json.dump(meta, open(META, "w"), ensure_ascii=False, indent=2)
print(f"写入 {NPY}  det={det:+.4f} ortho_err={ortho:.1e}")
print(R_fixed)

# 3) 经 vr_align 真实加载校验 + compute_delta_action 离线一致性 sanity
import sys
sys.path.insert(0, JHLI)
sys.path.insert(0, f"{JHLI}/lerobot_franka_teleop/lerobot_teleoperator_franka")
import vr_align
from scipy.spatial.transform import Rotation
from lerobot_teleoperator_franka import unityvr_mapping as m

R_load, meta_load = vr_align.load_rotation(NPY)
ok, oe, dt = vr_align.validate_rotation(np.asarray(R_load, float))
print(f"\nvr_align.load_rotation OK: det={dt:+.4f} ortho={oe:.1e} valid={ok}")
assert ok, "vr_align.validate_rotation 不通过"

# compute_delta_action 旋转项必须 == 换基忠实式 R_cal@rotvec(ΔR_oc); 并打印三轴对应
def T(rotvec):
    M = np.eye(4); M[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix(); return M
maxerr = 0.0
print("\noc(S系)单轴小转 -> base delta 主轴 (固定映射下应稳定):")
for ax, e in (("oc_x", [0.1, 0, 0]), ("oc_y", [0, 0.1, 0]), ("oc_z", [0, 0, 0.1])):
    pT, cT = T([0, 0, 0]), T(e)
    d = m.compute_delta_action(cT, pT, np.asarray(R_load, float), (1.0, 1.0), (1, 1, 1, 1, 1, 1))
    faithful = np.asarray(R_load, float) @ Rotation.from_matrix(cT[:3, :3] @ pT[:3, :3].T).as_rotvec()
    maxerr = max(maxerr, np.linalg.norm(d[3:] - faithful))
    j = int(np.argmax(np.abs(d[3:])))
    print(f"  {ax}+ -> delta_r={np.round(d[3:],4)}  主轴 base_{'xyz'[j]}{np.sign(d[3:][j]):+.0f}")
print(f"\n换基一致性 max‖delta_rot − R@rotvec(ΔR_oc)‖ = {maxerr:.2e} "
      f"({'OK' if maxerr < 1e-12 else 'FAIL'})")
print("\n安装完成。会话流程: 面朝 base +X 长按 Meta 重置 → 直接遥操(无需标定)。")
