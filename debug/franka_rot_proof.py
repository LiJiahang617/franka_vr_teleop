"""确定性离线证明: compute_delta_action 的旋转项命令的是【实际控制器增量旋转的逆】。
判据是刚体旋转的换基恒等式, 与任何地面真值/标定质量无关。无机器人/无 VR。"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(0)

# 现役 R_cal (16:17, 探针打印的真实矩阵)
R_cal = np.array([[ 0.0342,  0.0628, -0.9974],
                  [-0.9992, -0.0175, -0.0354],
                  [-0.0197,  0.9979,  0.0621]])
print(f"det(R_cal)={np.linalg.det(R_cal):+.4f}  (Kabsch 真旋转, 恒 +1)")

def to_transform(quat):
    """照抄 unity_vr_reader.to_transform 的旋转部分: S@R@S, S=diag(1,1,-1)。"""
    S = np.diag([1.0, 1.0, -1.0])
    return S @ Rot.from_quat(quat).as_matrix() @ S

def code_rot_delta(cur_T, prev_T, Rc):
    """照抄 unityvr_mapping.compute_delta_action 的 d[3:] 行(scaler=1, signs=1)。"""
    d_rot_oc = Rot.from_matrix(cur_T[:3, :3] @ prev_T[:3, :3].T).as_rotvec()
    return Rc @ (-d_rot_oc)                      # ← 现有公式 (Bug2 的 -)

def faithful_rot_delta(cur_T, prev_T, Rc):
    """物理忠实: 把控制器在 S 转换 oc 系做的增量旋转, 换基到 base。
    刚体换基: ΔR_base = Rc · ΔR_oc · Rcᵀ ; 其 rotvec = Rc @ rotvec(ΔR_oc) (Rc 真旋转)。"""
    dR_oc = cur_T[:3, :3] @ prev_T[:3, :3].T
    return Rc @ Rot.from_matrix(dR_oc).as_rotvec()

max_err_code, max_err_fix = 0.0, 0.0
for _ in range(2000):
    q0 = rng.normal(size=4); q0 /= np.linalg.norm(q0)
    # 小增量旋转(模拟一帧手腕动作)
    dq_axis = rng.normal(size=3); dq_axis /= np.linalg.norm(dq_axis)
    dq_ang = rng.uniform(0.005, 0.15)
    R1 = Rot.from_quat(q0).as_matrix()
    R2 = Rot.from_rotvec(dq_axis * dq_ang).as_matrix() @ R1
    q1 = Rot.from_matrix(R1).as_quat()
    q2 = Rot.from_matrix(R2).as_quat()
    prev_T = np.eye(4); prev_T[:3, :3] = to_transform(q1)
    cur_T  = np.eye(4); cur_T[:3, :3]  = to_transform(q2)

    faithful = faithful_rot_delta(cur_T, prev_T, R_cal)
    code     = code_rot_delta(cur_T, prev_T, R_cal)
    fixed    = faithful_rot_delta(cur_T, prev_T, R_cal)  # 提议修法 == 忠实式

    # 现公式 vs 忠实式: 应当恒等于「取负」(即命令了逆旋转)
    max_err_code = max(max_err_code, np.linalg.norm(code - (-faithful)))
    # 提议修法 vs 忠实式
    max_err_fix = max(max_err_fix, np.linalg.norm(fixed - faithful))

print(f"\n2000 次随机增量:")
print(f"  ‖现公式 - (−忠实式)‖ 最大 = {max_err_code:.2e}   "
      f"→ {'恒等(现公式==−忠实式==命令了逆旋转)' if max_err_code<1e-9 else '不恒等'}")
print(f"  ‖提议修法 - 忠实式‖     最大 = {max_err_fix:.2e}   "
      f"→ {'恒等(修法==忠实)' if max_err_fix<1e-12 else '不恒等'}")

# 单例直观: 控制器绕 oc-z 转 +10°, 看现公式 vs 忠实式
prev_T = np.eye(4); prev_T[:3, :3] = to_transform([0, 0, 0, 1])
cur_T  = np.eye(4); cur_T[:3, :3]  = to_transform(Rot.from_rotvec([0, 0, np.radians(10)]).as_quat())
print(f"\n单例 控制器 S-oc 系绕 +z 转 10°:")
print(f"  忠实式 base rotvec = {faithful_rot_delta(cur_T, prev_T, R_cal)}")
print(f"  现公式 base rotvec = {code_rot_delta(cur_T, prev_T, R_cal)}   (= −忠实式, 即逆向)")
