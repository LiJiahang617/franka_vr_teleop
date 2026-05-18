"""只读 VR→delta 客观复验(修复+重标定后用)。不连机器人、不发指令。
每 tick: 校验 compute_delta_action 旋转项 == R_cal@rotvec(ΔR_oc)(换基忠实式, 修复已生效);
结束: 从三手势聚合 oc→base 旋转轴对应, 报对应矩阵 det 是否 ≈+1(正交右手/手性一致)。
用户按住 trigger 依次 ①纯yaw ②纯roll ③纯pitch 各 ~3s, Ctrl-C 结束。"""
import sys, time
import numpy as np
from scipy.spatial.transform import Rotation

_JHLI = "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop"
sys.path.insert(0, _JHLI)
sys.path.insert(0, f"{_JHLI}/lerobot_franka_teleop/lerobot_teleoperator_franka")
import vr_align
from unity_vr_reader import UnityVRReader
from lerobot_teleoperator_franka import unityvr_mapping as m

R_cal, meta = vr_align.load_rotation(f"{_JHLI}/.stage3_oc2arm_R.npy")
R_cal = np.asarray(R_cal, float)
print(f"R_cal det={np.linalg.det(R_cal):+.4f}  saved={meta.get('quality')}")
np.set_printoptions(precision=4, suppress=True)

rd = UnityVRReader()
prev_T = None
AX = ["x", "y", "z"]
rows_oc, rows_d = [], []          # 收集 (rotvec_oc, delta_rot) 供聚合
ident_max = 0.0
print("\n>>> 按住 trigger 做受控单轴手势(①yaw ②roll ③pitch); Ctrl-C 结束。仅观测。\n")
t0 = time.time(); n = 0
try:
    while time.time() - t0 < 120:
        tr, btn = rd.get_transformations_and_buttons()
        if not m.is_enabled(btn) or "r" not in tr:
            prev_T = None; time.sleep(0.03); continue
        cur_T = np.asarray(tr["r"], float)
        if prev_T is not None:
            dR_oc = cur_T[:3, :3] @ prev_T[:3, :3].T
            rv_oc = Rotation.from_matrix(dR_oc).as_rotvec()
            d = m.compute_delta_action(cur_T, prev_T, R_cal, (1.0, 1.0), (1, 1, 1, 1, 1, 1))
            faithful = R_cal @ rv_oc                       # 换基忠实式
            ident_max = max(ident_max, np.linalg.norm(d[3:] - faithful))
            ang = np.degrees(np.linalg.norm(rv_oc))
            if ang > 1.5:
                n += 1
                rows_oc.append(rv_oc); rows_d.append(d[3:])
                a_oc = AX[int(np.argmax(np.abs(rv_oc)))]
                a_d = AX[int(np.argmax(np.abs(d[3:])))]
                so = np.sign(rv_oc[np.argmax(np.abs(rv_oc))])
                sd = np.sign(d[3:][np.argmax(np.abs(d[3:]))])
                print(f"[{n:03d}] |Δ|={ang:5.1f}° oc≈{a_oc}{so:+.0f}  ->  "
                      f"delta_r=[{d[3]:+.4f},{d[4]:+.4f},{d[5]:+.4f}] 主轴≈{a_d}{sd:+.0f}")
        prev_T = cur_T.copy()
        time.sleep(0.03)
except KeyboardInterrupt:
    pass

print(f"\n样本 {n}。fix-live 校验: max‖delta_rot − R_cal@rotvec(ΔR_oc)‖ = {ident_max:.2e} "
      f"({'OK 修复已生效(换基忠实式)' if ident_max < 1e-9 else '❌ 不一致, 代码未生效?'})")
if n >= 9:
    A = np.array(rows_oc); B = np.array(rows_d)
    # 最小二乘解 B ≈ M @ A^T 的 M (oc->base 旋转轴线性映射), 看是否 ≈ 正交真旋转
    M = np.linalg.lstsq(A, B, rcond=None)[0].T
    U, Sg, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    print(f"聚合 oc→base 旋转轴映射 M：\n{M}")
    print(f"  最近正交阵 det={np.linalg.det(Rm):+.4f}  奇异值={Sg}  "
          f"(det≈+1 且奇异值≈相等 → 正交右手、三轴手性一致=正确)")
    for i, axn in enumerate(["oc_x", "oc_y", "oc_z"]):
        j = int(np.argmax(np.abs(M[:, i])))
        print(f"  {axn} → base_{AX[j]} (系数 {M[j,i]:+.3f})")
else:
    print("样本不足(<9), 请三手势各多做几秒重跑。")
