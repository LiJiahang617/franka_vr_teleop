"""Phase3 决定性: 用成功 Bug2 episode 的真实记录, 逐帧关联
commanded action/delta_ee_pose[3:] (base 系指令旋转增量)
vs measured observations/arm/pose[3:] 的实际变化 (base 系 EE rotvec)。
若 z 反相关而 x,y 正相关 -> z 一直就是反的(非回归, Bug2 修复 z 错)。无机器人/无 VR。"""
import glob
import numpy as np
import scipy.spatial.transform as st

np.set_printoptions(precision=4, suppress=True)
p = sorted(glob.glob("/home/ubuntu/Desktop/jhli/_hdf5_episodes/ep*.h5"))[0]
import h5py
f = h5py.File(p, "r")
cmd = np.array(f["/action/delta_ee_pose"])      # (N,6) base 系指令增量
pose = np.array(f["/observations/arm/pose"])    # (N,6) [xyz, rotvec3] base 系测量
print(f"file={p}  N={len(cmd)}")

cmd_rot = cmd[:, 3:]                              # 指令旋转 (N,3)
meas_rv = pose[:, 3:]                             # 测量 EE rotvec (N,3)

# 测量的逐帧 base 系增量旋转: dR_i = R_{i+k} * R_i^-1 (取 k 帧后, 容忍阻抗/插值滞后)
def measured_delta(k):
    out = np.full((len(cmd), 3), np.nan)
    for i in range(len(cmd) - k):
        Ri = st.Rotation.from_rotvec(meas_rv[i])
        Rj = st.Rotation.from_rotvec(meas_rv[i + k])
        out[i] = (Rj * Ri.inv()).as_rotvec()
    return out

for k in (1, 2, 3, 5):
    md = measured_delta(k)
    # 只取指令旋转有效的帧 (norm 足够大, 避免噪声主导)
    cn = np.linalg.norm(cmd_rot, axis=1)
    thr = np.nanpercentile(cn[cn > 0], 60) if (cn > 0).any() else 0
    sel = (cn > max(thr, 1e-3)) & np.isfinite(md).all(axis=1)
    n = int(sel.sum())
    if n < 10:
        print(f"k={k}: 有效帧太少 ({n})")
        continue
    print(f"\n--- k={k}帧滞后, 有效帧 n={n} (指令|rot|>{max(thr,1e-3):.4f}) ---")
    for ax, j in (("rx", 0), ("ry", 1), ("rz", 2)):
        c = cmd_rot[sel, j]
        m = md[sel, j]
        # 符号一致率 + 相关系数
        sign_agree = np.mean(np.sign(c) == np.sign(m))
        cc = np.corrcoef(c, m)[0, 1] if np.std(c) > 1e-9 and np.std(m) > 1e-9 else np.nan
        verdict = "正(对)" if (np.isfinite(cc) and cc > 0.15) else ("反(REVERSED)" if (np.isfinite(cc) and cc < -0.15) else "弱/不确定")
        print(f"  {ax}: 符号一致率={sign_agree:5.1%}  相关={cc:+.3f}  -> {verdict}")
