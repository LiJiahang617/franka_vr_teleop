"""只读 VR→delta 旁路诊断 — 不连机器人、不发任何指令。
复用 UnityVRReader + 现役 R_cal + compute_delta_action，逐 enabled tick 打印:
  控制器本帧增量旋转在(S 转换后 oc 系)的主轴 + 原始 Unity 系主轴 + 输出 delta[3:]。
用户按住 trigger 依次做: ①纯 yaw(绕竖直转手腕) ②纯 roll ③纯 pitch，各 ~3s。
据"手实际绕哪轴 vs delta_r{x,y,z} 符号"定位 z 是否反 / 是否手性问题。"""
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

_JHLI = "/home/ubuntu/Desktop/jhli/franka_vr_teleop"
sys.path.insert(0, _JHLI)
sys.path.insert(0, f"{_JHLI}/franka_vr_teleop/lerobot_teleoperator_franka")

import vr_align
from unity_vr_reader import UnityVRReader
from lerobot_teleoperator_franka import unityvr_mapping as m

R_cal, meta = vr_align.load_rotation(f"{_JHLI}/.stage3_oc2arm_R.npy")
R_cal = np.asarray(R_cal, float)
print(f"R_cal det={np.linalg.det(R_cal):+.4f} (Kabsch 应 +1)  quality={meta.get('quality')}")
print(f"R_cal=\n{np.round(R_cal,4)}")
np.set_printoptions(precision=4, suppress=True)

rd = UnityVRReader()
prev_T = None
AX = ["x", "y", "z"]
print("\n>>> 按住 trigger(≥0.85) 做受控单轴手势；Ctrl-C 结束。仅观测，机器人不会动。\n")
t0 = time.time()
n = 0
try:
    while time.time() - t0 < 90:
        tr, btn = rd.get_transformations_and_buttons()
        if not m.is_enabled(btn) or "r" not in tr:
            prev_T = None
            time.sleep(0.03)
            continue
        cur_T = np.asarray(tr["r"], float)
        if prev_T is not None:
            # 本帧增量旋转: 在 S 转换后 oc 系
            dR_oc = Rotation.from_matrix(cur_T[:3, :3] @ prev_T[:3, :3].T)
            rv_oc = dR_oc.as_rotvec()
            # delta（最终 base 系，照 compute_delta_action）
            d = m.compute_delta_action(cur_T, prev_T, R_cal, (1.0, 1.0), (1, 1, 1, 1, 1, 1))
            ang = np.degrees(np.linalg.norm(rv_oc))
            if ang > 1.5:  # 只打有意义的转动
                a_oc = AX[int(np.argmax(np.abs(rv_oc)))]
                a_d = AX[int(np.argmax(np.abs(d[3:])))]
                n += 1
                print(f"[{n:03d}] |Δrot|={ang:5.1f}°  oc轴≈{a_oc}{np.sign(rv_oc[np.argmax(np.abs(rv_oc))]):+.0f} "
                      f"rv_oc={rv_oc}  ->  delta_r=[{d[3]:+.4f},{d[4]:+.4f},{d[5]:+.4f}] 主轴≈{a_d}")
        prev_T = cur_T.copy()
        time.sleep(0.03)
except KeyboardInterrupt:
    pass
print(f"\n采样结束，共 {n} 条有效转动样本。")
