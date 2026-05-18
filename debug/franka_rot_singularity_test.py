"""Phase3 最小验证: 复现 _send_action_cartesian 旋转数学, 测 near-pi 奇异处
小 base 系 z 增量是否符号翻转; 并读成功 episode 的 EE 角度对照。无机器人/无 VR。"""
import glob
import numpy as np
import scipy.spatial.transform as st

np.set_printoptions(precision=4, suppress=True)


def applied_increment(cur_rotvec, base_delta_rotvec):
    """完全照抄 franka.py:_send_action_cartesian + server 转换链路。
    返回: 实际施加于刚体的 base 系增量旋转(target * cur^-1)的 rotvec。"""
    current_rot = st.Rotation.from_rotvec(cur_rotvec)
    delta_rot = st.Rotation.from_rotvec(base_delta_rotvec)
    target_rotation = delta_rot * current_rot                 # franka.py:373/378
    target_rotvec = target_rotation.as_rotvec()               # franka.py:374/379
    # server robot_update_desired_ee_pose: from_rotvec -> as_quat
    sent_quat = st.Rotation.from_rotvec(target_rotvec).as_quat()
    sent_rot = st.Rotation.from_quat(sent_quat)
    # 实际刚体在 base 系经历的增量 = sent * current^-1
    applied = sent_rot * current_rot.inv()
    return applied.as_rotvec()


# 意图: 绕 base z 转 +0.05 rad (用户感知的 "yaw")
intended = np.array([0.0, 0.0, 0.05])

print("=== A) 当前 near-pi EE 姿态 (rotvec=[-2.9871,0.7698,0.1358], θ≈176.9°) ===")
cur_now = np.array([-2.9871, 0.7698, 0.1358])
print(f"  θ={np.degrees(np.linalg.norm(cur_now)):.2f}°")
for ax, dv in [("base +Z yaw", intended),
               ("base +X roll", np.array([0.05, 0, 0])),
               ("base +Y pitch", np.array([0, 0.05, 0]))]:
    app = applied_increment(cur_now, dv)
    print(f"  intended {ax}={dv} -> applied={app}  (z 符号: 意图 {np.sign(dv[2]):+.0f} 实际 {np.sign(app[2]):+.0f})")

print("\n=== B) 一个远离奇异的 EE 姿态 (θ≈60°, 类似成功run) ===")
cur_ok = st.Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_rotvec()
print(f"  θ={np.degrees(np.linalg.norm(cur_ok)):.2f}°")
for ax, dv in [("base +Z yaw", intended),
               ("base +X roll", np.array([0.05, 0, 0])),
               ("base +Y pitch", np.array([0, 0.05, 0]))]:
    app = applied_increment(cur_ok, dv)
    print(f"  intended {ax}={dv} -> applied={app}  (z 符号: 意图 {np.sign(dv[2]):+.0f} 实际 {np.sign(app[2]):+.0f})")

print("\n=== C) 扫 θ 从 150°→179°(绕当前轴), 看 applied(+z yaw) 何时翻 ===")
axis = cur_now / np.linalg.norm(cur_now)
for deg in [150, 160, 170, 175, 177, 178, 179, 179.5]:
    cur = axis * np.radians(deg)
    app = applied_increment(cur, intended)
    flip = "FLIP" if np.sign(app[2]) != np.sign(intended[2]) else "ok"
    print(f"  θ={deg:6.1f}°  applied_z={app[2]:+.5f}  |app|={np.linalg.norm(app):.5f}  {flip}")

print("\n=== D) 成功 Bug2 episode 的 EE 姿态角度 (observations/arm/pose) ===")
try:
    import h5py
    p = sorted(glob.glob("/home/ubuntu/Desktop/jhli/_hdf5_episodes/ep*.h5"))[0]
    f = h5py.File(p, "r")
    pose = np.array(f["/observations/arm/pose"])  # (N,6) [xyz, rotvec3]
    ang = np.degrees(np.linalg.norm(pose[:, 3:], axis=1))
    print(f"  {p}")
    print(f"  N={len(ang)}  EE 旋转角(deg): min={ang.min():.1f} max={ang.max():.1f} "
          f"mean={ang.mean():.1f}  首帧={ang[0]:.1f}")
    print(f"  距 π(180°) 最近={180 - ang.max():.1f}°  (越大越远离奇异)")
except Exception as e:
    print(f"  读 hdf5 失败: {type(e).__name__}: {e}")
