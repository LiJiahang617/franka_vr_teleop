"""UnityVR 世界系→base 的纯 delta 映射（无硬件，Route B 公式，可单测）。"""
import numpy as np
from scipy.spatial.transform import Rotation

# 位置(极矢量)固定映射, 与旋转(赝矢量, 走 R_cal 换基)分离 —— 验证参考
# realman vr_utils.RIGHT_POSITION_MATRIX=[[0,0,1],[-1,0,0],[0,1,0]](应用于原始
# Unity Δp, 验证有效) 与本会话用户实测反推一致(双重印证)。本管线 d_pos_oc=S@Δp
# (S=diag(1,1,-1), to_transform 已做), 故 _POS_MAP = RIGHT_POSITION_MATRIX @ S,
# det=+1。会话流程: 戴头显面朝 base+X 长按 Meta 重置世界系(±几度可接受), 无需标定。
_POS_MAP = np.array([[0., 0., -1.],
                     [-1., 0., 0.],
                     [0., 1., 0.]], dtype=float)


def compute_delta_action(cur_T, prev_T, R_cal, pose_scaler, channel_signs, *,
                         pos_axis_gain=(1., 1., 1.), rot_axis_gain=(1., 1., 1.)):
    """每 tick delta_ee_pose(6,)：oc 世界系位姿增量经已标定 R_cal 映射到 base。

    Args:
        cur_T, prev_T: 4x4（**必须是 UnityVRReader.to_transform 输出**: Unity 左手系
               经 S=diag(1,1,-1) 转右手系的位姿）。旋转项=刚体换基: 控制器在
               S-oc 系的增量旋转, 经真旋转 R_cal 换基到 base
               (rotvec(R_cal·ΔR·R_calᵀ)=R_cal@rotvec(ΔR))。位置(极矢量)与旋转
               (赝矢量)张量性不同, **分用不同矩阵**(验证参考 vr_utils 同款设计):
               位置走固定 _POS_MAP, 旋转走 R_cal@d_rot_oc。R_cal 现为固定坐标
               映射(非 SVD 标定), 配合面朝 base+X 长按 Meta 重置世界系。
        R_cal: 3x3 已标定 oc->base 旋转（vr_align）
        pose_scaler: [pos_scale, ori_scale] 全局标量增益
        channel_signs: 长 6 的 ±1
        pos_axis_gain: (keyword-only) 位置每轴增益 [gx, gy, gz]，默认 (1,1,1)。
               仅在已验通映射方向输出后逐轴缩放，不改 _POS_MAP 方向/手性（§10.2(0)
               红线，关联 lesson kabsch-cannot-absorb-handedness）。默认全 1 时输出
               逐字等价历史 pose_scaler 两标量行为。
        rot_axis_gain: (keyword-only) 旋转每轴增益 [grx, gry, grz]，默认 (1,1,1)。
               仅在 R_cal@d_rot_oc 换基输出后逐轴缩放，不改换基公式（§10.2(0)）。
    Returns:
        np.ndarray (6,) = [dx,dy,dz,drx,dry,drz]（base 系）
    """
    cur_T = np.asarray(cur_T, float)
    prev_T = np.asarray(prev_T, float)
    R_cal = np.asarray(R_cal, float)
    d_pos_oc = cur_T[:3, 3] - prev_T[:3, 3]
    d_rot_oc = Rotation.from_matrix(cur_T[:3, :3] @ prev_T[:3, :3].T).as_rotvec()
    d = np.zeros(6)
    d[:3] = _POS_MAP @ d_pos_oc  # 位置: 固定极矢量映射(验证参考+实测双印证), 不走 R_cal
    d[3:] = R_cal @ d_rot_oc  # 刚体换基: 控制器增量旋转(S-oc系)经真旋转 R_cal 忠实表达到 base (= rotvec(R_cal·ΔR_oc·R_calᵀ)); 旧式取负实为命令逆旋转(Bug2)
    s = np.asarray(channel_signs, float)
    ps, os_ = float(pose_scaler[0]), float(pose_scaler[1])
    pg = np.asarray(pos_axis_gain, float)  # §11.3 per-axis 位置增益（默认全1=历史行为）
    rg = np.asarray(rot_axis_gain, float)  # §11.3 per-axis 旋转增益（默认全1=历史行为）
    d[:3] = d[:3] * ps * pg * s[:3]
    d[3:] = d[3:] * os_ * rg * s[3:]
    return d


TRIGGER_ENABLE_TH = 0.85


def is_enabled(buttons, th=TRIGGER_ENABLE_TH):
    """Realman 约定: trigger(模拟量) >= 阈值 → 遥操使能/锚定（按住式）。"""
    rt = buttons.get("rightTrig", (0.0,))
    tv = rt[0] if isinstance(rt, (tuple, list)) and len(rt) > 0 else 0.0
    return float(tv) >= th


def next_gripper_closed(prev_closed, grip_prev, grip_now):
    """Realman 约定: grip 上升沿(松->按) 翻转夹爪开/合(toggle)。返回新的 closed 布尔。"""
    if (not grip_prev) and grip_now:
        return not prev_closed
    return prev_closed
