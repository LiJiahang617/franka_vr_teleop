import numpy as np
from scipy.spatial.transform import Rotation
import importlib.util, sys

# 直接加载模块文件，避免触发 __init__.py（后者依赖 lerobot 包未装）
_MAPPING_FILE = (
    "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop"
    "/lerobot_teleoperator_franka/lerobot_teleoperator_franka/unityvr_mapping.py"
)
spec = importlib.util.spec_from_file_location("unityvr_mapping", _MAPPING_FILE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _T(pos, rot=np.eye(3)):
    T = np.eye(4); T[:3, :3] = rot; T[:3, 3] = pos
    return T


def test_position_through_fixed_POS_MAP_and_scaler():
    # 位置走固定 _POS_MAP(=验证参考 RIGHT_POSITION_MATRIX∘S), 不再过 R_cal。
    R = Rotation.from_euler("z", 90, degrees=True).as_matrix()  # 仅作旋转项参数, 不应影响位置
    prev = _T([0, 0, 0]); cur = _T([0.10, 0.0, 0.0])            # d_pos_oc=[0.1,0,0]
    d = m.compute_delta_action(cur, prev, R, pose_scaler=[2.0, 1.0],
                               channel_signs=[1, 1, 1, 1, 1, 1])
    # _POS_MAP@[0.1,0,0]=[0,-0.1,0]; *pos_scaler2 -> [0,-0.2,0]; 与传入 R 无关
    assert np.allclose(d[:3], [0.0, -0.2, 0.0], atol=1e-9)
    assert np.allclose(d[3:], [0.0, 0.0, 0.0], atol=1e-9)


def test_rotation_delta_mapped_by_R():
    R = np.eye(3)
    prev = _T([0, 0, 0])
    cur = _T([0, 0, 0], Rotation.from_rotvec([0.0, 0.0, 0.2]).as_matrix())
    d = m.compute_delta_action(cur, prev, R, pose_scaler=[1.0, 3.0],
                               channel_signs=[1, 1, 1, 1, 1, 1])
    assert np.allclose(d[:3], [0, 0, 0], atol=1e-9)
    # 旋转为刚体换基: R=I 时 d_rot_oc=[0,0,0.2], R@d_rot_oc*ori_scale(3.0)=[0,0,0.6]。
    # (旧期望 -0.6 是 Bug2 误把赝矢量当极矢量取负 → 命令了逆旋转, 已纠正)
    assert np.allclose(d[3:], [0.0, 0.0, 0.6], atol=1e-6)


def test_channel_signs_applied():
    R = np.eye(3)
    d = m.compute_delta_action(_T([0.1, 0.2, 0.3]), _T([0, 0, 0]), R,
                               pose_scaler=[1.0, 1.0],
                               channel_signs=[-1, 1, -1, 1, 1, 1])
    # _POS_MAP@[0.1,0.2,0.3]=[-0.3,-0.1,0.2]; *signs[-1,1,-1] -> [0.3,-0.1,-0.2]
    assert np.allclose(d[:3], [0.3, -0.1, -0.2], atol=1e-9)


def test_is_enabled_trigger_threshold():
    assert m.is_enabled({"rightTrig": (0.9,)}) is True
    assert m.is_enabled({"rightTrig": (0.85,)}) is True
    assert m.is_enabled({"rightTrig": (0.84,)}) is False
    assert m.is_enabled({}) is False


def test_next_gripper_closed_toggle_on_rising_edge():
    # 初始 open(False); 上升沿翻转
    assert m.next_gripper_closed(False, False, True) is True   # 松->按: open->closed
    assert m.next_gripper_closed(True, False, True) is False   # 再按: closed->open
    # 按住不变 / 松开不变 / 下降沿不变
    assert m.next_gripper_closed(True, True, True) is True
    assert m.next_gripper_closed(False, True, False) is False
    assert m.next_gripper_closed(True, True, False) is True


def test_rotation_is_change_of_basis_of_controller_increment():
    """正确规格(换基恒等式): compute_delta_action 的旋转项(去 scaler/signs)必须等于
    把控制器在 S-oc 系做的增量旋转 ΔR_oc 换基到 base, 即
        rotvec(R_cal · ΔR_oc · R_calᵀ) == R_cal @ rotvec(ΔR_oc)。
    旋转是赝矢量, 与位置(极矢量)各按自身张量性变换 —— 不应被强行要求同手性
    (旧 test_rotation_handedness_consistent_with_position 把 Bug2 谬误写成了规格)。"""
    import importlib.util, numpy as np
    from scipy.spatial.transform import Rotation
    _s = importlib.util.spec_from_file_location(
        "_uvr", "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop/lerobot_teleoperator_franka/lerobot_teleoperator_franka/unity_vr_reader.py")
    _uvr = importlib.util.module_from_spec(_s); _s.loader.exec_module(_uvr)
    _sm = importlib.util.spec_from_file_location(
        "_uvm", "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop/"
        "lerobot_teleoperator_franka/lerobot_teleoperator_franka/unityvr_mapping.py")
    _uvm = importlib.util.module_from_spec(_sm); _sm.loader.exec_module(_uvm)

    def T(pos, quat):
        return _uvr.to_transform({"pos": pos, "quat": quat,
                                  "grip": 0.0, "A": 0, "B": 0, "trigger": 0.0})

    rng = np.random.default_rng(7)
    for _ in range(50):
        R_cal = Rotation.random(random_state=rng).as_matrix()  # 任意真旋转(det=+1)
        q0 = Rotation.random(random_state=rng).as_quat()
        dq = Rotation.from_rotvec(
            rng.normal(size=3) / np.linalg.norm(rng.normal(size=3)) * rng.uniform(0.01, 0.2))
        q1 = (dq * Rotation.from_quat(q0)).as_quat()
        pT, cT = T([0, 0, 0], q0.tolist()), T([0, 0, 0], q1.tolist())
        out = _uvm.compute_delta_action(cT, pT, R_cal, [1.0, 1.0], [1, 1, 1, 1, 1, 1])[3:]
        dR_oc = cT[:3, :3] @ pT[:3, :3].T
        expect = R_cal @ Rotation.from_matrix(dR_oc).as_rotvec()      # 换基忠实式
        assert np.allclose(out, expect, atol=1e-9), (
            f"旋转项非换基忠实式: out={out} expect={expect} "
            f"(差≈{np.linalg.norm(out-expect):.2e}; 若 ≈‖2·expect‖ 即命令了逆旋转)")


def test_position_matches_verified_RIGHT_POSITION_MATRIX():
    """正确规格: to_transform 输入下, 纯 Unity 平移映射 == 验证过的 Realman
    RIGHT_POSITION_MATRIX=[[0,0,1],[-1,0,0],[0,1,0]] @ Δp_unity (极矢量 LH->RH)。
    与本会话用户实测"位置 x,y 反"反推一致(双重印证)。"""
    import importlib.util, numpy as np
    _s = importlib.util.spec_from_file_location(
        "_uvr", "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop/lerobot_teleoperator_franka/lerobot_teleoperator_franka/unity_vr_reader.py")
    _uvr = importlib.util.module_from_spec(_s); _s.loader.exec_module(_uvr)
    Pv = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]], float)  # 验证参考

    from scipy.spatial.transform import Rotation as _R
    # 单位四元数被 to_transform 当零位姿; 用固定微小非单位朝向(prev/cur 同朝向,
    # 旋转增量为 0), 位置部分 T[:3,3]=S@pos 与朝向无关。
    _q = _R.from_rotvec([0.02, -0.01, 0.015]).as_quat().tolist()

    def T(pos):
        return _uvr.to_transform({"pos": pos, "quat": _q,
                                  "grip": 0.0, "A": 0, "B": 0, "trigger": 0.0})
    rng = np.random.default_rng(3)
    Rid = np.eye(3)
    for _ in range(20):
        dp = rng.normal(size=3) * 0.1
        pT, cT = T([0.0, 0.0, 0.0]), T(dp.tolist())
        out = m.compute_delta_action(cT, pT, Rid, [1.0, 1.0], [1, 1, 1, 1, 1, 1])[:3]
        assert np.allclose(out, Pv @ dp, atol=1e-9), (
            f"位置非验证参考极矢量映射: out={out} expect={Pv @ dp}")


# ===== Task 1: §11.3 per-axis 增益层测试 (Step 1 新增) =====

def test_pos_axis_gain_scales_each_position_axis_independently():
    """§11.3: pos_axis_gain 逐轴缩放位置各分量，互不影响。
    用先取无增益基线再乘的断言方式，避免硬编码 _POS_MAP 数值造成脆性。"""
    R = np.eye(3)
    prev = _T([0, 0, 0])
    cur = _T([0.1, 0.2, 0.3])
    signs = [1, 1, 1, 1, 1, 1]
    ps = [1.0, 1.0]
    # 先取无增益基线
    base = m.compute_delta_action(cur, prev, R, pose_scaler=ps, channel_signs=signs)
    # 施加逐轴增益
    pg = [2.0, 5.0, 10.0]
    gained = m.compute_delta_action(cur, prev, R, pose_scaler=ps, channel_signs=signs,
                                    pos_axis_gain=pg)
    # 位置各轴 = 基线逐轴 × pg
    assert np.allclose(gained[:3], base[:3] * np.array(pg), atol=1e-12), (
        f"位置增益不匹配: gained={gained[:3]} base={base[:3]} pg={pg}")
    # 旋转项不应受 pos_axis_gain 影响
    assert np.allclose(gained[3:], base[3:], atol=1e-12), (
        f"pos_axis_gain 不应影响旋转: gained[3:]={gained[3:]} base[3:]={base[3:]}")


def test_rot_axis_gain_scales_each_rotation_axis_independently():
    """§11.3: rot_axis_gain 逐轴缩放旋转各分量，仅改目标轴，其余轴不变。"""
    R = np.eye(3)
    prev = _T([0, 0, 0])
    # 仅绕 z 轴旋转 0.2 rad
    cur = _T([0, 0, 0], Rotation.from_rotvec([0.0, 0.0, 0.2]).as_matrix())
    signs = [1, 1, 1, 1, 1, 1]
    ps = [1.0, 1.0]
    # 无增益基线
    base = m.compute_delta_action(cur, prev, R, pose_scaler=ps, channel_signs=signs)
    # z 轴增益 = 3，其余 = 1
    rg = [1.0, 1.0, 3.0]
    gained = m.compute_delta_action(cur, prev, R, pose_scaler=ps, channel_signs=signs,
                                    rot_axis_gain=rg)
    # 旋转 z 分量应放大 3 倍
    assert np.allclose(gained[3:], base[3:] * np.array(rg), atol=1e-12), (
        f"旋转增益不匹配: gained[3:]={gained[3:]} base[3:]={base[3:]} rg={rg}")
    # 位置项不应受 rot_axis_gain 影响
    assert np.allclose(gained[:3], base[:3], atol=1e-12), (
        f"rot_axis_gain 不应影响位置: gained[:3]={gained[:3]} base[:3]={base[:3]}")


def test_axis_gain_default_equals_legacy_pose_scaler_behavior():
    """§11.3 向后兼容: 不传 pos/rot_axis_gain == 显式传 (1,1,1)，数值逐字等价。"""
    R = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    prev = _T([0.1, -0.2, 0.05])
    cur = _T([0.15, -0.18, 0.07], Rotation.from_rotvec([0.05, -0.03, 0.1]).as_matrix())
    ps = [2.5, 1.8]
    signs = [1, -1, 1, -1, 1, 1]
    # 不传增益（走默认）
    d_legacy = m.compute_delta_action(cur, prev, R, pose_scaler=ps, channel_signs=signs)
    # 显式传全 1 增益
    d_explicit = m.compute_delta_action(cur, prev, R, pose_scaler=ps, channel_signs=signs,
                                        pos_axis_gain=(1., 1., 1.),
                                        rot_axis_gain=(1., 1., 1.))
    assert np.allclose(d_legacy, d_explicit, atol=1e-12), (
        f"默认增益应与显式 (1,1,1) 逐字等价: legacy={d_legacy} explicit={d_explicit}")


def test_axis_gain_does_not_change_direction_or_handedness():
    """§11.3 §10.2(0) 红线守门: 正增益下每非零分量的符号（方向/手性）与无增益基线一致，
    且 gained[i] == base[i] * gain[i] 严格成立（增益是纯标量逐轴缩放，与方向正交）。
    30 随机用例覆盖任意 R_cal、任意位置/旋转增量。"""
    rng = np.random.default_rng(42)
    for _ in range(30):
        R_cal = Rotation.random(random_state=rng).as_matrix()
        pos_delta = rng.normal(size=3) * 0.1
        rot_delta = rng.normal(size=3) * 0.2
        prev = _T([0, 0, 0])
        cur = _T(pos_delta, Rotation.from_rotvec(rot_delta).as_matrix())
        signs = [1, 1, 1, 1, 1, 1]
        ps = [1.0, 1.0]
        # 无增益基线
        base = m.compute_delta_action(cur, prev, R_cal, pose_scaler=ps, channel_signs=signs)
        # 随机正增益（避免零增益混淆符号判断）
        pg = rng.uniform(0.5, 5.0, size=3)
        rg = rng.uniform(0.5, 5.0, size=3)
        gained = m.compute_delta_action(cur, prev, R_cal, pose_scaler=ps, channel_signs=signs,
                                        pos_axis_gain=pg, rot_axis_gain=rg)
        # 严格逐轴等式：gained == base * gain
        assert np.allclose(gained[:3], base[:3] * pg, atol=1e-12), (
            f"位置逐轴等式违反: gained={gained[:3]} base={base[:3]} pg={pg}")
        assert np.allclose(gained[3:], base[3:] * rg, atol=1e-12), (
            f"旋转逐轴等式违反: gained[3:]={gained[3:]} base[3:]={base[3:]} rg={rg}")
        # 符号守门：非零分量符号不变
        for i in range(3):
            if abs(base[i]) > 1e-10:
                assert np.sign(gained[i]) == np.sign(base[i]), (
                    f"位置轴 {i} 方向翻转! base={base[i]:.6f} gained={gained[i]:.6f} pg={pg[i]:.3f}")
        for i in range(3):
            if abs(base[3 + i]) > 1e-10:
                assert np.sign(gained[3 + i]) == np.sign(base[3 + i]), (
                    f"旋转轴 {i} 方向翻转! base={base[3+i]:.6f} gained={gained[3+i]:.6f} rg={rg[i]:.3f}")


def test_axis_gain_keyword_only_does_not_break_5_positional_call():
    """§11.3 向后兼容: 5 位置参调用仍正常工作，新参 keyword-only 不破坏既有调用。"""
    R = np.eye(3)
    prev = _T([0, 0, 0])
    cur = _T([0.05, -0.03, 0.1])
    # 5 位置参（unityvr_robot.py 与既有 7 测试的调用方式）
    d = m.compute_delta_action(cur, prev, R, [1.0, 1.0], [1, 1, 1, 1, 1, 1])
    assert d.shape == (6,), f"5位置参调用应返回 shape (6,)，实际 {d.shape}"
    # 结果与显式传默认增益一致
    d_default = m.compute_delta_action(cur, prev, R, [1.0, 1.0], [1, 1, 1, 1, 1, 1],
                                       pos_axis_gain=(1., 1., 1.),
                                       rot_axis_gain=(1., 1., 1.))
    assert np.allclose(d, d_default, atol=1e-12)


# ===== review-fix: §11.3 shape(3,)+finite 校验测试 (新增 2 个) =====

def test_axis_gain_rejects_wrong_shape():
    """pos/rot_axis_gain 非 (3,) shape 时必须 fail-loud 抛 ValueError。
    涵盖: 标量、len1、len2、len4、shape(3,1) 各畸形。"""
    R = np.eye(3)
    prev = _T([0, 0, 0])
    cur = _T([0.05, 0.0, 0.0])
    ps = [1.0, 1.0]
    signs = [1, 1, 1, 1, 1, 1]

    # pos_axis_gain 畸形用例
    bad_pos_cases = [
        2.0,            # 标量 → shape ()
        (2.0,),         # len1 → shape (1,)
        [1., 1.],       # len2 → shape (2,)
        [1., 1., 1., 1.],  # len4 → shape (4,)
        [[1.], [1.], [1.]],  # shape (3,1)
    ]
    for bad in bad_pos_cases:
        import pytest
        with pytest.raises(ValueError, match="pos_axis_gain 必须 shape"):
            m.compute_delta_action(cur, prev, R, ps, signs, pos_axis_gain=bad)

    # rot_axis_gain 畸形用例
    bad_rot_cases = [
        2.0,
        (2.0,),
        [1., 1.],
        [1., 1., 1., 1.],
        [[1.], [1.], [1.]],
    ]
    for bad in bad_rot_cases:
        import pytest
        with pytest.raises(ValueError, match="rot_axis_gain 必须 shape"):
            m.compute_delta_action(cur, prev, R, ps, signs, rot_axis_gain=bad)


def test_axis_gain_rejects_non_finite():
    """pos/rot_axis_gain 含 nan 或 inf 时必须 fail-loud 抛 ValueError。"""
    R = np.eye(3)
    prev = _T([0, 0, 0])
    cur = _T([0.05, 0.0, 0.0])
    ps = [1.0, 1.0]
    signs = [1, 1, 1, 1, 1, 1]
    import pytest

    # pos_axis_gain 含 nan
    with pytest.raises(ValueError, match="非有限"):
        m.compute_delta_action(cur, prev, R, ps, signs,
                               pos_axis_gain=[1., float("nan"), 1.])
    # pos_axis_gain 含 inf
    with pytest.raises(ValueError, match="非有限"):
        m.compute_delta_action(cur, prev, R, ps, signs,
                               pos_axis_gain=[1., float("inf"), 1.])
    # rot_axis_gain 含 nan
    with pytest.raises(ValueError, match="非有限"):
        m.compute_delta_action(cur, prev, R, ps, signs,
                               rot_axis_gain=[1., float("nan"), 1.])
    # rot_axis_gain 含 inf
    with pytest.raises(ValueError, match="非有限"):
        m.compute_delta_action(cur, prev, R, ps, signs,
                               rot_axis_gain=[1., float("inf"), 1.])
