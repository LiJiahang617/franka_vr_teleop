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
        "_uvr", "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop/unity_vr_reader.py")
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
        "_uvr", "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop/unity_vr_reader.py")
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
