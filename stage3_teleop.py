"""
Stage 3 安全遥操作测试 — 初始锚定方案 (anchored to init pose).

参考 Realman/vr_utils.py 的思路, 抛弃 per-tick delta 累加:
  RG 按下瞬间   : 锚定 init_vr_T (oculus frame 4x4), init_arm_pos / init_arm_R
  RG 持续按住   : target_arm = init_arm + POS_MATRIX @ (cur_vr_pos - init_vr_pos) * pos_signs * scaler_pos
                  target_arm_R = R(rotvec_arm) * init_arm_R
                                 其中 rotvec_arm = ROT_MATRIX @ rotvec(init_vr_R⁻¹ * cur_vr_R) * rot_signs * scaler_ori
  RG 松开       : 丢锚, target 保持不变 (机械臂 hold)

优点 (相比 delta 累加):
  - clamp/rate-limit 不丢位移 (target 绝对计算, 不再受历史误差累积)
  - 方向 debug 直观: rel_vr_pos 和 arm_target_dpos 一一对应
  - 用户抖动只影响 cur_vr 瞬时, 不写入 target 历史

仍保留: cartesian impedance + auto-restart on policy terminate + 50Hz keep-alive + CSV 日志 + 段汇总.
"""
import argparse
import csv
import datetime
import logging
import threading
import time
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot_robot_franka.franka_interface_client import FrankaInterfaceClient
from lerobot_teleoperator_franka.oculus.oculus_reader import OculusReader

import vr_align


# ====== 默认参数 ======
QUEST_IP = None         # USB
ROBOT_IP = "127.0.0.1"
ROBOT_PORT = 4242

SCALER_POS = 0.5
SCALER_ORI = 0.3
RATE_LIMIT_POS = 0.005                # 单 tick (20ms) 内 target 最大位移 m
RATE_LIMIT_ORI = np.deg2rad(5.0)     # 单 tick 最大姿态变化 rad
VR_JUMP_THRESHOLD = 0.05             # cur_vr - prev_vr 单 tick 上限 (m). 超过视为 Quest 追踪跳变, 丢弃

LOOP_HZ = 50
DURATION_S = 60.0
KEEPALIVE_DT = 0.05

OC2ARM_R_PATH = "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop/.stage3_oc2arm_R.npy"
PLATFORM_TOOLS_DIR = "/home/ubuntu/Desktop/jhli/platform-tools"
CAL_MIN_DISP = 0.12          # 每手势最小位移 m
CAL_ANGLE_TOL_DEG = 15.0     # 两手势夹角与已知夹角(90°)允许误差
DRIFT_WARN_DEG = 20.0        # 启动时 oc 姿态与标定基差超此值告警(含扭头)

# === 坐标变换矩阵预设 ===
# arm_axis = matrix @ vr_axis
# oculus_reader 注释: oculus X=右, Y=上, Z=朝用户(后); robot X=前, Y=左, Z=上
POS_MATRIX_PRESETS = {
    "oculus_reader": np.array([[0, 0, -1],   # arm_x = -oc_z
                               [-1, 0, 0],   # arm_y = -oc_x
                               [0, 1, 0]],   # arm_z = +oc_y
                              dtype=float),
    # Realman 风格 (Z 轴翻转, 验证 rail-berkeley APK 实际 Z 是否"前进 = 正")
    "realman":       np.array([[0, 0, 1],    # arm_x = +oc_z
                               [-1, 0, 0],
                               [0, 1, 0]],
                              dtype=float),
}

ROT_MATRIX_PRESETS = {
    "oculus_reader": np.array([[0, 0, 1],
                               [1, 0, 0],
                               [0, 1, 0]],
                              dtype=float),
    "realman":       np.array([[0, 0, 1],
                               [1, 0, 0],
                               [0, 1, 0]],
                              dtype=float),
}


def _ensure_platform_tools_path():
    """确保 OculusReader 内部 os.system(adb ...) 能找到本项目自带 adb。"""
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if PLATFORM_TOOLS_DIR not in parts:
        os.environ["PATH"] = PLATFORM_TOOLS_DIR + os.pathsep + os.environ.get("PATH", "")


# ====== 工具函数 ======
def _is_controller_dead(exc: BaseException) -> bool:
    s = str(exc).lower()
    return ("no controller running" in s
            or "controller is not running" in s
            or "polymetis server error" in s)


def _parse_signs(s: str) -> np.ndarray:
    parts = [int(x) for x in s.split(",")]
    assert len(parts) == 3 and all(p in (-1, 1) for p in parts), \
        f"signs 必须 3 个 ±1, 逗号分隔, 得到 {s}"
    return np.array(parts, dtype=float)


def _rate_limit_step(cur_6d: np.ndarray, desired_6d: np.ndarray,
                     max_pos: float, max_ori: float) -> np.ndarray:
    """target 单 tick 步进幅度限制. 不丢位移 (下个 tick 继续追)."""
    dpos = desired_6d[:3] - cur_6d[:3]
    n = np.linalg.norm(dpos)
    if n > max_pos:
        dpos = dpos * (max_pos / n)
    new_pos = cur_6d[:3] + dpos

    cur_R = R.from_rotvec(cur_6d[3:])
    desired_R = R.from_rotvec(desired_6d[3:])
    rel_R = desired_R * cur_R.inv()
    rel_rv = rel_R.as_rotvec()
    rn = np.linalg.norm(rel_rv)
    if rn > max_ori:
        rel_rv = rel_rv * (max_ori / rn)
    new_R = R.from_rotvec(rel_rv) * cur_R
    return np.concatenate([new_pos, new_R.as_rotvec()])


def _R_yaw_oc(yaw_deg: float) -> np.ndarray:
    """绕 oculus 垂直轴 (oc_Y) 的 3x3 旋转, 用于校正 yaw"""
    yaw_rad = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)


def _start_cartesian_and_warmup(c: FrankaInterfaceClient) -> np.ndarray:
    """启动 cartesian + warm-up. start_cartesian 是异步, 需要短暂等待
    再发首个 update, 否则会撞 "no controller running". 加 retry 兜底."""
    c.server.robot_start_cartesian_impedance_control(None, None)
    # 给 polymetis 时间把 TorchScript policy 注册成 current
    time.sleep(0.3)
    cur_ee = np.asarray(c.server.robot_get_ee_pose(), dtype=float)
    last_err = None
    for attempt in range(5):
        try:
            c.server.robot_update_desired_ee_pose(cur_ee.tolist())
            return cur_ee
        except Exception as e:
            last_err = e
            if "no controller running" in str(e).lower() or "controller is not running" in str(e).lower():
                time.sleep(0.2)
                continue
            raise
    raise RuntimeError(f"warm-up 失败 5 次: {last_err}")


def _safe_update(c: FrankaInterfaceClient, target_6d: np.ndarray, log: logging.Logger):
    try:
        c.server.robot_update_desired_ee_pose(target_6d.tolist())
        return target_6d, False
    except Exception as e:
        if _is_controller_dead(e):
            log.warning(f"controller terminated, 重启 cartesian: {e}")
            cur_ee = _start_cartesian_and_warmup(c)
            return cur_ee, True
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=DURATION_S)
    parser.add_argument("--quest-ip", default=QUEST_IP,
                        help="None / 空 = USB; 否则 WiFi adb 的 Quest IP")
    parser.add_argument("--scaler-pos", type=float, default=SCALER_POS)
    parser.add_argument("--scaler-ori", type=float, default=SCALER_ORI)
    parser.add_argument("--pos-matrix", choices=list(POS_MATRIX_PRESETS), default="oculus_reader")
    parser.add_argument("--rot-matrix", choices=list(ROT_MATRIX_PRESETS), default="oculus_reader")
    parser.add_argument("--pos-signs", type=str, default="1,1,1",
                        help="位置三轴 sign 调整 (作用在矩阵输出之后)")
    parser.add_argument("--rot-signs", type=str, default="-1,-1,1")
    parser.add_argument("--lock-rotation", action="store_true", help="锁住 target 姿态 = init_arm_R, 仅平移 (标定用)")
    parser.add_argument("--rate-pos", type=float, default=RATE_LIMIT_POS, help="单 tick 位置上限 m")
    parser.add_argument("--rate-ori", type=float, default=RATE_LIMIT_ORI, help="单 tick 姿态上限 rad")
    parser.add_argument("--yaw-deg", type=float, default=0.0,
                        help="绕 oculus 垂直轴 (oc_Y) 旋转 VR 参考系 N°. 用于校正你站位与 Franka base 的偏角. 可选 0/45/90/135/180/225/270/315 扫一轮看哪个对")
    parser.add_argument("--log-csv", type=str, default="")
    parser.add_argument("--vr-source", choices=["unity", "oculus"], default="unity",
                        help="VR 数据源: unity=世界系(头显独立,推荐) / oculus=旧 head-relative")
    args = parser.parse_args()

    # 基矩阵 (不带 yaw); yaw 用 _R_yaw_oc 实时合成, 支持运行中按 A/B 调整
    pos_M_base = POS_MATRIX_PRESETS[args.pos_matrix].copy()
    rot_M_base = ROT_MATRIX_PRESETS[args.rot_matrix].copy()
    yaw_deg = float(args.yaw_deg)
    pos_M = pos_M_base @ _R_yaw_oc(yaw_deg)
    rot_M = rot_M_base @ _R_yaw_oc(yaw_deg)
    pos_signs = _parse_signs(args.pos_signs)
    rot_signs = _parse_signs(args.rot_signs)
    scaler_pos = args.scaler_pos
    scaler_ori = args.scaler_ori

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("stage3")

    _loaded = vr_align.load_rotation(OC2ARM_R_PATH)
    _R = _loaded[0] if _loaded is not None else None
    pos_M, rot_M, pos_signs, rot_signs, _mode = vr_align.resolve_mapping(
        _R, pos_M, rot_M, pos_signs, rot_signs)
    if _mode == "calibrated":
        _q = _loaded[1].get("quality", {})
        _angle_err = _q.get("angle_err_deg", "?")
        _recon_max = _q.get("recon_max_deg", "?")
        log.info(f"已加载标定 R (角误差={_angle_err} deg 重建残差={_recon_max} deg), calibrated 映射")
    else:
        log.warning("未找到标定 R, 用 legacy yaw-only 映射 (按 A 走 2 手势标定)")

    print("=" * 64)
    print("Stage 3 遥操作测试 (初始锚定方案)")
    print(f"  时长 {args.duration:.0f}s, scaler=({scaler_pos},{scaler_ori}), "
          f"rate_limit={args.rate_pos*1000:.1f}mm/{np.rad2deg(args.rate_ori):.0f}°/step, {LOOP_HZ}Hz")
    print(f"  pos_matrix={args.pos_matrix} pos_signs={args.pos_signs}")
    print(f"  rot_matrix={args.rot_matrix} rot_signs={args.rot_signs} yaw_deg={args.yaw_deg}"
          + (" [LOCKED]" if args.lock_rotation else ""))
    print("⚠️  E-stop 在手边, 手柄全程在头显视野内, RG 按下才动")
    print("=" * 64)

    log.info(f"连 zerorpc {ROBOT_IP}:{ROBOT_PORT}")
    c = FrankaInterfaceClient(ip=ROBOT_IP, port=ROBOT_PORT)

    log.info("启动 cartesian + warm-up ...")
    target_6d = _start_cartesian_and_warmup(c)
    log.info(f"初始 EE pos={target_6d[:3]}, rotvec={target_6d[3:]}")

    log.info("启 OculusReader, 主线程 keep-alive ...")
    reader_box = [None]
    err_box = [None]

    def _init_reader():
        try:
            if args.vr_source == "unity":
                from unity_vr_reader import UnityVRReader
                reader_box[0] = UnityVRReader()
            else:
                reader_box[0] = OculusReader(ip_address=args.quest_ip)
        except BaseException as e:
            err_box[0] = e

    th = threading.Thread(target=_init_reader, daemon=True)
    th.start()
    while th.is_alive():
        target_6d, _ = _safe_update(c, target_6d, log)
        time.sleep(KEEPALIVE_DT)
    if err_box[0] is not None:
        raise err_box[0]
    oculus = reader_box[0]
    log.info("OculusReader OK")

    if _mode == "calibrated" and _loaded[1].get("oc_ref_rotvec"):
        _rv = None
        for _ in range(60):
            _tr, _ = oculus.get_transformations_and_buttons()
            if "r" in _tr and abs(float(np.linalg.det(_tr["r"][:3, :3])) - 1.0) < 1e-3:
                _rv = R.from_matrix(_tr["r"][:3, :3]).as_rotvec()
                break
            time.sleep(0.01)
        if _rv is not None:
            _ref = np.array(_loaded[1]["oc_ref_rotvec"], dtype=float)
            _drift = float(np.degrees(np.linalg.norm(
                (R.from_rotvec(_rv) * R.from_rotvec(_ref).inv()).as_rotvec())))
            if _drift > DRIFT_WARN_DEG:
                log.warning(f"当前 oc 姿态与标定基差 {_drift:.0f} deg "
                            f"(>{DRIFT_WARN_DEG:.0f}), 头部姿势可能变了, 建议按 A 重标")
            else:
                log.info(f"oc 姿态漂移检查 OK ({_drift:.0f} deg)")

    t_warm = time.time() + 1.0
    while time.time() < t_warm:
        target_6d, _ = _safe_update(c, target_6d, log)
        oculus.get_transformations_and_buttons()
        time.sleep(KEEPALIVE_DT)

    for n in (3, 2, 1):
        print(f"  {n} ...")
        t_end = time.time() + 1.0
        while time.time() < t_end:
            target_6d, _ = _safe_update(c, target_6d, log)
            time.sleep(KEEPALIVE_DT)
    print(">>> 开始! 按 RG 拖动手柄, 松开 RG 停, 30s 后自动结束.")
    print("-" * 64)

    log_path = args.log_csv or (
        f"/home/ubuntu/Desktop/jhli/_stage3_"
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")

    # 旁边写 meta.json
    import json
    meta_path = log_path.replace(".csv", ".meta.json")
    meta = {
        "started_at": datetime.datetime.now().isoformat(),
        "args": vars(args),
        "pos_matrix": pos_M.tolist(),
        "rot_matrix": rot_M.tolist(),
        "pos_signs": pos_signs.tolist(),
        "rot_signs": rot_signs.tolist(),
        "rate_limit_pos_m_per_tick": float(args.rate_pos),
        "rate_limit_ori_rad_per_tick": float(args.rate_ori),
        "loop_hz": LOOP_HZ,
        "initial_ee_6d": target_6d.tolist(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"meta -> {meta_path}")

    csv_f = open(log_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "t_rel_s", "loop", "rg", "anchored",
        "oc_px", "oc_py", "oc_pz", "oc_ox", "oc_oy", "oc_oz", "r_trig",
        "rel_vr_x", "rel_vr_y", "rel_vr_z",
        "rel_vr_rx", "rel_vr_ry", "rel_vr_rz",
        "rel_arm_x", "rel_arm_y", "rel_arm_z",
        "rel_arm_rx", "rel_arm_ry", "rel_arm_rz",
        "des_x", "des_y", "des_z", "des_rx", "des_ry", "des_rz",
        "tgt_x", "tgt_y", "tgt_z", "tgt_rx", "tgt_ry", "tgt_rz",
        "ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz",
        "ms_oc_read", "ms_send", "ms_read_ee",
    ])

    period = 1.0 / LOOP_HZ
    n_loop = 0
    n_rg_frames = 0
    n_restart = 0
    n_anchor = 0

    init_vr_T = None
    init_arm_pos = None
    init_arm_R = None
    prev_vr_pos = None
    n_vr_jump = 0
    prev_A = False
    prev_B = False
    n_cal = 0
    cal_state = "idle"          # idle | g1 | g2
    cal_g1_oc = None             # 手势1 起点 (oc)
    cal_g2_oc0 = None            # 手势2 起点 (oc)
    cal_d_oc_1 = None            # 手势1 位移 (oc)

    last_print = time.time()
    target_at_last_print = target_6d.copy()
    t_start = time.time()

    try:
        while time.time() - t_start < args.duration:
            t_tick = time.time()

            t0 = time.time()
            transforms, buttons = oculus.get_transformations_and_buttons()
            ms_oc = (time.time() - t0) * 1e3
            rg = bool(buttons.get("RG", False))
            r_trig = buttons.get("rightTrig", (0.0,))
            r_trig = float(r_trig[0]) if isinstance(r_trig, tuple) else float(r_trig)
            oc_pos = np.full(3, np.nan); oc_ori = np.full(3, np.nan)
            rel_vr_pos = np.zeros(3); rel_vr_rotvec = np.zeros(3)
            rel_arm_pos = np.zeros(3); rel_arm_rotvec = np.zeros(3)
            desired_6d = target_6d.copy()
            anchored_flag = 0

            # OculusReader 偶发返回零矩阵 (追踪丢失瞬间), 需校验
            cur_T_valid = False
            if "r" in transforms:
                cur_T = transforms["r"]
                det = float(np.linalg.det(cur_T[:3, :3]))
                if abs(det - 1.0) < 1e-3:
                    cur_T_valid = True
                    oc_pos = cur_T[:3, 3].copy()
                    oc_ori = R.from_matrix(cur_T[:3, :3]).as_rotvec()

            # ---- A 键: 2 方向手势 SVD 标定; B 键: 中止回 idle ----
            cur_A = bool(buttons.get("A", False))
            cur_B = bool(buttons.get("B", False))
            if cur_B and not prev_B and cal_state != "idle":
                cal_state = "idle"; cal_g1_oc = None; cal_g2_oc0 = None; cal_d_oc_1 = None
                log.info("[CAL] B: 中止标定, 回 idle (保留已生效 R)")
            if cur_A and not prev_A:
                if not cur_T_valid:
                    log.warning("[CAL] oc 无效, 忽略 A, 等追踪稳定")
                elif cal_state == "idle":
                    cal_g1_oc = oc_pos.copy(); cal_state = "g1"
                    log.info("[CAL] 手势1: 手柄对准【机械臂正前方】拉 >=12cm, 再按 A")
                elif cal_state == "g1":
                    _d1 = oc_pos - cal_g1_oc
                    if np.linalg.norm(_d1) < CAL_MIN_DISP:
                        log.warning(f"[CAL] 手势1 仅 {np.linalg.norm(_d1)*100:.1f}cm "
                                    f"<{CAL_MIN_DISP*100:.0f}cm, 作废回 idle")
                        cal_state = "idle"; cal_g1_oc = None
                    else:
                        cal_d_oc_1 = _d1.copy(); cal_g2_oc0 = oc_pos.copy(); cal_state = "g2"
                        log.info(f"[CAL] 手势1 OK |d_oc|={np.linalg.norm(_d1)*100:.1f}cm. "
                                 f"手势2: 手柄沿【真实竖直向上】拉 >=12cm, 再按 A")
                elif cal_state == "g2":
                    _d2 = oc_pos - cal_g2_oc0
                    if np.linalg.norm(_d2) < CAL_MIN_DISP:
                        log.warning(f"[CAL] 手势2 仅 {np.linalg.norm(_d2)*100:.1f}cm "
                                    f"<{CAL_MIN_DISP*100:.0f}cm, 作废回 idle")
                        cal_state = "idle"; cal_g1_oc = None; cal_g2_oc0 = None; cal_d_oc_1 = None
                    else:
                        _doc = [cal_d_oc_1, _d2]
                        _darm = [np.array([1., 0., 0.]), np.array([0., 0., 1.])]
                        _Rs = vr_align.solve_rotation(_doc, _darm)
                        _ok, _oe, _det = vr_align.validate_rotation(_Rs)
                        _oi, _ai, _rec = vr_align.gesture_pair_quality(_doc, _darm)
                        _ae = abs(_oi - _ai)
                        if _ok and _ae < CAL_ANGLE_TOL_DEG:
                            pos_M = _Rs.copy(); rot_M = _Rs.copy()
                            pos_signs = np.ones(3); rot_signs = np.ones(3)
                            _mode = "calibrated"
                            _ocref = R.from_matrix(cur_T[:3, :3]).as_rotvec()
                            vr_align.save_rotation(
                                OC2ARM_R_PATH, _Rs,
                                {"oc_inter_deg": _oi, "angle_err_deg": _ae,
                                 "recon_max_deg": _rec}, _ocref)
                            n_cal += 1
                            log.info(f"[CAL] OK 标定#{n_cal} 生效并存盘: 夹角={_oi:.1f} "
                                     f"(误差{_ae:.1f}) 重建={_rec:.2f} "
                                     f"ortho={_oe:.1e} det={_det:.3f}")
                        else:
                            log.warning(f"[CAL] 拒绝: ok={_ok} 夹角误差={_ae:.1f} "
                                        f"(阈值{CAL_ANGLE_TOL_DEG:.0f}). 两手势需非平行/方向别搞反, 重按 A")
                        cal_state = "idle"; cal_g1_oc = None; cal_g2_oc0 = None; cal_d_oc_1 = None
            prev_A = cur_A
            prev_B = cur_B

            if cur_T_valid:
                # VR 追踪跳变检测: cur_vr - prev_vr 模 > THRESHOLD 视为非物理运动, 丢弃
                if prev_vr_pos is not None:
                    jump = np.linalg.norm(oc_pos - prev_vr_pos)
                    if jump > VR_JUMP_THRESHOLD:
                        n_vr_jump += 1
                        log.warning(f"VR 跳变 #{n_vr_jump}: |Δvr|={jump*100:.1f}cm @ t={t_tick-t_start:.2f}s, 丢弃此 tick")
                        cur_T_valid = False  # 丢弃, 不更新 anchor / desired
                prev_vr_pos = oc_pos.copy()

            if cur_T_valid:
                if rg:
                    if init_vr_T is None:
                        init_vr_T = cur_T.copy()
                        cur_ee = np.asarray(c.server.robot_get_ee_pose(), dtype=float)
                        init_arm_pos = cur_ee[:3].copy()
                        init_arm_R = R.from_rotvec(cur_ee[3:])
                        n_anchor += 1
                        log.info(f"锚定 #{n_anchor}: arm={init_arm_pos}, vr={cur_T[:3,3]}")
                        desired_6d = np.concatenate([init_arm_pos, init_arm_R.as_rotvec()])
                    else:
                        rel_vr_pos = cur_T[:3, 3] - init_vr_T[:3, 3]
                        rel_vr_R = R.from_matrix(cur_T[:3, :3] @ init_vr_T[:3, :3].T)
                        rel_vr_rotvec = rel_vr_R.as_rotvec()
                        rel_arm_pos = (pos_M @ rel_vr_pos) * pos_signs * scaler_pos
                        rel_arm_rotvec = (rot_M @ rel_vr_rotvec) * rot_signs * scaler_ori
                        desired_pos = init_arm_pos + rel_arm_pos
                        if args.lock_rotation:
                            desired_R = init_arm_R
                            rel_arm_rotvec = np.zeros(3)
                        else:
                            desired_R = R.from_rotvec(rel_arm_rotvec) * init_arm_R
                        desired_6d = np.concatenate([desired_pos, desired_R.as_rotvec()])

                    anchored_flag = 1
                    n_rg_frames += 1
                else:
                    if init_vr_T is not None:
                        log.info("RG 松开, 丢锚, target hold")
                    init_vr_T = None
                    init_arm_pos = None
                    init_arm_R = None
            else:
                # cur_T 无效或跳变, 视同 RG 松开, 丢锚, target 保持
                init_vr_T = None
                init_arm_pos = None
                init_arm_R = None
                # 注意: prev_vr_pos 已在上面更新, 这里不清; 下个有效帧会重新比较

            target_6d = _rate_limit_step(target_6d, desired_6d, args.rate_pos, args.rate_ori)

            t1 = time.time()
            target_6d, restarted = _safe_update(c, target_6d, log)
            ms_send = (time.time() - t1) * 1e3
            if restarted:
                n_restart += 1
                init_vr_T = None
                init_arm_pos = None
                init_arm_R = None

            t2 = time.time()
            try:
                ee_act = np.asarray(c.server.robot_get_ee_pose(), dtype=float)
            except Exception:
                ee_act = np.full(6, np.nan)
            ms_ee = (time.time() - t2) * 1e3

            csv_w.writerow([
                f"{t_tick - t_start:.4f}", n_loop, int(rg), anchored_flag,
                f"{oc_pos[0]:.5f}", f"{oc_pos[1]:.5f}", f"{oc_pos[2]:.5f}",
                f"{oc_ori[0]:.5f}", f"{oc_ori[1]:.5f}", f"{oc_ori[2]:.5f}",
                f"{r_trig:.3f}",
                f"{rel_vr_pos[0]:.5f}", f"{rel_vr_pos[1]:.5f}", f"{rel_vr_pos[2]:.5f}",
                f"{rel_vr_rotvec[0]:.5f}", f"{rel_vr_rotvec[1]:.5f}", f"{rel_vr_rotvec[2]:.5f}",
                f"{rel_arm_pos[0]:.5f}", f"{rel_arm_pos[1]:.5f}", f"{rel_arm_pos[2]:.5f}",
                f"{rel_arm_rotvec[0]:.5f}", f"{rel_arm_rotvec[1]:.5f}", f"{rel_arm_rotvec[2]:.5f}",
                f"{desired_6d[0]:.5f}", f"{desired_6d[1]:.5f}", f"{desired_6d[2]:.5f}",
                f"{desired_6d[3]:.5f}", f"{desired_6d[4]:.5f}", f"{desired_6d[5]:.5f}",
                f"{target_6d[0]:.5f}", f"{target_6d[1]:.5f}", f"{target_6d[2]:.5f}",
                f"{target_6d[3]:.5f}", f"{target_6d[4]:.5f}", f"{target_6d[5]:.5f}",
                f"{ee_act[0]:.5f}", f"{ee_act[1]:.5f}", f"{ee_act[2]:.5f}",
                f"{ee_act[3]:.5f}", f"{ee_act[4]:.5f}", f"{ee_act[5]:.5f}",
                f"{ms_oc:.2f}", f"{ms_send:.2f}", f"{ms_ee:.2f}",
            ])

            n_loop += 1

            if t_tick - last_print >= 1.0:
                last_print = t_tick
                elapsed = t_tick - t_start
                p = target_6d[:3]
                d_pos = target_6d[:3] - target_at_last_print[:3]
                tag = "ANCH" if anchored_flag else "----"
                # 锚定时给出当前 rel_vr 和 rel_arm, 不锚则空
                if anchored_flag:
                    rv_now = (rel_vr_pos[0], rel_vr_pos[1], rel_vr_pos[2])
                    ra_now = (rel_arm_pos[0], rel_arm_pos[1], rel_arm_pos[2])
                    extra = (f" rel_vr=({rv_now[0]*100:+5.1f},{rv_now[1]*100:+5.1f},{rv_now[2]*100:+5.1f})cm"
                             f" rel_arm=({ra_now[0]*100:+5.1f},{ra_now[1]*100:+5.1f},{ra_now[2]*100:+5.1f})cm")
                else:
                    extra = ""
                print(f"  t={elapsed:5.1f}s loop={n_loop:4d} RG={n_rg_frames:4d} rst={n_restart} "
                      f"anch={n_anchor} [{tag}] "
                      f"tgt=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}){extra}")
                target_at_last_print = target_6d.copy()

            sleep_left = period - (time.time() - t_tick)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\n[Ctrl+C] 提前退出")

    csv_f.flush(); csv_f.close()

    elapsed = time.time() - t_start
    print("-" * 64)
    print(f"结束: 实际 {elapsed:.1f}s | loop={n_loop} | RG 帧={n_rg_frames} | "
          f"controller 重启={n_restart} | 锚定次数={n_anchor} | VR 跳变={n_vr_jump}")
    print(f"最终 target_ee pos = {target_6d[:3]}")
    print(f"CSV 日志: {log_path}")
    print("机械臂保持 cartesian impedance hold 当前 target. polymetis 未停.")
    print(f"最终 mapping={_mode}, yaw_deg = {yaw_deg:.1f}°, 标定成功 {n_cal} 次")

    try:
        import pandas as pd
        df = pd.read_csv(log_path)
        df["gap"] = (df.rg.astype(int).diff().fillna(0) != 0).cumsum()
        segs = [sg for _, sg in df[df.rg == 1].groupby("gap") if len(sg) >= 10]
        if segs:
            print()
            print(f"--- RG 摁下段汇总 ({len(segs)} 段, matrix={args.pos_matrix} pos_signs={args.pos_signs}) ---")
            print(f"{'#':>2} {'t_s':>5} {'dt':>4}  "
                  f"{'rel_vr_net(x,y,z)':>26}  "
                  f"{'rel_arm_net(x,y,z)':>26}  "
                  f"{'ee_net(x,y,z)':>26}")
            def _dom(v):
                # 主导轴 + sign, 返回 ('X', '+', mag)
                axes = ['X', 'Y', 'Z']
                i = int(np.argmax(np.abs(v)))
                return axes[i], ('+' if v[i] >= 0 else '-'), abs(float(v[i]))
            for i, sg in enumerate(segs):
                t0 = sg.t_rel_s.iloc[0]
                dt = sg.t_rel_s.iloc[-1] - t0
                rv = np.array([sg.rel_vr_x.iloc[-1], sg.rel_vr_y.iloc[-1], sg.rel_vr_z.iloc[-1]])
                ra = np.array([sg.rel_arm_x.iloc[-1], sg.rel_arm_y.iloc[-1], sg.rel_arm_z.iloc[-1]])
                ee = np.array([sg.ee_x.iloc[-1] - sg.ee_x.iloc[0],
                               sg.ee_y.iloc[-1] - sg.ee_y.iloc[0],
                               sg.ee_z.iloc[-1] - sg.ee_z.iloc[0]])
                rv_a, rv_s, rv_m = _dom(rv)
                ra_a, ra_s, ra_m = _dom(ra)
                ee_a, ee_s, ee_m = _dom(ee)
                # sanity: arm_dom 和 ee_dom 应当同轴同号, 否则机械臂跟得很差
                ok = (ra_a == ee_a and ra_s == ee_s)
                ok_str = "OK" if ok else "!! arm vs ee 不一致"
                print(f"{i:2d} {t0:5.1f} {dt:4.1f}s  "
                      f"({rv[0]:+.3f},{rv[1]:+.3f},{rv[2]:+.3f})  "
                      f"({ra[0]:+.3f},{ra[1]:+.3f},{ra[2]:+.3f})  "
                      f"({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f})  "
                      f"oc{rv_s}{rv_a}({rv_m*100:.1f}cm) -> ee{ee_s}{ee_a}({ee_m*100:.1f}cm) {ok_str}")
            print()
            print("解读: rel_vr_net = 段末手柄相对锚点的位移 (oculus frame)")
            print("      rel_arm_net = 映射到机械臂坐标的相对位移 (期望机械臂走这么多)")
            print("      ee_net = 段内 EE 实际走的距离")
            print("      若 rel_vr ≠ 0 但 rel_arm 方向反 → 矩阵或 sign 不对")
    except Exception as e:
        print(f"段汇总失败: {e}")


if __name__ == "__main__":
    main()
