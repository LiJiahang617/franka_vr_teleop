"""离线时间对齐转换器：以主相机图像时间戳为锚，把各模态重采到锚时间轴。

用法（CLI）：
    python align_offline.py --in ep.h5 --out aligned.npz --on-stale interpolate
    python align_offline.py --in ep.h5 --out aligned.npz --on-stale drop
    python align_offline.py --in ep.h5 --out aligned.npz --on-stale keep --cam-anchor exterior

设计要点：
  - 以主相机（默认 cam_anchor=第一个相机）的 timestamp(N,) 为锚时间轴（N_anchor 个点）
  - arm/effector/action 各模态通过线性插值（np.interp）重采到锚轴
  - arm.pose 中的 EE 旋转（indices 3:6，欧拉角 rad）用 SLERP（球面线性插值）
  - state_hifreq 保留原 240Hz 时间轴与数据，不重采到图像锚轴
  - on_stale：对被标记为 stale 的锚帧的处理策略
      - "interpolate"：忽略 stale 标记，正常插值（默认）
      - "drop"：丢弃 stale 锚帧，返回数组长度 < N_anchor
      - "keep"：原样保留 stale 帧（keep 与 interpolate 对 state/action 对齐结果相同，
                区别仅 keep 保留 stale 标记的语义意图）
  - 外推行为：锚轴超出某模态时间范围时，np.interp 与 SLERP 均采用端点保持（hold）

返回值（align_by_image_timestamp）：dict[str, np.ndarray]，键含：
  - "anchor_ts"            : (N_out,) 锚时间轴
  - "anchor_stale"         : (N_out,) bool，锚帧 stale 标记（drop 后全 False）
  - "anchor_indices"       : (N_out,) int，各输出帧在原始图像时间戳序列中的下标
                             interpolate/keep 模式 = np.arange(N_anchor)；
                             drop 模式 = np.where(~anchor_stale_orig)[0]（保留帧的原始下标）
  - "arm_joints"           : (N_out, 7) float64
  - "arm_joint_vel"        : (N_out, 7) float64
  - "arm_pose"             : (N_out, 6) float64（pos 线性 + rot SLERP）
  - "gripper_position"     : (N_out, 1) float64
  - "gripper_position_norm": (N_out, 1) float64
  - "gripper_cmd"          : (N_out, 1) float64
  - "action_delta_ee_pose" : (N_out, 6) float64
  - "state_hifreq_joints"  : (M, 7) float64（不重采，原始高频数据）
  - "state_hifreq_joint_vel": (M, 7) float64
  - "state_hifreq_pose"    : (M, 6) float64
  - "state_hifreq_timestamp": (M,) float64

对应 spec：docs/superpowers/specs/2026-05-19-franka-datacollection-completion-design.md §10.4
"""
import argparse
import logging
import sys

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _dedup_strictly_increasing(
    ts: np.ndarray, vals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """对时间轴去重，保留重复时间戳中的最后一个样本（最新数据），返回严格递增序列。

    schema v2 允许带 stale 的模态时间戳「非递增」——补帧复用上一帧戳时可能出现
    重复时间戳。scipy Slerp 要求严格递增，np.interp 遇重复 xp 不报错但结果不可靠。
    此 helper 统一规范化所有模态的源时间轴。

    Args:
        ts: 源时间轴 (N,)，允许非严格递增（含重复）
        vals: 源数据 (N,) 或 (N, D)

    Returns:
        (ts_unique, vals_unique)：严格递增的时间轴与对应数据
    """
    if len(ts) == 0:
        return ts, vals

    # 找每个时间戳值最后一次出现的原始索引：
    # 对 ts 逆序做 np.unique（保留第一次出现即逆序最后），再映射回原始索引
    ts_rev = ts[::-1]
    _, first_in_rev = np.unique(ts_rev, return_index=True)
    # 逆序索引 → 原始索引（最后一次出现）
    last_appearance = len(ts) - 1 - first_in_rev
    # 按原始时间顺序排序（保证严格递增）
    last_appearance_sorted = np.sort(last_appearance)
    ts_unique = ts[last_appearance_sorted]
    vals_unique = vals[last_appearance_sorted]
    return ts_unique, vals_unique


def _interp_cols(src_ts: np.ndarray, src_vals: np.ndarray, dst_ts: np.ndarray) -> np.ndarray:
    """对多列 src_vals(N, D) 逐列线性插值。

    Args:
        src_ts: 源时间轴 (N,)，必须严格递增
        src_vals: 源数据 (N, D)
        dst_ts: 目标时间轴 (M,)

    Returns:
        插值结果 (M, D)
    """
    D = src_vals.shape[1]
    out = np.empty((len(dst_ts), D), dtype=np.float64)
    for d in range(D):
        out[:, d] = np.interp(dst_ts, src_ts, src_vals[:, d])
    return out


def _slerp_euler(src_ts: np.ndarray, euler_rad: np.ndarray, dst_ts: np.ndarray) -> np.ndarray:
    """以 SLERP 对欧拉角（xyz 内旋 rad）做球面线性插值，输出欧拉角已 unwrap。

    参数中欧拉角被转换为四元数，SLERP 后再转回欧拉角，最后逐列 np.unwrap 减少
    分支切换处的数值跳变。SLERP 保证姿态球面连续，输出欧拉角已 unwrap，但接近
    万向锁时欧拉表示仍可能有数值跳变（这是欧拉角固有局限）。

    Args:
        src_ts: 源时间轴 (N,)，必须严格递增且 N >= 2
        euler_rad: 源欧拉角 (N, 3)，顺序 [rx, ry, rz]（'xyz' 内旋）
        dst_ts: 目标时间轴 (M,)

    Returns:
        SLERP 插值后的欧拉角 (M, 3)，已逐列 unwrap
    """
    # 转为 Rotation 对象（假设欧拉角编码为 xyz 内旋）
    rotations = Rotation.from_euler("xyz", euler_rad)
    slerp = Slerp(src_ts, rotations)
    # 将 dst_ts 裁剪到 [src_ts[0], src_ts[-1]]，防止外推
    dst_ts_clamped = np.clip(dst_ts, src_ts[0], src_ts[-1])
    interped = slerp(dst_ts_clamped)
    euler_out = interped.as_euler("xyz")
    # 逐列 unwrap 减少万向锁附近的数值跳变
    for col in range(euler_out.shape[1]):
        euler_out[:, col] = np.unwrap(euler_out[:, col])
    return euler_out


def _interp_modal(
    modal_name: str,
    src_ts: np.ndarray,
    src_vals: np.ndarray,
    dst_ts: np.ndarray,
) -> np.ndarray:
    """带 N<2 边界检查的多列线性插值。

    Args:
        modal_name: 模态名，用于错误信息
        src_ts: 源时间轴（规范化后严格递增）
        src_vals: 源数据 (N, D)
        dst_ts: 目标时间轴

    Returns:
        插值或广播结果 (len(dst_ts), D)

    Raises:
        ValueError: src_ts 长度为 0 时抛出，错误信息包含 modal_name
    """
    N = len(src_ts)
    if N == 0:
        raise ValueError(f"{modal_name} 模态规范化后无有效帧（0 帧），无法插值")
    if N == 1:
        # 单帧广播到全锚时间轴
        return np.tile(src_vals[0], (len(dst_ts), 1))
    return _interp_cols(src_ts, src_vals, dst_ts)


def _slerp_modal(
    modal_name: str,
    src_ts: np.ndarray,
    euler_rad: np.ndarray,
    dst_ts: np.ndarray,
) -> np.ndarray:
    """带 N<2 边界检查的 SLERP 欧拉角插值。

    Args:
        modal_name: 模态名，用于错误信息
        src_ts: 源时间轴（规范化后严格递增）
        euler_rad: 源欧拉角 (N, 3)
        dst_ts: 目标时间轴

    Returns:
        插值或广播结果 (len(dst_ts), 3)

    Raises:
        ValueError: src_ts 长度为 0 时抛出，错误信息包含 modal_name
    """
    N = len(src_ts)
    if N == 0:
        raise ValueError(f"{modal_name} 模态规范化后无有效帧（0 帧），无法插值")
    if N == 1:
        # 单旋转广播到全锚时间轴
        return np.tile(euler_rad[0], (len(dst_ts), 1))
    return _slerp_euler(src_ts, euler_rad, dst_ts)


# ---------------------------------------------------------------------------
# Task 8-A 辅助：hw_timestamp 有效性检测与锚时间轴选择
# ---------------------------------------------------------------------------

# 线性相关判据（与 spike_realsense_hw_timestamp.py 决策树一致）
_HW_TS_R2_THRESHOLD = 0.9999


def _select_anchor_ts(
    sw_ts: np.ndarray,
    hw_ts: np.ndarray | None,
) -> np.ndarray:
    """选择锚时间轴，返回值**始终在 monotonic 秒域**。

    hw_ts 线性相关达标（R² > 0.9999）时，把 hw_ts 经回归逆变换映射回 monotonic 秒域
    后返回（保留 hw 戳的低抖动特性，同时与 arm_ts/eff_ts/act_ts 单位统一）；
    否则直接 fallback 到 sw_ts（已是 monotonic 秒）。

    Args:
        sw_ts: 软件时间戳（秒，monotonic），shape (N,)。
        hw_ts: 硬件时间戳（毫秒，global_time），shape (N,)，或 None（字段缺失）。

    Returns:
        锚时间轴 (N,) float64，**单位：monotonic 秒**。
        hw 达标时为 hw_ts 去抖后映射回 monotonic 秒的值；否则即 sw_ts 原值。
    """
    if hw_ts is None or len(hw_ts) < 2:
        return sw_ts

    # 线性回归：hw_c ≈ slope * sw_c + intercept（中心化减少数值误差）
    hw_ts_s = hw_ts / 1000.0  # 毫秒 → 秒，与 sw_ts 单位统一后做回归
    sw_c = sw_ts - sw_ts[0]
    hw_c = hw_ts_s - hw_ts_s[0]

    A = np.vstack([sw_c, np.ones(len(sw_c))]).T
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, hw_c, rcond=None)
    except np.linalg.LinAlgError:
        return sw_ts

    slope, intercept = float(coeffs[0]), float(coeffs[1])
    hw_pred = slope * sw_c + intercept
    ss_res = float(np.sum((hw_c - hw_pred) ** 2))
    ss_tot = float(np.sum((hw_c - hw_c.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if r2 > _HW_TS_R2_THRESHOLD:
        # hw_ts 线性相关达标：把 hw_ts 经回归逆变换映射回 monotonic 秒域。
        # 正变换：hw_c = slope * sw_c + intercept
        # 逆变换：sw_c_hat = (hw_c - intercept) / slope
        # → anchor[i] = sw_c_hat[i] + sw_ts[0]
        if abs(slope) < 1e-10:
            # slope 过小（hw_ts 几乎不随 sw_ts 变化）：逆变换数值不稳定，fallback
            return sw_ts
        sw_c_hat = (hw_c - intercept) / slope
        return (sw_c_hat + sw_ts[0]).astype(np.float64)

    # fallback：hw_ts 不可用或线性相关不达标
    return sw_ts



def _project_hw_to_monotonic(
    eff_ts_monotonic: np.ndarray,
    eff_ts_hardware: np.ndarray,
) -> np.ndarray:
    """用线性回归把硬件时戳轴映射到 monotonic 时戳轴。

    Args:
        eff_ts_monotonic: (N,) effector 字段 polymetis 接收时刻（monotonic 秒）
        eff_ts_hardware:  (N,) libfranka 硬件 push 物理时戳（since robot start 秒）

    Returns:
        (N,) 投影到 monotonic 域的精确时戳

    Raises:
        ValueError: 输入长度不一致或 < 2 帧
    """
    if eff_ts_monotonic.shape != eff_ts_hardware.shape:
        raise ValueError(
            f"形状不一致: monotonic {eff_ts_monotonic.shape} vs hardware {eff_ts_hardware.shape}"
        )
    N = len(eff_ts_monotonic)
    if N < 2:
        raise ValueError(f"至少需要 2 帧做线性回归，N={N}")
    slope, intercept = np.polyfit(eff_ts_hardware, eff_ts_monotonic, deg=1)
    projected = slope * eff_ts_hardware + intercept
    return projected

# ---------------------------------------------------------------------------
# 核心对齐函数
# ---------------------------------------------------------------------------

def align_by_image_timestamp(
    h5_path: str,
    on_stale: str = "interpolate",
    cam_anchor: str | None = None,
) -> dict:
    """以主相机图像时间戳为锚，把各模态重采到锚时间轴。

    Args:
        h5_path: v2 hdf5 路径
        on_stale: stale 帧处理策略（"interpolate" | "drop" | "keep"）。
            keep 与 interpolate 对 state/action 对齐结果相同，区别仅 keep 保留 stale
            标记的语义意图。
        cam_anchor: 锚相机名（None 表示取第一个相机按字典序）

    Returns:
        dict[str, np.ndarray]，包含各模态对齐后的数组（见模块文档），
        额外包含 "anchor_indices" 键：各输出帧在原始图像时间戳序列中的下标。

    Raises:
        ValueError: 如果 on_stale 无效、相机不存在、数据为空等
    """
    if on_stale not in ("interpolate", "drop", "keep"):
        raise ValueError(f"on_stale 必须是 interpolate/drop/keep，得到：{on_stale!r}")

    with h5py.File(h5_path, "r") as f:
        # --- 确定锚相机 ---
        rgb_group = f["observations/camera/rgb"]
        cam_names = sorted(rgb_group.keys())
        if len(cam_names) == 0:
            raise ValueError("hdf5 中没有相机数据")
        if cam_anchor is None:
            cam_anchor = cam_names[0]
        elif cam_anchor not in cam_names:
            raise ValueError(f"锚相机 {cam_anchor!r} 不存在，可用：{cam_names}")

        sw_ts_raw = f[f"observations/camera/rgb/{cam_anchor}/timestamp"][...]
        anchor_stale = f[f"observations/camera/rgb/{cam_anchor}/stale"][...]

        # Task 8-A：优先用 hw_timestamp 作为锚时间轴（线性相关达标时）；否则 fallback sw_ts。
        # 线性相关判据：hw_ts 与 sw_ts 线性回归 R² > 0.9999（与 spike 决策树一致）。
        hw_ts_raw = None
        hw_ts_key = f"observations/camera/rgb/{cam_anchor}/hw_timestamp"
        if hw_ts_key in f:
            hw_ts_raw = f[hw_ts_key][...]

        anchor_ts = _select_anchor_ts(sw_ts_raw, hw_ts_raw)

        # --- 各模态原始数据 ---
        arm_ts = f["observations/arm/timestamp"][...]
        arm_joints = f["observations/arm/joints"][...]         # (N, 7)
        arm_joint_vel = f["observations/arm/joint_vel"][...]   # (N, 7)
        arm_pose = f["observations/arm/pose"][...]             # (N, 6) [px, py, pz, rx, ry, rz]

        eff_ts = f["observations/effector/timestamp"][...]
        gripper_pos = f["observations/effector/position"][...]       # (N, 1)
        gripper_norm = f["observations/effector/position_norm"][...] # (N, 1)

        # Phase D Sub2: 读取 effector hw_timestamp（若存在）
        eff_hw_ts_key = "observations/effector/hw_timestamp"
        eff_hw_ts_raw = f[eff_hw_ts_key][...] if eff_hw_ts_key in f else None

        act_ts = f["action/timestamp"][...]
        act_delta_ee = f["action/delta_ee_pose"][...]  # (N, 6)
        act_gripper_cmd = f["action/gripper_cmd"][...]  # (N, 1)

        # state_hifreq（不重采，原样返回）
        hifreq_joints = f["observations/state_hifreq/joints"][...]
        hifreq_joint_vel = f["observations/state_hifreq/joint_vel"][...]
        hifreq_pose = f["observations/state_hifreq/pose"][...]
        hifreq_ts = f["observations/state_hifreq/timestamp"][...]

    # --- on_stale 处理：确定锚时间轴与 anchor_indices ---
    N_anchor = len(anchor_ts)
    if on_stale == "drop":
        # 丢弃 stale 锚帧；anchor_indices 为被保留帧的原始下标
        keep_mask = ~anchor_stale
        anchor_ts_use = anchor_ts[keep_mask]
        anchor_stale_out = anchor_stale[keep_mask]  # 全 False
        anchor_indices = np.where(keep_mask)[0].astype(np.intp)
    else:
        # interpolate / keep：保留所有锚帧；anchor_indices 即 0..N_anchor-1
        anchor_ts_use = anchor_ts
        anchor_stale_out = anchor_stale
        anchor_indices = np.arange(N_anchor, dtype=np.intp)

    if len(anchor_ts_use) == 0:
        raise ValueError("锚时间轴在 on_stale='drop' 后为空（所有帧都是 stale）")

    # Phase D Sub2: effector hw_timestamp 精确对齐分支（含 gate 防误用坏数据）
    # codex review HIGH #2：原逻辑 R² 只 log 不 gate，finite-but-invalid hw_ts 会静默劣化训练数据
    # 新增 3 道 gate：唯一性 / 严格非降 / R² ≥ 0.99；任一不过 → 退回 eff_ts + WARNING
    eff_ts_for_interp = eff_ts   # 默认旧路径
    if eff_hw_ts_raw is not None:
        # Gate 1: shape + finite
        if eff_hw_ts_raw.shape != eff_ts.shape or not np.all(np.isfinite(eff_hw_ts_raw)):
            log.warning(
                "[align] effector hw_timestamp 含 NaN 或形状不匹配，退回 effector_ts；"
                "如需精确同步请确认 polymetis ENABLE_GRIPPER_HW_TIMESTAMP=ON 录新数据"
            )
        else:
            # Gate 2: 至少 2 个 unique 值（防全常数）
            n_unique = len(np.unique(eff_hw_ts_raw))
            if n_unique < 2:
                log.warning(
                    f"[align] effector hw_timestamp 退化常数（unique={n_unique}），"
                    f"退回 effector_ts；polymetis cpp ENABLE_GRIPPER_HW_TIMESTAMP=OFF？"
                )
            # Gate 3: 严格非降（允许重复，禁止递减）
            elif np.any(np.diff(eff_hw_ts_raw) < 0):
                n_violations = int(np.sum(np.diff(eff_hw_ts_raw) < 0))
                log.warning(
                    f"[align] effector hw_timestamp 非单调非降"
                    f"（{n_violations} 处递减），退回 effector_ts；"
                    f"polymetis 时戳来源异常"
                )
            else:
                # Gate 4: 残差 stddev（wall-clock 域） < 50ms — 真实精度判据
                # 不用 R² 因为：hw_ts 可能多帧卡死同值(事件触发)，R² 对此不敏感
                # （894 帧 517 帧步进=0 仍能拿到 R²=0.99，但残差 850ms）
                # 残差 stddev 才反映 align 后真实物理时刻偏差。
                try:
                    projected = _project_hw_to_monotonic(eff_ts, eff_hw_ts_raw)
                    residual = eff_ts - projected
                    res_stddev_ms = float(np.sqrt(np.mean(residual ** 2)) * 1000)
                    res_max_ms = float(np.max(np.abs(residual)) * 1000)
                    if res_stddev_ms <= 50.0:
                        eff_ts_for_interp = projected
                        log.info(
                            f"[align] effector 用 hw_timestamp 精确对齐 "
                            f"(残差 stddev={res_stddev_ms:.2f}ms, max={res_max_ms:.2f}ms)"
                        )
                    else:
                        log.warning(
                            f"[align] effector hw_timestamp 残差过大"
                            f"(stddev={res_stddev_ms:.2f}ms > 50ms, max={res_max_ms:.2f}ms)，"
                            f"退回 effector_ts；可能 hw_ts 多帧卡死同值"
                            f"（事件触发，与 wall-clock 不强线性）"
                        )
                except Exception as e:
                    log.warning(
                        f"[align] _project_hw_to_monotonic 失败 ({e})，退回 effector_ts"
                    )
    else:
        log.info(
            "[align] hdf5 无 effector hw_timestamp 字段（旧数据或 polymetis 未 rebuild）；"
            "建议 rebuild polymetis (ENABLE_GRIPPER_HW_TIMESTAMP=ON) 重录以启用精确对齐"
        )

    # --- 规范化各模态源时间轴（去除重复时间戳，保留最后一个样本） ---
    # schema v2 stale 补帧可能产生重复时间戳，scipy Slerp 要求严格递增
    arm_ts_u, arm_joints_u = _dedup_strictly_increasing(arm_ts, arm_joints)
    arm_ts_u2, arm_joint_vel_u = _dedup_strictly_increasing(arm_ts, arm_joint_vel)
    arm_ts_u3, arm_pose_u = _dedup_strictly_increasing(arm_ts, arm_pose)

    eff_ts_u, gripper_pos_u = _dedup_strictly_increasing(eff_ts_for_interp, gripper_pos)
    eff_ts_u2, gripper_norm_u = _dedup_strictly_increasing(eff_ts_for_interp, gripper_norm)

    act_ts_u, act_delta_ee_u = _dedup_strictly_increasing(act_ts, act_delta_ee)
    act_ts_u2, act_gripper_cmd_u = _dedup_strictly_increasing(act_ts, act_gripper_cmd)

    # --- 各模态插值到锚时间轴（带 N<2 边界处理） ---

    # arm：pos 线性插值，rot SLERP
    interp_joints = _interp_modal("arm", arm_ts_u, arm_joints_u, anchor_ts_use)
    interp_joint_vel = _interp_modal("arm", arm_ts_u2, arm_joint_vel_u, anchor_ts_use)

    # arm_pose：前 3 列（位置）线性，后 3 列（欧拉角）SLERP
    interp_pos = _interp_modal("arm", arm_ts_u3, arm_pose_u[:, :3], anchor_ts_use)
    interp_euler = _slerp_modal("arm", arm_ts_u3, arm_pose_u[:, 3:], anchor_ts_use)
    interp_pose = np.concatenate([interp_pos, interp_euler], axis=1)

    # effector：线性插值
    interp_gripper_pos = _interp_modal("effector", eff_ts_u, gripper_pos_u, anchor_ts_use)
    interp_gripper_norm = _interp_modal("effector", eff_ts_u2, gripper_norm_u, anchor_ts_use)

    # action（delta_ee_pose + gripper_cmd）：线性插值
    interp_delta_ee = _interp_modal("action", act_ts_u, act_delta_ee_u, anchor_ts_use)
    interp_gripper_cmd = _interp_modal("action", act_ts_u2, act_gripper_cmd_u, anchor_ts_use)

    return {
        "anchor_ts": anchor_ts_use,
        "anchor_stale": anchor_stale_out,
        "anchor_indices": anchor_indices,
        "arm_joints": interp_joints,
        "arm_joint_vel": interp_joint_vel,
        "arm_pose": interp_pose,
        "gripper_position": interp_gripper_pos,
        "gripper_position_norm": interp_gripper_norm,
        "gripper_cmd": interp_gripper_cmd,
        "action_delta_ee_pose": interp_delta_ee,
        "state_hifreq_joints": hifreq_joints,
        "state_hifreq_joint_vel": hifreq_joint_vel,
        "state_hifreq_pose": hifreq_pose,
        "state_hifreq_timestamp": hifreq_ts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="以主相机图像时间戳为锚，对 v2 hdf5 做离线时间对齐并写出对齐后的 npy/npz"
    )
    p.add_argument("--in", dest="h5_in", required=True, help="输入 v2 hdf5 路径")
    p.add_argument("--out", dest="out", required=True, help="输出 .npz 路径（含对齐后各模态数组）")
    p.add_argument(
        "--on-stale",
        choices=["interpolate", "drop", "keep"],
        default="interpolate",
        help=(
            "stale 帧处理策略（默认 interpolate）。"
            "keep 与 interpolate 对 state/action 对齐结果相同，区别仅 keep 保留 stale 标记的语义意图。"
        ),
    )
    p.add_argument(
        "--cam-anchor",
        default=None,
        help="锚相机名（默认取第一个相机，按字典序）",
    )
    return p


def main(argv=None):
    """CLI 入口。"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    print(f"[align_offline] 读取 {args.h5_in!r}，on_stale={args.on_stale!r}, "
          f"cam_anchor={args.cam_anchor!r}")
    aligned = align_by_image_timestamp(
        args.h5_in,
        on_stale=args.on_stale,
        cam_anchor=args.cam_anchor,
    )

    np.savez(args.out, **aligned)
    n_out = len(aligned["anchor_ts"])
    n_stale = int(aligned["anchor_stale"].sum())
    print(f"[align_offline] 对齐完成：N_out={n_out}，stale={n_stale}，输出 → {args.out!r}")


if __name__ == "__main__":
    sys.exit(main())
