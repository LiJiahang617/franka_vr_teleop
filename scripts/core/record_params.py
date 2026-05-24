"""录制超参纯函数(可单测, 不依赖硬件/lerobot)。"""
import math

import numpy as np


def resolve_record_fps(cli_fps, cfg_fps):
    """录制频率单一来源: CLI 给了用 CLI(临时覆盖), 否则用 cfg(唯一真值)。
    相机 fps / 循环节拍 / hdf5 target_fps 都应取本函数结果, 保证同源一致。
    """
    fps = float(cli_fps) if cli_fps is not None else float(cfg_fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"record fps 必须为有限正数, 得到 {fps}")
    return fps


def extract_joint_vel(obs, dof=7):
    """从 get_observation 的 obs 取 joint 速度; 缺失(未接通)则零填(向后兼容)。"""
    if all(f"joint_{i+1}.vel" in obs for i in range(dof)):
        return np.array([float(obs[f"joint_{i+1}.vel"]) for i in range(dof)],
                        dtype=np.float64)
    return np.zeros(dof, dtype=np.float64)


def realsense_fps(fps):
    """RealSense 相机帧率适配.

    RealSense 硬件支持的整数帧率: 6/15/30/60/90 (D435 等). 用户 yaml record.fps
    控制 record loop 节拍 (可以是 20/25 等任意 Hz), 但 cam stream 必须用支持值.
    实现: 取 >= fps 的最小合法值 (cam 跑得比 loop 快, record loop 抽样最新帧).
    例: fps=20 -> cam 30Hz; fps=10 -> cam 15Hz; fps=35 -> cam 60Hz.
    """
    f = int(round(float(fps)))
    valid = [6, 15, 30, 60, 90]
    if f in valid:
        return f
    for v in valid:
        if v >= f:
            return v
    return 90


def parse_reset_config(rec_raw: dict):
    """从 record yaml 的 raw dict 解析 reset 配置（纯函数, 可离线单测）。

    Returns (reset_between_episodes: bool, reset_wait: float)。
    严格解析: 防 yaml 引号字符串 "false" 被 bool() 误判为 True 而在用户
    以为关闭时仍执行真机 robot.reset() 回 HOME（高危静默误动作）。

    Raises ValueError: 非法 reset_between_episodes / reset_wait（非有限或<0）。
    """
    v = rec_raw.get("reset_between_episodes", True)
    if isinstance(v, bool):
        rbe = v
    elif v is None:
        rbe = True
    elif isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "on"):
            rbe = True
        elif s in ("false", "0", "no", "off"):
            rbe = False
        else:
            raise ValueError(
                f"record.reset_between_episodes 非法: {v!r}（应为 bool 或 "
                "true/false/1/0/yes/no/on/off）"
            )
    else:
        raise ValueError(
            f"record.reset_between_episodes 必须为 bool, got {type(v).__name__}={v!r}"
        )

    w = rec_raw.get("reset_wait", 1.0)
    if w is None:
        rw = 1.0
    else:
        try:
            rw = float(w)
        except (TypeError, ValueError):
            raise ValueError(f"record.reset_wait 非数字: {w!r}")
        import math as _math
        if not _math.isfinite(rw) or rw < 0:
            raise ValueError(
                f"record.reset_wait 必须为有限非负数, got {rw!r}"
            )
    return rbe, rw


# ================================================================
# 严格解析 helpers（Phase C review-fix: 真机配置鲁棒，fail-loud）
# ================================================================

DEFAULT_STATE_HIFREQ_RATE = 240


def parse_bool(value, default: bool = True, *, key_name: str = "value"):
    """严格解析配置 bool: 防 yaml 引号字符串 "false" 被 bool() 误判 True。

    True: True/"true"/"True"/"1"/"yes"/"on"；False: False/"false"/"False"/"0"/"no"/"off"；
    None: 取 default。其它(int/list/...): raise ValueError 带 key_name 上下文。
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
        raise ValueError(
            f"{key_name} 非法 bool 字符串: {value!r}（应为 true/false/1/0/yes/no/on/off）"
        )
    raise ValueError(f"{key_name} 必须为 bool, got {type(value).__name__}={value!r}")


def parse_section_dict(value, *, key_name: str) -> dict:
    """归一化 yaml 子段为 dict: None→{}, dict→自身, 其他→ValueError（fail-loud 防链式 .get 崩）。"""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise ValueError(
        f"{key_name} 必须为 dict 或 null, got {type(value).__name__}={value!r}"
    )


def parse_positive_int(value, default: int, *, key_name: str) -> int:
    """严格正整数: None→default; bool 拒绝(避 True/False→1/0 误用); 其它必须可转 int 且 >0。"""
    if value is None:
        return default
    if isinstance(value, bool):  # bool 是 int 子类，特判防 True→1/False→0 误用
        raise ValueError(f"{key_name} 必须为 int>0, got bool {value!r}")
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key_name} 必须可转 int, got {type(value).__name__}={value!r}")
    if v <= 0:
        raise ValueError(f"{key_name} 必须 > 0, got {v}")
    return v


def parse_axis_gain(value, *, key_name: str, default=(1.0, 1.0, 1.0)) -> list:
    """严格解析 per-axis 增益: None→default; list/tuple 必须 len==3 元素全有限数。

    config-load 时 fail-loud (vs T1 compute_delta_action 的运行时 fail) = 两层防御，
    早失败避坏增益喂真机运动（同 Phase B-T5 parse_reset_config 真机配置鲁棒 ethos）。
    """
    if value is None:
        return [float(x) for x in default]
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{key_name} 必须为 list/tuple, got {type(value).__name__}={value!r}"
        )
    if len(value) != 3:
        raise ValueError(
            f"{key_name} 必须 len==3 [x,y,z]/[rx,ry,rz], got len={len(value)}: {value!r}"
        )
    out = []
    for i, x in enumerate(value):
        if isinstance(x, bool):  # bool 子类特判
            raise ValueError(f"{key_name}[{i}] 必须为数字, got bool {x!r}")
        try:
            xf = float(x)
        except (TypeError, ValueError):
            raise ValueError(f"{key_name}[{i}] 必须可转 float, got {type(x).__name__}={x!r}")
        if not math.isfinite(xf):
            raise ValueError(f"{key_name}[{i}] 必须有限(非 nan/inf), got {xf!r}")
        out.append(xf)
    return out


def resolve_record_overrides(
    *,
    cli_episodes,
    cli_episode_sec,
    cli_out_dir,
    cli_task_name,
    cli_oc2base,
    record_cfg,
    out_dir_fallback: str,
) -> dict:
    """CLI None 仅覆盖语义：CLI 给了用 CLI（临时覆盖），否则读 RecordConfig（唯一真值）。

    严格 `is None` 判断，禁用 `cli or cfg` falsy 误判（0/""/False 均视为显式给定）。
    out_dir 二级回退：CLI None 且 cfg.out_dir None → 用 out_dir_fallback 常量。
    task_name CLI None → 回退 cfg.task_description（yaml 里 task.description）。

    Args:
        cli_episodes:      CLI --episodes 值（None=未给）。
        cli_episode_sec:   CLI --episode-sec 值（None=未给）。
        cli_out_dir:       CLI --out-dir 值（None=未给）。
        cli_task_name:     CLI --task-name 值（None=未给）。
        cli_oc2base:       CLI --oc2base-R 值（None=未给）。
        record_cfg:        RecordConfig 实例（含 num_episodes/episode_time_sec/
                           out_dir/task_description/oc2base_path）。
        out_dir_fallback:  当 CLI 和 cfg.out_dir 均为 None 时的二级回退路径常量。

    Returns:
        dict 含 episodes/episode_sec/out_dir/task_name/oc2base_path 解析结果。
    """
    # episodes: CLI 给了覆盖，否则 cfg.num_episodes
    episodes = cli_episodes if cli_episodes is not None else record_cfg.num_episodes

    # episode_sec: CLI 给了覆盖，否则 cfg.episode_time_sec
    episode_sec = cli_episode_sec if cli_episode_sec is not None else record_cfg.episode_time_sec

    # out_dir: 二级回退 — CLI > cfg.out_dir > fallback 常量
    if cli_out_dir is not None:
        out_dir = cli_out_dir
    elif record_cfg.out_dir is not None:
        out_dir = record_cfg.out_dir
    else:
        out_dir = out_dir_fallback

    # task_name: CLI 给了覆盖，否则 cfg.task_description（yaml task.description）
    task_name = cli_task_name if cli_task_name is not None else record_cfg.task_description

    # oc2base_path: CLI 给了覆盖，否则 cfg.oc2base_path（接通 RecordConfig.oc2base_path）
    oc2base_path = cli_oc2base if cli_oc2base is not None else record_cfg.oc2base_path

    return {
        "episodes": episodes,
        "episode_sec": episode_sec,
        "out_dir": out_dir,
        "task_name": task_name,
        "oc2base_path": oc2base_path,
    }
