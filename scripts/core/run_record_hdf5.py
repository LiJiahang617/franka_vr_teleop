"""hdf5 录制入口：复用 run_record 的 robot/teleop/相机构造，sink=HDF5EpisodeWriter。

与既有 run_record.py 并存，不改其逻辑。
读取同一份 record_cfg.yaml，用 RecordConfig 解析配置，
把 LeRobotDataset sink 替换为 HDF5EpisodeWriter 写 franka-hdf5-v2。

观测字段对齐说明（来自 franka.py get_observation 实读）：
  - joint 位置: joint_1.pos ... joint_7.pos (float，单独 key)
  - joint 速度: joint_1.vel ... joint_7.vel (float，已接通 robot_get_joint_velocities)
  - ee pose:   ee_pose.x/y/z/rx/ry/rz (float，单独 key)
  - 夹爪状态:  gripper_state_norm ([0,1]), gripper_max_open 来自 cfg
  - 夹爪指令:  gripper_cmd_bin (get_action 返回)
  - 相机图像:  cam.read() 返回 numpy array，用 cv2.imencode 编码为 jpeg bytes

多线程采集架构（Phase D Task 4）：
  - robot_state 线程：单线程内串行调 zerorpc 读关节+夹爪+EE位姿，返回 dict
  - wrist_cam 线程：独立调用 wrist 相机 read()，与 robot_state 真并行
  - exterior_cam 线程：独立调用 exterior 相机 read()，与 robot_state 真并行
  - state_hifreq：Task 4 不接（M=0 占位，Task 7 范畴）
"""
import argparse
import copy
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np
import yaml

from pathlib import Path as _Path
# run_record_hdf5.py 在 <repo>/scripts/core/ ; scripts 目录(=parents[1])上 path
# 供 core.* / run_record 解析(结构固定, 用 __file__ 相对优于硬编码/env)。
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

# 纯逻辑依赖（无硬件）：可在模块顶层 import，测试加载安全
from core import paths as _paths
from core.acquisition import AcquisitionHub, HistoryCollectorThread, SensorThread
from core.async_saver import AsyncEpisodeSaver
from core.hdf5_writer import write_episode
from core.record_params import resolve_record_fps, extract_joint_vel, realsense_fps, parse_reset_config, resolve_record_overrides

# 硬件依赖（franka/lerobot 真实包）：延迟到函数内 import，避免测试加载时爆
# RecordConfig → from run_record import RecordConfig  (在 build_robot_and_teleop/main 内)
# FrankaConfig, Franka → from lerobot_robot_franka   (在 build_robot_and_teleop 内)
# RealSenseCameraConfig → from lerobot.cameras.realsense  (在 build_robot_and_teleop 内)
# create_teleop → from lerobot_teleoperator_franka    (在 build_robot_and_teleop 内)

log = logging.getLogger("rec_hdf5")
logging.basicConfig(level=logging.INFO, format="%(message)s")

def _preflight_abort(robot, teleop, reason: str) -> None:
    """预检失败：尽力断开所有已建资源后退出(2)。

    逐个 try/except 确保 robot/teleop 都尝试 disconnect（一个抛不影响另一个），
    最后稳定 sys.exit(2)（开录前可行动报错，非中途静默失败）。
    """
    log.error(f"[PREFLIGHT] {reason}")
    for name, obj in (("robot", robot), ("teleop", teleop)):
        try:
            obj.disconnect()
        except Exception as e:  # noqa: BLE001 — 清理尽力而为，不掩盖原 reason
            log.warning(f"[PREFLIGHT] {name}.disconnect() 异常(忽略继续清理): {e}")
    sys.exit(2)




def build_robot_and_teleop(record_cfg, fps: float):
    """按既有 run_record.py 同款构造 robot 和 teleop。

    硬件相关 import 在本函数内延迟执行，避免模块加载时依赖真实硬件包。

    Returns:
        (robot, teleop, gripper_max_open)
    """
    # 延迟 import 硬件依赖
    from lerobot_robot_franka import FrankaConfig, Franka
    from lerobot.cameras.configs import ColorMode, Cv2Rotation
    from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
    from lerobot_teleoperator_franka import create_teleop

    # 相机配置 (yaml.cameras.wrist/exterior 独立子段, 各自 width/height/fps/rotate_deg)
    def _rotate_deg_to_enum(deg):
        m = {0: Cv2Rotation.NO_ROTATION, 90: Cv2Rotation.ROTATE_90,
             180: Cv2Rotation.ROTATE_180, 270: Cv2Rotation.ROTATE_270, -90: Cv2Rotation.ROTATE_270}
        if int(deg) not in m:
            raise ValueError(f"rotate_deg 必须为 0/90/180/270, 实得 {deg}")
        return m[int(deg)]

    wrist_image_cfg = RealSenseCameraConfig(
        serial_number_or_name=record_cfg.wrist_cam_serial,
        fps=realsense_fps(getattr(record_cfg, "wrist_fps", fps)),
        width=getattr(record_cfg, "wrist_width", record_cfg.width),
        height=getattr(record_cfg, "wrist_height", record_cfg.height),
        color_mode=ColorMode.RGB,
        use_depth=False,
        rotation=_rotate_deg_to_enum(getattr(record_cfg, "wrist_rotate_deg", 0)),
    )
    exterior_image_cfg = RealSenseCameraConfig(
        serial_number_or_name=record_cfg.exterior_cam_serial,
        fps=realsense_fps(getattr(record_cfg, "exterior_fps", fps)),
        width=getattr(record_cfg, "exterior_width", record_cfg.width),
        height=getattr(record_cfg, "exterior_height", record_cfg.height),
        color_mode=ColorMode.RGB,
        use_depth=False,
        rotation=_rotate_deg_to_enum(getattr(record_cfg, "exterior_rotate_deg", 0)),
    )
    camera_config = {"wrist_image": wrist_image_cfg, "exterior_image": exterior_image_cfg}

    # 机器人配置（与 run_record.py 完全一致）
    # smoothing_alpha: robot 段优先, 否则用 teleop 段 (兼容); 都没就用 dataclass 默认 0.4
    _smoothing = (getattr(record_cfg, "robot_smoothing_alpha", 0.0) or
                  getattr(record_cfg, "smoothing_alpha", 0.4))
    robot_config = FrankaConfig(
        robot_ip=record_cfg.robot_ip,
        cameras=camera_config,
        debug=record_cfg.debug,
        close_threshold=record_cfg.close_threshold,
        use_gripper=record_cfg.use_gripper,
        gripper_reverse=record_cfg.gripper_reverse,
        gripper_bin_threshold=record_cfg.gripper_bin_threshold,
        gripper_max_open=record_cfg.gripper_max_open,
        control_mode=record_cfg.control_mode,
        execute_mode=record_cfg.execute_mode,
        home_joint_position=record_cfg.home_joint_position,
        smoothing_alpha=float(_smoothing),
    )
    robot = Franka(robot_config)

    # teleop 配置（与 run_record.py 完全一致）
    teleop_config = record_cfg.create_teleop_config()
    teleop = create_teleop(teleop_config)

    robot.connect()
    teleop.connect()

    return robot, teleop, record_cfg.gripper_max_open


def _encode_jpg(img: np.ndarray) -> np.ndarray:
    """将相机 RGB 图像编码为 jpeg bytes (uint8 array)。

    相机 color_mode=ColorMode.RGB → img 为 RGB; cv2.imencode 按 OpenCV 惯例
    默认输入 BGR, 故须先 RGB→BGR。否则下游 hdf5_lerobot_map._decode 的
    imdecode(BGR)+cvtColor(BGR2RGB) 会净多一次 R↔B 互换 (黄变青)。
    """
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg') 失败")
    return np.frombuffer(buf.tobytes(), np.uint8)


def _make_robot_state_read_fn(robot, zerorpc_lock=None):
    """构造 robot_state 采集线程的 read_fn。

    设计：zerorpc client（robot._robot）非线程安全，所有 zerorpc 调用必须持
    zerorpc_lock 才能执行，与主线程 send_action 串行互斥。
    若 robot._robot 不存在（FakeRobot 等测试替身），回退到 robot.get_observation()
    并从结果 dict 中提取 arm/effector 字段（同样持锁，保持对称一致）。

    robot_state read_fn 只产观测状态字段（joints/joint_vel/ee_pose/gripper_norm），
    不产动作字段（gripper_cmd 由主循环从 teleop action 取）。

    Args:
        robot: Franka 实例（含 _robot zerorpc client）或测试替身。
        zerorpc_lock: threading.Lock，保护所有 zerorpc client 调用。
                      None 时不加锁（仅兼容旧测试，新代码应始终传入）。

    Returns:
        Callable[[], dict]：返回含 arm/effector 观测字段的 dict。
    """
    zerorpc_client = getattr(robot, "_robot", None)

    if zerorpc_lock is None:
        # 无锁占位（测试回退兼容，不影响 FakeRobot 路径）
        import contextlib
        lock_ctx = contextlib.nullcontext
    else:
        lock_ctx = lambda: zerorpc_lock  # noqa: E731

    if zerorpc_client is not None:
        # 真机路径：所有 zerorpc 调用必须持锁（与主线程 send_action 串行互斥）
        def _read_via_zerorpc():
            with lock_ctx():
                joint_pos = zerorpc_client.robot_get_joint_positions()
                joint_vel = zerorpc_client.robot_get_joint_velocities()
                ee_pose = zerorpc_client.robot_get_ee_pose()
                try:
                    gripper_state = zerorpc_client.gripper_get_state()
                    gripper_norm = max(0.0, min(1.0, gripper_state["width"] / robot.config.gripper_max_open))
                    gripper_hw_ts = gripper_state.get("timestamp")   # 旧 polymetis 无此字段 → None
                except Exception:
                    # 夹爪不可用时仅 gripper_norm 置零（不连带影响已读的 joints/ee_pose）
                    gripper_norm = 0.0
                    gripper_hw_ts = None
            return {
                "joints": np.array(joint_pos, dtype=np.float64),
                "joint_vel": np.array(joint_vel, dtype=np.float64),
                "ee_pose": np.array(ee_pose, dtype=np.float64),
                "gripper_norm": gripper_norm,
                "gripper_hw_ts": gripper_hw_ts,
            }
        return _read_via_zerorpc
    else:
        # 测试回退路径：从 get_observation() 提取 arm/effector 字段
        def _read_via_get_observation():
            with lock_ctx():
                obs = robot.get_observation()
            joints = np.array([obs[f"joint_{i+1}.pos"] for i in range(7)], dtype=np.float64)
            joint_vel_arr = np.array([obs.get(f"joint_{i+1}.vel", 0.0) for i in range(7)], dtype=np.float64)
            ee_pose = np.array(
                [obs[f"ee_pose.{ax}"] for ax in ["x", "y", "z", "rx", "ry", "rz"]],
                dtype=np.float64,
            )
            gripper_norm = float(obs.get("gripper_state_norm") or 0.0)
            return {
                "joints": joints,
                "joint_vel": joint_vel_arr,
                "ee_pose": ee_pose,
                "gripper_norm": gripper_norm,
                "gripper_hw_ts": None,   # FakeRobot 路径无硬件戳
            }
        return _read_via_get_observation


def _make_camera_read_fn(cam):
    """构造单路相机采集线程的 read_fn。

    Task 8-A：支持 RealsenseHwWrapper（read() 返回 (rgb, hw_ts_ms) 元组）
    和普通相机（read() 返回 ndarray）两种接口，统一包装为 (rgb, hw_ts_or_None) 元组。

    Args:
        cam: 相机对象，需有 read() 方法。
             若 getattr(cam, 'HW_WRAPPER', False) 为 True（RealsenseHwWrapper），
             cam.read() 应返回 (rgb_ndarray, hw_ts_ms) 元组；
             否则视为普通相机，cam.read() 返回 rgb_ndarray。

    Returns:
        Callable[[], tuple[np.ndarray | None, float | None]]：
        返回 (rgb, hw_ts_or_None) 元组；hw_ts 单位毫秒（float），普通相机为 None。
    """
    is_hw_wrapper = getattr(cam, "HW_WRAPPER", False)

    if is_hw_wrapper:
        def _read():
            # RealsenseHwWrapper.read() 已返回 (rgb, hw_ts_ms) 元组
            return cam.read()
    else:
        def _read():
            # 普通相机：read() 返回 ndarray，包装为 (rgb, None) 元组
            return cam.read(), None

    return _read



def _make_state_hifreq_read_fn(robot, zerorpc_lock):
    """构造 state_hifreq 240Hz 累积采集线程的 read_fn。

    设计：zerorpc client（robot._robot）非线程安全，所有 zerorpc 调用必须持
    zerorpc_lock 才能执行。与 robot_state 线程、主循环 send_action 串行互斥。
    若 robot._robot 不存在（FakeRobot 等测试替身），回退到 robot.get_observation()
    并提取 arm 字段（同样持锁，保持对称）。

    返回的 dict 含 joints/joint_vel/ee_pose（数据）和 poly_ts（polymetis 侧时间戳）。
    真机若 zerorpc 接口无 poly_ts，则 poly_ts 用 time.monotonic() 占位（Task 9 验证）。
    HistoryCollectorThread 会从返回 dict 中取 "poly_ts" 键；若无则也用 monotonic 占位。

    Args:
        robot: Franka 实例（含 _robot zerorpc client）或测试替身。
        zerorpc_lock: threading.Lock，保护所有 zerorpc client 调用（必须非 None）。

    Returns:
        Callable[[], dict]：返回含 joints/joint_vel/ee_pose/poly_ts 的 dict。
    """
    import time as _time

    zerorpc_client = getattr(robot, "_robot", None)

    if zerorpc_client is not None:
        # 真机路径：串行 3 次 zerorpc 调用，持锁与其他线程互斥
        def _read_hifreq_zerorpc():
            with zerorpc_lock:
                joint_pos = zerorpc_client.robot_get_joint_positions()
                joint_vel = zerorpc_client.robot_get_joint_velocities()
                ee_pose = zerorpc_client.robot_get_ee_pose()
                # polymetis 接口暂不转发 poly_ts；用 monotonic 占位（Task 9 验证）
                poly_ts = _time.monotonic()
            joints_arr = np.array(joint_pos, dtype=np.float64)
            jvel_arr = np.array(joint_vel, dtype=np.float64)
            ee_arr = np.array(ee_pose, dtype=np.float64)
            # Imp4：形状校验，防止真机返回意外维度静默写坏 hdf5
            if joints_arr.shape != (7,):
                raise ValueError(
                    f"robot_get_joint_positions 返回形状 {joints_arr.shape}，期望 (7,)"
                )
            if jvel_arr.shape != (7,):
                raise ValueError(
                    f"robot_get_joint_velocities 返回形状 {jvel_arr.shape}，期望 (7,)"
                )
            if ee_arr.shape != (6,):
                raise ValueError(
                    f"robot_get_ee_pose 返回形状 {ee_arr.shape}，期望 (6,)"
                )
            return {
                "joints": joints_arr,
                "joint_vel": jvel_arr,
                "ee_pose": ee_arr,
                "poly_ts": poly_ts,
            }
        return _read_hifreq_zerorpc
    else:
        # 测试回退路径：从 get_observation() 提取
        def _read_hifreq_fallback():
            with zerorpc_lock:
                obs = robot.get_observation()
            poly_ts = _time.monotonic()
            return {
                "joints": np.array(
                    [obs[f"joint_{i+1}.pos"] for i in range(7)], dtype=np.float64
                ),
                "joint_vel": np.array(
                    [obs.get(f"joint_{i+1}.vel", 0.0) for i in range(7)], dtype=np.float64
                ),
                "ee_pose": np.array(
                    [obs[f"ee_pose.{ax}"] for ax in ["x", "y", "z", "rx", "ry", "rz"]],
                    dtype=np.float64,
                ),
                "poly_ts": poly_ts,
            }
        return _read_hifreq_fallback

def record_episode(robot, teleop, fps: float, max_sec: float,
                   gripper_max_open: float, cam_names: list,
                   *, stop_flag=None, frame_observer=None,
                   hifreq_rate: float = 0.0):
    """录制一个 episode，每 tick 从 AcquisitionHub 拉 snapshot 拼帧返回帧列表。

    多线程采集架构（Phase D Task 4 核心）：
    - robot_state 线程（含 arm/effector）：单线程内串行调 zerorpc 或测试回退
      get_observation()，与相机线程真并行。zerorpc 非线程安全，绝不并发。
    - wrist_cam / exterior_cam 线程：各自独立调 cam.read()，真并行。
    - 主循环按 fps 节拍 hub.snapshot(now) 拼帧，hub.stop() 优雅退出无僵尸。

    图像在 _encode_jpg 中编码（cvtColor(RGB2BGR)->imencode），编码在 deepcopy 前完成。
    frame_observer 在 _encode_jpg 之前调用原始 RGB（守门测试约束不变）。

    Args:
        robot: Franka 实例（含 cameras 属性和 _robot zerorpc client）
        teleop: teleop 实例
        fps: 目标帧率
        max_sec: 最长录制时间（秒）
        gripper_max_open: 夹爪最大开度（米），用于将 norm 转换为 gripper_m
        cam_names: 相机名列表，需与写盘时 cam_names 一致
        stop_flag: 可选 callable()->bool，返回 True 时提前结束当前 ep
        frame_observer: 可选 Callable[[str, np.ndarray], None]，每帧每路 cam
                        在 _encode_jpg 之前调用，传入 (cam_name, rgb_ndarray)。
                        默认 None=零行为变化。

    Args:
        hifreq_rate: state_hifreq 采集频率（Hz），>0 时启动 HistoryCollectorThread
                     持 zerorpc_lock 累积采集，0 时不采（M=0 占位）。

    Returns:
        tuple[list[dict], dict | None]：
        - buf: 帧列表，每帧含 ts/joints/joint_vel/ee_pose/gripper_m/gripper_norm/
               gripper_cmd/delta_ee_pose/cams（JPEG 编码）及 v2 多模态独立 ts/stale 字段。
        - state_hifreq_block: dict（joints/joint_vel/pose/timestamp/poly_ts/wrench）
               或 None（hifreq_rate=0 时）。写盘前传给 write_episode。
    """
    # --- 构造 SensorThread + AcquisitionHub ---
    # 共享 stop_event，hub.stop() 时一次性停所有线程
    stop_event = threading.Event()

    # zerorpc_lock：保护所有 zerorpc client 访问（robot._robot 非线程安全）
    # 主线程 send_action 与 robot_state inline read 持锁互斥（真机路径）；
    # FakeRobot 测试路径（robot._robot 为 None）下 SensorThread 也持此锁。
    zerorpc_lock = threading.Lock()

    # zerorpc 真机 gating（lesson 2026-05-23-zerorpc-gevent-thread-affinity）：
    # zerorpc Client 基于 gevent，Hub 是 thread-local 绑 OS 线程；后台 SensorThread/
    # HistoryCollectorThread 调 robot._robot.* 会触发 gevent Waiter.switch 跨线程断言。
    # 真机路径下 robot_state 走主线程 inline read、state_hifreq 强制关；
    # FakeRobot 测试路径（无 _robot）保留原 SensorThread + HistoryCollector 行为。
    zerorpc_isolated = getattr(robot, "_robot", None) is not None
    if zerorpc_isolated and hifreq_rate > 0:
        log.warning(
            "[REC] 真机 zerorpc 检测到，state_hifreq %dHz 暂关"
            "（gevent thread-affinity 限制，Phase D Task 7 修通后启用）",
            int(hifreq_rate),
        )
        hifreq_rate = 0.0

    # robot_state read_fn：真机/测试都需要（真机主循环 inline 调；测试 SensorThread 调）
    robot_state_fn = _make_robot_state_read_fn(robot, zerorpc_lock=zerorpc_lock)

    sensors: dict[str, SensorThread] = {}
    if not zerorpc_isolated:
        # 测试路径：保留 robot_state SensorThread（FakeRobot 不走 zerorpc）
        sensors["robot_state"] = SensorThread(
            name="robot_state",
            read_fn=robot_state_fn,
            target_rate=fps,
            stop_event=stop_event,
        )

    # 各路相机线程：cameras 不走 zerorpc（librealsense direct），两路径都建
    for cn in cam_names:
        cam = robot.cameras.get(cn)
        if cam is not None and hasattr(cam, "read"):
            sensors[cn] = SensorThread(
                name=cn,
                read_fn=_make_camera_read_fn(cam),
                target_rate=fps,
                stop_event=stop_event,
            )

    hub = AcquisitionHub(sensors)

    # state_hifreq 累积采集线程：仅测试路径启用（真机上面已强制关 hifreq_rate）
    hifreq_collector = None
    if hifreq_rate > 0:
        hifreq_fn = _make_state_hifreq_read_fn(robot, zerorpc_lock)
        hifreq_collector = HistoryCollectorThread(
            name="state_hifreq",
            read_fn=hifreq_fn,
            target_rate=hifreq_rate,
            stop_event=stop_event,
        )

    buf = []
    period = 1.0 / fps
    t_end = time.monotonic() + max_sec

    try:
        # 预热：等待所有线程至少采到一帧（最多等 max(2*period, 0.5) 秒，相机首帧可能慢）
        warm_timeout = max(2.0 * period, 0.5)
        deadline_warm = time.monotonic() + warm_timeout
        while time.monotonic() < deadline_warm:
            snap = hub.snapshot(time.monotonic())
            if len(snap) == len(sensors):
                break
            time.sleep(period * 0.1)

        # 预热结束后检查哪些 sensor 仍未采到首帧，记 warning
        snap_check = hub.snapshot(time.monotonic())
        for sensor_name in sensors:
            if sensor_name not in snap_check:
                log.warning(
                    f"[WARMUP] sensor '{sensor_name}' 预热超时未采到首帧，"
                    "将降级为占位/stale"
                )

        while time.monotonic() < t_end:
            # 键盘提前结束钩子
            if stop_flag is not None and stop_flag():
                break

            t0 = time.monotonic()

            # 采集 teleop action
            action = teleop.get_action()

            # 发送 action 到机器人（持 zerorpc_lock，与后台读状态线程串行互斥）
            with zerorpc_lock:
                robot.send_action(action)

            # 从 AcquisitionHub 取 snapshot（各模态各自最新时刻）
            now = time.monotonic()
            snap = hub.snapshot(now)

            # --- 解包 robot_state 模态（arm/effector 共用同一线程时刻）---
            # 真机：主线程 inline read（zerorpc gevent thread-affinity 限制）
            # 测试：走 hub.snapshot 拿 SensorThread 数据
            rs_default = (
                {
                    "joints": np.zeros(7, np.float64),
                    "joint_vel": np.zeros(7, np.float64),
                    "ee_pose": np.zeros(6, np.float64),
                    "gripper_norm": 0.0,
                    "gripper_hw_ts": None,   # inline read 异常时占位，避免 KeyError
                },
                now,
                True,
            )
            if zerorpc_isolated:
                try:
                    rs_data = robot_state_fn()    # robot_state_fn 内部持 zerorpc_lock
                    rs_ts = time.monotonic()
                    rs_stale = False
                except Exception as exc:
                    # 与 SensorThread 一致的容错：read 失败时占位+stale=True，不挂主循环
                    log.warning("[REC] inline robot_state read 异常（占位重试）：%s", exc)
                    rs_data, rs_ts, rs_stale = rs_default
            else:
                rs_data, rs_ts, rs_stale = snap.get("robot_state", rs_default)
            joints = np.asarray(rs_data["joints"], np.float64)
            joint_vel = np.asarray(rs_data["joint_vel"], np.float64)
            ee_pose = np.asarray(rs_data["ee_pose"], np.float64)
            gripper_norm = float(rs_data["gripper_norm"])
            gripper_m = gripper_norm * gripper_max_open
            # Imp1：gripper_cmd 来自 teleop action（动作/指令语义），非观测侧
            gripper_cmd = float(action.get("gripper_cmd_bin", 0.0))

            # arm_ts 和 effector_ts 共用 robot_state 线程的时间戳（schema v2 合规：数值接近）
            arm_ts = rs_ts
            effector_ts = rs_ts
            arm_stale = rs_stale
            effector_stale = rs_stale

            # delta_ee_pose action 数组
            delta_ee_pose = np.array(
                [action.get(f"delta_ee_pose.{ax}", 0.0) for ax in ["x", "y", "z", "rx", "ry", "rz"]],
                dtype=np.float64,
            )

            # --- 解包各相机模态 ---
            cams = {}
            cam_ts = {}
            cam_stale_dict = {}
            cam_hw_ts = {}
            for cn in cam_names:
                if cn in snap:
                    data, c_ts, c_stale = snap[cn]
                else:
                    # 相机线程无 read() 或未采到帧：占位
                    data, c_ts, c_stale = None, now, True

                # Task 8-A：_make_camera_read_fn 统一返回 (rgb, hw_ts_or_None) 元组
                # hw_ts 单位毫秒（float）；普通相机为 None（fallback 软件戳）
                if isinstance(data, tuple) and len(data) == 2:
                    img, hw_ts_val = data
                else:
                    img, hw_ts_val = data, None

                if img is not None and isinstance(img, np.ndarray):
                    # frame_observer 在 _encode_jpg 之前调用原始 RGB（守门测试约束不变）
                    if frame_observer is not None:
                        frame_observer(cn, img)
                    cams[cn] = _encode_jpg(img)
                else:
                    cams[cn] = np.zeros((4,), np.uint8)

                cam_ts[cn] = c_ts
                cam_stale_dict[cn] = c_stale
                # hw_ts 有效时写真硬件戳（毫秒）；否则 fallback 软件戳（秒，与 c_ts 单位同）
                cam_hw_ts[cn] = hw_ts_val if hw_ts_val is not None else c_ts

            # Task 5：透传 zerorpc gripper 硬件时间戳（旧 polymetis 无此字段时为 None）
            effector_hw_ts = rs_data.get("gripper_hw_ts")   # None 或 float（秒，since robot start）

            buf.append(dict(
                ts=now,  # Minor1：帧时刻用 snapshot 时刻，非编码后时刻
                # v2 各模态独立时间戳字段
                arm_ts=arm_ts,
                effector_ts=effector_ts,
                effector_hw_ts=effector_hw_ts,  # Task 5：gripper 硬件戳（旧接口 None）
                cam_ts=cam_ts,
                cam_hw_ts=cam_hw_ts,  # Task 8-A 实填真硬件戳；普通相机 fallback 软件戳
                # v2 stale 字段（各模态独立，多线程时可能非零）
                arm_stale=arm_stale,
                effector_stale=effector_stale,
                cam_stale=cam_stale_dict,
                joints=joints,
                joint_vel=joint_vel,
                ee_pose=ee_pose,
                gripper_m=gripper_m,
                gripper_norm=gripper_norm,
                gripper_cmd=gripper_cmd,
                delta_ee_pose=delta_ee_pose,
                cams=cams,
            ))

            dt = period - (time.monotonic() - t0)
            if dt > 0:
                time.sleep(dt)

    finally:
        # 优雅退出：置 stop_event，join 所有采集线程（timeout=2s），无僵尸
        hub.stop()
        # state_hifreq 采集线程共享同一 stop_event，hub.stop() 已置位；
        # 只需再 join 等线程完成当前 read_fn 后退出（HistoryCollectorThread.stop 内部再置一次无害）
        if hifreq_collector is not None:
            hifreq_collector.stop(timeout=2.0)

    # --- 组装 state_hifreq_block（M>0）或 None（M=0 占位）---
    state_hifreq_block = None
    if hifreq_collector is not None:
        history = hifreq_collector.get_history()
        if history:
            ts_arr = np.array([s.ts for s in history], dtype=np.float64)
            poly_arr = np.array([s.poly_ts for s in history], dtype=np.float64)
            joints_arr = np.stack([s.data["joints"] for s in history])  # (M,7)
            jvel_arr = np.stack([s.data["joint_vel"] for s in history])  # (M,7)
            pose_arr = np.stack([s.data["ee_pose"] for s in history])    # (M,6)
            state_hifreq_block = {
                "joints": joints_arr,
                "joint_vel": jvel_arr,
                "pose": pose_arr,
                "timestamp": ts_arr,
                "poly_ts": poly_arr,
                "wrench": np.zeros((len(history), 6), dtype=np.float64),  # Phase F 实填
            }
            log.info(
                f"[HIFREQ] state_hifreq 采集 M={len(history)} 帧"
                f"（目标 {hifreq_rate}Hz × {max_sec}s ≈ {hifreq_rate * max_sec:.0f}）"
            )
        else:
            _last_err = hifreq_collector.last_error
            if _last_err is not None:
                log.warning(
                    "[HIFREQ] state_hifreq collector 无采样（last_error: %s）", _last_err
                )
            else:
                log.warning("[HIFREQ] state_hifreq collector 无采样（可能录制时间太短）")

    return buf, state_hifreq_block


def run_episodes(robot, teleop, saver, *, fps, episode_sec, gripper_max_open,
                 cam_names, out_dir, task_name, oc2base_R, vr_source,
                 episodes, decide, reset_fn=None, reset_wait=0.0,
                 stop_flag=None, frame_observer=None):
    """episode 循环编排：录完→deepcopy→submit→新 buffer（非阻塞）。

    "采集"与"落盘"解耦：
    - record_episode 只负责一条 ep 的采集并返回 list[frame]（cams 已编码）。
    - deepcopy(整个 payload) 在 buffer 复用（buf=None）前完成，frames+meta 全脱钩外部引用。
    - saver.submit(path, payload) 入队即返回，不等写盘（由 AsyncEpisodeSaver 后台完成）。
    - 丢弃 = 不 submit，不产文件。
    - 进程退出前由调用方（main 的 with AsyncEpisodeSaver）close() join 排空。

    **stop 语义**：decide(ep) 在 record_episode 返回后调用；返回 'stop' = 停止且
    **不提交当前刚录的 ep**（视为未显式 keep）。已 submit 的历史 ep 由
    with AsyncEpisodeSaver 退出 close()/join 排空保证零丢失。最终键位→keep/discard/stop
    的 UX 映射由 Task4 定义（Task4 可在需要时新增"保存当前再停"路径）。

    **背压语义**：saver.submit 为 put_nowait O(1) 非阻塞；队列满抛 QueueFullError
    （不静默丢，符合 spec §3.2 快速失败背压），不阻塞录制循环。

    **reset 语义**：keep 与 discard 后（非末条、非 stop）均调用 reset_fn 回 home
    （丢弃坏 ep 后仍需回 home 再重录）。

    Args:
        robot: Franka 实例
        teleop: teleop 实例
        saver: 实现 submit(path, payload) 的存盘器（AsyncEpisodeSaver 或 mock）
        fps: 目标帧率
        episode_sec: 每条 episode 最长时间（秒）
        gripper_max_open: 夹爪最大开度（米）
        cam_names: 相机名列表
        out_dir: 输出目录
        task_name: 任务名称
        oc2base_R: 3x3 标定旋转矩阵（ndarray）
        vr_source: VR 来源标识（字符串）
        episodes: 录制 episode 总数
        decide: Callable[[int], str]，返回 "keep"/"discard"/"stop"
                （Task 4 由键盘 events 驱动；本 Task 测试注入 lambda）
        reset_fn: 可选 Callable，episode 间调用回 home（Task 3 占位 hook；
                  None=不 reset）
        reset_wait: reset 后等待时间（秒）
        stop_flag: 可选 callable()->bool，传给 record_episode 提前结束当前 ep
                   （Task 4 由 EpisodeDecider.episode_stop_flag() 提供；
                   None=按 episode_sec 计时结束，headless 安全）
        frame_observer: 可选 Callable[[str, np.ndarray], None]，每帧每路 cam
                        在编码前透传给 record_episode（Task 5 UI hook；
                        默认 None=零行为变化，既有测试全绿守门）
    """
    for ep in range(episodes):
        buf, hifreq_block = record_episode(robot, teleop, fps, episode_sec, gripper_max_open,
                                           cam_names, stop_flag=stop_flag,
                                           frame_observer=frame_observer,
                                           hifreq_rate=240.0)

        action = decide(ep)

        if action == "stop":
            # 停止：不 submit，不 reset，直接退出循环
            log.info(f"[REC] episode {ep} 停止录制")
            break
        elif action == "discard":
            # 丢弃：不 submit，不产文件
            log.info(f"[REC] episode {ep} 丢弃（不写盘）")
            buf = None
        else:
            # keep：deepcopy 必须在 buf 复用/清空前，编码已在 record_episode 内完成
            path = f"{out_dir}/ep{ep:04d}_{int(time.time())}.h5"
            payload = copy.deepcopy({  # 整体 deepcopy：frames+meta 一次隔离，消除 oc2base_R/cam_names 别名风险
                "frames": buf,
                "meta": dict(
                    task_name=task_name,
                    target_fps=fps,
                    oc2base_R=oc2base_R,
                    quality={},
                    vr_source=vr_source,
                    cam_names=cam_names,
                    state_hifreq_block=hifreq_block,  # Task 7-A：M>0 实填，None→write_episode 占位
                ),
            })  # deepcopy 时序：在 buf=None 前
            saver.submit(path, payload)
            log.info(f"[REC] episode {ep} 已入队写盘 → {path}")
            buf = None  # 释放本地引用；后台线程持有 deepcopy 快照

        # episode 间 reset（非末尾、非 stop 后）
        if reset_fn is not None and ep < episodes - 1:
            reset_fn()
            if reset_wait > 0:
                time.sleep(reset_wait)


def main():
    ap = argparse.ArgumentParser(description="hdf5 录制入口（franka-hdf5-v2）")
    ap.add_argument("--config", required=True, help="record_cfg.yaml 路径")
    ap.add_argument("--fps", type=float, default=None, help="录制帧率(默认读 cfg.fps; 给了则临时覆盖)")
    ap.add_argument("--episodes", type=int, default=None, help="录制 episode 数(默认读 cfg.task.num_episodes; 给了则临时覆盖)")
    ap.add_argument("--episode-sec", type=float, default=None, help="每 episode 最长时间（秒）(默认读 cfg.time.episode_time_sec; 给了则临时覆盖)")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录(默认读 cfg.out_dir; 给了则临时覆盖)")
    ap.add_argument("--task-name", default=None, help="任务名称写入 hdf5(默认读 cfg.task.description; 给了则临时覆盖)")
    # 标定文件（oc2base_R），Task3 用；此处允许缺失并用单位矩阵占位
    ap.add_argument("--oc2base-R", default=None,
                    help="oc2base_R .npy 路径（缺失则用单位矩阵）")
    a = ap.parse_args()

    # 延迟 import 硬件依赖（RecordConfig 来自 record_config，需 lerobot 真实包）
    from record_config import RecordConfig

    with open(a.config) as fh:
        raw = yaml.safe_load(fh)
    record_cfg = RecordConfig(raw["record"])
    fps = resolve_record_fps(a.fps, record_cfg.fps)
    log.info(f"[REC] 录制频率单一来源 fps={fps}（相机/循环/写盘同源）")

    # CLI None 仅覆盖：各录制超参从 RecordConfig 读（单一真值），CLI 给了才临时覆盖
    # 严格 is None 判断，禁 cli or cfg（0/""/False falsy 误判）；参 resolve_record_fps 范式
    overrides = resolve_record_overrides(
        cli_episodes=a.episodes,
        cli_episode_sec=a.episode_sec,
        cli_out_dir=a.out_dir,
        cli_task_name=a.task_name,
        cli_oc2base=a.oc2base_R,
        record_cfg=record_cfg,
        out_dir_fallback=_paths.HDF5_EPISODES_DIR,
    )
    episodes = overrides["episodes"]
    episode_sec = overrides["episode_sec"]
    out_dir = overrides["out_dir"]
    task_name = overrides["task_name"]
    oc2base_path = overrides["oc2base_path"]
    log.info(f"[REC] episodes={episodes}, episode_sec={episode_sec}, out_dir={out_dir}")
    log.info(f"[REC] task_name={task_name!r}, oc2base_path={oc2base_path!r}")

    # reset 配置：读 RecordConfig（T3 已迁入，单一真源，parse_reset_config 已在底层调用）
    reset_between_episodes = record_cfg.reset_between_episodes
    reset_wait_sec = record_cfg.reset_wait

    # 标定矩阵：经 resolve_record_overrides 接通 RecordConfig.oc2base_path（CLI 覆盖优先）
    # 文件缺失降级为 np.eye(3)+warning（Phase A 语义保留，不强制要求标定文件存在）
    if oc2base_path is not None and os.path.exists(oc2base_path):
        R = np.load(oc2base_path)
    else:
        log.warning("[REC] oc2base_R 未提供或文件不存在，使用单位矩阵占位")
        R = np.eye(3)

    robot, teleop, gripper_max_open = build_robot_and_teleop(record_cfg, fps)
    os.makedirs(out_dir, exist_ok=True)

    # 相机名与 HDF5 schema 对应：wrist_image, exterior_image
    cam_names = list(robot.cameras.keys())
    log.info(f"[REC] 检测到相机: {cam_names}")

    # sink：闭包调 write_episode（Task 2 抽出的模块级函数）
    def sink(path, payload):
        meta = dict(payload["meta"])
        state_hifreq_block = meta.pop("state_hifreq_block", None)
        write_episode(path, payload["frames"], state_hifreq_block=state_hifreq_block, **meta)

    # §11.2 预检门：robot.connect 后、录制前运行，任一不过 → sys.exit(2)（开录前 ~10s 拦截）
    # 目的：把"中途静默失败"变"启动期可行动报错"，避免录完才发现夹爪/色彩异常
    from core import preflight as pf
    from tools.hdf5_lerobot_map import _decode as _hdf5_decode  # 接线错=模块级 bug，fail-loud

    # 0. 控制器预检：幂等启动 cartesian impedance controller，避免主循环 send_action 时
    # 报 "no controller running"（_run_polymetis_rw.sh 后台异步启动偶发失败的兜底）
    if record_cfg.controller_preflight_enabled:
        log.info("[PREFLIGHT] 启动/确认 cartesian impedance controller...")
        _arm_client = getattr(robot, "_robot", None)
        if _arm_client is None:
            _preflight_abort(
                robot, teleop,
                "无法获取 arm zerorpc client(robot._robot 缺失)→检查 robot 连接/wrapper",
            )
        controller_verdict = pf.run_controller_preflight(
            client=_arm_client,
            polymetis_python=record_cfg.controller_preflight_python,
            polymetis_conda_prefix=record_cfg.controller_preflight_conda_prefix,
        )
        if not controller_verdict.ok:
            _preflight_abort(robot, teleop, f"控制器预检失败: {controller_verdict.reason}")
        log.info(f"[PREFLIGHT] {controller_verdict.reason}")
    else:
        log.info("[PREFLIGHT] 控制器预检已禁用 (yaml record.controller_preflight.enabled=false)")

    # 1. 夹爪预检（仅当 use_gripper=True；zerorpc client 由 robot._robot 获取，预检在循环外一次性完成）
    if getattr(record_cfg, "use_gripper", True):
        log.info("[PREFLIGHT] 运行夹爪预检（进程存活/连接就绪/width 真变）…")
        _gripper_client = getattr(robot, "_robot", None)
        if _gripper_client is None:
            _preflight_abort(
                robot, teleop,
                "无法获取夹爪 zerorpc client(robot._robot 缺失)→检查 robot 连接/wrapper",
            )
        gripper_verdict = pf.run_gripper_preflight(
            client=_gripper_client,
            proc_probe=pf.default_proc_probe,
            log_probe=lambda: pf.default_log_probe(
                "/home/ubuntu/Desktop/jhli/_gripper_live.log"
            ),
        )
        if not gripper_verdict.ok:
            _preflight_abort(robot, teleop, f"夹爪预检失败: {gripper_verdict.reason}")

    # 2. 色彩预检（默认开启；RecordConfig.color_preflight 单一来源，yaml 可设 color_preflight: false 关闭）
    color_preflight_enabled = record_cfg.color_preflight
    if color_preflight_enabled:
        log.info("[PREFLIGHT] 采首帧运行色彩通道序预检…")
        encoded_pf = []
        try:
            obs_pf = robot.get_observation()
            for camera_name in cam_names:
                img = obs_pf.get(camera_name)
                if img is not None and isinstance(img, np.ndarray):
                    encoded_pf.append(_encode_jpg(img))
        except Exception as e:  # noqa: BLE001 — 仅相机观测/采帧失败=弱降级(色彩判据不确定)，继续录制
            log.warning(f"[PREFLIGHT] 色彩预检采帧异常（弱判据，继续录制）: {e}")
            encoded_pf = []
        if encoded_pf:
            # _hdf5_decode: cv2.imdecode(IMREAD_COLOR)(BGR)→cvtColor(BGR2RGB)(RGB)
            # 接线错由顶部 import 已 fail-loud，此处调用异常=真 bug，不被 warning 吞
            color_verdict = pf.run_color_preflight(
                decode_fn=_hdf5_decode,
                encoded_frames=encoded_pf,
            )
            if not color_verdict.ok:
                _preflight_abort(robot, teleop, f"色彩预检失败: {color_verdict.reason}")
        else:
            log.warning("[PREFLIGHT] 色彩预检跳过（无可用相机帧）")

    log.info("[PREFLIGHT] 夹爪/色彩预检通过，开始录制")

    # 终端键盘监听（复用 run_record.py 既有模式）
    # headless 时 listener=None、events 全 False → EpisodeDecider 安全降级为计时保存
    from lerobot.utils.control_utils import init_keyboard_listener
    from core.episode_keyboard import EpisodeDecider

    listener, events = init_keyboard_listener()
    dec = EpisodeDecider(events)
    log.info("[REC] 键盘控制：→ 结束并保存当前 ep | ← 结束并丢弃 | Esc 停止录制")

    # decide：读取当前 events 状态，keep/discard 后 reset 逐 ep 标志（stop 不 reset）
    def decide(ep):
        action = dec.decide_after_episode()
        # stop 故意不 reset：stop_recording 是全局停止标志，保留以让 run_episodes 跳出循环（勿"顺手"在此清理）
        if action in ("keep", "discard"):
            dec.reset_episode_flags()
        return action

    try:
        # with 上下文保证进程退出前 close() join 排空（数据零丢）
        with AsyncEpisodeSaver(sink=sink, maxsize=5) as saver:
            run_episodes(
                robot, teleop, saver,
                fps=fps,
                episode_sec=episode_sec,
                gripper_max_open=gripper_max_open,
                cam_names=cam_names,
                out_dir=out_dir,
                task_name=task_name,
                oc2base_R=R,
                vr_source=record_cfg.control_mode,
                episodes=episodes,
                decide=decide,
                reset_fn=robot.reset if reset_between_episodes else None,
                reset_wait=reset_wait_sec,
                stop_flag=dec.episode_stop_flag(),
            )
    finally:
        robot.disconnect()
        teleop.disconnect()


if __name__ == "__main__":
    main()
