"""hdf5 录制入口：复用 run_record 的 robot/teleop/相机构造，sink=HDF5EpisodeWriter。

与既有 run_record.py 并存，不改其逻辑。
读取同一份 record_cfg.yaml，用 RecordConfig 解析配置，
把 LeRobotDataset sink 替换为 HDF5EpisodeWriter 写 franka-hdf5-v1。

观测字段对齐说明（来自 franka.py get_observation 实读）：
  - joint 位置: joint_1.pos ... joint_7.pos (float，单独 key)
  - joint 速度: joint_1.vel ... joint_7.vel (float，已接通 robot_get_joint_velocities)
  - ee pose:   ee_pose.x/y/z/rx/ry/rz (float，单独 key)
  - 夹爪状态:  gripper_state_norm ([0,1]), gripper_max_open 来自 cfg
  - 夹爪指令:  gripper_cmd_bin (get_action 返回)
  - 相机图像:  cam.read() 返回 numpy array，用 cv2.imencode 编码为 jpeg bytes
"""
import argparse
import copy
import logging
import os
import sys
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

    # 相机配置（与 run_record.py 完全一致）
    wrist_image_cfg = RealSenseCameraConfig(
        serial_number_or_name=record_cfg.wrist_cam_serial,
        fps=realsense_fps(fps),
        width=record_cfg.width,
        height=record_cfg.height,
        color_mode=ColorMode.RGB,
        use_depth=False,
        rotation=Cv2Rotation.NO_ROTATION,
    )
    exterior_image_cfg = RealSenseCameraConfig(
        serial_number_or_name=record_cfg.exterior_cam_serial,
        fps=realsense_fps(fps),
        width=record_cfg.width,
        height=record_cfg.height,
        color_mode=ColorMode.RGB,
        use_depth=False,
        rotation=Cv2Rotation.NO_ROTATION,
    )
    camera_config = {"wrist_image": wrist_image_cfg, "exterior_image": exterior_image_cfg}

    # 机器人配置（与 run_record.py 完全一致）
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


def record_episode(robot, teleop, fps: float, max_sec: float,
                   gripper_max_open: float, cam_names: list,
                   *, stop_flag=None, frame_observer=None) -> list:
    """录制一个 episode，每 tick 收 action/obs 拼 frame 返回帧列表。

    图像在 _encode_jpg 中编码（cvtColor(RGB2BGR)->imencode），
    编码在 deepcopy 前完成（由 run_episodes 在 submit 前 deepcopy）。

    Args:
        robot: Franka 实例
        teleop: teleop 实例
        fps: 目标帧率
        max_sec: 最长录制时间（秒）
        gripper_max_open: 夹爪最大开度（米），用于将 norm 转换为 gripper_m
        cam_names: 相机名列表，需与写盘时 cam_names 一致
        stop_flag: 可选 callable()->bool，返回 True 时提前结束当前 ep  # Task4 中途中断预留, 当前未接线
                   （Task 4 键盘接入；本 Task 默认 None=按 max_sec 结束）
        frame_observer: 可选 Callable[[str, np.ndarray], None]，每帧每路 cam
                        在 _encode_jpg 之前调用，传入 (cam_name, rgb_ndarray)。
                        默认 None=零行为变化（既有代码路径完全不变）。
                        注意：observer 收到的是 obs[cn] 原始引用，勿在内部修改。

    Returns:
        list[dict]：采集的帧列表，每帧含 ts/joints/joint_vel/ee_pose/
                    gripper_m/gripper_norm/gripper_cmd/delta_ee_pose/cams；
                    cams[cn] 已是 JPEG 编码后的 uint8 bytes。
    """
    buf = []
    period = 1.0 / fps
    t_end = time.monotonic() + max_sec
    while time.monotonic() < t_end:
        # 键盘提前结束钩子（Task 4 接入；stop_flag=None 时跳过判断）
        if stop_flag is not None and stop_flag():
            break

        t0 = time.monotonic()

        # 采集 teleop action
        action = teleop.get_action()

        # 发送 action 到机器人
        robot.send_action(action)

        # 采集机器人观测（包含相机图像）
        obs = robot.get_observation()

        # 拼接 joint 位置数组（joint_1.pos ... joint_7.pos）
        joints = np.array([obs[f"joint_{i+1}.pos"] for i in range(7)], dtype=np.float64)

        # joint_vel: 已接通 robot_get_joint_velocities; 缺失则零填(向后兼容)
        joint_vel = extract_joint_vel(obs)

        # ee_pose 数组
        ee_pose = np.array(
            [obs[f"ee_pose.{ax}"] for ax in ["x", "y", "z", "rx", "ry", "rz"]],
            dtype=np.float64,
        )

        # 夹爪状态：gripper_state_norm * gripper_max_open → gripper_m
        gripper_norm = float(obs.get("gripper_state_norm") or 0.0)
        gripper_m = gripper_norm * gripper_max_open

        # 夹爪指令
        gripper_cmd = float(action.get("gripper_cmd_bin", 0.0))

        # delta_ee_pose action 数组
        delta_ee_pose = np.array(
            [action.get(f"delta_ee_pose.{ax}", 0.0) for ax in ["x", "y", "z", "rx", "ry", "rz"]],
            dtype=np.float64,
        )

        # 相机图像：编码为 jpeg bytes（编码在 deepcopy 前，满足 deepcopy 时序要求）
        cams = {}
        for cn in cam_names:
            img = obs.get(cn)
            if img is not None and isinstance(img, np.ndarray):
                # frame_observer 在 _encode_jpg 之前调用原始 RGB 数据（Task 5 hook）
                if frame_observer is not None:
                    frame_observer(cn, img)
                cams[cn] = _encode_jpg(img)
            else:
                # 占位（相机未就绪时）
                cams[cn] = np.zeros((4,), np.uint8)

        buf.append(dict(
            ts=time.monotonic(),
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

    return buf


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
        buf = record_episode(robot, teleop, fps, episode_sec, gripper_max_open,
                             cam_names, stop_flag=stop_flag,
                             frame_observer=frame_observer)

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
    ap = argparse.ArgumentParser(description="hdf5 录制入口（franka-hdf5-v1）")
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

    # 延迟 import 硬件依赖（RecordConfig 来自 run_record，需 lerobot 真实包）
    from run_record import RecordConfig

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
        write_episode(path, payload["frames"], **payload["meta"])

    # §11.2 预检门：robot.connect 后、录制前运行，任一不过 → sys.exit(2)（开录前 ~10s 拦截）
    # 目的：把"中途静默失败"变"启动期可行动报错"，避免录完才发现夹爪/色彩异常
    from core import preflight as pf
    from tools.hdf5_lerobot_map import _decode as _hdf5_decode  # 接线错=模块级 bug，fail-loud

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
