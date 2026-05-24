"""hdf5 录制 Web UI 入口：组装 RecorderController + Flask app，
通过浏览器控制录制（替代键盘模式），与 run_record_hdf5.py 并存。

选择模式：
  - UI 模式（本文件）：yaml cfg.ui.enabled=true，浏览器访问 http://<host>:<port>
  - 终端键盘模式（run_record_hdf5.py）：yaml cfg.ui.enabled=false 或直接调用旧入口

复用 run_record_hdf5.py 中：
  - build_robot_and_teleop：构造 Franka + teleop（延迟 import 硬件依赖）
  - run_episodes：episode 循环编排（签名/语义零改动）
  - _preflight_abort：预检失败的清理退出
  - write_episode：HDF5 写盘函数
  - _encode_jpg：RGB→BGR→JPEG 编码（守 RGB 顺序 lesson）

RecorderController 持有 events dict，UI 按钮写入 events，
EpisodeDecider 读取 events，语义与终端键盘逐字等价。
"""
import argparse
import logging
import os
import sys
import threading

import numpy as np
import yaml
from werkzeug.serving import make_server  # H3 fix: graceful shutdown 替 app.run

from pathlib import Path as _Path

# scripts/ 目录加入 sys.path（与 run_record_hdf5.py 完全一致）
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

# 纯逻辑依赖（无硬件），顶层 import 安全
from core import paths as _paths
from core.async_saver import AsyncEpisodeSaver
from core.hdf5_writer import write_episode
from core.record_params import resolve_record_fps, resolve_record_overrides

# 从既有入口 import 复用函数（DRY，守 "禁止 fork 漂移" 红线）
# build_robot_and_teleop / run_episodes / _preflight_abort / _encode_jpg
# 均通过 importlib 动态加载（与 conftest.py / UI 子包保持一致的加载范式）
import importlib.util as _ilu

_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
_UI_DIR = os.path.join(os.path.dirname(_CORE_DIR), "ui")

# 加载 run_record_hdf5 模块（动态加载避免命名冲突，同时保持路径相对安全）
_rrh_spec = _ilu.spec_from_file_location(
    "run_record_hdf5",
    os.path.join(_CORE_DIR, "run_record_hdf5.py"),
)
_rrh_mod = _ilu.module_from_spec(_rrh_spec)
_rrh_spec.loader.exec_module(_rrh_mod)

# 从 run_record_hdf5 直接取函数引用（DRY：改一处两入口同步）
build_robot_and_teleop = _rrh_mod.build_robot_and_teleop
run_episodes = _rrh_mod.run_episodes
_preflight_abort = _rrh_mod._preflight_abort
_encode_jpg = _rrh_mod._encode_jpg

# 加载 UI 子包（control_panel / recorder_controller）
_cp_spec = _ilu.spec_from_file_location(
    "ui_control_panel",
    os.path.join(_UI_DIR, "control_panel.py"),
)
_cp_mod = _ilu.module_from_spec(_cp_spec)
_cp_spec.loader.exec_module(_cp_mod)
build_app = _cp_mod.build_app

_rc_spec = _ilu.spec_from_file_location(
    "ui_recorder_controller",
    os.path.join(_UI_DIR, "recorder_controller.py"),
)
_rc_mod = _ilu.module_from_spec(_rc_spec)
_rc_spec.loader.exec_module(_rc_mod)
RecorderController = _rc_mod.RecorderController

log = logging.getLogger("rec_hdf5_ui")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Bug 4 fix (Codex C): Werkzeug access log 每请求一行 (30Hz × 3 路由 ≈ 90/s) 噪音过大.
# 调到 WARNING: 屏蔽 200/304 access log, 仍保留 4xx/5xx 客户端/服务器错误 (诊断必需).
logging.getLogger("werkzeug").setLevel(logging.WARNING)


def main():
    """UI 模式录制入口：解析 cfg → 装配 controller → 起 Flask 服务器。

    与 run_record_hdf5.py 的 main 并存；由 cfg.ui.enabled 或 --ui 标志区分。
    """
    ap = argparse.ArgumentParser(
        description="hdf5 录制 Web UI 入口（浏览器控制录制）"
    )
    ap.add_argument("--config", required=True, help="record_cfg.yaml 路径")
    ap.add_argument("--fps", type=float, default=None,
                    help="录制帧率（默认读 cfg.fps；给了则临时覆盖）")
    ap.add_argument("--episodes", type=int, default=None,
                    help="录制 episode 数（默认读 cfg.task.num_episodes；给了则临时覆盖）")
    ap.add_argument("--episode-sec", type=float, default=None,
                    help="每 episode 最长时间（秒）（默认读 cfg.time.episode_time_sec；给了则临时覆盖）")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录（默认读 cfg.out_dir；给了则临时覆盖）")
    ap.add_argument("--task-name", default=None,
                    help="任务名称写入 hdf5（默认读 cfg.task.description；给了则临时覆盖）")
    ap.add_argument("--oc2base-R", default=None,
                    help="oc2base_R .npy 路径（缺失则用单位矩阵）")
    a = ap.parse_args()

    # 延迟 import 硬件依赖（RecordConfig 来自 record_config，需 lerobot 真实包）
    from record_config import RecordConfig

    with open(a.config) as fh:
        raw = yaml.safe_load(fh)
    record_cfg = RecordConfig(raw["record"])

    # 校验 ui.enabled：false 时快速报错，提示用旧入口
    ui_cfg = record_cfg.ui_config
    if not ui_cfg["enabled"]:
        log.error(
            "[UI] cfg ui.enabled=false，请改 yaml（record.ui.enabled: true）"
            " 或用 run_record_hdf5.py 键盘模式"
        )
        sys.exit(2)

    fps = resolve_record_fps(a.fps, record_cfg.fps)
    log.info(f"[UI] 录制频率单一来源 fps={fps}")

    # CLI None 仅覆盖（与 run_record_hdf5.py main 完全一致）
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
    log.info(f"[UI] episodes={episodes}, episode_sec={episode_sec}, out_dir={out_dir}")
    log.info(f"[UI] task_name={task_name!r}")

    reset_between_episodes = record_cfg.reset_between_episodes
    reset_wait_sec = record_cfg.reset_wait

    # 标定矩阵（与 run_record_hdf5.py 同款降级逻辑）
    if oc2base_path is not None and os.path.exists(oc2base_path):
        R = np.load(oc2base_path)
    else:
        log.warning("[UI] oc2base_R 未提供或文件不存在，使用单位矩阵占位")
        R = np.eye(3)

    # 构造 robot + teleop（延迟 import 硬件包，与 run_record_hdf5.py 完全一致）
    robot, teleop, gripper_max_open = build_robot_and_teleop(record_cfg, fps)
    os.makedirs(out_dir, exist_ok=True)

    cam_names = list(robot.cameras.keys())
    log.info(f"[UI] 检测到相机: {cam_names}")

    # events dict（由 UI 按钮写入；与 EpisodeDecider 三键逐字等价）
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    # §11.2 预检门：robot.connect 后、录制前运行，任一不过 → _preflight_abort 退出
    # 目的：把"中途静默失败"变"启动期可行动报错"，避免录完才发现夹爪/色彩异常
    from core import preflight as pf
    from tools.hdf5_lerobot_map import _decode as _hdf5_decode  # 接线错=模块级 bug，fail-loud

    # 0. 控制器预检 (Bug VR-控不了末端 真根因): 幂等启动 cartesian impedance controller.
    # UI 入口 fork run_record_hdf5.py 时漏了这一步, 导致 send_action 报 "no controller running".
    # 与 run_record_hdf5.py main 第 777-792 行同款.
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

    # 1. 夹爪预检（仅当 use_gripper=True；zerorpc client 由 robot._robot 获取）
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

    log.info("[PREFLIGHT] 夹爪/色彩预检通过，开始装配录制器")

    # 装配 RecorderController（桥接 UI 与录制器）
    controller = RecorderController(events, fps=fps)

    # _wrapped_run_episodes：用真 saver + 真 run_episodes，内部管理 AsyncEpisodeSaver 生命周期
    # Bug 2 fix: 把 yaml.reset_between_episodes 闭包捕获, 录制时仅当 True 才让 run_episodes 自动 reset.
    # attach_record_args.reset_fn 永远是 robot.reset (home 按钮始终可用).
    def _wrapped_run_episodes(robot_, teleop_, saver_, **kwargs):
        """run_episodes 包装：通过 AsyncEpisodeSaver 管理 saver 生命周期。

        Bug 2: kwargs["reset_fn"] 由 attach_record_args 传入 robot.reset; 但
        若 yaml reset_between_episodes=false, 这里强制覆盖为 None, run_episodes
        ep 间不自动 reset. home 按钮 (_handle_home_cmd) 仍能调 _record_args["reset_fn"].
        """
        def _sink(path, payload):
            write_episode(path, payload["frames"], **payload["meta"])

        if not reset_between_episodes:
            kwargs["reset_fn"] = None  # 仅切断 run_episodes 的自动 reset

        with AsyncEpisodeSaver(sink=_sink, maxsize=5) as saver_real:
            run_episodes(robot_, teleop_, saver_real, **kwargs)

    # 装配录制参数（Task 5 范式：attach_record_args → start()）
    # 直接传入 _wrapped_run_episodes，无需事后覆盖私有属性 _record_args
    controller.attach_record_args(
        robot=robot,
        teleop=teleop,
        saver=None,  # saver 由 _wrapped_run_episodes 内部管理；此处占位
        run_episodes_fn=_wrapped_run_episodes,
        fps=fps,
        episode_sec=episode_sec,
        gripper_max_open=gripper_max_open,
        cam_names=cam_names,
        out_dir=out_dir,
        task_name=task_name,
        oc2base_R=R,
        vr_source=record_cfg.control_mode,
        # M1 (Codex 复审): 撤销强制 episodes=1, 改为 yaml.task.num_episodes 决定.
        # UI 模式语义: 每次点开始录 yaml 配置的 num_episodes 条 ep (yaml 默认 1).
        # 用户可在 yaml 改为更大批量, save/discard 后会自动开下一条 ep.
        # 配合 yaml reset_between_episodes=false 时 ep 间不自动 reset (停留位姿).
        episodes=episodes,
        reset_fn=robot.reset,   # Bug 2: home 按钮始终可用; ep 间 auto-reset 由 _wrapped 控
        reset_wait=reset_wait_sec,
    )

    # 构造 Flask app（复用 build_app，已配置 Cache-Control after_request）
    app = build_app(controller=controller)

    log.info(
        f"[UI] Flask 服务器启动：http://{ui_cfg['host']}:{ui_cfg['port']}"
        " （浏览器打开，急停在手）"
    )

    # 架构修订（2026-05-24）：Flask 移子线程，主线程跑命令消费循环
    # 原因：原 controller daemon thread 跑 run_episodes 触发 zerorpc gevent
    # thread-affinity 死锁（lesson 2026-05-24-phaseE-ui-zerorpc-gevent-daemon-thread.md）。
    # zerorpc client 在 build_robot_and_teleop 主线程创建，必须由主线程消费命令。
    #
    # H3 fix: 用 werkzeug.make_server 拿到 server 句柄，finally 里 shutdown() 让
    # serve_forever 循环退出。注意：werkzeug ThreadedWSGIServer 工作线程 daemon_threads
    # 默认 True，handler 是 daemon 子线程，shutdown() 后随主进程退；本路径主要保证
    # serve_forever 干净退出，不强承诺等所有在飞 handler 完成。Stop/Start handler 自身
    # 是 ≤几 ms 的轻操作（只写 events + queue），实际风险极小。
    http_server = make_server(
        ui_cfg["host"], ui_cfg["port"], app, threaded=True
    )
    flask_thread = threading.Thread(
        target=http_server.serve_forever,
        # M1 (Codex 复审): daemon=True 兜底防 shutdown 卡死挂进程。主停机靠下方
        # http_server.shutdown() + join，daemon=True 仅是兜底，主进程退时强清线程。
        daemon=True,
        name="flask-server",
    )
    flask_thread.start()

    try:
        # prepare() 仅切状态机 INITIALIZING → WAITING（不启 daemon thread）
        # H4 fix: prepare 现 fail-loud，IllegalTransition 直接抛到此处由 finally 兜底
        controller.prepare()
        # preview sampler 用 daemon thread read cam（不调 zerorpc，OK）
        controller.start_preview_sampler(robot.cameras)
        # 主线程阻塞消费命令队列 — 所有 zerorpc 调用主线程发生，绕开 gevent thread-affinity
        controller.consume_commands_blocking()
    except KeyboardInterrupt:
        log.info("[UI] Ctrl+C，优雅退出")
    finally:
        # 有序清理：①停 Flask serve_forever（阻止新 /api 请求）→ ②停 preview
        # → ③stop_recording → ④断硬件。
        log.info("[UI] 停 Flask serve_forever（≤5s）...")
        try:
            http_server.shutdown()
        except Exception as e:
            log.warning(f"[UI] http_server.shutdown 异常: {e}")
        flask_thread.join(timeout=5)
        if flask_thread.is_alive():
            log.warning("[UI] Flask thread 5s 内未退出")
        # H2 fix: 每步独立 try/except，前一步异常不阻断后续 cleanup。
        # 含义：zerorpc 调用在 KI 后状态可能不一致，disconnect 自身可能抛；
        # 我们记录异常但保证 robot/teleop 都被尝试断开。
        for _name, _fn in [
            ("stop_preview_sampler", controller.stop_preview_sampler),
            ("stop_recording", controller.stop_recording),
            ("robot.disconnect", robot.disconnect),
            ("teleop.disconnect", teleop.disconnect),
        ]:
            try:
                _fn()
            except Exception as e:
                log.warning(f"[UI] cleanup {_name} 异常（继续后续清理）: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
