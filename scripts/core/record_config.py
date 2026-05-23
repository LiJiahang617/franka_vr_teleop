"""RecordConfig：录制会话配置类（独立模块，Route B 专用）。

从原 run_record.py 迁出，只保留 unityvr 控制模式，
甩掉对旧遥操 Config 类（Dynamixel/SpaceMouse/Oculus）和 lerobot 训练模块的 import。
"""
from typing import Dict, Any
from dataclasses import field
from lerobot_teleoperator_franka import UnityVRTeleopConfig
from core.record_params import (
    parse_reset_config,
    parse_bool,
    parse_section_dict,
    parse_positive_int,
    parse_axis_gain,
    DEFAULT_STATE_HIFREQ_RATE,
)

# ================================================================
# Phase E Task 6: ui 段严格解析（模块级纯函数，可离线单测）
# ================================================================

# ui 段字段默认值（避魔法数字散落 RecordConfig.__init__）
_UI_DEFAULTS = {
    "enabled": False,           # 默认关；True → 走 UI 模式，False = 键盘模式（现有 run_record）
    "host": "0.0.0.0",          # Flask 绑定 IP；本机访问 "127.0.0.1"，远程则 "0.0.0.0"
    "port": 5055,               # Flask 端口（避 polymetis 50051 / zerorpc 4242 / gripper 50052）
    "preview_max_w": 320,       # 预览图像最大宽度（像素）；实际按比例缩放
    "preview_max_h": 240,       # 预览图像最大高度（像素）
    "preview_quality": 60,      # JPEG 预览压缩质量（1-100）；值越低带宽越低
    "status_poll_hz": 30,       # 前端 setInterval 轮询频率参考（前端当前写死 33ms）
}


def _parse_ui_config(raw) -> dict:
    """严格解析 record.ui 子段，fail-loud 防畸形配置喂真机（Phase C 范式）。

    Args:
        raw: yaml 解析后的 ui 值（None=无 ui 段, dict=有 ui 段, 其它=ValueError）。

    Returns:
        七字段 dict，缺省字段回退 _UI_DEFAULTS。

    Raises:
        ValueError: raw 非 None/dict，或字段类型非法。
    """
    # parse_section_dict: None→{}, dict→自身, 其他→ValueError（fail-loud）
    ui_raw = parse_section_dict(raw, key_name="record.ui")

    return {
        # bool 严格解析：防 yaml "false" 字符串被 bool() 误判为 True
        "enabled": parse_bool(
            ui_raw.get("enabled"),
            default=_UI_DEFAULTS["enabled"],
            key_name="record.ui.enabled",
        ),
        # host: str 类型，无需特殊解析；None→默认
        "host": str(ui_raw["host"]) if ui_raw.get("host") is not None
                else _UI_DEFAULTS["host"],
        # port: 正整数，严格拒绝字符串/负数/零
        "port": parse_positive_int(
            ui_raw.get("port"),
            default=_UI_DEFAULTS["port"],
            key_name="record.ui.port",
        ),
        # preview_max_w: 正整数
        "preview_max_w": parse_positive_int(
            ui_raw.get("preview_max_w"),
            default=_UI_DEFAULTS["preview_max_w"],
            key_name="record.ui.preview_max_w",
        ),
        # preview_max_h: 正整数
        "preview_max_h": parse_positive_int(
            ui_raw.get("preview_max_h"),
            default=_UI_DEFAULTS["preview_max_h"],
            key_name="record.ui.preview_max_h",
        ),
        # preview_quality: 正整数（1-100 合理性由 Task 7 消费方校验，此处只保证 >0）
        "preview_quality": parse_positive_int(
            ui_raw.get("preview_quality"),
            default=_UI_DEFAULTS["preview_quality"],
            key_name="record.ui.preview_quality",
        ),
        # status_poll_hz: 正整数
        "status_poll_hz": parse_positive_int(
            ui_raw.get("status_poll_hz"),
            default=_UI_DEFAULTS["status_poll_hz"],
            key_name="record.ui.status_poll_hz",
        ),
    }


class RecordConfig:
    """录制会话配置类（Route B 专用，仅支持 unityvr 控制模式）。"""

    def __init__(self, cfg: Dict[str, Any]):
        storage = cfg["storage"]
        task = cfg["task"]
        time = cfg["time"]
        cam = cfg["cameras"]
        robot = cfg["robot"]
        policy = cfg["policy"]
        teleop = cfg["teleop"]

        # Global config
        self.repo_id: str = cfg["repo_id"]
        self.debug: bool = cfg.get("debug", True)
        self.fps: str = cfg.get("fps", 15)
        self.user_info: str = cfg.get("user_notes", None)
        self.run_mode: str = cfg.get("run_mode", "run_record")
        self.rename_map: dict[str, str] = field(default_factory=dict)

        # Teleop config - 仅支持 unityvr
        self.control_mode = teleop.get("control_mode", "unityvr")
        self._parse_teleop_config(teleop)

        # Policy config（保留接口，Route B run_record_hdf5 不实际使用 policy 字段）
        self._parse_policy_config(policy)

        # Robot config
        self.robot_ip: str = robot["ip"]
        self.use_gripper: bool = robot["use_gripper"]
        self.close_threshold = robot["close_threshold"]
        self.gripper_reverse: bool = robot["gripper_reverse"]
        self.gripper_bin_threshold: float = robot["gripper_bin_threshold"]
        self.gripper_max_open: float = robot.get("gripper_max_open", 0.08)
        self.execute_mode: str = robot.get("execute_mode", "ee_pose")  # "ee_pose" or "joint"

        # Task config
        self.num_episodes: int = task.get("num_episodes", 1)
        self.display: bool = task.get("display", True)
        self.task_description: str = task.get("description", "default task")
        self.resume: bool = task.get("resume", False)
        self.resume_dataset: str = task.get("resume_dataset", "")

        # Time config
        self.episode_time_sec: int = time.get("episode_time_sec", 60)
        self.reset_time_sec: int = time.get("reset_time_sec", 10)
        # 注意：yaml 实际键为 save_meta_period，此处 save_mera_period 为既有拼写 bug，
        # Phase C 不修（修改会破坏 run_record.py 另一入口行为 = 违反向后兼容红线）
        # TODO(Phase 后续/统一入口时): 修正拼写为 save_meta_period 并改 yaml
        self.save_mera_period: int = time.get("save_mera_period", 1)

        # Cameras config
        self.wrist_cam_serial: str = cam["wrist_cam_serial"]
        self.exterior_cam_serial: str = cam["exterior_cam_serial"]
        self.width: int = cam["width"]
        self.height: int = cam["height"]

        # Storage config
        self.push_to_hub: bool = storage.get("push_to_hub", False)

        # Phase C 扩展字段（严格解析, 全 fail-loud, 守 Phase B-T5 真机配置鲁棒 ethos）
        self.out_dir = cfg.get("out_dir", None)  # None=下游(Task4 resolve_record_overrides)
                                                  # 回退 paths.HDF5_EPISODES_DIR;
                                                  # contract: 消费方须 is None 检查并回退
        depth_cfg = parse_section_dict(cfg.get("depth"), key_name="record.depth")
        self.depth_enabled = parse_bool(
            depth_cfg.get("enabled"), default=False, key_name="record.depth.enabled"
        )
        sh_cfg = parse_section_dict(cfg.get("state_hifreq"), key_name="record.state_hifreq")
        self.state_hifreq_enabled = parse_bool(
            sh_cfg.get("enabled"), default=False, key_name="record.state_hifreq.enabled"
        )
        self.state_hifreq_rate = parse_positive_int(
            sh_cfg.get("rate"), default=DEFAULT_STATE_HIFREQ_RATE, key_name="record.state_hifreq.rate"
        )
        # depth/state_hifreq 仅 RecordConfig 占位键，不写 hdf5/不进 schema，Phase D 才消费

        # reset 配置：经 parse_reset_config 纯函数解析（单一真源在 record_params）
        self.reset_between_episodes, self.reset_wait = parse_reset_config(cfg)

        # 颜色预检开关（严格解析防 yaml 引号 "false" 误判；缺省 True = 预检开启）
        self.color_preflight = parse_bool(
            cfg.get("color_preflight"), default=True, key_name="record.color_preflight"
        )

        # 控制器预检配置（严格解析；缺省向后兼容旧 yaml 无 controller_preflight 段）
        cp_cfg = parse_section_dict(cfg.get("controller_preflight"), key_name="record.controller_preflight")
        self.controller_preflight_enabled = parse_bool(
            cp_cfg.get("enabled"), default=True, key_name="record.controller_preflight.enabled"
        )
        self.controller_preflight_python = str(
            cp_cfg.get("polymetis_python") or
            "/home/ubuntu/Desktop/jhli/envs/polymetis-local/bin/python"
        )
        self.controller_preflight_conda_prefix = str(
            cp_cfg.get("polymetis_conda_prefix") or
            "/home/ubuntu/Desktop/jhli/envs/polymetis-local"
        )

        # Phase E Task 6: Web UI 配置段（严格解析，缺省全默认，向后兼容旧 yaml 无 ui 段）
        self.ui_config = _parse_ui_config(cfg.get("ui"))

    def _parse_teleop_config(self, teleop: Dict[str, Any]) -> None:
        """解析遥操配置（仅支持 unityvr）。"""
        if self.control_mode == "unityvr":
            uvr_cfg = teleop.get("unityvr_config", {})
            self.use_gripper = uvr_cfg.get("use_gripper", True)
            self.pose_scaler = uvr_cfg.get("pose_scaler", [1.0, 1.0])
            self.channel_signs = uvr_cfg.get("channel_signs", [1, 1, 1, 1, 1, 1])
            self.oc2base_path = uvr_cfg.get("oc2base_path",
                "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop/.stage3_oc2arm_R.npy")
            self.unityvr_robot_ip = uvr_cfg.get("robot_ip", "127.0.0.1")
            self.unityvr_robot_port = uvr_cfg.get("robot_port", 4242)
            # §11.3 每轴增益（严格解析: len==3+finite+numeric, config-load 时 fail-loud;
            # T1 compute_delta_action 运行时 fail-loud 校验=两层防御; §10.2(0) 红线: 不改方向/手性）
            self.pos_axis_gain = parse_axis_gain(
                uvr_cfg.get("pos_axis_gain"),
                key_name="record.teleop.unityvr_config.pos_axis_gain",
            )
            self.rot_axis_gain = parse_axis_gain(
                uvr_cfg.get("rot_axis_gain"),
                key_name="record.teleop.unityvr_config.rot_axis_gain",
            )
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}（仅支持 unityvr）")

    def _parse_policy_config(self, policy: Dict[str, Any]) -> None:
        """解析 policy 配置（保留接口，Route B 录制不实际使用 policy 执行）。"""
        policy_type = policy["type"]
        if policy_type == "act":
            from lerobot.policies import ACTConfig
            self.policy = ACTConfig(
                device=policy["device"],
                push_to_hub=policy["push_to_hub"],
            )
        elif policy_type == "diffusion":
            from lerobot.policies import DiffusionConfig
            self.policy = DiffusionConfig(
                device=policy["device"],
                push_to_hub=policy["push_to_hub"],
            )
        else:
            raise ValueError(f"No config for policy type: {policy_type}")

        if policy.get("pretrained_path"):
            self.policy.pretrained_path = policy["pretrained_path"]

    def create_teleop_config(self):
        """创建遥操配置对象（仅 unityvr）。"""
        if self.control_mode == "unityvr":
            return UnityVRTeleopConfig(
                use_gripper=self.use_gripper,
                pose_scaler=self.pose_scaler,
                channel_signs=self.channel_signs,
                oc2base_path=self.oc2base_path,
                robot_ip=self.unityvr_robot_ip,
                robot_port=self.unityvr_robot_port,
                pos_axis_gain=self.pos_axis_gain,
                rot_axis_gain=self.rot_axis_gain,
            )
        else:
            raise ValueError(f"Unsupported control mode: {self.control_mode}")
