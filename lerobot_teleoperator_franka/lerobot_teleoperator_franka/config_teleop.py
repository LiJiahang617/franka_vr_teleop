from dataclasses import dataclass, field
from typing import List

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("lerobot_teleoperator_franka")
@dataclass
class BaseTeleopConfig(TeleoperatorConfig):
    """所有遥操模式的基础配置。"""
    control_mode: str = "unityvr"
    use_gripper: bool = True


@TeleoperatorConfig.register_subclass("unityvr_teleop")
@dataclass
class UnityVRTeleopConfig(BaseTeleopConfig):
    """世界系 Unity VR 遥操配置（头部无关）。"""
    control_mode: str = "unityvr"
    pose_scaler: List[float] = field(default_factory=lambda: [1.0, 1.0])
    channel_signs: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    oc2base_path: str = "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop/.stage3_oc2arm_R.npy"
    robot_ip: str = "127.0.0.1"
    robot_port: int = 4242
    # §11.3 每轴运动灵敏度（默认全1=等价历史 pose_scaler 两标量行为，无新键时零改动）
    # 仅在已验通映射方向输出后逐轴缩放，不改 _POS_MAP/R_cal 方向/手性（§10.2(0) 红线）
    pos_axis_gain: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])  # [gx, gy, gz]
    rot_axis_gain: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])  # [grx, gry, grz]
    # VR 食指扳机激活阈值 [0..1] (高于此值开始发 delta_ee_pose)
    trigger_threshold: float = 0.85
    # send_action EMA 平滑系数 (0..1); 越小越平滑, 越大越跟手
    smoothing_alpha: float = 0.4
