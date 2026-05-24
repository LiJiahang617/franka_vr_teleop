from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("franka_robot")
@dataclass
class FrankaConfig(RobotConfig):
    use_gripper: bool = True
    gripper_reverse: bool = True
    robot_ip: str = "192.168.1.104"
    gripper_bin_threshold: float = 0.98
    gripper_max_open: float = 0.0801  # gripper max open width in meters
    debug: bool = True
    close_threshold: float = 0.7
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    control_mode: str = "isoteleop"
    # Execute mode for oculus: "ee_pose" (cartesian impedance) or "joint" (joint impedance via IK)
    execute_mode: str = "ee_pose"
    # HOME 关节位姿 (rad, 7 维); 默认 Franka2 示教位; "回 Home" 按钮用. None=使用模块默认.
    home_joint_position: list = field(default_factory=lambda: [
        -0.032383, 0.309742, -0.028457, -1.616216, 0.001244, 1.563408, 0.832192,
    ])
    # send_action EMA 平滑系数 (0..1); 越小越平滑越延迟, 越大越跟手
    smoothing_alpha: float = 0.4

