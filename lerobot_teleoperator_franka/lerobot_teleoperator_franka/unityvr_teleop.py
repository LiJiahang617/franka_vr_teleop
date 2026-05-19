"""UnityVR 世界系遥操作（镜像 OculusTeleop，头显朝向无关）。"""
import logging
from typing import Any, Dict

from .base_teleop import BaseTeleop
from .config_teleop import UnityVRTeleopConfig
from .unityvr_robot import UnityVRRobot

logger = logging.getLogger(__name__)


class UnityVRTeleop(BaseTeleop):
    """世界系 Quest 控制器遥操作：delta_ee_pose（base）+ 夹爪。RG 按下才记。"""

    config_class = UnityVRTeleopConfig
    name = "UnityVRTeleop"

    def __init__(self, config: UnityVRTeleopConfig):
        super().__init__(config)
        self.unityvr_robot: UnityVRRobot = None

    def _get_teleop_name(self) -> str:
        return "UnityVRTeleop"

    @property
    def action_features(self) -> dict:
        features = {}
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"delta_ee_pose.{axis}"] = float
        for i in range(7):
            features[f"joint_{i+1}.pos"] = float
        features["gripper_cmd_bin"] = float
        return features

    def _connect_impl(self) -> None:
        self.unityvr_robot = UnityVRRobot(
            oc2base_path=self.cfg.oc2base_path,
            pose_scaler=self.cfg.pose_scaler,
            channel_signs=self.cfg.channel_signs,
            use_gripper=self.cfg.use_gripper,
            robot_ip=self.cfg.robot_ip,
            robot_port=self.cfg.robot_port,
            pos_axis_gain=self.cfg.pos_axis_gain,   # §11.3 per-axis 位置增益透传
            rot_axis_gain=self.cfg.rot_axis_gain,   # §11.3 per-axis 旋转增益透传
        )
        logger.info("[TELEOP] UnityVR connected (world-frame, head-independent)")

    def _disconnect_impl(self) -> None:
        pass

    def _get_action_impl(self) -> Dict[str, Any]:
        return self.unityvr_robot.get_observations()
