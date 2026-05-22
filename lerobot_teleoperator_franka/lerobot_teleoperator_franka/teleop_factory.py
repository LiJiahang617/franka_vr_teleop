#!/usr/bin/env python

"""
Factory for creating teleoperation instances.
"""

from .base_teleop import BaseTeleop
from .config_teleop import (
    BaseTeleopConfig,
    UnityVRTeleopConfig,
)
from .unityvr_teleop import UnityVRTeleop


def create_teleop(config: BaseTeleopConfig) -> BaseTeleop:
    """
    Create a teleoperation instance based on the configuration.

    Args:
        config: Teleoperation configuration (UnityVRTeleopConfig)

    Returns:
        A teleoperation instance (UnityVRTeleop)

    Raises:
        ValueError: If the control mode is not supported
    """
    if isinstance(config, UnityVRTeleopConfig) or config.control_mode == "unityvr":
        return UnityVRTeleop(config if isinstance(config, UnityVRTeleopConfig) else UnityVRTeleopConfig())

    else:
        raise ValueError(f"Unsupported control mode: {config.control_mode}. "
                         f"Supported modes: unityvr")


def create_teleop_config(control_mode: str, **kwargs) -> BaseTeleopConfig:
    """
    Create a teleoperation configuration based on the control mode.

    Args:
        control_mode: The teleoperation mode ("unityvr")
        **kwargs: Configuration parameters specific to each mode

    Returns:
        A teleoperation configuration instance

    Raises:
        ValueError: If the control mode is not supported
    """
    if control_mode == "unityvr":
        return UnityVRTeleopConfig(**kwargs)
    else:
        raise ValueError(f"Unsupported control mode: {control_mode}. "
                         f"Supported modes: unityvr")


# Convenience function to get action features for a control mode
def get_action_features(control_mode: str, use_gripper: bool = True) -> dict:
    """
    Get the action features for a given control mode.

    Args:
        control_mode: The teleoperation mode ("unityvr")
        use_gripper: Whether gripper is used

    Returns:
        Dictionary of action features

    Raises:
        ValueError: If the control mode is not supported
    """
    if control_mode == "unityvr":
        features = {}
        for axis in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"delta_ee_pose.{axis}"] = float
        if use_gripper:
            features["gripper_cmd_bin"] = float
        return features

    else:
        raise ValueError(f"Unsupported control mode: {control_mode}")
