import logging
import time
import threading
from pathlib import Path
from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.robot import Robot
from .config_franka import FrankaConfig
from typing import Any, Dict
import yaml
from .franka_interface_client import FrankaInterfaceClient
from scipy.spatial.transform import Rotation as R
import numpy as np
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig

HOME_JOINT_POSITION = np.array(
    [-0.032383, 0.309742, -0.028457, -1.616216, 0.001244, 1.563408, 0.832192]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class Franka(Robot):
    config_class = FrankaConfig
    name = "franka"

    def __init__(self, config: FrankaConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)

        self.config = config
        self._is_connected = False
        self._robot = None
        self._initial_pose = None
        self._prev_observation = None
        self._num_joints = 7
        self._gripper_force = 20
        self._gripper_speed = 0.2
        self._gripper_epsilon = 1.0
        self._gripper_position = 1
        self._dt = 0.002
        self._last_gripper_position = 1
        
        # 动作平滑：指数移动平均 (EMA) 滤波器
        # 平滑系数从 config 读 (默认 0.4, yaml record.robot.smoothing_alpha 可改)
        self._smoothing_alpha = getattr(config, "smoothing_alpha", 0.4)
        self._smoothed_delta = None  # 上一次平滑后的 delta

        # VR enable 开关 (ros2_realman_ws 同款设计):
        # 默认 False, send_action 跳过 update_desired_ee_pose 但保留夹爪命令.
        # 主控通过 enable_arm_control()/disable_arm_control() 切换;
        # enable 时主动启 joint impedance controller (防 polymetis stale terminate).
        self._arm_control_enabled: bool = False

    def enable_arm_control(self) -> bool:
        """启用臂控: 启 joint impedance controller + 置 enabled=True.

        必须在 connect 之后调. polymetis controller 启动后, 后续 send_action
        持续 update_current_policy 维持 controller alive (不会 stale terminate).

        Returns:
            True 启用成功; False 启动 controller 失败.
        """
        if not self.is_connected or self._robot is None:
            logger.warning("[ROBOT] enable_arm_control 失败: robot 未连接")
            return False

        # 探活: 如果 controller 已健康在跑(preflight 启过), 直接复用,
        # 不再重复 SetController. 实机日志确认: 在已有 joint impedance 上
        # hot-swap 同类型新 policy 会触发 libfranka communication_constraints_violation
        # reflex → 自动 error recovery (~1s) → polymetis validRobotContext 超 1s 阈值
        # → "Terminating custom controller" → 后续 update 全部 "no controller running".
        # 故策略: 探活 update 成功就只翻 flag; 仅在确实没有 controller 时才 start.
        #
        # 注意: FrankaInterfaceClient.robot_update_desired_ee_pose 包装器吞异常 + 要求
        # np.ndarray, 不能用来做探活. 直接走 zerorpc server 拿到 raise 行为.
        probe_ok = False
        try:
            ee = self._robot.robot_get_ee_pose()
            self._robot.server.robot_update_desired_ee_pose(list(ee))
            probe_ok = True
        except Exception as probe_exc:
            logger.info(f"[ROBOT] 现有 controller 不可用 ({type(probe_exc).__name__}: {str(probe_exc)[:120]}), 启新 joint impedance")

        if probe_ok:
            self._arm_control_enabled = True
            logger.info("[ROBOT] 臂控已启用 (复用现有 joint impedance controller)")
            return True

        # 没有可用 controller, 先 terminate 残留 (无残留时会抛, 忽略) 再 start
        try:
            self._robot.robot_terminate_current_policy()
        except Exception:
            pass
        try:
            self._robot.robot_start_joint_impedance_control()
            self._arm_control_enabled = True
            logger.info("[ROBOT] 臂控已启用 (新 joint impedance 已 load)")
            return True
        except Exception as e:
            logger.error(f"[ROBOT] enable_arm_control 失败: {e}")
            return False

    def disable_arm_control(self) -> None:
        """禁用臂控: 置 enabled=False (send_action 跳过 update_desired_ee_pose).

        不主动 terminate controller (避免与其他进程冲突);
        polymetis 在无 update 时自动 stale terminate (~1s).
        """
        self._arm_control_enabled = False
        logger.info("[ROBOT] 臂控已禁用 (send_action 仅处理夹爪)")

    def is_arm_control_enabled(self) -> bool:
        return self._arm_control_enabled

    def open_gripper_max(self, blocking: bool = True) -> bool:
        """把夹爪开到 config.gripper_max_open. 录制起点用, 保证每条 ep 始于全开.

        Args:
            blocking: True 等到夹爪 settle 才返回 (~0.5s); False 入队立即返回.

        Returns:
            True 调用成功; False 未连接或异常.
        """
        if not self.is_connected or self._robot is None:
            return False
        try:
            self._robot.gripper_goto(
                width=self.config.gripper_max_open,
                speed=self._gripper_speed,
                force=self._gripper_force,
                blocking=blocking,
            )
            return True
        except Exception as e:
            logger.warning(f"[ROBOT] open_gripper_max 失败: {e}")
            return False

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")

        # Connect to robot
        self._robot = self._check_franka_connection(self.config.robot_ip)
        
        # Initialize gripper
        if self.config.use_gripper:
            self._gripper = self._check_gripper_connection(self.config.robot_ip)


        # Connect cameras
        logger.info("\n===== [CAM] Initializing Cameras =====")
        for cam_name, cam in self.cameras.items():
            cam.connect()
            logger.info(f"[CAM] {cam_name} connected successfully.")
        logger.info("===== [CAM] Cameras Initialized Successfully =====\n")

        self.is_connected = True
        logger.info(f"[INFO] {self.name} env initialized. Control: {self.config.control_mode}\n")


    def _check_gripper_connection(self, robot_ip: str):
        logger.info("\n===== [GRIPPER] Initializing gripper...")
        self._robot.gripper_initialize()
        print("Homing gripper")
        self._robot.gripper_goto(width=self.config.gripper_max_open, speed=self._gripper_speed, force=self._gripper_force, blocking=True)
        logger.info("===== [GRIPPER] Gripper initialized successfully.\n")
        return None


    def _check_franka_connection(self, robot_ip: str):
        try:
            logger.info("\n===== [ROBOT] Connecting to Franka robot =====")
            
            franka = FrankaInterfaceClient(ip=robot_ip, port=4242)
            franka.robot_start_joint_impedance_control()

            joint_positions = franka.robot_get_joint_positions()
            if joint_positions is not None and len(joint_positions) == 7:
                formatted_joints = [round(j, 4) for j in joint_positions]
                logger.info(f"[ROBOT] Current joint positions: {formatted_joints}")
                logger.info("===== [ROBOT] Franka connected successfully =====\n")
            else:
                logger.info("===== [ERROR] Failed to read joint positions. Check connection or remote control mode =====")

        except Exception as e:
            logger.info("===== [ERROR] Failed to connect to Franka robot =====")
            logger.info(f"Exception: {e}\n")

        return franka


    def reset(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        # Reset robot
        # ee_positions_reset = np.array(
        # [0.40581301, 0.0, 0.44111654, -2.22150303, -2.15458315, 0.0]
        # )
        # print(f"\nMoving ee to: {ee_positions_reset} ...\n")
        # self._robot.robot_move_to_ee_pose(pose=ee_positions_reset, time_to_go=2.0)
        # self._robot.gripper_goto(
        #     width=robot_config.gripper_max_open,
        #     speed=robot_config.gripper_speed,
        #     force=robot_config.gripper_force,
        #     blocking=True
        # )

        # joint_positions = np.array([1.58472168, -1.56486702, -1.74356186, -2.634835, -0.11180906, 4.2022109, -1.51133597])
        print(f"\nMoving joint positions to: {HOME_JOINT_POSITION} ...\n")
        self._robot.robot_move_to_joint_positions(positions = np.asarray(getattr(self.config, "home_joint_position", None) or HOME_JOINT_POSITION, dtype=np.float64), time_to_go=None)
        self._robot.gripper_goto(
            width=self.config.gripper_max_open,
            speed=self._gripper_speed,
            force=self._gripper_force,
            blocking=True
        )
        # self._robot.gripper_goto(width=self.config.gripper_max_open, speed=self._gripper_speed, force=self._gripper_force, blocking=True)
        logger.info("===== [ROBOT] Robot reset successfully =====\n")


    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            # joint positions
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
            "joint_6.pos": float,
            "joint_7.pos": float,
            # gripper state
            "gripper_state_norm": float, # raw position in [0,1]
            "gripper_cmd_bin": float, # action command bin (0 or 1)
            # # joint velocities
            # "joint_1.vel": float,
            # "joint_2.vel": float,
            # "joint_3.vel": float,
            # "joint_4.vel": float,
            # "joint_5.vel": float,
            # "joint_6.vel": float,
            # "joint_7.vel": float,
            # end effector pose
            "ee_pose.x": float,
            "ee_pose.y": float,
            "ee_pose.z": float,
            "ee_pose.rx": float,
            "ee_pose.ry": float,
            "ee_pose.rz": float,
        }
        # return {
        #     "ee_pose.x": float,
        #     "ee_pose.y": float,
        #     "ee_pose.z": float,
        #     "ee_pose.rx": float,
        #     "ee_pose.ry": float,
        #     "ee_pose.rz": float,
        #     "gripper_state_norm": float, # raw position in [0,1]
        # }

    @property
    def action_features(self) -> dict[str, type]:
        """Return action features based on control mode."""
        if self.config.control_mode == "unityvr":
            features = {}
            # Delta EE pose (always present)
            for axis in ["x", "y", "z", "rx", "ry", "rz"]:
                features[f"delta_ee_pose.{axis}"] = float
            if self.config.use_gripper:
                features["gripper_cmd_bin"] = float
            return features
        else:
            raise ValueError(f"Unsupported control mode: {self.config.control_mode}")

    def _handle_gripper(self, gripper_value: float, is_binary: bool = True) -> None:
        """Handle gripper control with common logic."""
        if not self.config.use_gripper:
            return
        
        if is_binary:
            gripper_position = gripper_value
        else:
            gripper_position = 0.0 if gripper_value < self.config.close_threshold else 1.0
        
        if self.config.gripper_reverse:
            gripper_position = 1 - gripper_position

        try:
            if gripper_position != self._last_gripper_position:
                self._robot.gripper_goto(
                    width=gripper_position * self.config.gripper_max_open,
                    speed=self._gripper_speed,
                    force=self._gripper_force,
                )
                self._last_gripper_position = gripper_position
            
            gripper_state = self._robot.gripper_get_state()
            gripper_state_norm = max(0.0, min(1.0, gripper_state["width"] / self.config.gripper_max_open))
            if self.config.gripper_reverse:
                gripper_state_norm = 1 - gripper_state_norm
            self._gripper_position = gripper_state_norm
        except Exception as e:
            logger.warning(f"[GRIPPER] zerorpc error: {e}")

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.control_mode == "unityvr":
            self._send_action_cartesian(action)
        else:
            raise ValueError(f"Unsupported control mode: {self.config.control_mode}")

        return action

    def _send_action_cartesian(self, action: dict[str, Any]) -> None:
        """Send action in spacemouse/oculus mode (delta ee pose).

        VR enable: _arm_control_enabled=False 时跳过 update_desired_ee_pose
        (避免 polymetis 报 no controller running), 但仍处理 gripper 命令.
        """
        # VR disable 短路: 跳过臂控, 仍处理夹爪 (用户可独立测试夹爪)
        if not self._arm_control_enabled:
            if "gripper_cmd_bin" in action:
                self._handle_gripper(action["gripper_cmd_bin"], is_binary=True)
            return

        # Check for reset request
        if action.get("reset_requested", False):
            logger.info("[ROBOT] Reset requested, moving to home position...")
            try:
                # ee_positions_reset= np.array(
                #     [0.55581301, 0.00308523, 0.44111654, -2.22150303, -2.15458315, 0.00646556]
                # )
                # self._robot.robot_move_to_ee_pose(pose = ee_positions_reset, time_to_go=2.0)
                # self._robot.gripper_goto(
                #     width=self.config.gripper_max_open,
                #     speed=self._gripper_speed,
                #     force=self._gripper_force,
                #     blocking=True
                # )
                self._robot.robot_move_to_joint_positions(positions = np.asarray(getattr(self.config, "home_joint_position", None) or HOME_JOINT_POSITION, dtype=np.float64), time_to_go=None)
                self._robot.gripper_goto(
                    width=self.config.gripper_max_open,
                    speed=self._gripper_speed,
                    force=self._gripper_force,
                    blocking=True
                )
                self._robot.robot_start_joint_impedance_control()
            except Exception as e:
                logger.warning(f"[ROBOT] Reset failed: {e}, trying to restart controller...")
                try:
                    self._robot.robot_start_joint_impedance_control()
                except Exception as e2:
                    logger.error(f"[ROBOT] Failed to restart controller: {e2}")
            return
        
        delta_ee_pose = np.array([action[f"delta_ee_pose.{axis}"] for axis in ["x", "y", "z", "rx", "ry", "rz"]])

        # --- EMA 动作平滑 ---
        if np.linalg.norm(delta_ee_pose) < 1e-6:
            # 输入为零（RG 没按），重置平滑状态
            self._smoothed_delta = None
        else:
            if self._smoothed_delta is None:
                self._smoothed_delta = delta_ee_pose.copy()
            else:
                alpha = self._smoothing_alpha
                self._smoothed_delta = alpha * delta_ee_pose + (1 - alpha) * self._smoothed_delta
            delta_ee_pose = self._smoothed_delta.copy()

        if not self.config.debug:
            import scipy.spatial.transform as st

            try:
                ee_pose = self._robot.robot_get_ee_pose()
            except Exception as e:
                logger.warning(f"[ROBOT] Failed to get ee pose: {e}")
                if "gripper_cmd_bin" in action:
                    self._handle_gripper(action["gripper_cmd_bin"], is_binary=True)
                return

            # 计算位置和旋转的变化量
            position_delta = np.linalg.norm(delta_ee_pose[:3])
            rotation_delta = np.linalg.norm(delta_ee_pose[3:])
            
            # 设置阈值：位置变化超过 0.03m 或旋转变化超过 0.2rad 时进行插值
            max_position_step = 0.02  # 每步最大位置变化 (米)
            max_rotation_step = 0.1   # 每步最大旋转变化 (弧度)
            
            # 计算需要的插值步数
            position_steps = max(1, int(np.ceil(position_delta / max_position_step))) if position_delta > 0 else 1
            rotation_steps = max(1, int(np.ceil(rotation_delta / max_rotation_step))) if rotation_delta > 0 else 1
            num_steps = max(position_steps, rotation_steps)
            
            # 如果动作太大，进行插值
            if num_steps > 1:
                logger.debug(f"[ROBOT] Large delta detected, interpolating with {num_steps} steps")
                
                for step in range(1, num_steps + 1):
                    alpha = step / num_steps
                    interpolated_delta = delta_ee_pose * alpha
                    
                    target_position = ee_pose[:3] + interpolated_delta[:3]
                    current_rot = st.Rotation.from_rotvec(ee_pose[3:])
                    delta_rot = st.Rotation.from_rotvec(interpolated_delta[3:])
                    target_rotation = delta_rot * current_rot
                    target_rotvec = target_rotation.as_rotvec()
                    target_ee_pose = np.concatenate([target_position, target_rotvec])
                    try:
                        self._robot.robot_update_desired_ee_pose(target_ee_pose)
                    except Exception as e:
                        logger.warning(f"[ROBOT] zerorpc error during interpolation step {step}: {e}")
                        break
                    time.sleep(0.01)  # 每步间隔 10ms
            else:
                # 正常小动作或零 delta (keep-alive): 直接执行 update.
                # 零 delta 时 target = 当前 ee_pose (无运动), 仅用于保持 polymetis
                # controller alive (无 update 超 ~1s 会 stale terminate, 导致
                # vr_enable 后用户首次按 trigger 时 controller 已死 → 末端不动).
                target_position = ee_pose[:3] + delta_ee_pose[:3]
                current_rot = st.Rotation.from_rotvec(ee_pose[3:])
                delta_rot = st.Rotation.from_rotvec(delta_ee_pose[3:])
                target_rotation = delta_rot * current_rot
                target_rotvec = target_rotation.as_rotvec()
                target_ee_pose = np.concatenate([target_position, target_rotvec])
                try:
                    self._robot.robot_update_desired_ee_pose(target_ee_pose)
                except Exception as e:
                    logger.warning(f"[ROBOT] zerorpc error: {e}")
        
        if "gripper_cmd_bin" in action:
            self._handle_gripper(action["gripper_cmd_bin"], is_binary=True)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        try:
            # Read joint positions
            joint_position = self._robot.robot_get_joint_positions()
            # Read joint velocities
            joint_velocity = self._robot.robot_get_joint_velocities()
            # Read end effector pose
            ee_pose = self._robot.robot_get_ee_pose()
        except Exception as e:
            logger.warning(f"[ROBOT] zerorpc error in get_observation: {e}")
            # 返回上次的观测值作为 fallback
            if self._prev_observation is not None:
                return self._prev_observation
            else:
                raise

        # Prepare observation dictionary
        obs_dict = {}
        for i in range(len(joint_position)):
            obs_dict[f"joint_{i+1}.pos"] = float(joint_position[i])
            obs_dict[f"joint_{i+1}.vel"] = float(joint_velocity[i])

        for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
            obs_dict[f"ee_pose.{axis}"] = float(ee_pose[i])
  
        # for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
        #     obs_dict[f"ee_vel.{axis}"] = float(ee_speed[i])

        if self.config.use_gripper:

            obs_dict["gripper_state_norm"] = self._gripper_position
            obs_dict["gripper_cmd_bin"] = self._last_gripper_position
        else:
            obs_dict["gripper_state_norm"] = None
            obs_dict["gripper_cmd_bin"] = None

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        self._prev_observation = obs_dict

        return obs_dict

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        for cam in self.cameras.values():
            cam.disconnect()

        self.is_connected = False
        logger.info(f"[INFO] ===== All {self.name} connections have been closed =====")

    def calibrate(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        return self.is_connected
    
    def configure(self) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = value

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
           cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def cameras(self):
        return self._cameras

    @cameras.setter
    def cameras(self, value):
        self._cameras = value

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

if __name__ == "__main__":
    import numpy as np
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    class RecordConfig:
        def __init__(self, cfg: Dict[str, Any]):
            robot = cfg["robot"]
            cam = cfg["cameras"]
            self.fps: str = cfg.get("fps", 15)

            # robot config
            self.robot_ip = robot["ip"]
            self.use_gripper = robot["use_gripper"]
            self.close_threshold = robot["close_threshold"]
            self.gripper_bin_threshold = robot["gripper_bin_threshold"]
            self.gripper_reverse = robot["gripper_reverse"]
            self.control_mode = robot["control_mode"]


            # cameras config
            self.wrist_cam_serial: str = cam["wrist_cam_serial"]
            self.exterior_cam_serial: str = cam["exterior_cam_serial"]
            self.width: int = cam["width"]
            self.height: int = cam["height"]


    with open(Path(__file__).parent / "config" / "cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)


    record_cfg = RecordConfig(cfg["record"])

    # Create RealSenseCamera configurations
    wrist_image_cfg = RealSenseCameraConfig(serial_number_or_name=record_cfg.wrist_cam_serial,
                                    fps=record_cfg.fps,
                                    width=record_cfg.width,
                                    height=record_cfg.height,
                                    color_mode=ColorMode.RGB,
                                    use_depth=False,
                                    rotation=Cv2Rotation.NO_ROTATION)

    exterior_image_cfg = RealSenseCameraConfig(serial_number_or_name=record_cfg.exterior_cam_serial,
                                    fps=record_cfg.fps,
                                    width=record_cfg.width,
                                    height=record_cfg.height,
                                    color_mode=ColorMode.RGB,
                                    use_depth=False,
                                    rotation=Cv2Rotation.NO_ROTATION)

    # Create the robot and teleoperator configurations
    camera_config = {"wrist_image": wrist_image_cfg, "exterior_image": exterior_image_cfg}

    robot_config = FrankaConfig(
            robot_ip=record_cfg.robot_ip,
            cameras = camera_config,
            debug = False,
            close_threshold = record_cfg.close_threshold,
            use_gripper = record_cfg.use_gripper,
            gripper_reverse = record_cfg.gripper_reverse,
            gripper_bin_threshold = record_cfg.gripper_bin_threshold,
            control_mode = record_cfg.control_mode
        )
    franka = Franka(robot_config)
    franka.connect()