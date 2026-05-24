"""UnityVRRobot：世界系 Quest 控制器 → delta_ee_pose+夹爪 观测，复用 vr_align 标定 R。"""
import logging

import numpy as np

# §11.1: vr_align/unity_vr_reader 已入本包，包内相对导入，无需 sys.path hack
from . import vr_align
from .unity_vr_reader import UnityVRReader

from . import unityvr_mapping as _m

logger = logging.getLogger(__name__)
DOF = 7


class UnityVRRobot:
    def __init__(self, oc2base_path="/home/ubuntu/Desktop/jhli/franka_vr_teleop/.stage3_oc2arm_R.npy",
                 pose_scaler=(1.0, 1.0), channel_signs=(1, 1, 1, 1, 1, 1),
                 use_gripper=True, robot_ip="127.0.0.1", robot_port=4242,
                 pos_axis_gain=(1., 1., 1.), rot_axis_gain=(1., 1., 1.),
                 trigger_threshold=0.85):
        """初始化 UnityVRRobot。

        Args:
            oc2base_path: vr_align 标定 R 文件路径。
            pose_scaler: [pos_scale, ori_scale] 全局标量增益。
            channel_signs: 长 6 的 ±1，各轴方向符号。
            use_gripper: 是否启用夹爪切换逻辑。
            robot_ip: Franka zerorpc 服务 IP。
            robot_port: Franka zerorpc 服务端口。
            pos_axis_gain: §11.3 每轴位置增益 [gx, gy, gz]，默认 (1,1,1)=历史行为。
                仅在已验通映射方向输出后逐轴缩放，不改方向/手性（§10.2(0) 红线）。
            rot_axis_gain: §11.3 每轴旋转增益 [grx, gry, grz]，默认 (1,1,1)=历史行为。
        """
        loaded = vr_align.load_rotation(oc2base_path)
        if loaded is None:
            raise RuntimeError(
                f"未找到标定 R: {oc2base_path}。先跑 2 手势 SVD 标定生成它"
                f"（stage3_teleop.py --vr-source unity 按 A 两手势），再录制。")
        self._R = np.asarray(loaded[0], float)
        self._meta = loaded[1]
        self._reader = UnityVRReader()
        self._pose_scaler = list(pose_scaler)
        self._channel_signs = list(channel_signs)
        self._pos_axis_gain = list(pos_axis_gain)    # §11.3 per-axis 位置增益
        self._rot_axis_gain = list(rot_axis_gain)    # §11.3 per-axis 旋转增益
        self._trigger_threshold = float(trigger_threshold)  # VR 食指扳机激活阈值 [0..1]
        self._use_gripper = use_gripper
        self._prev_T = None
        self._gripper_closed = False
        self._grip_prev = False
        # 测量关节用（dataset 格式一致性；MVP 笛卡尔 execute, 不做 IK）
        from lerobot_robot_franka.franka_interface_client import FrankaInterfaceClient
        self._client = FrankaInterfaceClient(ip=robot_ip, port=robot_port)
        logger.info(f"[UnityVRRobot] R 已载 (quality={self._meta.get('quality')})")

    def _measured_joints(self):
        try:
            j = np.asarray(self._client.robot_get_joint_positions(), float).reshape(-1)
            if j.shape[0] >= DOF:
                return j[:DOF]
        except Exception as e:
            logger.warning(f"[UnityVRRobot] 读关节失败: {e}")
        return np.zeros(DOF)

    def get_observations(self):
        """返回 dict，键与 UnityVRTeleop.action_features 完全一致。"""
        transforms, buttons = self._reader.get_transformations_and_buttons()
        enabled = _m.is_enabled(buttons, th=self._trigger_threshold)
        delta = np.zeros(6)
        if enabled and ("r" in transforms):
            cur_T = transforms["r"]
            if self._prev_T is not None:
                delta = _m.compute_delta_action(
                    cur_T, self._prev_T, self._R,
                    self._pose_scaler, self._channel_signs,
                    pos_axis_gain=self._pos_axis_gain,   # §11.3 per-axis 位置增益透传
                    rot_axis_gain=self._rot_axis_gain)   # §11.3 per-axis 旋转增益透传
            self._prev_T = np.asarray(cur_T, float).copy()
        else:
            self._prev_T = None  # 松开/无效→丢锚, 防跳变

        grip_now = bool(buttons.get("RG", False))
        if self._use_gripper:
            self._gripper_closed = _m.next_gripper_closed(
                self._gripper_closed, self._grip_prev, grip_now)
        self._grip_prev = grip_now
        gripper = 1.0 if self._gripper_closed else 0.0

        joints = self._measured_joints()
        obs = {}
        for i, ax in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
            obs[f"delta_ee_pose.{ax}"] = float(delta[i])
        for i in range(DOF):
            obs[f"joint_{i+1}.pos"] = float(joints[i])
        obs["gripper_cmd_bin"] = float(gripper)
        return obs
