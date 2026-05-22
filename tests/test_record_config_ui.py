"""Task 6: RecordConfig.ui_config strict-parse 测试（TDD Phase E）。

与 test_record_config_phasec.py 同一 monkeypatch 范式：注入假模块后
importlib 加载 run_record.py，避免 lerobot/franka 硬件依赖。

验证:
  1. 无 ui 段时 ui_config 返回全默认值（向后兼容）
  2. 有 ui 段时字段被正确解析（覆盖默认）
  3. port 为非整数字符串时 config-load 阶段 fail-loud
  4. 读取真实 yaml 文件后 ui_config 七个字段齐全
"""
import importlib.util
import os
import sys
import types
from pathlib import Path as _Path

import pytest
import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RR_PATH = os.path.join(_REPO, "scripts/core/record_config.py")


def _fake_module(name, **attrs):
    """构造带指定属性的假模块（复用 Phase C 范式）。"""
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class _FakeACTConfig:
    """最小假 ACTConfig。"""
    def __init__(self, device=None, push_to_hub=False):
        self.device = device
        self.push_to_hub = push_to_hub


def _load_run_record():
    """注入假依赖后 importlib 加载 run_record.py，返回模块（Phase C 同款）。"""
    from dataclasses import dataclass, field as dc_field

    @dataclass
    class FakeUnityVRTeleopConfig:
        use_gripper: bool = True
        pose_scaler: list = dc_field(default_factory=lambda: [1.0, 1.0])
        channel_signs: list = dc_field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
        oc2base_path: str = ""
        robot_ip: str = "127.0.0.1"
        robot_port: int = 4242
        pos_axis_gain: list = dc_field(default_factory=lambda: [1.0, 1.0, 1.0])
        rot_axis_gain: list = dc_field(default_factory=lambda: [1.0, 1.0, 1.0])

    ltf = _fake_module("lerobot_teleoperator_franka")
    ltf.DynamixelTeleopConfig = object
    ltf.SpacemouseTeleopConfig = object
    ltf.OculusTeleopConfig = object
    ltf.UnityVRTeleopConfig = FakeUnityVRTeleopConfig
    ltf.create_teleop = None
    ltf.create_teleop_config = None

    mocks = {
        "lerobot_robot_franka": _fake_module(
            "lerobot_robot_franka", FrankaConfig=object, Franka=object),
        "lerobot_teleoperator_franka": ltf,
        "lerobot": _fake_module("lerobot"),
        "lerobot.cameras": _fake_module("lerobot.cameras"),
        "lerobot.cameras.configs": _fake_module(
            "lerobot.cameras.configs", ColorMode=object, Cv2Rotation=object),
        "lerobot.cameras.realsense": _fake_module("lerobot.cameras.realsense"),
        "lerobot.cameras.realsense.camera_realsense": _fake_module(
            "lerobot.cameras.realsense.camera_realsense", RealSenseCameraConfig=object),
        "lerobot.scripts": _fake_module("lerobot.scripts"),
        "lerobot.scripts.lerobot_record": _fake_module(
            "lerobot.scripts.lerobot_record", record_loop=None),
        "lerobot.processor": _fake_module(
            "lerobot.processor", make_default_processors=None),
        "lerobot.utils": _fake_module("lerobot.utils"),
        "lerobot.utils.visualization_utils": _fake_module(
            "lerobot.utils.visualization_utils", init_rerun=None),
        "lerobot.utils.control_utils": _fake_module(
            "lerobot.utils.control_utils",
            init_keyboard_listener=None,
            sanity_check_dataset_robot_compatibility=None),
        "lerobot.utils.constants": _fake_module(
            "lerobot.utils.constants", HF_LEROBOT_HOME=_Path("/tmp/fake")),
        "lerobot.datasets": _fake_module("lerobot.datasets"),
        "lerobot.datasets.lerobot_dataset": _fake_module(
            "lerobot.datasets.lerobot_dataset", LeRobotDataset=object),
        "lerobot.datasets.utils": _fake_module(
            "lerobot.datasets.utils", hw_to_dataset_features=None),
        "lerobot.configs": _fake_module("lerobot.configs"),
        "lerobot.configs.policies": _fake_module(
            "lerobot.configs.policies", PreTrainedConfig=object),
        "lerobot.policies": _fake_module(
            "lerobot.policies",
            ACTConfig=_FakeACTConfig,
            DiffusionConfig=_FakeACTConfig),
        "lerobot.policies.factory": _fake_module(
            "lerobot.policies.factory", make_policy=None, make_pre_post_processors=None),
        "lerobot.processor.rename_processor": _fake_module(
            "lerobot.processor.rename_processor", rename_stats=None),
        "send2trash": _fake_module("send2trash", send2trash=None),
        "scripts": _fake_module("scripts"),
        "scripts.utils": _fake_module("scripts.utils"),
        "scripts.utils.dataset_utils": _fake_module(
            "scripts.utils.dataset_utils",
            generate_dataset_name=None, update_dataset_info=None),
        "scripts.utils.teleop_joint_offsets": _fake_module(
            "scripts.utils.teleop_joint_offsets",
            get_start_joints=None, compute_joint_offsets=None),
    }

    scripts_dir = os.path.join(_REPO, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    saved = {k: sys.modules.pop(k, None) for k in ["record_config"]}
    sys.modules.update(mocks)
    try:
        spec = importlib.util.spec_from_file_location("record_config", _RR_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["record_config"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k in mocks:
            sys.modules.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


def _minimal_cfg(rec_overrides=None):
    """构造最小合法 unityvr cfg dict（不含 ui 段，向后兼容基准）。"""
    cfg = {
        "repo_id": "test/test",
        "debug": True,
        "fps": 30,
        "run_mode": "run_record",
        "storage": {"push_to_hub": False},
        "task": {"description": "test task", "num_episodes": 5},
        "time": {"episode_time_sec": 60, "reset_time_sec": 10, "save_meta_period": 1},
        "cameras": {
            "wrist_cam_serial": "abc", "exterior_cam_serial": "def",
            "width": 424, "height": 240,
        },
        "robot": {
            "ip": "127.0.0.1", "use_gripper": True, "close_threshold": 0.02,
            "gripper_reverse": False, "gripper_bin_threshold": 0.5,
        },
        "policy": {"type": "act", "device": "cpu", "push_to_hub": False},
        "teleop": {
            "control_mode": "unityvr",
            "unityvr_config": {
                "use_gripper": True,
                "pose_scaler": [3.0, 2.0],
                "channel_signs": [1, 1, 1, -1, -1, 1],
                "oc2base_path": "/tmp/fake.npy",
                "robot_ip": "127.0.0.1",
                "robot_port": 4242,
            },
        },
    }
    if rec_overrides:
        cfg.update(rec_overrides)
    return cfg


# ================================================================
# Task 6 核心测试
# ================================================================

def test_ui_section_defaults_when_absent():
    """无 ui 段时 ui_config 返回全默认值，不报错（向后兼容旧 yaml）。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg())
    ui = rc.ui_config
    assert ui["enabled"] is False
    assert ui["host"] == "0.0.0.0"
    assert ui["port"] == 5055
    assert ui["preview_max_w"] == 320
    assert ui["preview_max_h"] == 240
    assert ui["preview_quality"] == 60
    assert ui["status_poll_hz"] == 30


def test_ui_section_parsed_strict_when_present():
    """有 ui 段时字段被正确解析（覆盖默认）。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg(rec_overrides={
        "ui": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 6060,
            "preview_max_w": 160,
            "preview_max_h": 120,
            "preview_quality": 75,
            "status_poll_hz": 15,
        }
    }))
    ui = rc.ui_config
    assert ui["enabled"] is True
    assert ui["host"] == "127.0.0.1"
    assert ui["port"] == 6060
    assert ui["preview_max_w"] == 160
    assert ui["preview_max_h"] == 120
    assert ui["preview_quality"] == 75
    assert ui["status_poll_hz"] == 15


def test_ui_section_strict_fails_loud_on_bad_type():
    """port 为非整数字符串时 config-load 阶段 fail-loud（ValueError/TypeError/KeyError 均可）。"""
    m = _load_run_record()
    with pytest.raises((ValueError, TypeError, KeyError)):
        m.RecordConfig(_minimal_cfg(rec_overrides={"ui": {"port": "not_an_int"}}))


def test_existing_yaml_loads_with_ui_default():
    """读取真实 yaml 文件后 ui_config 七个字段齐全（yaml 有 ui 段走解析，无则走默认）。"""
    m = _load_run_record()
    yaml_path = os.path.join(_REPO, "scripts/config/record_cfg_unityvr.yaml")
    with open(yaml_path) as fh:
        raw = yaml.safe_load(fh)
    rc = m.RecordConfig(raw["record"])
    ui = rc.ui_config
    # 无论 yaml 是否有 ui 段，七个字段必须存在且类型正确
    for k in ("enabled", "host", "port", "preview_max_w", "preview_max_h",
              "preview_quality", "status_poll_hz"):
        assert k in ui, f"ui_config 缺少字段: {k}"
    # 类型断言
    assert isinstance(ui["enabled"], bool), f"enabled 应为 bool，实际 {type(ui['enabled'])}"
    assert isinstance(ui["host"], str), f"host 应为 str，实际 {type(ui['host'])}"
    assert isinstance(ui["port"], int), f"port 应为 int，实际 {type(ui['port'])}"
    assert isinstance(ui["preview_max_w"], int), f"preview_max_w 应为 int"
    assert isinstance(ui["preview_max_h"], int), f"preview_max_h 应为 int"
    assert isinstance(ui["preview_quality"], int), f"preview_quality 应为 int"
    assert isinstance(ui["status_poll_hz"], int), f"status_poll_hz 应为 int"
