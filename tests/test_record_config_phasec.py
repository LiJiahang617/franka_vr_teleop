"""Task 3: RecordConfig 扩展全量数采超参 + §11.3 增益键 TDD 测试。

测试策略：
- run_record.py 顶部有大量 lerobot/franka 重依赖，importlib 直接加载会爆
- 改用 monkeypatch sys.modules 注入最小假包后 importlib 加载 run_record
- 构造 minimal cfg dict，断言 RecordConfig 新字段值 == 期望/默认
- 覆盖：out_dir/depth_enabled/state_hifreq/reset_*/color_preflight/pos_rot_axis_gain 解析
  + 缺键全回退默认（向后兼容）
  + unityvr 分支 pos_axis_gain/rot_axis_gain 透传到 create_teleop_config()
"""
import importlib.util
import os
import sys
import types
from pathlib import Path as _Path

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RR_PATH = os.path.join(_REPO, "scripts/core/record_config.py")


def _fake_module(name, **attrs):
    """构造带指定属性的假模块。"""
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class _FakeACTConfig:
    """最小假 ACTConfig，接受 device/push_to_hub 参数。"""
    def __init__(self, device=None, push_to_hub=False):
        self.device = device
        self.push_to_hub = push_to_hub


class _FakeDiffusionConfig:
    """最小假 DiffusionConfig。"""
    def __init__(self, device=None, push_to_hub=False):
        self.device = device
        self.push_to_hub = push_to_hub


def _make_fake_unityvr_teleop_config(**extra_defaults):
    """返回可记录 pos/rot_axis_gain 的 FakeUnityVRTeleopConfig 类。"""
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
        trigger_threshold: float = 0.85
        smoothing_alpha: float = 0.4

    return FakeUnityVRTeleopConfig


# ================================================================
# 共享：注入假模块 + 加载 run_record 模块
# ================================================================
def _load_run_record(fake_unityvr_cls=None):
    """注入假依赖后 importlib 加载 run_record.py，返回模块。

    每次调用前先清理上次注册的同名模块，保证测试独立。
    """
    if fake_unityvr_cls is None:
        fake_unityvr_cls = _make_fake_unityvr_teleop_config()

    ltf = _fake_module("lerobot_teleoperator_franka")
    ltf.DynamixelTeleopConfig = object
    ltf.SpacemouseTeleopConfig = object
    ltf.OculusTeleopConfig = object
    ltf.UnityVRTeleopConfig = fake_unityvr_cls
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
            DiffusionConfig=_FakeDiffusionConfig),
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

    # 每次重新加载（新对象），避免跨测试缓存
    saved = {k: sys.modules.pop(k, None) for k in ["record_config"]}
    sys.modules.update(mocks)
    try:
        spec = importlib.util.spec_from_file_location("record_config", _RR_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["record_config"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        # 清理注入（保持测试隔离）
        for k in mocks:
            sys.modules.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


def _minimal_cfg(control_mode="unityvr", uvr_overrides=None, rec_overrides=None):
    """构造最小合法 cfg dict（unityvr 模式）。"""
    uvr_cfg = {
        "use_gripper": True,
        "pose_scaler": [3.0, 2.0],
        "channel_signs": [1, 1, 1, -1, -1, 1],
        "oc2base_path": "/tmp/fake.npy",
        "robot_ip": "127.0.0.1",
        "robot_port": 4242,
    }
    if uvr_overrides:
        uvr_cfg.update(uvr_overrides)

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
            "control_mode": control_mode,
            "unityvr_config": uvr_cfg,
        },
    }
    if rec_overrides:
        cfg.update(rec_overrides)
    return cfg


# ================================================================
# Step 1 失败测试：新字段解析
# ================================================================

def test_record_config_out_dir_default_none():
    """out_dir 缺省 → None（不新增 required 键，向后兼容）。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg())
    # 新字段还不存在时测试 FAIL；实现后测试 PASS
    assert hasattr(rc, "out_dir"), "RecordConfig 应有 out_dir 属性"
    assert rc.out_dir is None, f"out_dir 缺省应为 None，实际 {rc.out_dir!r}"


def test_record_config_out_dir_from_cfg():
    """out_dir 在 cfg 中指定 → 正确解析。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg(rec_overrides={"out_dir": "/data/episodes"}))
    assert rc.out_dir == "/data/episodes", f"实际 {rc.out_dir!r}"


def test_record_config_depth_enabled_default_false():
    """depth.enabled 缺省 → False。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg())
    assert hasattr(rc, "depth_enabled"), "RecordConfig 应有 depth_enabled 属性"
    assert rc.depth_enabled is False, f"实际 {rc.depth_enabled!r}"


def test_record_config_depth_enabled_from_cfg():
    """depth.enabled=true → True。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg(rec_overrides={"depth": {"enabled": True}}))
    assert rc.depth_enabled is True


def test_record_config_state_hifreq_defaults():
    """state_hifreq 缺省 → enabled=False, rate=240。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg())
    assert hasattr(rc, "state_hifreq_enabled"), "RecordConfig 应有 state_hifreq_enabled"
    assert hasattr(rc, "state_hifreq_rate"), "RecordConfig 应有 state_hifreq_rate"
    assert rc.state_hifreq_enabled is False
    assert rc.state_hifreq_rate == 240


def test_record_config_state_hifreq_from_cfg():
    """state_hifreq 在 cfg 中指定 → 正确解析。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg(rec_overrides={
        "state_hifreq": {"enabled": True, "rate": 500}
    }))
    assert rc.state_hifreq_enabled is True
    assert rc.state_hifreq_rate == 500


def test_record_config_reset_between_episodes_from_cfg():
    """reset_between_episodes 从 cfg 经 parse_reset_config 解析 → True/False 正确。"""
    m = _load_run_record()
    # 默认（yaml 有 reset_between_episodes: true）
    rc = m.RecordConfig(_minimal_cfg(rec_overrides={"reset_between_episodes": True}))
    assert hasattr(rc, "reset_between_episodes"), "RecordConfig 应有 reset_between_episodes"
    assert rc.reset_between_episodes is True
    # 字符串 false（parse_reset_config 核心测试：防 bool('false')==True）
    rc2 = m.RecordConfig(_minimal_cfg(rec_overrides={"reset_between_episodes": "false"}))
    assert rc2.reset_between_episodes is False


def test_record_config_reset_wait_default():
    """reset_wait 缺省 → 1.0。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg())
    assert hasattr(rc, "reset_wait"), "RecordConfig 应有 reset_wait"
    assert rc.reset_wait == 1.0


def test_record_config_color_preflight_default_true():
    """color_preflight 缺省 → True。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg())
    assert hasattr(rc, "color_preflight"), "RecordConfig 应有 color_preflight"
    assert rc.color_preflight is True


def test_record_config_color_preflight_from_cfg():
    """color_preflight=false → False。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg(rec_overrides={"color_preflight": False}))
    assert rc.color_preflight is False


def test_record_config_unityvr_pos_axis_gain_default():
    """unityvr 分支 pos_axis_gain 缺省 → [1.0, 1.0, 1.0]。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg())
    assert hasattr(rc, "pos_axis_gain"), "unityvr RecordConfig 应有 pos_axis_gain"
    assert list(rc.pos_axis_gain) == [1.0, 1.0, 1.0], f"实际 {rc.pos_axis_gain}"


def test_record_config_unityvr_rot_axis_gain_default():
    """unityvr 分支 rot_axis_gain 缺省 → [1.0, 1.0, 1.0]。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg())
    assert hasattr(rc, "rot_axis_gain"), "unityvr RecordConfig 应有 rot_axis_gain"
    assert list(rc.rot_axis_gain) == [1.0, 1.0, 1.0], f"实际 {rc.rot_axis_gain}"


def test_record_config_unityvr_pos_axis_gain_from_cfg():
    """unityvr_config.pos_axis_gain 指定 → 正确解析。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg(uvr_overrides={"pos_axis_gain": [2.0, 3.0, 4.0]}))
    assert list(rc.pos_axis_gain) == [2.0, 3.0, 4.0], f"实际 {rc.pos_axis_gain}"


def test_record_config_unityvr_rot_axis_gain_from_cfg():
    """unityvr_config.rot_axis_gain 指定 → 正确解析。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg(uvr_overrides={"rot_axis_gain": [0.5, 1.5, 2.5]}))
    assert list(rc.rot_axis_gain) == [0.5, 1.5, 2.5], f"实际 {rc.rot_axis_gain}"


def test_record_config_create_teleop_config_passes_axis_gain():
    """create_teleop_config() unityvr 分支透传 pos/rot_axis_gain 到 UnityVRTeleopConfig。"""
    # 构造记录初始化参数的 FakeUnityVRTeleopConfig
    captured = {}

    from dataclasses import dataclass, field as dc_field

    @dataclass
    class TrackingUnityVRTeleopConfig:
        use_gripper: bool = True
        pose_scaler: list = dc_field(default_factory=lambda: [1.0, 1.0])
        channel_signs: list = dc_field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
        oc2base_path: str = ""
        robot_ip: str = "127.0.0.1"
        robot_port: int = 4242
        pos_axis_gain: list = dc_field(default_factory=lambda: [1.0, 1.0, 1.0])
        rot_axis_gain: list = dc_field(default_factory=lambda: [1.0, 1.0, 1.0])
        trigger_threshold: float = 0.85
        smoothing_alpha: float = 0.4

        def __post_init__(self):
            captured["pos_axis_gain"] = list(self.pos_axis_gain)
            captured["rot_axis_gain"] = list(self.rot_axis_gain)

    m = _load_run_record(fake_unityvr_cls=TrackingUnityVRTeleopConfig)
    rc = m.RecordConfig(_minimal_cfg(uvr_overrides={
        "pos_axis_gain": [2.0, 3.0, 4.0],
        "rot_axis_gain": [5.0, 6.0, 7.0],
    }))
    rc.create_teleop_config()
    assert captured.get("pos_axis_gain") == [2.0, 3.0, 4.0], (
        f"create_teleop_config 未透传 pos_axis_gain，实际 {captured.get('pos_axis_gain')}")
    assert captured.get("rot_axis_gain") == [5.0, 6.0, 7.0], (
        f"create_teleop_config 未透传 rot_axis_gain，实际 {captured.get('rot_axis_gain')}")


def test_record_config_backward_compatible_missing_keys():
    """向后兼容：cfg 不含任何新键时 RecordConfig 不报错，全取安全默认。"""
    m = _load_run_record()
    # 使用不含新键的 cfg（模拟旧 record_cfg.yaml）
    rc = m.RecordConfig(_minimal_cfg())
    # 不抛异常即成功；额外断言几个关键默认
    assert rc.out_dir is None
    assert rc.depth_enabled is False
    assert rc.state_hifreq_enabled is False
    assert rc.color_preflight is True
    assert list(rc.pos_axis_gain) == [1.0, 1.0, 1.0]
    assert list(rc.rot_axis_gain) == [1.0, 1.0, 1.0]


def test_record_config_pose_scaler_still_default():
    """新字段不影响既有 pose_scaler 解析（向后兼容守门）。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg())
    assert list(rc.pose_scaler) == [3.0, 2.0], f"pose_scaler 被破坏，实际 {rc.pose_scaler}"


# ================================================================
# Phase C review-fix 新增边界测试（验证 helpers 在 RecordConfig 层生效）
# ================================================================

def test_depth_null_not_attribute_error():
    """cfg depth: None → parse_section_dict 转空 dict，不 AttributeError，depth_enabled=False。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg(rec_overrides={"depth": None}))
    assert rc.depth_enabled is False


def test_depth_non_dict_raises():
    """cfg depth: "bad" → parse_section_dict → ValueError（fail-loud）。"""
    m = _load_run_record()
    with pytest.raises(ValueError):
        m.RecordConfig(_minimal_cfg(rec_overrides={"depth": "bad"}))


def test_state_hifreq_null_safe():
    """cfg state_hifreq: None → parse_section_dict 转空 dict，取默认 enabled=False, rate=240。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg(rec_overrides={"state_hifreq": None}))
    assert rc.state_hifreq_enabled is False
    assert rc.state_hifreq_rate == 240


def test_state_hifreq_rate_zero_raises():
    """state_hifreq.rate: 0 → parse_positive_int → ValueError（fail-loud）。"""
    m = _load_run_record()
    with pytest.raises(ValueError, match="必须 > 0"):
        m.RecordConfig(_minimal_cfg(rec_overrides={"state_hifreq": {"enabled": True, "rate": 0}}))


def test_state_hifreq_rate_negative_raises():
    """state_hifreq.rate: -1 → parse_positive_int → ValueError（fail-loud）。"""
    m = _load_run_record()
    with pytest.raises(ValueError, match="必须 > 0"):
        m.RecordConfig(_minimal_cfg(rec_overrides={"state_hifreq": {"enabled": True, "rate": -1}}))


def test_color_preflight_string_false_correctly_false():
    """color_preflight: "false"（yaml 引号字符串）→ parse_bool → False，防 bool("false")==True 误判。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg(rec_overrides={"color_preflight": "false"}))
    assert rc.color_preflight is False, (
        f'color_preflight="false" 应解析为 False，实际 {rc.color_preflight!r}（parse_bool 未生效？）'
    )


def test_pos_axis_gain_wrong_length_raises_at_config():
    """pos_axis_gain len!=3 → config-load 时即 ValueError（非延后到运行时）。"""
    m = _load_run_record()
    with pytest.raises(ValueError, match="len==3"):
        m.RecordConfig(_minimal_cfg(uvr_overrides={"pos_axis_gain": [1.0, 2.0]}))


def test_pos_axis_gain_nan_raises_at_config():
    """pos_axis_gain 含 nan → config-load 时即 ValueError（有限性检查）。"""
    m = _load_run_record()
    with pytest.raises(ValueError, match="有限"):
        m.RecordConfig(_minimal_cfg(uvr_overrides={"pos_axis_gain": [1.0, float("nan"), 1.0]}))


# ================================================================
# controller_preflight 配置解析测试
# ================================================================

def test_record_config_controller_preflight_defaults():
    """yaml 缺 controller_preflight 段 → 默认 enabled=True + 默认路径。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg())
    assert hasattr(rc, "controller_preflight_enabled"), "RecordConfig 应有 controller_preflight_enabled"
    assert rc.controller_preflight_enabled is True, f"默认应为 True，实际 {rc.controller_preflight_enabled!r}"
    assert hasattr(rc, "controller_preflight_python"), "RecordConfig 应有 controller_preflight_python"
    # 兼容 expandvars 后 (含 'polymetis' 或 expansion 失败时含 '${POLYMETIS_ENV}')
    assert ("polymetis" in rc.controller_preflight_python or
            "POLYMETIS_ENV" in rc.controller_preflight_python), (
        f"默认 python 路径应含 polymetis，实际 {rc.controller_preflight_python!r}"
    )
    assert hasattr(rc, "controller_preflight_conda_prefix"), "RecordConfig 应有 controller_preflight_conda_prefix"


def test_record_config_controller_preflight_disabled():
    """yaml enabled=false → cfg.controller_preflight_enabled is False。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg(rec_overrides={
        "controller_preflight": {"enabled": False}
    }))
    assert rc.controller_preflight_enabled is False, (
        f"enabled=false 应解析为 False，实际 {rc.controller_preflight_enabled!r}"
    )


def test_record_config_controller_preflight_custom_paths():
    """yaml 自定义路径 → cfg 字段反映。"""
    m = _load_run_record()
    rc = m.RecordConfig(_minimal_cfg(rec_overrides={
        "controller_preflight": {
            "polymetis_python": "/opt/custom/venv/bin/python",
            "polymetis_conda_prefix": "/opt/custom/venv",
        }
    }))
    assert rc.controller_preflight_python == "/opt/custom/venv/bin/python", (
        f"实际 {rc.controller_preflight_python!r}"
    )
    assert rc.controller_preflight_conda_prefix == "/opt/custom/venv", (
        f"实际 {rc.controller_preflight_conda_prefix!r}"
    )
