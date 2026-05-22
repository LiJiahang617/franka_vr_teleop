"""Task 5: record_cfg_unityvr.yaml 补全验证——锁定"仅改 yaml 即生效"。

测试策略：
1. yaml.safe_load 真 record_cfg_unityvr.yaml，断言含新键
2. RecordConfig(raw["record"]) 解析后字段 == yaml 值
3. resolve_record_overrides(CLI 全 None) 取 yaml 的 episodes/episode_sec/out_dir/task
   → 验证"仅改 yaml 即生效"（构造 yaml 副本改值，断言 overrides 跟随）

红线：
- yaml 键名严格与 RecordConfig.get 键一致（避免拼写不符既有 bug 再发）
- 仅 yaml + 新测试两文件改动，不改 RecordConfig/run_record_hdf5/unityvr_mapping/schema
- run_record.py 走 record_cfg.yaml（另一入口），不受影响
"""
import importlib.util
import os
import sys
import types
from pathlib import Path as _Path

import pytest
import yaml

# ================================================================
# 路径和文件定位
# ================================================================
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_YAML_PATH = os.path.join(_REPO, "scripts/config/record_cfg_unityvr.yaml")
_RR_PATH = os.path.join(_REPO, "scripts/core/record_config.py")
_RP_PATH = os.path.join(_REPO, "scripts/core/record_params.py")

# ================================================================
# 辅助：加载 record_params（无 lerobot 依赖，直接 importlib）
# ================================================================
def _load_record_params():
    scripts_dir = os.path.join(_REPO, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    # 若 core.record_params 已在 sys.modules 直接取
    if "core.record_params" in sys.modules:
        return sys.modules["core.record_params"]
    spec = importlib.util.spec_from_file_location("core.record_params", _RP_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["core.record_params"] = mod
    spec.loader.exec_module(mod)
    return mod


# ================================================================
# 辅助：注入假依赖后 importlib 加载 run_record（复用 test_record_config_phasec 范式）
# ================================================================
def _fake_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class _FakeACTConfig:
    def __init__(self, device=None, push_to_hub=False):
        self.device = device
        self.push_to_hub = push_to_hub


class _FakeDiffusionConfig:
    def __init__(self, device=None, push_to_hub=False):
        self.device = device
        self.push_to_hub = push_to_hub


def _make_fake_unityvr_teleop_config():
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

    return FakeUnityVRTeleopConfig


def _load_run_record():
    """注入假依赖后加载 run_record.py，每次 fresh 加载。"""
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


# ================================================================
# 共享：读取真 yaml
# ================================================================
def _load_yaml():
    """读取 record_cfg_unityvr.yaml，返回 raw dict。"""
    with open(_YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ================================================================
# Step 1: 断言 yaml 含新键（FAIL 前 yaml 无这些键）
# ================================================================

class TestYamlContainsNewKeys:
    """断言 yaml 含 Phase C + §11.3 引入的全部新键。"""

    def test_yaml_has_out_dir(self):
        """yaml record.out_dir 键存在。"""
        raw = _load_yaml()
        assert "out_dir" in raw["record"], (
            "record_cfg_unityvr.yaml record 段缺 out_dir 键（Task5 Step3 补全前失败）"
        )

    def test_yaml_out_dir_value(self):
        """yaml record.out_dir 值符合规范（非 null）。"""
        raw = _load_yaml()
        # 期望有明确路径而非 null
        assert raw["record"]["out_dir"] is not None, (
            "out_dir 值不应为 null；应为实际 hdf5 输出目录路径"
        )

    def test_yaml_has_depth_section(self):
        """yaml record.depth 段存在（Phase D 占位）。"""
        raw = _load_yaml()
        assert "depth" in raw["record"], "record 段缺 depth 占位键"
        assert "enabled" in raw["record"]["depth"], "record.depth 缺 enabled 键"

    def test_yaml_depth_enabled_false(self):
        """yaml record.depth.enabled 默认 false（Phase D 前不采集）。"""
        raw = _load_yaml()
        assert raw["record"]["depth"]["enabled"] is False, (
            f"depth.enabled 应为 false，实际 {raw['record']['depth']['enabled']!r}"
        )

    def test_yaml_has_state_hifreq_section(self):
        """yaml record.state_hifreq 段存在（Phase D 占位）。"""
        raw = _load_yaml()
        assert "state_hifreq" in raw["record"], "record 段缺 state_hifreq 占位键"
        assert "enabled" in raw["record"]["state_hifreq"], "record.state_hifreq 缺 enabled 键"
        assert "rate" in raw["record"]["state_hifreq"], "record.state_hifreq 缺 rate 键"

    def test_yaml_state_hifreq_enabled_false(self):
        """yaml record.state_hifreq.enabled 默认 false。"""
        raw = _load_yaml()
        assert raw["record"]["state_hifreq"]["enabled"] is False, (
            f"state_hifreq.enabled 应为 false，实际 {raw['record']['state_hifreq']['enabled']!r}"
        )

    def test_yaml_state_hifreq_rate_240(self):
        """yaml record.state_hifreq.rate 默认 240。"""
        raw = _load_yaml()
        assert raw["record"]["state_hifreq"]["rate"] == 240, (
            f"state_hifreq.rate 应为 240，实际 {raw['record']['state_hifreq']['rate']!r}"
        )

    def test_yaml_has_color_preflight(self):
        """yaml record.color_preflight 键存在。"""
        raw = _load_yaml()
        assert "color_preflight" in raw["record"], "record 段缺 color_preflight 键"

    def test_yaml_color_preflight_true(self):
        """yaml record.color_preflight 默认 true（§11.2 色彩预检启用）。"""
        raw = _load_yaml()
        assert raw["record"]["color_preflight"] is True, (
            f"color_preflight 应为 true，实际 {raw['record']['color_preflight']!r}"
        )

    def test_yaml_has_pos_axis_gain(self):
        """yaml record.teleop.unityvr_config.pos_axis_gain 键存在（§11.3）。"""
        raw = _load_yaml()
        uvr_cfg = raw["record"]["teleop"]["unityvr_config"]
        assert "pos_axis_gain" in uvr_cfg, (
            "unityvr_config 缺 pos_axis_gain 键（Task5 Step3 补全前失败）"
        )

    def test_yaml_pos_axis_gain_default_unit(self):
        """yaml pos_axis_gain 默认 [1.0, 1.0, 1.0]（等价历史 pose_scaler 行为）。"""
        raw = _load_yaml()
        gain = raw["record"]["teleop"]["unityvr_config"]["pos_axis_gain"]
        assert list(gain) == [1.0, 1.0, 1.0], (
            f"pos_axis_gain 默认应为 [1.0, 1.0, 1.0]，实际 {gain}"
        )

    def test_yaml_has_rot_axis_gain(self):
        """yaml record.teleop.unityvr_config.rot_axis_gain 键存在（§11.3）。"""
        raw = _load_yaml()
        uvr_cfg = raw["record"]["teleop"]["unityvr_config"]
        assert "rot_axis_gain" in uvr_cfg, "unityvr_config 缺 rot_axis_gain 键"

    def test_yaml_rot_axis_gain_default_unit(self):
        """yaml rot_axis_gain 默认 [1.0, 1.0, 1.0]。"""
        raw = _load_yaml()
        gain = raw["record"]["teleop"]["unityvr_config"]["rot_axis_gain"]
        assert list(gain) == [1.0, 1.0, 1.0], (
            f"rot_axis_gain 默认应为 [1.0, 1.0, 1.0]，实际 {gain}"
        )


# ================================================================
# Step 2: RecordConfig 解析后字段 == yaml 值
# ================================================================

class TestRecordConfigParsesYamlValues:
    """RecordConfig(raw["record"]) 解析后字段 == yaml 中的值。"""

    def _make_rc(self, raw_overrides=None):
        """从真 yaml 加载 RecordConfig，可选 raw 覆盖（用于"仅改 yaml"验证）。"""
        raw = _load_yaml()
        if raw_overrides:
            # 浅合并 record 段的顶层键
            raw["record"].update(raw_overrides)
        m = _load_run_record()
        return m.RecordConfig(raw["record"])

    def test_rc_out_dir_matches_yaml(self):
        """RecordConfig.out_dir == yaml record.out_dir。"""
        raw = _load_yaml()
        rc = self._make_rc()
        assert rc.out_dir == raw["record"]["out_dir"], (
            f"RecordConfig.out_dir={rc.out_dir!r} 与 yaml {raw['record']['out_dir']!r} 不一致"
        )

    def test_rc_depth_enabled_matches_yaml(self):
        """RecordConfig.depth_enabled == yaml record.depth.enabled（False）。"""
        rc = self._make_rc()
        assert rc.depth_enabled is False

    def test_rc_state_hifreq_enabled_matches_yaml(self):
        """RecordConfig.state_hifreq_enabled == yaml record.state_hifreq.enabled（False）。"""
        rc = self._make_rc()
        assert rc.state_hifreq_enabled is False

    def test_rc_state_hifreq_rate_matches_yaml(self):
        """RecordConfig.state_hifreq_rate == yaml record.state_hifreq.rate（240）。"""
        rc = self._make_rc()
        assert rc.state_hifreq_rate == 240

    def test_rc_color_preflight_matches_yaml(self):
        """RecordConfig.color_preflight == yaml record.color_preflight（True）。"""
        rc = self._make_rc()
        assert rc.color_preflight is True

    def test_rc_pos_axis_gain_matches_yaml(self):
        """RecordConfig.pos_axis_gain == yaml unityvr_config.pos_axis_gain ([1,1,1])。"""
        raw = _load_yaml()
        rc = self._make_rc()
        yaml_gain = list(raw["record"]["teleop"]["unityvr_config"]["pos_axis_gain"])
        assert list(rc.pos_axis_gain) == yaml_gain, (
            f"pos_axis_gain: rc={list(rc.pos_axis_gain)} yaml={yaml_gain}"
        )

    def test_rc_rot_axis_gain_matches_yaml(self):
        """RecordConfig.rot_axis_gain == yaml unityvr_config.rot_axis_gain ([1,1,1])。"""
        raw = _load_yaml()
        rc = self._make_rc()
        yaml_gain = list(raw["record"]["teleop"]["unityvr_config"]["rot_axis_gain"])
        assert list(rc.rot_axis_gain) == yaml_gain, (
            f"rot_axis_gain: rc={list(rc.rot_axis_gain)} yaml={yaml_gain}"
        )

    def test_rc_num_episodes_matches_yaml(self):
        """RecordConfig.num_episodes == yaml task.num_episodes（已有键，回归守门）。"""
        raw = _load_yaml()
        rc = self._make_rc()
        assert rc.num_episodes == raw["record"]["task"]["num_episodes"]

    def test_rc_episode_time_sec_matches_yaml(self):
        """RecordConfig.episode_time_sec == yaml time.episode_time_sec（已有键，回归守门）。"""
        raw = _load_yaml()
        rc = self._make_rc()
        assert rc.episode_time_sec == raw["record"]["time"]["episode_time_sec"]

    def test_rc_task_description_matches_yaml(self):
        """RecordConfig.task_description == yaml task.description（已有键，回归守门）。"""
        raw = _load_yaml()
        rc = self._make_rc()
        assert rc.task_description == raw["record"]["task"]["description"]


# ================================================================
# Step 3: resolve_record_overrides（CLI 全 None）取 yaml 值——"仅改 yaml 即生效"
# ================================================================

class TestYamlOnlyDrivesOverrides:
    """验证 resolve_record_overrides(CLI 全 None) 完全由 yaml 驱动。"""

    def _make_rc_and_rp(self, yaml_rec_overrides=None):
        """从真 yaml 构造 RecordConfig + 加载 record_params。"""
        raw = _load_yaml()
        if yaml_rec_overrides:
            raw["record"].update(yaml_rec_overrides)
        m = _load_run_record()
        rc = m.RecordConfig(raw["record"])
        rp = _load_record_params()
        return rc, rp

    def _fallback_dir(self):
        """模拟 _paths.HDF5_EPISODES_DIR 的值（实际路径，测试用固定字符串替代）。"""
        return "/home/ubuntu/Desktop/jhli/_hdf5_episodes"

    def _call_overrides(self, rc, rp, cli_overrides=None):
        """调用 resolve_record_overrides（CLI 全 None，仅 yaml 驱动）。"""
        cli = dict(
            cli_episodes=None, cli_episode_sec=None,
            cli_out_dir=None, cli_task_name=None, cli_oc2base=None,
        )
        if cli_overrides:
            cli.update(cli_overrides)
        return rp.resolve_record_overrides(
            **cli, record_cfg=rc, out_dir_fallback=self._fallback_dir(),
        )

    def test_overrides_episodes_from_yaml(self):
        """CLI None → episodes 来自 yaml task.num_episodes。"""
        rc, rp = self._make_rc_and_rp()
        raw = _load_yaml()
        res = self._call_overrides(rc, rp)
        assert res["episodes"] == raw["record"]["task"]["num_episodes"], (
            f"episodes 应等于 yaml 值 {raw['record']['task']['num_episodes']}，实际 {res['episodes']}"
        )

    def test_overrides_episode_sec_from_yaml(self):
        """CLI None → episode_sec 来自 yaml time.episode_time_sec。"""
        rc, rp = self._make_rc_and_rp()
        raw = _load_yaml()
        res = self._call_overrides(rc, rp)
        assert res["episode_sec"] == raw["record"]["time"]["episode_time_sec"]

    def test_overrides_out_dir_from_yaml(self):
        """CLI None → out_dir 来自 yaml record.out_dir（非 fallback）。"""
        rc, rp = self._make_rc_and_rp()
        res = self._call_overrides(rc, rp)
        raw = _load_yaml()
        expected_out_dir = raw["record"]["out_dir"]
        assert res["out_dir"] == expected_out_dir, (
            f"out_dir 应为 yaml 值 {expected_out_dir!r}，实际 {res['out_dir']!r}"
        )

    def test_overrides_task_name_from_yaml_description(self):
        """CLI None → task_name 来自 yaml task.description。"""
        rc, rp = self._make_rc_and_rp()
        raw = _load_yaml()
        res = self._call_overrides(rc, rp)
        assert res["task_name"] == raw["record"]["task"]["description"], (
            f"task_name={res['task_name']!r} vs yaml={raw['record']['task']['description']!r}"
        )

    def test_yaml_out_dir_change_propagates(self):
        """改 yaml out_dir 值 → overrides.out_dir 跟随改变（仅改 yaml 即生效）。"""
        new_dir = "/tmp/test_episodes_new"
        rc, rp = self._make_rc_and_rp(yaml_rec_overrides={"out_dir": new_dir})
        res = self._call_overrides(rc, rp)
        assert res["out_dir"] == new_dir, (
            f"out_dir 未跟随 yaml 改变，实际 {res['out_dir']!r}（应为 {new_dir!r}）"
        )

    def test_yaml_episodes_change_propagates(self):
        """改 yaml task.num_episodes 值 → overrides.episodes 跟随改变。"""
        # 直接在 yaml dict 层构造 override
        raw = _load_yaml()
        raw["record"]["task"]["num_episodes"] = 99
        m = _load_run_record()
        rc = m.RecordConfig(raw["record"])
        rp = _load_record_params()
        res = self._call_overrides(rc, rp)
        assert res["episodes"] == 99, (
            f"episodes 应跟随 yaml 改为 99，实际 {res['episodes']}"
        )

    def test_cli_out_dir_overrides_yaml(self):
        """CLI 给 out_dir → 覆盖 yaml out_dir（CLI 优先于 yaml）。"""
        rc, rp = self._make_rc_and_rp()
        res = self._call_overrides(rc, rp, cli_overrides={"cli_out_dir": "/cli/override"})
        assert res["out_dir"] == "/cli/override", (
            f"CLI out_dir 应覆盖 yaml，实际 {res['out_dir']!r}"
        )

    def test_cli_episodes_overrides_yaml(self):
        """CLI 给 episodes → 覆盖 yaml num_episodes。"""
        rc, rp = self._make_rc_and_rp()
        res = self._call_overrides(rc, rp, cli_overrides={"cli_episodes": 3})
        assert res["episodes"] == 3

    def test_out_dir_fallback_only_when_both_null(self):
        """yaml out_dir 为 None 且 CLI None → 使用 fallback 常量（二级回退）。"""
        rc, rp = self._make_rc_and_rp(yaml_rec_overrides={"out_dir": None})
        res = self._call_overrides(rc, rp)
        assert res["out_dir"] == self._fallback_dir(), (
            f"yaml 和 CLI 都 None 时应用 fallback，实际 {res['out_dir']!r}"
        )
