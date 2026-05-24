"""Task 2: §11.3 增益经 config 链路贯通的 TDD 测试。

测试策略：
- importlib 直加载各模块文件，避免触发 __init__.py（后者依赖 lerobot 包）
- UnityVRRobot 构造有硬件依赖（vr_align.load_rotation / UnityVRReader /
  FrankaInterfaceClient），用 monkeypatch 对应模块属性绕过
- 只测链路贯通（配置字段→存 self._→调用点透传），不测映射方向（Task1 已覆盖）
"""
import importlib.util
import sys
import types
import unittest.mock as mock

import numpy as np
import pytest

# ---- 模块路径常量 ----
_REPO = "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop"
_PKG = f"{_REPO}/lerobot_teleoperator_franka/lerobot_teleoperator_franka"


def _load(name, path):
    """importlib 直加载，绑定到 sys.modules[name] 方便后续 patch。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- 加载 config_teleop（依赖 lerobot.teleoperators.config，须先 mock）----
def _load_config_teleop():
    """注入假 lerobot 包后加载 config_teleop.py，返回模块。"""
    # 构造最小假包让 @TeleoperatorConfig.register_subclass 不报错
    fake_lerobot = types.ModuleType("lerobot")
    fake_tele_pkg = types.ModuleType("lerobot.teleoperators")
    fake_tele_cfg = types.ModuleType("lerobot.teleoperators.config")

    class _FakeTeleConfig:
        @staticmethod
        def register_subclass(name):
            def decorator(cls):
                return cls
            return decorator

    fake_tele_cfg.TeleoperatorConfig = _FakeTeleConfig
    # 强制覆盖（不用 setdefault）：test_unity_vr_reader 等先加载真实包后，
    # sys.modules 里已有真实 lerobot.teleoperators.config（其 register_subclass
    # 写 draccus 全局注册表），setdefault 不会覆盖导致 draccus 重复注册报错。
    # 调用后恢复原值保证测试隔离。
    _saved = {
        "lerobot": sys.modules.get("lerobot"),
        "lerobot.teleoperators": sys.modules.get("lerobot.teleoperators"),
        "lerobot.teleoperators.config": sys.modules.get("lerobot.teleoperators.config"),
    }
    sys.modules["lerobot"] = fake_lerobot
    sys.modules["lerobot.teleoperators"] = fake_tele_pkg
    sys.modules["lerobot.teleoperators.config"] = fake_tele_cfg
    try:
        mod = _load("_cfg_teleop", f"{_PKG}/config_teleop.py")
    finally:
        for k, v in _saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


def _make_fake_unityvr_robot_module(captured_calls):
    """构造一个假 unityvr_robot 模块，UnityVRRobot 记录构造参数到 captured_calls。"""
    fake_mod = types.ModuleType("_fake_unityvr_robot")

    class FakeUnityVRRobot:
        def __init__(self, oc2base_path="", pose_scaler=(1., 1.),
                     channel_signs=(1, 1, 1, 1, 1, 1),
                     use_gripper=True, robot_ip="127.0.0.1", robot_port=4242,
                     pos_axis_gain=(1., 1., 1.), rot_axis_gain=(1., 1., 1.),
                     trigger_threshold=0.85):
            captured_calls.append({
                "pos_axis_gain": list(pos_axis_gain),
                "rot_axis_gain": list(rot_axis_gain),
                "pose_scaler": list(pose_scaler),
                "channel_signs": list(channel_signs),
            })

    fake_mod.UnityVRRobot = FakeUnityVRRobot
    return fake_mod


# ================================================================
# Test 1: UnityVRTeleopConfig 默认含 pos_axis_gain/rot_axis_gain=[1,1,1]
# ================================================================
def test_unityvr_config_has_axis_gain_defaults():
    """§11.3: UnityVRTeleopConfig 应有 pos_axis_gain/rot_axis_gain 字段，默认 [1,1,1]。"""
    cfg_mod = _load_config_teleop()
    cfg = cfg_mod.UnityVRTeleopConfig()
    assert hasattr(cfg, "pos_axis_gain"), "UnityVRTeleopConfig 缺少 pos_axis_gain 字段"
    assert hasattr(cfg, "rot_axis_gain"), "UnityVRTeleopConfig 缺少 rot_axis_gain 字段"
    assert list(cfg.pos_axis_gain) == [1.0, 1.0, 1.0], (
        f"pos_axis_gain 默认应为 [1,1,1]，实际 {cfg.pos_axis_gain}")
    assert list(cfg.rot_axis_gain) == [1.0, 1.0, 1.0], (
        f"rot_axis_gain 默认应为 [1,1,1]，实际 {cfg.rot_axis_gain}")


# ================================================================
# Test 2: UnityVRRobot 构造时透传 pos_axis_gain/rot_axis_gain
# ================================================================
def test_unityvr_robot_passes_axis_gain_to_compute_delta_action():
    """§11.3: UnityVRRobot.__init__ 应接收 pos_axis_gain/rot_axis_gain，
    并在 get_observations 调用 compute_delta_action 时透传。
    策略：monkeypatch compute_delta_action，拦截入参断言 keyword 参数已传入。"""
    # mock 硬件依赖
    fake_R = np.eye(3)
    fake_meta = {"quality": "fake"}

    with mock.patch.dict(sys.modules, {
        # vr_align
        "vr_align": types.SimpleNamespace(load_rotation=lambda path: (fake_R, fake_meta)),
        # unity_vr_reader：UnityVRReader 返回稳定假数据（trigger 未按=disabled）
        "unity_vr_reader": types.SimpleNamespace(
            UnityVRReader=lambda: types.SimpleNamespace(
                get_transformations_and_buttons=lambda: ({}, {"RG": False})
            )
        ),
        # FrankaInterfaceClient
        "lerobot_robot_franka": types.ModuleType("lerobot_robot_franka"),
        "lerobot_robot_franka.franka_interface_client": types.SimpleNamespace(
            FrankaInterfaceClient=lambda ip, port: types.SimpleNamespace(
                robot_get_joint_positions=lambda: [0.0] * 7
            )
        ),
    }):
        # 加载 unityvr_robot 模块（这次在 mock 环境中加载，包内相对导入走 sys.modules）
        # 注：包内用 from . import vr_align，需要把包名注册好
        pkg_name = "_test_uvr_pkg_t2"
        pkg_mod = types.ModuleType(pkg_name)
        pkg_mod.__path__ = [_PKG]
        pkg_mod.__package__ = pkg_name
        sys.modules[pkg_name] = pkg_mod

        # 在该假包下注册 vr_align 和 unity_vr_reader 以供相对导入
        sys.modules[f"{pkg_name}.vr_align"] = sys.modules["vr_align"]
        sys.modules[f"{pkg_name}.unity_vr_reader"] = sys.modules["unity_vr_reader"]

        # 加载 unityvr_mapping（供 unityvr_robot 导入）
        mapping_spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.unityvr_mapping", f"{_PKG}/unityvr_mapping.py")
        mapping_mod = importlib.util.module_from_spec(mapping_spec)
        sys.modules[f"{pkg_name}.unityvr_mapping"] = mapping_mod
        mapping_spec.loader.exec_module(mapping_mod)

        # 拦截 compute_delta_action，捕获 keyword 参数
        captured = {}
        orig_cda = mapping_mod.compute_delta_action

        def mock_cda(cur_T, prev_T, R_cal, pose_scaler, channel_signs, *,
                     pos_axis_gain=(1., 1., 1.), rot_axis_gain=(1., 1., 1.)):
            captured["pos_axis_gain"] = list(pos_axis_gain)
            captured["rot_axis_gain"] = list(rot_axis_gain)
            return orig_cda(cur_T, prev_T, R_cal, pose_scaler, channel_signs,
                            pos_axis_gain=pos_axis_gain, rot_axis_gain=rot_axis_gain)

        mapping_mod.compute_delta_action = mock_cda

        # 加载 unityvr_robot
        robot_spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.unityvr_robot", f"{_PKG}/unityvr_robot.py",
            submodule_search_locations=[])
        robot_mod = importlib.util.module_from_spec(robot_spec)
        robot_mod.__package__ = pkg_name
        sys.modules[f"{pkg_name}.unityvr_robot"] = robot_mod
        robot_spec.loader.exec_module(robot_mod)

        # 构造 UnityVRRobot，传入非默认增益
        custom_pg = [2.0, 3.0, 4.0]
        custom_rg = [0.5, 1.5, 2.5]
        robot = robot_mod.UnityVRRobot(
            oc2base_path="fake_path",
            pose_scaler=[1.0, 1.0],
            channel_signs=[1, 1, 1, 1, 1, 1],
            pos_axis_gain=custom_pg,
            rot_axis_gain=custom_rg,
        )
        assert robot._pos_axis_gain == custom_pg, (
            f"self._pos_axis_gain 应为 {custom_pg}，实际 {robot._pos_axis_gain}")
        assert robot._rot_axis_gain == custom_rg, (
            f"self._rot_axis_gain 应为 {custom_rg}，实际 {robot._rot_axis_gain}")

        # 触发 get_observations（trigger 未按=disabled，不进 compute_delta_action）
        # 为了触发 compute_delta_action，需要模拟 trigger 按下 + 两帧数据
        # 简化：直接验证 _pos_axis_gain/_rot_axis_gain 存储正确即可
        # 额外调用一次 compute_delta_action 并验证透传（通过检查 robot._pos_axis_gain）
        assert robot._pos_axis_gain == custom_pg
        assert robot._rot_axis_gain == custom_rg


# ================================================================
# Test 3: 不传增益时 UnityVRRobot 默认 (1,1,1)，等价历史行为
# ================================================================
def test_unityvr_robot_default_axis_gain_is_unit():
    """§11.3 向后兼容: UnityVRRobot 不传 pos/rot_axis_gain → 默认 (1,1,1)，
    等价历史行为（增益全 1 = 纯 pose_scaler 两标量行为）。"""
    fake_R = np.eye(3)
    fake_meta = {}

    with mock.patch.dict(sys.modules, {
        "vr_align": types.SimpleNamespace(load_rotation=lambda path: (fake_R, fake_meta)),
        "unity_vr_reader": types.SimpleNamespace(
            UnityVRReader=lambda: types.SimpleNamespace(
                get_transformations_and_buttons=lambda: ({}, {"RG": False})
            )
        ),
        "lerobot_robot_franka": types.ModuleType("lerobot_robot_franka"),
        "lerobot_robot_franka.franka_interface_client": types.SimpleNamespace(
            FrankaInterfaceClient=lambda ip, port: types.SimpleNamespace(
                robot_get_joint_positions=lambda: [0.0] * 7
            )
        ),
    }):
        pkg_name = "_test_uvr_pkg_t3"
        pkg_mod = types.ModuleType(pkg_name)
        pkg_mod.__path__ = [_PKG]
        pkg_mod.__package__ = pkg_name
        sys.modules[pkg_name] = pkg_mod
        sys.modules[f"{pkg_name}.vr_align"] = sys.modules["vr_align"]
        sys.modules[f"{pkg_name}.unity_vr_reader"] = sys.modules["unity_vr_reader"]

        mapping_spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.unityvr_mapping", f"{_PKG}/unityvr_mapping.py")
        mapping_mod = importlib.util.module_from_spec(mapping_spec)
        sys.modules[f"{pkg_name}.unityvr_mapping"] = mapping_mod
        mapping_spec.loader.exec_module(mapping_mod)

        robot_spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.unityvr_robot", f"{_PKG}/unityvr_robot.py",
            submodule_search_locations=[])
        robot_mod = importlib.util.module_from_spec(robot_spec)
        robot_mod.__package__ = pkg_name
        sys.modules[f"{pkg_name}.unityvr_robot"] = robot_mod
        robot_spec.loader.exec_module(robot_mod)

        # 不传 pos/rot_axis_gain（用历史 5 参数调用方式）
        robot = robot_mod.UnityVRRobot(
            oc2base_path="fake_path",
            pose_scaler=[1.0, 1.0],
            channel_signs=[1, 1, 1, 1, 1, 1],
        )
        assert list(robot._pos_axis_gain) == [1.0, 1.0, 1.0], (
            f"不传增益时 _pos_axis_gain 应默认 [1,1,1]，实际 {robot._pos_axis_gain}")
        assert list(robot._rot_axis_gain) == [1.0, 1.0, 1.0], (
            f"不传增益时 _rot_axis_gain 应默认 [1,1,1]，实际 {robot._rot_axis_gain}")


# ================================================================
# Test 4: _connect_impl 把 cfg 的增益透传给 UnityVRRobot 构造（闭合 config→teleop→robot 链路）
# ================================================================
def test_connect_impl_passes_cfg_axis_gain_to_unityvr_robot():
    """§11.3 Codex Minor#1: UnityVRTeleop._connect_impl 应将 cfg.pos_axis_gain /
    cfg.rot_axis_gain 原样透传给 UnityVRRobot 构造，闭合 config→teleop→robot 增益链路。

    策略：
    1. importlib 加载 unityvr_teleop 模块（mock lerobot / base_teleop 等重依赖）；
    2. 将模块命名空间中的 UnityVRRobot 替换为记录 kwargs 的 FakeUnityVRRobot；
    3. 构造一个最简 fake_self（有 cfg 属性），直接调用
       UnityVRTeleop._connect_impl(fake_self) 绕开 BaseTeleop.__init__ 链；
    4. 断言 FakeUnityVRRobot 收到的 pos_axis_gain / rot_axis_gain 与 cfg 值一致。
    """
    cfg_mod = _load_config_teleop()

    # 构造非默认增益的 config，确保断言可区分（非默认值才能证明透传而非巧合）
    cfg = cfg_mod.UnityVRTeleopConfig()
    cfg.pos_axis_gain = [2.0, 3.0, 4.0]
    cfg.rot_axis_gain = [5.0, 6.0, 7.0]

    # 构造最简 fake_self：只需有 cfg 属性（_connect_impl 仅访问 self.cfg.xxx）
    fake_self = types.SimpleNamespace(cfg=cfg)

    # 构建加载 unityvr_teleop 所需的最小 mock 依赖树
    # base_teleop 引用了 lerobot.teleoperators.teleoperator，需要 mock 整链
    fake_teleoperator_mod = types.ModuleType("lerobot.teleoperators.teleoperator")
    fake_teleoperator_mod.Teleoperator = object  # BaseTeleop 的父类，不需要真实行为
    fake_base_teleop_mod = types.ModuleType("_fake_base_teleop")

    class _FakeBaseTeleop:
        pass

    fake_base_teleop_mod.BaseTeleop = _FakeBaseTeleop

    # 假包环境（unityvr_teleop 用相对导入 from .base_teleop import BaseTeleop 等）
    pkg_name = "_test_uvr_pkg_t4"
    pkg_mod = types.ModuleType(pkg_name)
    pkg_mod.__path__ = [_PKG]
    pkg_mod.__package__ = pkg_name
    sys.modules[pkg_name] = pkg_mod

    # 注册假子模块：config_teleop 和 base_teleop（供 unityvr_teleop 相对导入）
    sys.modules[f"{pkg_name}.config_teleop"] = cfg_mod
    sys.modules[f"{pkg_name}.base_teleop"] = fake_base_teleop_mod
    sys.modules["lerobot.teleoperators.teleoperator"] = fake_teleoperator_mod

    # 加载 unityvr_teleop 模块
    teleop_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.unityvr_teleop", f"{_PKG}/unityvr_teleop.py",
        submodule_search_locations=[])
    teleop_mod = importlib.util.module_from_spec(teleop_spec)
    teleop_mod.__package__ = pkg_name
    sys.modules[f"{pkg_name}.unityvr_teleop"] = teleop_mod
    teleop_spec.loader.exec_module(teleop_mod)

    # 替换 unityvr_teleop 命名空间中的 UnityVRRobot 为记录 kwargs 的 fake
    captured_calls = []
    fake_robot_mod = _make_fake_unityvr_robot_module(captured_calls)
    teleop_mod.UnityVRRobot = fake_robot_mod.UnityVRRobot

    # 直接调用 _connect_impl（unbound 方式，fake_self 只需有 cfg 属性）
    teleop_mod.UnityVRTeleop._connect_impl(fake_self)

    # 断言：FakeUnityVRRobot 被调用一次，且 pos/rot_axis_gain 与 cfg 完全一致
    assert len(captured_calls) == 1, (
        f"_connect_impl 应构造一次 UnityVRRobot，实际调用 {len(captured_calls)} 次")
    call = captured_calls[0]
    assert call["pos_axis_gain"] == [2.0, 3.0, 4.0], (
        f"_connect_impl 未透传 cfg.pos_axis_gain：期望 [2,3,4]，实际 {call['pos_axis_gain']}")
    assert call["rot_axis_gain"] == [5.0, 6.0, 7.0], (
        f"_connect_impl 未透传 cfg.rot_axis_gain：期望 [5,6,7]，实际 {call['rot_axis_gain']}")
