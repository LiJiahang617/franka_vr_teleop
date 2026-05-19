import importlib.util
import os

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "_franka_paths_under_test", os.path.join(_P, "scripts/core/paths.py"))
paths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paths)


def _reload_paths():
    """按当前环境变量重新执行 paths 模块（spec_from_file_location 加载的模块
    不在标准 finder，直接重执行等价于 reload）。"""
    _spec.loader.exec_module(paths)


def test_constants_are_absolute():
    for name in ["JHLI_ROOT", "REPO_ROOT", "SCRIPTS_DIR", "HDF5_EPISODES_DIR",
                 "LEROBOT_OUT", "OC2ARM_R_PATH", "SERVICES_DIR"]:
        v = getattr(paths, name)
        assert isinstance(v, str) and os.path.isabs(v), f"{name}={v!r} 非绝对路径"


def test_ports():
    assert paths.ARM_PORT == 50051
    assert paths.ZERORPC_PORT == 4242
    assert paths.GRIPPER_PORT == 50052


def test_repo_under_jhli_and_derived():
    assert paths.REPO_ROOT == paths.JHLI_ROOT + "/lerobot_franka_teleop"
    assert paths.SCRIPTS_DIR == paths.REPO_ROOT + "/scripts"
    assert paths.SERVICES_DIR == paths.REPO_ROOT + "/scripts/services"
    assert paths.OC2ARM_R_PATH == paths.REPO_ROOT + "/.stage3_oc2arm_R.npy"


def _set_env(key, value):
    """设 env 并返回恢复用的原值(None 表示原本不存在)。"""
    return os.environ.get(key), os.environ.__setitem__(key, value)


def _restore_env(key, prev):
    if prev is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = prev


def test_env_override():
    prev = os.environ.get("FRANKA_JHLI_ROOT")
    os.environ["FRANKA_JHLI_ROOT"] = "/tmp/xx_jhli"
    try:
        _reload_paths()
        assert paths.JHLI_ROOT == "/tmp/xx_jhli"
        assert paths.REPO_ROOT == "/tmp/xx_jhli/lerobot_franka_teleop"
    finally:
        _restore_env("FRANKA_JHLI_ROOT", prev)
        _reload_paths()  # 恢复 paths 到当前环境默认, 不污染后续测试


def test_independent_env_overrides():
    """FRANKA_HDF5_EPISODES_DIR / FRANKA_LEROBOT_OUT / FRANKA_OC2ARM_R 各自
    独立覆盖, 且不影响其它派生常量(单一真值: 局部覆盖不串扰)。"""
    cases = [
        ("FRANKA_HDF5_EPISODES_DIR", "/tmp/xx_hdf5", "HDF5_EPISODES_DIR"),
        ("FRANKA_LEROBOT_OUT", "/tmp/xx_lerobot", "LEROBOT_OUT"),
        ("FRANKA_OC2ARM_R", "/tmp/xx_oc2arm.npy", "OC2ARM_R_PATH"),
    ]
    for env_key, val, attr in cases:
        prev = os.environ.get(env_key)
        os.environ[env_key] = val
        try:
            _reload_paths()
            assert getattr(paths, attr) == val, (env_key, attr)
            # 仅该常量受影响, REPO_ROOT 等不被串改
            assert paths.REPO_ROOT == paths.JHLI_ROOT + "/lerobot_franka_teleop"
        finally:
            _restore_env(env_key, prev)
            _reload_paths()


def test_oc2arm_r_exists_default():
    # 默认(未 override)时标定文件须真实存在(Phase A 已置 repo 路径;
    # 这是对 §11.1 根因"路径指向不存在标定文件"的部署不变量守门, 故意如此)。
    if os.environ.get("FRANKA_JHLI_ROOT") is None:
        assert os.path.isfile(paths.OC2ARM_R_PATH), f"标定缺失 {paths.OC2ARM_R_PATH}"
