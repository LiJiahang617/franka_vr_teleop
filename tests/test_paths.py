import importlib.util
import os
import sys

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "paths", os.path.join(_P, "scripts/core/paths.py"))
paths = importlib.util.module_from_spec(_spec)
# 注册到 sys.modules，使模块可被引用
sys.modules["paths"] = paths
_spec.loader.exec_module(paths)


def _reload_paths():
    """重新执行 paths 模块（环境变量改变后刷新模块状态）。"""
    _spec.loader.exec_module(paths)


def test_constants_are_absolute():
    for name in ["JHLI_ROOT", "REPO_ROOT", "HDF5_EPISODES_DIR",
                 "LEROBOT_OUT", "OC2ARM_R_PATH", "SERVICES_DIR"]:
        v = getattr(paths, name)
        assert isinstance(v, str) and os.path.isabs(v), f"{name}={v!r} 非绝对路径"


def test_ports():
    assert paths.ARM_PORT == 50051
    assert paths.ZERORPC_PORT == 4242
    assert paths.GRIPPER_PORT == 50052


def test_repo_under_jhli_and_derived():
    assert paths.REPO_ROOT == paths.JHLI_ROOT + "/lerobot_franka_teleop"
    assert paths.SERVICES_DIR == paths.REPO_ROOT + "/scripts/services"
    assert paths.OC2ARM_R_PATH == paths.REPO_ROOT + "/.stage3_oc2arm_R.npy"


def test_env_override():
    # spec_from_file_location 加载的模块不在标准 finder，直接重执行等价于 reload。
    # 用 try/finally 显式管理 env 和模块状态，避免 pytest finalizer 执行顺序问题。
    os.environ["FRANKA_JHLI_ROOT"] = "/tmp/xx_jhli"
    try:
        _reload_paths()
        assert paths.JHLI_ROOT == "/tmp/xx_jhli"
        assert paths.REPO_ROOT == "/tmp/xx_jhli/lerobot_franka_teleop"
    finally:
        del os.environ["FRANKA_JHLI_ROOT"]
        _reload_paths()  # 恢复默认状态


def test_oc2arm_r_exists_default():
    # 默认(未 override)时标定文件须真实存在(Phase A 已置 repo 路径)
    if os.environ.get("FRANKA_JHLI_ROOT") is None:
        assert os.path.isfile(paths.OC2ARM_R_PATH), f"标定缺失 {paths.OC2ARM_R_PATH}"
