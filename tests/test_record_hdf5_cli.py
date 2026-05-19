"""test_record_hdf5_cli.py: 验证 run_record_hdf5.py argparse CLI 缺省值全为 None。

不依赖硬件：importlib 加载模块后直接取 main() 内的 ArgumentParser，
用 parse_args([]) 检查缺省值（--config 是 required 会报错，
所以直接检查 add_argument 调用，或 parse_args 时给 --config）。
"""
import importlib.util, os, sys, argparse

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _load_module():
    """加载 run_record_hdf5.py（无硬件，sys.modules mock 硬件依赖）。"""
    sys.path.insert(0, os.path.join(_P, "scripts"))
    spec = importlib.util.spec_from_file_location(
        "rrh2", os.path.join(_P, "scripts/core/run_record_hdf5.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _build_parser():
    """重建与 main() 等价的 ArgumentParser（从源码读 add_argument 逻辑）。

    策略：由于 main() 含 hardware import，不直接调；
    直接用 argparse 重建并验证源码中 default 值，
    或改为 parse_known_args 避免 required error。
    实际策略：解析 _load_module() 不用，直接读 parser 从 ap.parse_args(['--config','x'])。
    """
    m = _load_module()
    # 猴补丁 main 不可用；改为解析 argparse 元数据：
    # 构造与 main() 同款 parser，检查 default
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--episode-sec", type=float, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--task-name", default=None)
    ap.add_argument("--oc2base-R", default=None)
    return ap


def _parse_module_args(extra=None):
    """从模块源码提取实际 ArgumentParser 的 defaults。

    做法：补丁 argparse.ArgumentParser.__init__ 捕获 parser 实例，
    然后调用 main 的部分——太复杂。
    改用：直接读源码文件，grep add_argument 行中的 default= 值。
    最简单：从 run_record_hdf5.py 源码 ast 解析 main() 里的 add_argument calls。
    但最可靠：把 ArgumentParser 提取为模块级函数（实现步骤做）。

    当前（Step 1 失败测试）：假设 main() 里有可提取的 parser，
    或我们直接 grep 源码验证 default 值。
    """
    import re
    src = open(os.path.join(_P, "scripts/core/run_record_hdf5.py")).read()
    # 期望: default=None for all 4 args
    return src


def test_cli_episodes_default_none():
    """--episodes 缺省值应为 None（非 1）。"""
    src = _parse_module_args()
    import re
    # 找 --episodes 的 add_argument 行，断言 default=None
    # 匹配模式: add_argument("--episodes", ..., default=None, ...)
    match = re.search(r'add_argument\(["\']--episodes["\'][^)]*default=([^,)]+)', src)
    assert match is not None, "--episodes add_argument 未找到"
    default_val = match.group(1).strip()
    assert default_val == "None", f"--episodes default 应为 None, got {default_val!r}"


def test_cli_episode_sec_default_none():
    """--episode-sec 缺省值应为 None（非 60.0）。"""
    src = _parse_module_args()
    import re
    match = re.search(r'add_argument\(["\']--episode-sec["\'][^)]*default=([^,)]+)', src)
    assert match is not None, "--episode-sec add_argument 未找到"
    default_val = match.group(1).strip()
    assert default_val == "None", f"--episode-sec default 应为 None, got {default_val!r}"


def test_cli_out_dir_default_none():
    """--out-dir 缺省值应为 None（非 HDF5_EPISODES_DIR 常量）。"""
    src = _parse_module_args()
    import re
    match = re.search(r'add_argument\(["\']--out-dir["\'][^)]*default=([^,)]+)', src)
    assert match is not None, "--out-dir add_argument 未找到"
    default_val = match.group(1).strip()
    assert default_val == "None", f"--out-dir default 应为 None, got {default_val!r}"


def test_cli_task_name_default_none():
    """--task-name 缺省值应为 None（非 "task"）。"""
    src = _parse_module_args()
    import re
    match = re.search(r'add_argument\(["\']--task-name["\'][^)]*default=([^,)]+)', src)
    assert match is not None, "--task-name add_argument 未找到"
    default_val = match.group(1).strip()
    assert default_val == "None", f"--task-name default 应为 None, got {default_val!r}"


def test_cli_fps_default_already_none():
    """--fps 缺省值已经是 None（回归守门）。"""
    src = _parse_module_args()
    import re
    match = re.search(r'add_argument\(["\']--fps["\'][^)]*default=([^,)]+)', src)
    assert match is not None, "--fps add_argument 未找到"
    default_val = match.group(1).strip()
    assert default_val == "None", f"--fps default 应为 None, got {default_val!r}"


def test_cli_oc2base_r_default_already_none():
    """--oc2base-R 缺省值已经是 None（回归守门）。"""
    src = _parse_module_args()
    import re
    match = re.search(r'add_argument\(["\']--oc2base-R["\'][^)]*default=([^,)]+)', src)
    assert match is not None, "--oc2base-R add_argument 未找到"
    default_val = match.group(1).strip()
    assert default_val == "None", f"--oc2base-R default 应为 None, got {default_val!r}"
