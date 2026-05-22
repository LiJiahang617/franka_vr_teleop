"""Task 7: 入口模块 run_record_hdf5_ui.py 的离线 smoke 测试。

验证：
1. 模块文件存在且可 import（无启动副作用）
2. main 函数存在
3. --help 秒退且不启动 Flask 服务器
"""
import importlib.util
import os

# 入口文件路径（相对 tests/ 目录一级上跳到 repo 根）
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ENTRY = os.path.join(_REPO, "scripts/core/run_record_hdf5_ui.py")


def test_entry_module_exists_and_imports():
    """入口文件存在且 import 无副作用（不调 app.run / 不连硬件）。"""
    assert os.path.exists(_ENTRY), f"入口文件不存在: {_ENTRY}"
    spec = importlib.util.spec_from_file_location("rrh_ui", _ENTRY)
    # 仅 import 不执行 main；main 内部禁 startup side-effect
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main"), "入口模块缺少 main 函数"


def test_entry_argparse_help_does_not_start_server():
    """--help 秒退，不阻塞（不能调 app.run），且输出含 config 相关文字。"""
    import subprocess

    python = os.environ.get(
        "PYTEST_PYTHON",
        "/home/ubuntu/Desktop/jhli/envs/franka-teleop/bin/python",
    )
    r = subprocess.run(
        [python, _ENTRY, "--help"],
        capture_output=True,
        text=True,
        timeout=10,  # 超时 10s = 服务器已阻塞
    )
    assert r.returncode == 0, f"--help 返回非零: {r.returncode}\nstderr={r.stderr}"
    assert "config" in r.stdout.lower(), (
        f"--help 输出未含 'config':\n{r.stdout}"
    )
