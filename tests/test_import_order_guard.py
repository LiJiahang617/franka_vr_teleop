"""回归: 仓库根含与 editable 包同名的外层命名空间目录(无 __init__.py)。

一旦仓库根进 sys.path, 标准 PathFinder 把它当命名空间包返回 spec, 遮蔽 PEP660
editable 安装的真包 → `from lerobot_robot_franka import FrankaConfig` 报
"cannot import name ... (unknown location)"。

修复: run_record_hdf5.py 必须在 `sys.path.insert(0, <repo>)` 之前先 import
lerobot_robot_franka / lerobot_teleoperator_franka, 把真包锁进 sys.modules。
根因详见 docs/lessons/2026-05-19-namespace-dir-shadows-editable-install.md
"""
import re
import subprocess
import sys

REPO = "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop"


def _probe(code):
    return subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)


def test_repo_root_on_syspath_shadows_editable_without_guard():
    # 不加守卫: 先插仓库根再 import → 命名空间包(__file__ 为 None, 无 FrankaConfig)
    r = _probe(
        f"import sys; sys.path.insert(0, {REPO!r});"
        "import lerobot_robot_franka as m;"
        "print(\"SHADOWED\" if getattr(m, \"__file__\", None) is None else \"REAL\")"
    )
    assert r.stdout.strip() == "SHADOWED", (r.stdout, r.stderr)





def test_vr_modules_importable_from_teleop_package():
    # §11.1 守门: vr_align/unity_vr_reader/unityvr_robot 三个子模块均经包路径
    # 解析到 editable 真包内层目录的真文件——非 repo-根单层命名空间遮蔽副本。
    # 判别签名: 真包是双层嵌套 .../lerobot_teleoperator_franka/lerobot_teleoperator_franka/
    # (dir 名与其父目录名均为 lerobot_teleoperator_franka); repo-根遮蔽副本父目录是
    # repo 根(名不为 lerobot_teleoperator_franka)。注: 顶层包对象在某些 cwd 下是
    # 命名空间包(__file__=None, 合法), 故只断言子模块真文件, 不碰顶层 __file__。
    r = _probe(
        "import pathlib;"
        "import lerobot_teleoperator_franka.vr_align as a;"
        "import lerobot_teleoperator_franka.unity_vr_reader as u;"
        "import lerobot_teleoperator_franka.unityvr_robot as m;"
        "pa=pathlib.Path(a.__file__).resolve().parent;"
        "pu=pathlib.Path(u.__file__).resolve().parent;"
        "pm=pathlib.Path(m.__file__).resolve().parent;"
        "ok=(pa==pu==pm)"
        " and pa.name=='lerobot_teleoperator_franka'"
        " and pa.parent.name=='lerobot_teleoperator_franka'"
        " and m.vr_align.__name__=='lerobot_teleoperator_franka.vr_align'"
        " and m.UnityVRReader.__module__=='lerobot_teleoperator_franka.unity_vr_reader';"
        "print('OK' if ok else 'BAD a=%s u=%s m=%s'%(a.__file__,u.__file__,m.__file__))"
    )
    assert r.stdout.strip() == "OK", (r.stdout, r.stderr)


def test_run_record_hdf5_no_repo_root_syspath():
    # §11.1: run_record_hdf5.py 不得再把 repo 根塞 sys.path(只允许 scripts),
    # 且 Task3 期 pre-import 守卫块已随根因消失而删净。
    src = open(f"{REPO}/scripts/core/run_record_hdf5.py").read()
    assert 'sys.path.insert(0, "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop")' \
        not in src, "run_record_hdf5 仍有 repo-根 sys.path.insert"
    assert "import lerobot_robot_franka  # noqa: F401" not in src, \
        "Task3 pre-import 守卫块未删净"


def test_editable_pkg_resolves_even_if_repo_root_on_path():
    # §11.1 核心保证(对抗式): 即便人为把 repo 根插到 sys.path[0],
    # lerobot_teleoperator_franka 子模块仍解析到双层嵌套 editable 真包内文件,
    # 非 repo-根单层命名空间遮蔽副本。(顶层命名空间包 __file__=None 合法,
    # 故只断言子模块真文件 + 双层嵌套签名, 与 Task2 守门③同口径。)
    r = _probe(
        f"import sys, pathlib; sys.path.insert(0, {REPO!r});"
        "import lerobot_teleoperator_franka.vr_align as a;"
        "import lerobot_teleoperator_franka.unity_vr_reader as u;"
        "p=pathlib.Path(a.__file__).resolve().parent;"
        "ok=(p==pathlib.Path(u.__file__).resolve().parent)"
        " and p.name=='lerobot_teleoperator_franka'"
        " and p.parent.name=='lerobot_teleoperator_franka';"
        "print('OK' if ok else 'SHADOWED a=%s u=%s'%(a.__file__,u.__file__))"
    )
    assert r.stdout.strip() == "OK", (r.stdout, r.stderr)
