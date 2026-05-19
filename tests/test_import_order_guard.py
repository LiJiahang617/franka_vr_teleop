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



def test_run_record_hdf5_preimports_before_syspath_insert():
    # 结构回归: run_record_hdf5.py 里 pre-import 必须出现在注入仓库根之前,
    # 防止以后有人调换顺序又把真包遮蔽掉。
    src = open(f"{REPO}/scripts/core/run_record_hdf5.py").read()
    i_pre = src.find("import lerobot_robot_franka")
    i_insert = src.find(
        "sys.path.insert(0, \"/home/ubuntu/Desktop/jhli/lerobot_franka_teleop\")")
    assert i_pre != -1, "缺少 pre-import 守卫"
    assert i_insert != -1, "未找到仓库根 sys.path.insert"
    assert i_pre < i_insert, "pre-import 必须在 sys.path.insert(repo) 之前"


def test_vr_modules_importable_from_teleop_package():
    # §11.1: vr_align/unity_vr_reader 已入 lerobot_teleoperator_franka 包,
    # 任意 cwd 经包路径可导入（无需 repo-根-on-sys.path）。
    r = _probe(
        "import lerobot_teleoperator_franka.vr_align as a;"
        "import lerobot_teleoperator_franka.unity_vr_reader as u;"
        "print('OK' if a.__file__ and u.__file__ else 'BAD')"
    )
    assert r.stdout.strip() == "OK", (r.stdout, r.stderr)
