"""§11.1-T4 守门: debug 起停脚本里 _run_*.sh 启动参数必须是固定绝对
scripts/services/ 路径(防整合期陈旧路径回归: repo-根裸路径/相对/./ 等)。
"""
import os
import re
import shlex
from pathlib import Path

REPO = "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop"
SERVICES = f"{REPO}/scripts/services"

# 预期被守护的 debug 起停脚本(显式列举, 缺失即失败, 不静默跳过)
EXPECTED_SCRIPTS = [
    "debug/franka_start_arm.sh",
    "debug/franka_start_zerorpc.sh",
    "debug/franka_start_gripper.sh",
    "debug/franka_clean_restart.sh",
]


def _launch_run_args(text):
    """提取每个含 `bash ... _run_*.sh` 启动行里 bash 之后的脚本参数 token。

    用 shlex 解析行(容忍引号/./等), 找到 'bash'(或 setsid 后的 bash) 紧随的
    首个以 _run_ 开头或以 /_run_*.sh / _run_*.sh 结尾的参数。
    """
    args = []
    for line in text.splitlines():
        if "bash" not in line or "_run_" not in line:
            continue
        try:
            toks = shlex.split(line, comments=False, posix=True)
        except ValueError:
            # 解析失败(异常引号)也视为可疑, 交由调用方按"未取到合法绝对参数"判失败
            toks = re.findall(r"\S+", line)
        for i, t in enumerate(toks):
            if t == "bash" and i + 1 < len(toks):
                nxt = toks[i + 1]
                if "_run_" in nxt and nxt.endswith(".sh"):
                    args.append(nxt)
    return args


def test_expected_debug_scripts_exist():
    for rel in EXPECTED_SCRIPTS:
        p = os.path.join(REPO, rel)
        assert os.path.isfile(p), f"预期 debug 脚本缺失: {p}"


def test_debug_launch_uses_fixed_absolute_services_path():
    bad = []
    for rel in EXPECTED_SCRIPTS:
        sh = os.path.join(REPO, rel)
        if not os.path.isfile(sh):
            bad.append((rel, "脚本缺失"))
            continue
        text = Path(sh).read_text(encoding="utf-8")
        run_args = _launch_run_args(text)
        if not run_args:
            bad.append((os.path.basename(sh), "未找到 bash _run_*.sh 启动参数"))
            continue
        for a in run_args:
            # 必须是固定绝对 services 目录下脚本, 且文件真实存在
            if not a.startswith(SERVICES + "/"):
                bad.append((os.path.basename(sh),
                            f"_run 启动参数非固定绝对 services 路径: {a!r}"))
            elif not os.path.isfile(a):
                bad.append((os.path.basename(sh), f"目标服务脚本不存在: {a}"))
    assert not bad, f"debug 脚本陈旧/无效 _run 路径: {bad}"
