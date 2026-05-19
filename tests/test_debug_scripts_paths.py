"""
§11.1-T4: debug 起停脚本路径回归测试。
验证 franka_start_*.sh 和 franka_clean_restart.sh 中所有 _run_*.sh 引用
均已改为绝对 scripts/services/ 路径，且目标文件实际存在。
"""
import glob
import os
import re

REPO = "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop"


def test_debug_start_scripts_reference_existing_services():
    """debug 起停脚本中 _run_*.sh 必须使用绝对路径且文件存在，禁止裸名。"""
    bad = []
    target_scripts = (
        glob.glob(f"{REPO}/debug/franka_*start*.sh")
        + [f"{REPO}/debug/franka_clean_restart.sh"]
    )
    for sh in target_scripts:
        if not os.path.isfile(sh):
            continue
        txt = open(sh).read()
        # 绝对路径引用须指向真实文件
        for m in re.finditer(r"(/[\w./-]*scripts/services/_run_[\w]+\.sh)", txt):
            if not os.path.isfile(m.group(1)):
                bad.append((os.path.basename(sh), f"文件不存在: {m.group(1)}"))
        # 不允许裸名 _run_*.sh 残留
        if re.search(r"(setsid +)?bash +_run_[\w]+\.sh", txt):
            bad.append((os.path.basename(sh), "裸 _run_*.sh 未修正"))
    assert not bad, f"debug 脚本陈旧/无效路径: {bad}"
