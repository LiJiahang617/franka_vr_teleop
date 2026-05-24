"""回归: record_cfg_unityvr.yaml 里 oc2base_path 必须指向真实存在的标定文件。

整合期 "jhli -> jhli/franka_vr_teleop" 路径机械改写曾漏掉这行(陈旧 jhli
根路径无文件), 致真机 teleop connect 时 RuntimeError "未找到标定 R"。此测试
在离线阶段就拦住这类陈旧/失配的标定路径, 不必等真机才发现。
"""
import os

import yaml

REPO = "/home/ubuntu/Desktop/jhli/franka_vr_teleop"
CFG = f"{REPO}/scripts/config/record_cfg_unityvr.yaml"


def test_unityvr_oc2base_path_file_exists():
    raw = yaml.safe_load(open(CFG))
    uvr = raw["record"]["teleop"]["unityvr_config"]
    p = uvr["oc2base_path"]
    assert os.path.isfile(p), f"oc2base_path 指向的标定文件不存在: {p}"


def test_unityvr_oc2base_path_is_repo_canonical():
    # 规范位置 = repo 内(与 config_teleop.py / run_record.py 代码默认一致),
    # 不是陈旧 jhli 根路径。
    raw = yaml.safe_load(open(CFG))
    p = raw["record"]["teleop"]["unityvr_config"]["oc2base_path"]
    assert p == f"{REPO}/.stage3_oc2arm_R.npy", f"oc2base_path 非规范 repo 路径: {p}"
