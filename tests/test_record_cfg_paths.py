"""回归: record_cfg_unityvr.yaml 里 oc2base_path 必须指向真实存在的标定文件。

整合期 "jhli -> jhli/franka_vr_teleop" 路径机械改写曾漏掉这行(陈旧 jhli
根路径无文件), 致真机 teleop connect 时 RuntimeError "未找到标定 R"。此测试
在离线阶段就拦住这类陈旧/失配的标定路径, 不必等真机才发现。
"""
import os
import pathlib

import yaml

REPO = "/home/ubuntu/Desktop/jhli/franka_vr_teleop"
CFG = f"{REPO}/scripts/config/record_cfg_unityvr.yaml"


def test_unityvr_oc2base_path_file_exists():
    raw = yaml.safe_load(open(CFG))
    uvr = raw["record"]["teleop"]["unityvr_config"]
    p = uvr["oc2base_path"]
    assert os.path.isfile(p), f"oc2base_path 指向的标定文件不存在: {p}"


def test_unityvr_oc2base_path_is_repo_canonical():
    """oc2base_path 在 yaml 中是相对项目根的路径 (开源后用户可改)."""
    import yaml
    p = pathlib.Path(__file__).resolve().parents[1] / "scripts/config/record_cfg_unityvr.yaml"
    raw = yaml.safe_load(p.read_text())
    oc_path = raw["record"]["teleop"]["unityvr_config"]["oc2base_path"]
    # 接受相对路径 "./.stage3_oc2arm_R.npy" 或绝对 (本机/用户路径)
    assert ".stage3_oc2arm_R.npy" in oc_path, f"oc2base_path 应含 .stage3_oc2arm_R.npy: {oc_path}"

