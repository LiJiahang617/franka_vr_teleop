"""franka-hdf5-v1 episodes → 标准 LeRobotDataset。lerobot 怪问题隔离于此一处。

⚠️ 版本注意(2026-05-19, 用户决策): 本脚本用 franka2 环境内 lerobot(v3.0), 产出
   codebase_version=v3.0 数据集(meta=tasks.parquet/episodes/chunk/file.parquet/
   stats.json; 文件名 file-NNN; videos/{key}/chunk/)。用户既有训练/可视化管线
   (RoboCOIN visualize_dataset / realman 参考集 / GR00T modality.json)均为 v2.1,
   与 v3.0 不互通(lerobot 对 codebase_version 强校验)。按用户决策本脚本保留产
   v3.0 不改; 产 v2.1 由【独立 v2.1 转换器】负责(见 spec §11.5, 与 Codex 协作设计)。

用法:
  envs/franka-teleop/bin/python lerobot_franka_teleop/scripts/tools/hdf5_to_lerobot.py \\
      --in _hdf5_episodes --repo-id local/franka_x --fps 30

注：lerobot hw_to_dataset_features 将 float 键聚合为向量，图像键加前缀
    observation.images.{cam}。frame dict 键格式见 hdf5_lerobot_map.py。
"""
import argparse
import glob
import logging
import sys
import h5py

from pathlib import Path as _Path
# hdf5_to_lerobot.py 在 <repo>/scripts/tools/ ; scripts 目录(=parents[1])上 path
# 供 `from tools.hdf5_lerobot_map import ...`(下一行)解析; schema 加载另由
# schema_loader 经 __file__ 相对定位, 与此 sys.path 互不依赖。
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from core.schema_loader import load_franka_hdf5_schema
S = load_franka_hdf5_schema()
from tools.hdf5_lerobot_map import build_feature_specs, hdf5_frame_to_lerobot

log = logging.getLogger("h5->lerobot")


def _cam_names(h5):
    return sorted(list(h5["observations/camera/rgb"].keys()))


def _cam_hw(h5, cam_names):
    """从首帧解码图像得到实际 H/W，避免硬编码尺寸造成 features 不匹配。"""
    import cv2, numpy as np
    hw = {}
    for c in cam_names:
        raw = h5[f"observations/camera/rgb/{c}/images"][0]
        arr = np.frombuffer(bytes(raw), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        hw[c] = (img.shape[0], img.shape[1], 3)
    return hw


def convert(in_dir, repo_id, fps, root, task="task"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import hw_to_dataset_features

    eps = sorted(glob.glob(f"{in_dir}/*.h5"))
    if not eps:
        raise SystemExit(f"无 episode: {in_dir}")

    # 用首个 episode 定 features（包含实际图像尺寸）
    with h5py.File(eps[0], "r") as f0:
        cams = _cam_names(f0)
        cam_hw = _cam_hw(f0, cams)

    a_hw, o_hw = build_feature_specs(cams, cam_hw=cam_hw)
    feats = {
        **hw_to_dataset_features(a_hw, "action"),
        **hw_to_dataset_features(o_hw, "observation", use_video=True),
    }
    ds = LeRobotDataset.create(repo_id=repo_id, fps=int(fps), features=feats, root=root)

    for ep in eps:
        v = S.validate_episode(ep)
        if v:
            log.warning(f"跳过不合规 {ep}: {v}")
            continue
        with h5py.File(ep, "r") as f:
            N = f["observations/timestamp"].shape[0]
            for i in range(N):
                ds.add_frame(hdf5_frame_to_lerobot(f, i, cams, task=task))
        ds.save_episode()
        log.info(f"已转 {ep} ({N} 帧)")
    log.info(f"完成 -> {root or '默认 HF home'} repo={repo_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="/home/ubuntu/Desktop/jhli/_hdf5_episodes")
    ap.add_argument("--repo-id", default="local/franka_unityvr")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--root", default="/home/ubuntu/Desktop/jhli/_lerobot_out")
    ap.add_argument("--task", default="task", help="任务描述字符串（写入 lerobot task 字段）")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    convert(a.in_dir, a.repo_id, a.fps, a.root, task=a.task)


if __name__ == "__main__":
    main()
