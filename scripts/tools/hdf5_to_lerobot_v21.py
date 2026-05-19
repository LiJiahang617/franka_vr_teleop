"""franka-hdf5-v1 → lerobot v2.1 转换器（独立实现，不依赖任何版本 lerobot）。

功能模块：
  v21_meta    - info.json / tasks.jsonl / episodes.jsonl / episodes_stats.jsonl 构建
  v21_parquet - episode 帧写入 parquet（state/action/5 元列，无图像列）
  v21_video   - episode 视频导出（ffmpeg libx264 yuv420p，同 realman 参数）
  convert     - 顶层转换逻辑（串联 meta/parquet/video）
  main        - CLI 入口

CLI 用法：
  python scripts/tools/hdf5_to_lerobot_v21.py \\
    --in-dir /path/to/hdf5 --out /path/to/out \\
    --fps 30 --task "franka task" --robot-type franka --state-layout native
"""

import json
import sys
from pathlib import Path

import numpy as np

# 自举：确保 <repo>/scripts 在 sys.path 中（同 v3.0 模式）
_REPO_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_REPO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_REPO_SCRIPTS))

# ──────────────────────────────────────────────────────────────────────────────
# v21_meta：元数据构建纯函数
# ──────────────────────────────────────────────────────────────────────────────

def build_info_json(
    robot_type: str,
    fps: int,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    cam_specs: dict,  # {cam_name: (H, W, C)}
    action_names: list,
    state_names: list,
) -> dict:
    """构建 meta/info.json 字典，对标 realman v2.1 参考集结构。

    Args:
        robot_type: 机器人类型，例如 "franka"
        fps: 帧率
        total_episodes: episode 总数
        total_frames: 帧总数
        total_tasks: task 总数
        cam_specs: 各相机规格 {cam_name: (H, W, C)}
        action_names: action 特征名列表（7D）
        state_names: observation.state 特征名列表（14D）

    Returns:
        符合 lerobot v2.1 schema 的 info.json 字典
    """
    # 构建 features dict
    features = {}

    # observation.state：数值向量 block
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [len(state_names)],
        "names": state_names,
    }

    # action：数值向量 block
    features["action"] = {
        "dtype": "float32",
        "shape": [len(action_names)],
        "names": action_names,
    }

    # 每相机 video block（含完整 info 子 dict）
    for cam_name, (h, w, c) in cam_specs.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [h, w, c],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": h,
                "video.width": w,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": c,
                "has_audio": False,
            },
        }

    # 5 元列 block（timestamp float32，其余 int64，shape=[1], names=None）
    for col, dtype in [
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ]:
        features[col] = {
            "dtype": dtype,
            "shape": [1],
            "names": None,
        }

    return {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(cam_specs),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def build_tasks_jsonl(tasks: list) -> list:
    """构建 tasks.jsonl 行列表。

    Args:
        tasks: task 字符串列表

    Returns:
        每行 {"task_index": int, "task": str} 的列表
    """
    return [{"task_index": i, "task": t} for i, t in enumerate(tasks)]


def build_episodes_jsonl(episodes: list) -> list:
    """构建 episodes.jsonl 行列表。

    Args:
        episodes: [(episode_index, [task_str, ...], length), ...] 列表

    Returns:
        每行 {"episode_index": int, "tasks": [str], "length": int} 的列表
    """
    return [
        {"episode_index": ep_idx, "tasks": tasks, "length": length}
        for ep_idx, tasks, length in episodes
    ]


def compute_episode_stats(feature_arrays: dict) -> dict:
    """计算 episode 各 feature 的统计值（沿 axis=0）。

    Args:
        feature_arrays: {feature_name: np.ndarray shape (N, D)} 的字典

    Returns:
        {feature_name: {"min": [...], "max": [...], "mean": [...], "std": [...], "count": [N]}}
    """
    stats = {}
    for key, arr in feature_arrays.items():
        arr = np.asarray(arr, dtype=np.float64)
        stats[key] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(arr.shape[0])],
        }
    return stats


def write_meta_files(out_dir: Path, info: dict, tasks: list, episodes: list, stats: list) -> None:
    """将 4 个 meta 文件写入 out_dir/meta/。

    Args:
        out_dir: 输出根目录
        info: info.json 字典
        tasks: tasks.jsonl 行列表（每行一个 dict）
        episodes: episodes.jsonl 行列表（每行一个 dict）
        stats: episodes_stats.jsonl 行列表（每行一个 dict）
    """
    meta_dir = Path(out_dir) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # info.json（indent=4）
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")

    # tasks.jsonl（每行 json.dumps）
    (meta_dir / "tasks.jsonl").write_text(
        "\n".join(json.dumps(row) for row in tasks) + "\n", encoding="utf-8"
    )

    # episodes.jsonl（每行 json.dumps）
    (meta_dir / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in episodes) + "\n", encoding="utf-8"
    )

    # episodes_stats.jsonl（每行 json.dumps）
    (meta_dir / "episodes_stats.jsonl").write_text(
        "\n".join(json.dumps(row) for row in stats) + "\n", encoding="utf-8"
    )
