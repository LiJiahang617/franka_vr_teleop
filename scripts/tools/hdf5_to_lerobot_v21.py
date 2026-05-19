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
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# 自举：确保 <repo>/scripts 在 sys.path 中（同 v3.0 模式）
_REPO_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_REPO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_REPO_SCRIPTS))

from tools.hdf5_lerobot_map import OBS_STATE_KEYS, hdf5_frame_to_lerobot

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

    chunks_size = 1000
    return {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(cam_specs),
        "total_chunks": max(1, math.ceil(total_episodes / chunks_size)),
        "chunks_size": chunks_size,
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
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] == 0:
            raise ValueError(f"compute_episode_stats: feature {key!r} 空数组(0 帧)无法统计")
        stats[key] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(arr.shape[0])],
        }
    return stats


def write_meta_files(out_dir: Path, info: dict, tasks: list, episodes: list, stats_rows: list) -> None:
    """将 4 个 meta 文件写入 out_dir/meta/。

    Args:
        out_dir: 输出根目录
        info: info.json 字典
        tasks: tasks.jsonl 行列表（每行一个 dict）
        episodes: episodes.jsonl 行列表（每行一个 dict）
        stats_rows: episodes_stats.jsonl 行列表，每行 {'episode_index': int, 'stats':
            {feature: {min/max/mean/std/count}}}（由 convert 用 compute_episode_stats
            结果包装，**勿直接传 compute_episode_stats 返回的 dict**）
    """
    meta_dir = Path(out_dir) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # info.json（indent=4）
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4, allow_nan=False), encoding="utf-8")

    # tasks.jsonl（每行 json.dumps）
    (meta_dir / "tasks.jsonl").write_text(
        "\n".join(json.dumps(row, allow_nan=False) for row in tasks) + "\n", encoding="utf-8"
    )

    # episodes.jsonl（每行 json.dumps）
    (meta_dir / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row, allow_nan=False) for row in episodes) + "\n", encoding="utf-8"
    )

    # episodes_stats.jsonl（每行 json.dumps）
    (meta_dir / "episodes_stats.jsonl").write_text(
        "\n".join(json.dumps(row, allow_nan=False) for row in stats_rows) + "\n", encoding="utf-8"
    )


# ──────────────────────────────────────────────────────────────────────────────
# v21_parquet：episode 帧写入（复用 hdf5_lerobot_map，无图像列）
# ──────────────────────────────────────────────────────────────────────────────

# realman state 重排索引（由 OBS_STATE_KEYS 推导，防止 key 序漂移）
# native layout: joints(0-6), ee_pose(7-12), gripper_norm(13)
# realman layout: joints(0-6), gripper_norm(13→7), ee_pose(7-12→8-13)
_REALMAN_STATE_ORDER = (
    [f"joint_{i+1}.pos" for i in range(7)]
    + ["gripper_norm"]
    + [f"ee_pose.{a}" for a in ("x", "y", "z", "rx", "ry", "rz")]
)
_REALMAN_STATE_IDX = [OBS_STATE_KEYS.index(k) for k in _REALMAN_STATE_ORDER]
assert sorted(_REALMAN_STATE_IDX) == list(range(14)), "置换不合法，OBS_STATE_KEYS 结构已变动"  # 自检


def episode_to_parquet(
    h5_path: str,
    out_parquet: str,
    episode_index: int,
    task_index: int,
    fps: float,
    cam_names: list,
    task: str,
    index_base: int,
    state_layout: str = "native",
) -> tuple:
    """将 franka-hdf5-v1 episode 写为 lerobot v2.1 parquet。

    复用 hdf5_lerobot_map.hdf5_frame_to_lerobot 取帧（图像丢弃，仅 state/action 入列）。
    parquet 列（精确顺序）：
      observation.state  fixed_size_list<float32>[14]
      action             fixed_size_list<float32>[7]
      timestamp          float32，row0=0.0，i/fps
      frame_index        int64
      episode_index      int64
      index              int64（index_base + i 全局连续）
      task_index         int64

    Args:
        h5_path: franka-hdf5-v1 文件路径
        out_parquet: 输出 parquet 路径（父目录需已存在）
        episode_index: episode 索引
        task_index: task 索引
        fps: 帧率，用于计算 timestamp
        cam_names: 相机名列表（用于读帧，图像不入 parquet）
        task: task 字符串（传给 hdf5_frame_to_lerobot）
        index_base: 全局帧起始偏移（跨 episode 连续）
        state_layout: "native"（默认）或 "realman"（仅重排 observation.state 列序）

    Returns:
        (N, stat_arrays)：N 为帧数；stat_arrays = {feature: np.ndarray (N, D)}，
        供 compute_episode_stats 使用。observation.state (N,14), action (N,7),
        其余元列均 (N,1)。
    """
    if state_layout not in ("native", "realman"):
        raise ValueError(f"state_layout 必须为 'native' 或 'realman'，实际: {state_layout!r}")

    with h5py.File(h5_path, "r") as h5:
        N = int(h5["observations/arm/joints"].shape[0])
        if N == 0:
            raise ValueError(f"episode_to_parquet: {h5_path} 0 帧，无法转换")

        states = []
        actions = []
        timestamps = []
        frame_indices = []
        episode_indices = []
        indices = []
        task_indices = []

        for i in range(N):
            frame = hdf5_frame_to_lerobot(h5, i, cam_names=cam_names, task=task)

            # state 重排（action 不经此分支，保持 7D 原样）
            state = frame["observation.state"]  # np.float32 (14,)
            if state_layout == "realman":
                state = state[_REALMAN_STATE_IDX]

            action = frame["action"]  # np.float32 (7,)

            states.append(state.tolist())
            actions.append(action.tolist())
            timestamps.append(np.float32(i / fps).item())  # 故意 float32 对齐 realman timestamp dtype
            frame_indices.append(i)
            episode_indices.append(episode_index)
            indices.append(index_base + i)
            task_indices.append(task_index)

    # pyarrow fixed_size_list 类型
    fsl14 = pa.list_(pa.float32(), 14)
    fsl7 = pa.list_(pa.float32(), 7)

    # 严格按列名顺序构建 schema（对标 realman parquet 实测）
    schema = pa.schema([
        pa.field("observation.state", fsl14),
        pa.field("action", fsl7),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
    ])

    table = pa.table(
        {
            "observation.state": pa.array(states, type=fsl14),
            "action": pa.array(actions, type=fsl7),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index": pa.array(indices, type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
        },
        schema=schema,
    )

    pq.write_table(table, out_parquet)

    # 构建 stat_arrays（2D，供 compute_episode_stats）
    states_np = np.array(states, dtype=np.float32)    # (N, 14)
    actions_np = np.array(actions, dtype=np.float32)  # (N, 7)
    stat_arrays = {
        "observation.state": states_np,
        "action": actions_np,
        "timestamp": np.array(timestamps, dtype=np.float32).reshape(N, 1),
        "frame_index": np.array(frame_indices, dtype=np.int64).reshape(N, 1),
        "episode_index": np.array(episode_indices, dtype=np.int64).reshape(N, 1),
        "index": np.array(indices, dtype=np.int64).reshape(N, 1),
        "task_index": np.array(task_indices, dtype=np.int64).reshape(N, 1),
    }
    return N, stat_arrays
