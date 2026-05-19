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
import glob
import logging
import os
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# 自举：确保 <repo>/scripts 和 repo 根在 sys.path 中
_REPO_SCRIPTS = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_REPO_SCRIPTS))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from franka_hdf5_schema import validate_episode
from tools.hdf5_lerobot_map import ACTION_KEYS, OBS_STATE_KEYS, hdf5_frame_to_lerobot

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


# ──────────────────────────────────────────────────────────────────────────────
# v21_video：episode 视频导出（ffmpeg libx264 yuv420p，同 realman 实测参数）
# ──────────────────────────────────────────────────────────────────────────────

def episode_to_videos(
    h5_path: str,
    out_dir: str,
    episode_chunk: int,
    episode_index: int,
    cam_names: list,
    fps: int,
) -> dict:
    """将 franka-hdf5-v1 episode 各相机图像编码为 h264 mp4 文件。

    与 map._decode 通道约定一致：jpeg bytes → BGR(cv2.imdecode) → RGB(cvtColor)
    以 rgb24 喂 ffmpeg（流式逐帧写 stdin），产 yuv420p h264（同 realman 实测参数）。
    规避 SVT-AV1 <32px 崩溃：始终用 libx264。
    原子写：先输出 <name>.mp4.tmp，成功后 os.replace 为最终 mp4。

    Args:
        h5_path: franka-hdf5-v1 文件路径
        out_dir: 输出根目录
        episode_chunk: chunk 索引（用于路径模板）
        episode_index: episode 索引（用于路径模板）
        cam_names: 相机名列表，对应 hdf5 中 observations/camera/rgb/{cam}/images
        fps: 帧率

    Returns:
        {cam_name: Path} — 每相机 mp4 输出路径 dict
    """
    out_paths = {}

    with h5py.File(h5_path, "r") as h5:
        for cam_name in cam_names:
            ds = h5[f"observations/camera/rgb/{cam_name}/images"]
            n_frames = ds.shape[0]
            if n_frames == 0:
                raise ValueError(f"episode_to_videos: {cam_name} 0 帧，无法编码")

            # 第 0 帧确定 H, W，同时校验解码成功
            frame0_bytes = bytes(ds[0])
            frame0_bgr = cv2.imdecode(np.frombuffer(frame0_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame0_bgr is None:
                raise ValueError(f"episode_to_videos: {cam_name} 第 0 帧 jpeg 解码失败")
            H, W = frame0_bgr.shape[:2]

            # 准备输出目录和路径（先写 tmp，成功后原子改名）
            vid_dir = (
                Path(out_dir)
                / "videos"
                / f"chunk-{episode_chunk:03d}"
                / f"observation.images.{cam_name}"
            )
            vid_dir.mkdir(parents=True, exist_ok=True)
            out_mp4 = vid_dir / f"episode_{episode_index:06d}.mp4"
            tmp_mp4 = vid_dir / f"episode_{episode_index:06d}.mp4.tmp"

            # ffmpeg 命令：读 rgb24 stdin，输出 libx264 yuv420p mp4（写 tmp）
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{W}x{H}",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-r", str(fps),
                "-an",
                "-f", "mp4",
                str(tmp_mp4),
            ]

            # 流式写 stdin：逐帧 jpeg → BGR → RGB → proc.stdin，避免全量缓冲
            timeout_s = max(120, n_frames * 2)
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except FileNotFoundError:
                raise RuntimeError("episode_to_videos: 系统未找到 ffmpeg, 请确认已安装且在 PATH")

            try:
                # 第 0 帧已解码，转 RGB 后写入
                frame0_rgb = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2RGB)
                proc.stdin.write(frame0_rgb.tobytes())

                for i in range(1, n_frames):
                    buf = bytes(ds[i])
                    bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise ValueError(f"episode_to_videos: {cam_name} 第 {i} 帧 jpeg 解码失败")
                    if bgr.shape[:2] != (H, W):
                        raise ValueError(
                            f"episode_to_videos: {cam_name} 第 {i} 帧尺寸 {bgr.shape[:2]} != 首帧 {(H, W)}"
                        )
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    proc.stdin.write(rgb.tobytes())

                proc.stdin.close()
                err = proc.stderr.read()
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    raise RuntimeError(f"episode_to_videos: ffmpeg 超时（{cam_name}，{timeout_s}s）")

            except BrokenPipeError:
                # ffmpeg 提前退出（如格式错误），读 stderr 后抛出
                err = proc.stderr.read()
                proc.wait()
                raise RuntimeError(
                    f"episode_to_videos: ffmpeg 提前退出（{cam_name}）: "
                    f"{err.decode(errors='replace')[-2000:]}"
                )
            except Exception:
                # 任何异常：确保 proc 终止，删 tmp，向上抛
                try:
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass
                tmp_mp4.unlink(missing_ok=True)
                raise

            if proc.returncode != 0:
                tmp_mp4.unlink(missing_ok=True)
                raise RuntimeError(
                    f"episode_to_videos: ffmpeg 编码失败（{cam_name}）: "
                    f"{err.decode(errors='replace')[-2000:]}"
                )

            # 原子改名：tmp → 最终 mp4
            os.replace(tmp_mp4, out_mp4)
            out_paths[cam_name] = out_mp4

    return out_paths


# ──────────────────────────────────────────────────────────────────────────────
# convert：顶层转换逻辑（串联 meta / parquet / video）
# ──────────────────────────────────────────────────────────────────────────────

def convert(
    in_dir,
    out,
    fps: float = 30.0,
    task: str = "task",
    robot_type: str = "franka",
    state_layout: str = "native",
) -> None:
    """将 in_dir 下所有 franka-hdf5-v1 episode 转换为 lerobot v2.1 数据集。

    不合规 episode（validate_episode 返回非空 violations）预校验跳过，不分配输出索引。
    通过校验但处理失败的 episode（parquet/video 异常）则 fail-loud 中止整个 convert。

    Args:
        in_dir: 含 .h5 文件的目录（Path 或 str）
        out: 输出根目录（Path 或 str）
        fps: 帧率（必须为整数，用于 parquet timestamp 与 info.json）
        task: 任务描述字符串（所有 episode 共用）
        robot_type: 机器人类型（写入 info.json robot_type 字段）
        state_layout: "native" 或 "realman"（仅影响 state 列序/命名，action 恒 7D 不变）

    Raises:
        ValueError: fps 非整数；in_dir 中无 .h5 文件；所有 episode 均不合规
    """
    # fps 整数校验（realman v2.1 约定）
    if float(fps) != int(fps):
        raise ValueError(f"convert: fps 必须为整数(realman v2.1 约定), got {fps}")
    fps = int(fps)

    in_dir = Path(in_dir)
    out = Path(out)

    # 排序保证 episode 顺序确定性
    h5_files = sorted(glob.glob(str(in_dir / "*.h5")))
    if not h5_files:
        raise ValueError(f"no hdf5 in {in_dir}")

    # ── state/action names（决定 info.json 中 features names 字段）──
    if state_layout == "realman":
        # 对标 realman info.json 命名规范
        state_names = (
            [f"joint_{i+1}_rad" for i in range(7)]
            + ["gripper_open"]
            + ["eef_pos_x", "eef_pos_y", "eef_pos_z"]
            + ["eef_rot_euler_x", "eef_rot_euler_y", "eef_rot_euler_z"]
        )
    else:
        # native：直接用 OBS_STATE_KEYS（human-readable 短名）
        state_names = list(OBS_STATE_KEYS)

    # action_names 恒由 ACTION_KEYS 衍生，与 state_layout 无关
    action_names = list(ACTION_KEYS)

    # ── 创建输出目录结构 ──
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # ── 逐 episode 转换 ──
    episodes_rows = []   # [(out_ei, [task], N), ...]
    stats_rows = []      # [{"episode_index": out_ei, "stats": {...}}, ...]
    total_frames = 0
    index_base = 0       # 全局帧索引累加器
    out_ei = 0           # 输出 episode 索引（仅对通过校验+成功处理的 episode 递增）

    # 首个通过校验的 episode 确定相机名与尺寸（用于后续一致性校验）
    cam_names = None
    H = W = None
    cam_specs = None

    for src_idx, h5_path in enumerate(h5_files):
        # ── 预校验：不合规直接跳过，不分配 out 索引 ──
        violations = validate_episode(h5_path)
        if violations:
            logging.warning(
                f"convert: 跳过不合规 episode {h5_path} "
                f"(violations: {violations})"
            )
            continue

        # ── 首个合规 episode：确定相机名与图像尺寸 ──
        if cam_names is None:
            with h5py.File(h5_path, "r") as h5:
                cam_names = sorted(h5["observations/camera/rgb"].keys())
                first_bytes = bytes(h5[f"observations/camera/rgb/{cam_names[0]}/images"][0])
                frame0_bgr = cv2.imdecode(np.frombuffer(first_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame0_bgr is None:
                    raise ValueError(
                        f"convert: 首 episode 首帧 jpeg 解码失败 ({h5_path})"
                    )
                H, W = frame0_bgr.shape[:2]
            cam_specs = {cam: (H, W, 3) for cam in cam_names}
        else:
            # ── 后续 episode：校验相机名与尺寸一致性 ──
            with h5py.File(h5_path, "r") as h5:
                ep_cams = sorted(h5["observations/camera/rgb"].keys())
                first_bytes = bytes(h5[f"observations/camera/rgb/{ep_cams[0]}/images"][0])
                frame_bgr = cv2.imdecode(np.frombuffer(first_bytes, np.uint8), cv2.IMREAD_COLOR)
            if ep_cams != cam_names or (frame_bgr is not None and frame_bgr.shape[:2] != (H, W)):
                h2, w2 = (frame_bgr.shape[:2] if frame_bgr is not None else (None, None))
                raise ValueError(
                    f"convert: {h5_path} 相机/尺寸与首 episode 不一致: "
                    f"cams={ep_cams} hw={(h2, w2)} vs {cam_names}/{(H, W)}"
                )

        out_parquet = out / "data" / "chunk-000" / f"episode_{out_ei:06d}.parquet"

        # parquet 转换（fail-loud：通过校验的 episode 处理失败属 bug/数据损坏）
        N, stat_arrays = episode_to_parquet(
            h5_path=h5_path,
            out_parquet=str(out_parquet),
            episode_index=out_ei,
            task_index=0,
            fps=fps,
            cam_names=cam_names,
            task=task,
            index_base=index_base,
            state_layout=state_layout,
        )

        # 视频导出（fail-loud）
        episode_to_videos(
            h5_path=h5_path,
            out_dir=str(out),
            episode_chunk=0,
            episode_index=out_ei,
            cam_names=cam_names,
            fps=fps,
        )

        # 收集元数据
        episodes_rows.append((out_ei, [task], N))
        stats_rows.append({
            "episode_index": out_ei,
            "stats": compute_episode_stats(stat_arrays),
        })
        total_frames += N
        index_base += N
        out_ei += 1

    if out_ei == 0:
        raise ValueError("convert: 无有效 episode(全部未通过 schema 校验)")

    total_episodes = len(episodes_rows)

    # ── 写 meta 文件 ──
    info = build_info_json(
        robot_type=robot_type,
        fps=fps,
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=1,
        cam_specs=cam_specs,
        action_names=action_names,
        state_names=state_names,
    )
    tasks_rows = build_tasks_jsonl([task])
    episodes_built = build_episodes_jsonl(episodes_rows)
    write_meta_files(out, info, tasks_rows, episodes_built, stats_rows)


# ──────────────────────────────────────────────────────────────────────────────
# main：CLI 入口
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI 入口：解析参数并调用 convert()。"""
    import argparse
    from core import paths

    parser = argparse.ArgumentParser(
        description="franka-hdf5-v1 → lerobot v2.1 转换器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--in-dir",
        default=paths.HDF5_EPISODES_DIR,
        help="含 .h5 文件的目录（默认: core.paths.HDF5_EPISODES_DIR）",
    )
    parser.add_argument(
        "--out",
        default=paths.LEROBOT_OUT,
        help="输出根目录（默认: core.paths.LEROBOT_OUT）",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="帧率")
    parser.add_argument("--task", default="task", help="任务描述字符串")
    parser.add_argument("--robot-type", default="franka", help="机器人类型（写入 info.json）")
    parser.add_argument(
        "--state-layout",
        choices=["native", "realman"],
        default="native",
        help="observation.state 列序/命名（native=OBS_STATE_KEYS 原序，realman=对标 realman 命名）",
    )

    args = parser.parse_args()
    convert(
        in_dir=args.in_dir,
        out=args.out,
        fps=args.fps,
        task=args.task,
        robot_type=args.robot_type,
        state_layout=args.state_layout,
    )


if __name__ == "__main__":
    main()
