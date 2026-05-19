r"""tests/test_v21_structure_diff.py — Task5 结构 diff 守门测试。

合成 2 个合规 franka-hdf5-v1 episode（帧数不等，2 相机），调 convert()，
断言产物与 realman v2.1 结构无缺项。

覆盖：
  - meta/info.json 顶层键 ⊇ realman 必需键集；codebase_version=="v2.1"
  - features 中 state/action/video block/元列 结构同构（video info 子键集 == realman）
  - data_path/video_path 字面值与 realman 逐字一致
  - tasks.jsonl / episodes.jsonl / episodes_stats.jsonl 每行键集 == realman 对应
  - episodes_stats.jsonl 每数值 feature stats 含 {min,max,mean,std,count}，
    长度 == feature shape[0]，count == [N]
  - 目录命名严格正则：data/chunk-\d{3}/episode_\d{6}.parquet
                      videos/chunk-\d{3}/observation\.images\.[\w]+/episode_\d{6}\.mp4
"""
import json
import re
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

# conftest 已把 <repo>/scripts 入 sys.path
from tools.hdf5_to_lerobot_v21 import convert

# ──────────────────────────────────────────────────────────────────────────────
# 合成 franka-hdf5-v1 生成器（完整合规，与 test_v21_cli.py 同源）
# ──────────────────────────────────────────────────────────────────────────────

def _make_hdf5(h5_path: Path, n_frames: int, cams=("exterior_image", "wrist_image"), img_hw=(16, 24)):
    """生成完整合规 franka-hdf5-v1 文件。"""
    import franka_hdf5_schema as S

    H, W = img_hw
    img_bgr = np.zeros((H, W, 3), np.uint8)
    ok, enc = cv2.imencode(".jpg", img_bgr)
    assert ok
    jb = np.frombuffer(enc.tobytes(), np.uint8)
    vlen_u8 = h5py.special_dtype(vlen=np.dtype("uint8"))

    with h5py.File(h5_path, "w") as f:
        # infos 组（schema 校验需要）
        inf = f.create_group("infos")
        inf.create_dataset("schema_version", data=np.bytes_(S.SCHEMA_VERSION))
        ti = inf.create_group("task_info")
        ti.create_dataset("task_name", data=np.bytes_("pick"))
        ti.create_dataset("collection_frequency", data=np.array([30.0, 30.0], np.float64))
        ti.create_dataset("total_frames", data=np.int64(n_frames))
        ti.create_dataset("robot", data=np.bytes_("franka_panda"))
        inf.create_group("camera_params")
        cal = inf.create_group("calibration")
        cal.create_dataset("oc2base_R", data=np.eye(3, dtype=np.float64))
        cal.create_dataset("quality", data=np.bytes_("{}"))
        cal.create_dataset("vr_source", data=np.bytes_("unity"))

        obs = f.create_group("observations")
        obs.create_dataset("timestamp", data=(np.arange(n_frames, dtype=np.float64) + 1).reshape(n_frames, 1))
        arm = obs.create_group("arm")
        arm.create_dataset("joints", data=np.arange(n_frames * 7, dtype=np.float64).reshape(n_frames, 7) * 0.1)
        arm.create_dataset("joint_vel", data=np.zeros((n_frames, 7), np.float64))
        arm.create_dataset("pose", data=np.arange(n_frames * 6, dtype=np.float64).reshape(n_frames, 6) * 0.2)
        arm.create_dataset("timestamp", data=np.arange(n_frames, dtype=np.float64))
        eff = obs.create_group("effector")
        eff.create_dataset("position", data=np.zeros((n_frames, 1), np.float64))
        eff.create_dataset("position_norm",
                           data=(np.arange(n_frames, dtype=np.float64) * 0.1 + 0.5).reshape(n_frames, 1))
        eff.create_dataset("type", data=np.array([b"gripper"] * n_frames, dtype=h5py.special_dtype(vlen=bytes)))
        eff.create_dataset("timestamp", data=np.arange(n_frames, dtype=np.float64))

        cam_grp = obs.create_group("camera")
        rgb = cam_grp.create_group("rgb")
        for c in cams:
            g = rgb.create_group(c)
            d = g.create_dataset("images", (n_frames,), dtype=vlen_u8)
            for i in range(n_frames):
                d[i] = jb
            g.create_dataset("timestamp", data=np.arange(n_frames, dtype=np.float64))

        hf = obs.create_group("state_hifreq")
        for k, sh in [("joints", (0, 7)), ("joint_vel", (0, 7)), ("pose", (0, 6)),
                      ("timestamp", (0,)), ("poly_ts", (0,))]:
            hf.create_dataset(k, data=np.zeros(sh, np.float64))

        act = f.create_group("action")
        act.create_dataset("delta_ee_pose",
                           data=np.arange(n_frames * 6, dtype=np.float64).reshape(n_frames, 6) * 0.05)
        act.create_dataset("gripper_cmd",
                           data=(np.arange(n_frames, dtype=np.float64) * 0.25).reshape(n_frames, 1))
        act.create_dataset("timestamp", data=np.arange(n_frames, dtype=np.float64))


# ──────────────────────────────────────────────────────────────────────────────
# 期望集：来自【已确证事实】中的 realman v2.1 字段清单
# ──────────────────────────────────────────────────────────────────────────────

# info.json 顶层必需键
_INFO_REQUIRED_KEYS = {
    "codebase_version", "robot_type", "total_episodes", "total_frames",
    "total_tasks", "total_videos", "total_chunks", "chunks_size", "fps",
    "splits", "data_path", "video_path", "features",
}

# data_path / video_path realman 字面值
_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

# video info 必需子键集（== realman）
_VIDEO_INFO_KEYS = {
    "video.height", "video.width", "video.codec", "video.pix_fmt",
    "video.is_depth_map", "video.fps", "video.channels", "has_audio",
}

# 数值 feature block 必需键（state / action）
_NUMERIC_FEATURE_KEYS = {"dtype", "shape", "names"}

# 元列名称集
_META_COLS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}

# tasks.jsonl 每行必需键
_TASKS_ROW_KEYS = {"task_index", "task"}

# episodes.jsonl 每行必需键
_EPISODES_ROW_KEYS = {"episode_index", "tasks", "length"}

# episodes_stats.jsonl 每行必需键
_STATS_ROW_KEYS = {"episode_index", "stats"}

# 每数值 feature 统计必需键
_STATS_FEATURE_KEYS = {"min", "max", "mean", "std", "count"}

# ──────────────────────────────────────────────────────────────────────────────
# fixture：合成 2 个 ep（帧数不等）→ convert → 返回 out_dir
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def out_dir():
    """合成 2 个合规 hdf5（N=3, N=5），调 convert，返回产物根目录。"""
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = Path(tmp) / "in"
        out = Path(tmp) / "out"
        in_dir.mkdir()
        _make_hdf5(in_dir / "ep000.h5", n_frames=3, img_hw=(16, 24))
        _make_hdf5(in_dir / "ep001.h5", n_frames=5, img_hw=(16, 24))
        convert(in_dir=in_dir, out=out, fps=30, task="pick test",
                robot_type="franka", state_layout="native")
        yield out


# ──────────────────────────────────────────────────────────────────────────────
# 测试：info.json 结构 diff
# ──────────────────────────────────────────────────────────────────────────────

def test_info_top_level_keys(out_dir):
    """info.json 顶层键集合 ⊇ realman 必需键集。"""
    info = json.loads((out_dir / "meta" / "info.json").read_text())
    missing = _INFO_REQUIRED_KEYS - set(info.keys())
    assert not missing, f"info.json 缺顶层键: {missing}"


def test_info_codebase_version(out_dir):
    """codebase_version 必须精确为 'v2.1'。"""
    info = json.loads((out_dir / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v2.1", f"codebase_version={info['codebase_version']!r}"


def test_info_data_video_path(out_dir):
    """data_path / video_path 字面值与 realman 逐字一致。"""
    info = json.loads((out_dir / "meta" / "info.json").read_text())
    assert info["data_path"] == _DATA_PATH, f"data_path={info['data_path']!r}"
    assert info["video_path"] == _VIDEO_PATH, f"video_path={info['video_path']!r}"


def test_info_features_numeric_blocks(out_dir):
    """features 中 observation.state / action 为 {dtype,shape,names} 结构。"""
    info = json.loads((out_dir / "meta" / "info.json").read_text())
    feats = info["features"]
    for key in ("observation.state", "action"):
        assert key in feats, f"features 缺 {key}"
        block = feats[key]
        missing = _NUMERIC_FEATURE_KEYS - set(block.keys())
        assert not missing, f"features.{key} 缺键: {missing}"
        assert block["dtype"] == "float32", f"features.{key}.dtype={block['dtype']!r}"
        assert isinstance(block["shape"], list) and len(block["shape"]) == 1, \
            f"features.{key}.shape 应为长度1列表，实为 {block['shape']}"
        assert isinstance(block["names"], list) and len(block["names"]) == block["shape"][0], \
            f"features.{key}.names 长度应=={block['shape'][0]}"


def test_info_features_video_blocks(out_dir):
    """每相机 video feature block dtype=='video'，info 子键集 == realman。"""
    info = json.loads((out_dir / "meta" / "info.json").read_text())
    feats = info["features"]
    cam_keys = [k for k in feats if k.startswith("observation.images.")]
    assert len(cam_keys) == 2, f"期望 2 个相机 feature，实得: {cam_keys}"
    for key in cam_keys:
        block = feats[key]
        assert block["dtype"] == "video", f"{key}.dtype={block['dtype']!r}"
        assert isinstance(block["shape"], list) and len(block["shape"]) == 3, \
            f"{key}.shape 应为 [H,W,C] 3元素，实为 {block['shape']}"
        assert block["names"] == ["height", "width", "channels"], \
            f"{key}.names={block['names']!r}"
        assert "info" in block, f"{key} 缺 info 子键"
        info_keys = set(block["info"].keys())
        missing = _VIDEO_INFO_KEYS - info_keys
        extra = info_keys - _VIDEO_INFO_KEYS
        assert not missing, f"{key}.info 缺键: {missing}"
        assert not extra, f"{key}.info 多余键: {extra}"
        assert block["info"]["video.codec"] == "h264", f"{key}.info.video.codec={block['info']['video.codec']!r}"
        assert block["info"]["video.pix_fmt"] == "yuv420p", \
            f"{key}.info.video.pix_fmt={block['info']['video.pix_fmt']!r}"
        assert block["info"]["video.is_depth_map"] is False
        assert block["info"]["has_audio"] is False


def test_info_features_meta_cols(out_dir):
    """5 元列 block 为 {dtype,shape:[1],names:null}。"""
    info = json.loads((out_dir / "meta" / "info.json").read_text())
    feats = info["features"]
    for col in _META_COLS:
        assert col in feats, f"features 缺元列 {col}"
        block = feats[col]
        assert block["shape"] == [1], f"features.{col}.shape={block['shape']!r}"
        assert block["names"] is None, f"features.{col}.names={block['names']!r}"


# ──────────────────────────────────────────────────────────────────────────────
# 测试：tasks.jsonl 结构 diff
# ──────────────────────────────────────────────────────────────────────────────

def test_tasks_jsonl_row_keys(out_dir):
    """tasks.jsonl 每行键集 == {task_index, task}。"""
    lines = (out_dir / "meta" / "tasks.jsonl").read_text().strip().splitlines()
    assert lines, "tasks.jsonl 为空"
    for i, line in enumerate(lines):
        row = json.loads(line)
        assert set(row.keys()) == _TASKS_ROW_KEYS, \
            f"tasks.jsonl 第 {i} 行键集 {set(row.keys())} != {_TASKS_ROW_KEYS}"


# ──────────────────────────────────────────────────────────────────────────────
# 测试：episodes.jsonl 结构 diff
# ──────────────────────────────────────────────────────────────────────────────

def test_episodes_jsonl_row_keys(out_dir):
    """episodes.jsonl 每行键集 == {episode_index, tasks, length}。"""
    lines = (out_dir / "meta" / "episodes.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2, f"期望 2 行 episodes.jsonl，实得 {len(lines)}"
    for i, line in enumerate(lines):
        row = json.loads(line)
        assert set(row.keys()) == _EPISODES_ROW_KEYS, \
            f"episodes.jsonl 第 {i} 行键集 {set(row.keys())} != {_EPISODES_ROW_KEYS}"


# ──────────────────────────────────────────────────────────────────────────────
# 测试：episodes_stats.jsonl 结构 diff
# ──────────────────────────────────────────────────────────────────────────────

def test_stats_jsonl_row_keys(out_dir):
    """episodes_stats.jsonl 每行键集 == {episode_index, stats}。"""
    lines = (out_dir / "meta" / "episodes_stats.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2, f"期望 2 行 episodes_stats.jsonl，实得 {len(lines)}"
    for i, line in enumerate(lines):
        row = json.loads(line)
        assert set(row.keys()) == _STATS_ROW_KEYS, \
            f"episodes_stats.jsonl 第 {i} 行键集 {set(row.keys())} != {_STATS_ROW_KEYS}"


def test_stats_feature_keys(out_dir):
    """每数值 feature stats 含 {min,max,mean,std,count}。"""
    lines = (out_dir / "meta" / "episodes_stats.jsonl").read_text().strip().splitlines()
    for i, line in enumerate(lines):
        row = json.loads(line)
        stats = row["stats"]
        for feat_key, feat_stats in stats.items():
            missing = _STATS_FEATURE_KEYS - set(feat_stats.keys())
            assert not missing, \
                f"第 {i} 行 stats[{feat_key}] 缺键: {missing}"


def test_stats_shape_and_count(out_dir):
    """数值 feature stats min/max/mean/std 长度==shape[0]，count==[N]。"""
    info = json.loads((out_dir / "meta" / "info.json").read_text())
    feats = info["features"]
    ep_lengths = {3, 5}  # 合成的两个 ep

    lines = (out_dir / "meta" / "episodes_stats.jsonl").read_text().strip().splitlines()
    for i, line in enumerate(lines):
        row = json.loads(line)
        stats = row["stats"]
        # 取本 ep 帧数（episodes.jsonl 对应行）
        ep_line = json.loads((out_dir / "meta" / "episodes.jsonl")
                             .read_text().strip().splitlines()[i])
        n = ep_line["length"]

        for feat_key, feat_stats in stats.items():
            if feat_key not in feats:
                continue
            block = feats[feat_key]
            if block["dtype"] in ("video",):
                continue  # 视频列不入 stats
            dim = block["shape"][0]
            for stat_name in ("min", "max", "mean", "std"):
                val = feat_stats[stat_name]
                assert isinstance(val, list) and len(val) == dim, \
                    f"第 {i} 行 stats[{feat_key}][{stat_name}] 长度应=={dim}，实为 {len(val) if isinstance(val,list) else type(val)}"
            count = feat_stats["count"]
            assert count == [n], \
                f"第 {i} 行 stats[{feat_key}].count 应==[{n}]，实为 {count}"


# ──────────────────────────────────────────────────────────────────────────────
# 测试：目录命名正则
# ──────────────────────────────────────────────────────────────────────────────

def test_data_file_naming(out_dir):
    r"""data 文件路径严格匹配 data/chunk-\d{3}/episode_\d{6}.parquet。"""
    data_files = list(out_dir.rglob("*.parquet"))
    assert data_files, "没有找到 parquet 文件"
    pat = re.compile(r"data/chunk-\d{3}/episode_\d{6}\.parquet")
    for f in data_files:
        rel = f.relative_to(out_dir).as_posix()
        assert pat.fullmatch(rel), f"data 文件路径不匹配正则: {rel!r}"


def test_video_file_naming(out_dir):
    r"""video 文件路径严格匹配 videos/chunk-\d{3}/observation\.images\.[\w]+/episode_\d{6}\.mp4。"""
    video_files = list(out_dir.rglob("*.mp4"))
    assert video_files, "没有找到 mp4 文件"
    pat = re.compile(r"videos/chunk-\d{3}/observation\.images\.[\w]+/episode_\d{6}\.mp4")
    for f in video_files:
        rel = f.relative_to(out_dir).as_posix()
        assert pat.fullmatch(rel), f"video 文件路径不匹配正则: {rel!r}"
