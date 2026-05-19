import numpy as np
from tools.hdf5_to_lerobot_v21 import (
    build_info_json, build_tasks_jsonl, build_episodes_jsonl,
    compute_episode_stats,
)

def test_info_json_schema():
    info = build_info_json(
        robot_type="franka", fps=30,
        total_episodes=2, total_frames=10, total_tasks=1,
        cam_specs={"exterior_image": (240, 424, 3), "wrist_image": (240, 424, 3)},
        action_names=[f"a{i}" for i in range(7)],
        state_names=[f"s{i}" for i in range(14)],
    )
    assert info["codebase_version"] == "v2.1"
    assert info["robot_type"] == "franka"
    for k in ["total_episodes","total_frames","total_tasks","total_videos",
              "total_chunks","chunks_size","fps","splits","data_path",
              "video_path","features"]:
        assert k in info, f"info.json 缺顶层键 {k}"
    assert info["data_path"] == "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    assert info["video_path"] == "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    assert info["total_videos"] == 2 * 2
    assert info["splits"] == {"train": "0:2"}
    f = info["features"]
    assert f["observation.state"]["dtype"] == "float32"
    assert f["observation.state"]["shape"] == [14]
    assert len(f["observation.state"]["names"]) == 14
    assert f["action"]["dtype"] == "float32"
    assert f["action"]["shape"] == [7]
    vk = "observation.images.exterior_image"
    assert f[vk]["dtype"] == "video"
    assert f[vk]["shape"] == [240, 424, 3]
    assert f[vk]["names"] == ["height", "width", "channels"]
    vi = f[vk]["info"]
    assert vi["video.codec"] == "h264"
    assert vi["video.pix_fmt"] == "yuv420p"
    assert vi["video.fps"] == 30
    assert vi["video.is_depth_map"] is False
    assert vi["has_audio"] is False
    for mc, dt in [("timestamp","float32"),("frame_index","int64"),
                   ("episode_index","int64"),("index","int64"),
                   ("task_index","int64")]:
        assert f[mc]["dtype"] == dt
        assert f[mc]["shape"] == [1]
        assert f[mc]["names"] is None

def test_tasks_jsonl():
    assert build_tasks_jsonl(["pick up the cube"]) == [{"task_index": 0, "task": "pick up the cube"}]

def test_episodes_jsonl():
    rows = build_episodes_jsonl([(0, ["pick"], 5), (1, ["pick"], 7)])
    assert rows[0] == {"episode_index": 0, "tasks": ["pick"], "length": 5}
    assert rows[1]["length"] == 7

def test_episode_stats_shape_and_values():
    state = np.array([[1., 2.], [3., 4.], [5., 6.]], np.float32)
    s = compute_episode_stats({"observation.state": state})["observation.state"]
    assert s["min"] == [1.0, 2.0]
    assert s["max"] == [5.0, 6.0]
    assert s["mean"] == [3.0, 4.0]
    assert s["count"] == [3]
    assert len(s["std"]) == 2 and all(x >= 0 for x in s["std"])
