"""tests/test_v21_cli.py — Task4 TDD：顶层 convert + CLI 端对端测试。

合成 2 个 franka-hdf5-v1 episode（N=3, N=4），覆盖：
  - convert() 后 meta 四文件、data/parquet、videos/mp4 全部存在
  - info.json 关键字段正确
  - index 跨 episode 全局连续（ep0: 0..2, ep1: 3..6）
  - episodes_stats.jsonl 每行含 state/action stats
  - CLI subprocess 退出 0
  - argparse 默认路径来自 core.paths（FRANKA_JHLI_ROOT env override）
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

# conftest 已把 <repo>/scripts 入 sys.path
from tools.hdf5_to_lerobot_v21 import convert

# 测试用 Python 解释器路径
_PY = sys.executable

# 脚本路径（通过 sys.path 中的 scripts 目录反推）
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_SCRIPT = _SCRIPTS_DIR / "tools" / "hdf5_to_lerobot_v21.py"

# ──────────────────────────────────────────────────────────────────────────────
# 合成 franka-hdf5-v1 生成器（完整合规，含 infos/schema_version）
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

        # observations
        obs = f.create_group("observations")
        obs.create_dataset("timestamp", data=(np.arange(n_frames, dtype=np.float64) + 1).reshape(n_frames, 1))
        arm = obs.create_group("arm")
        arm.create_dataset("joints", data=np.arange(n_frames * 7, dtype=np.float64).reshape(n_frames, 7) * 0.1)
        arm.create_dataset("joint_vel", data=np.zeros((n_frames, 7), np.float64))
        arm.create_dataset("pose", data=np.arange(n_frames * 6, dtype=np.float64).reshape(n_frames, 6) * 0.2)
        arm.create_dataset("timestamp", data=np.arange(n_frames, dtype=np.float64))
        eff = obs.create_group("effector")
        eff.create_dataset("position", data=np.zeros((n_frames, 1), np.float64))
        eff.create_dataset("position_norm", data=(np.arange(n_frames, dtype=np.float64) * 0.1 + 0.5).reshape(n_frames, 1))
        eff.create_dataset("type", data=np.array([b"gripper"] * n_frames, dtype=h5py.special_dtype(vlen=bytes)))
        eff.create_dataset("timestamp", data=np.arange(n_frames, dtype=np.float64))

        # 相机图像
        cam_grp = obs.create_group("camera")
        rgb = cam_grp.create_group("rgb")
        for c in cams:
            g = rgb.create_group(c)
            d = g.create_dataset("images", (n_frames,), dtype=vlen_u8)
            for i in range(n_frames):
                d[i] = jb
            g.create_dataset("timestamp", data=np.arange(n_frames, dtype=np.float64))

        # state_hifreq（空）
        hf = obs.create_group("state_hifreq")
        for k, sh in [("joints", (0, 7)), ("joint_vel", (0, 7)), ("pose", (0, 6)),
                      ("timestamp", (0,)), ("poly_ts", (0,))]:
            hf.create_dataset(k, data=np.zeros(sh, np.float64))

        # action
        act = f.create_group("action")
        act.create_dataset("delta_ee_pose", data=np.arange(n_frames * 6, dtype=np.float64).reshape(n_frames, 6) * 0.05)
        act.create_dataset("gripper_cmd", data=(np.arange(n_frames, dtype=np.float64) * 0.25).reshape(n_frames, 1))
        act.create_dataset("timestamp", data=np.arange(n_frames, dtype=np.float64))


def _make_two_episodes(in_dir: Path, cams=("exterior_image", "wrist_image")):
    """在 in_dir 生成 2 个 episode hdf5：ep0=3帧, ep1=4帧。"""
    in_dir.mkdir(parents=True, exist_ok=True)
    _make_hdf5(in_dir / "ep_000.h5", n_frames=3, cams=cams)
    _make_hdf5(in_dir / "ep_001.h5", n_frames=4, cams=cams)


# ──────────────────────────────────────────────────────────────────────────────
# 辅助：读 jsonl 文件
# ──────────────────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Fixture
# ──────────────────────────────────────────────────────────────────────────────

CAMS = ("exterior_image", "wrist_image")
EP0_N = 3
EP1_N = 4
TOTAL_N = EP0_N + EP1_N  # 7


@pytest.fixture(scope="module")
def converted_out(tmp_path_factory):
    """调用 convert()，返回 (in_dir, out_dir)；module 级共享，只转换一次。"""
    tmp = tmp_path_factory.mktemp("cli_test")
    in_dir = tmp / "in"
    out_dir = tmp / "out"
    _make_two_episodes(in_dir, cams=CAMS)
    convert(
        in_dir=in_dir,
        out=out_dir,
        fps=30,
        task="pick",
        robot_type="franka",
        state_layout="native",
    )
    return in_dir, out_dir


# ──────────────────────────────────────────────────────────────────────────────
# Test: meta 文件存在性
# ──────────────────────────────────────────────────────────────────────────────

class TestMetaFiles:
    def test_info_json_exists(self, converted_out):
        _, out = converted_out
        assert (out / "meta" / "info.json").exists()

    def test_tasks_jsonl_exists(self, converted_out):
        _, out = converted_out
        assert (out / "meta" / "tasks.jsonl").exists()

    def test_episodes_jsonl_exists(self, converted_out):
        _, out = converted_out
        assert (out / "meta" / "episodes.jsonl").exists()

    def test_episodes_stats_jsonl_exists(self, converted_out):
        _, out = converted_out
        assert (out / "meta" / "episodes_stats.jsonl").exists()


# ──────────────────────────────────────────────────────────────────────────────
# Test: info.json 字段
# ──────────────────────────────────────────────────────────────────────────────

class TestInfoJson:
    @pytest.fixture(autouse=True)
    def _load(self, converted_out):
        _, out = converted_out
        self.info = json.loads((out / "meta" / "info.json").read_text())

    def test_codebase_version(self):
        assert self.info["codebase_version"] == "v2.1"

    def test_robot_type(self):
        assert self.info["robot_type"] == "franka"

    def test_total_episodes(self):
        assert self.info["total_episodes"] == 2

    def test_total_frames(self):
        assert self.info["total_frames"] == TOTAL_N

    def test_total_tasks(self):
        assert self.info["total_tasks"] == 1

    def test_fps(self):
        assert self.info["fps"] == 30

    def test_features_state_shape(self):
        f = self.info["features"]
        assert f["observation.state"]["shape"] == [14]
        assert f["observation.state"]["dtype"] == "float32"

    def test_features_action_shape(self):
        f = self.info["features"]
        assert f["action"]["shape"] == [7]
        assert f["action"]["dtype"] == "float32"

    def test_features_cameras(self):
        f = self.info["features"]
        for cam in CAMS:
            key = f"observation.images.{cam}"
            assert key in f, f"features 缺相机 {key}"
            assert f[key]["dtype"] == "video"


# ──────────────────────────────────────────────────────────────────────────────
# Test: parquet 文件存在性
# ──────────────────────────────────────────────────────────────────────────────

class TestParquetFiles:
    def test_ep0_parquet_exists(self, converted_out):
        _, out = converted_out
        assert (out / "data" / "chunk-000" / "episode_000000.parquet").exists()

    def test_ep1_parquet_exists(self, converted_out):
        _, out = converted_out
        assert (out / "data" / "chunk-000" / "episode_000001.parquet").exists()


# ──────────────────────────────────────────────────────────────────────────────
# Test: index 跨 episode 全局连续
# ──────────────────────────────────────────────────────────────────────────────

class TestGlobalIndex:
    @pytest.fixture(autouse=True)
    def _load(self, converted_out):
        _, out = converted_out
        chunk = out / "data" / "chunk-000"
        self.df0 = pq.read_table(chunk / "episode_000000.parquet").to_pydict()
        self.df1 = pq.read_table(chunk / "episode_000001.parquet").to_pydict()

    def test_ep0_row_count(self):
        assert len(self.df0["index"]) == EP0_N

    def test_ep1_row_count(self):
        assert len(self.df1["index"]) == EP1_N

    def test_index_ep0_starts_at_zero(self):
        assert list(self.df0["index"]) == list(range(0, EP0_N))

    def test_index_ep1_continues_from_ep0(self):
        # ep1 index 从 EP0_N 开始，全局连续
        assert list(self.df1["index"]) == list(range(EP0_N, EP0_N + EP1_N))

    def test_episode_index_ep0_all_zero(self):
        assert all(v == 0 for v in self.df0["episode_index"])

    def test_episode_index_ep1_all_one(self):
        assert all(v == 1 for v in self.df1["episode_index"])

    def test_combined_index_is_full_range(self):
        combined = list(self.df0["index"]) + list(self.df1["index"])
        assert combined == list(range(TOTAL_N))


# ──────────────────────────────────────────────────────────────────────────────
# Test: episodes.jsonl 长度
# ──────────────────────────────────────────────────────────────────────────────

class TestEpisodesJsonl:
    @pytest.fixture(autouse=True)
    def _load(self, converted_out):
        _, out = converted_out
        self.rows = _read_jsonl(out / "meta" / "episodes.jsonl")

    def test_two_rows(self):
        assert len(self.rows) == 2

    def test_ep0_length(self):
        assert self.rows[0]["length"] == EP0_N

    def test_ep1_length(self):
        assert self.rows[1]["length"] == EP1_N

    def test_tasks_field(self):
        assert self.rows[0]["tasks"] == ["pick"]
        assert self.rows[1]["tasks"] == ["pick"]


# ──────────────────────────────────────────────────────────────────────────────
# Test: tasks.jsonl
# ──────────────────────────────────────────────────────────────────────────────

def test_tasks_jsonl_content(converted_out):
    _, out = converted_out
    rows = _read_jsonl(out / "meta" / "tasks.jsonl")
    assert len(rows) == 1
    assert rows[0]["task"] == "pick"
    assert rows[0]["task_index"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# Test: episodes_stats.jsonl
# ──────────────────────────────────────────────────────────────────────────────

class TestEpisodesStatsJsonl:
    @pytest.fixture(autouse=True)
    def _load(self, converted_out):
        _, out = converted_out
        self.rows = _read_jsonl(out / "meta" / "episodes_stats.jsonl")

    def test_two_rows(self):
        assert len(self.rows) == 2

    def test_each_row_has_stats(self):
        for row in self.rows:
            assert "stats" in row
            assert "episode_index" in row

    def test_state_stats_present(self):
        for row in self.rows:
            stats = row["stats"]
            assert "observation.state" in stats
            s = stats["observation.state"]
            for k in ("min", "max", "mean", "std", "count"):
                assert k in s, f"stats 缺 {k}"
            assert len(s["min"]) == 14
            assert len(s["max"]) == 14
            assert s["count"] == [row["episode_index"] == 0 and EP0_N or EP1_N]

    def test_action_stats_present(self):
        for row in self.rows:
            stats = row["stats"]
            assert "action" in stats
            s = stats["action"]
            assert len(s["min"]) == 7
            assert len(s["max"]) == 7


# ──────────────────────────────────────────────────────────────────────────────
# Test: 视频文件存在性
# ──────────────────────────────────────────────────────────────────────────────

class TestVideoFiles:
    def test_ep0_videos_exist(self, converted_out):
        _, out = converted_out
        for cam in CAMS:
            p = out / "videos" / "chunk-000" / f"observation.images.{cam}" / "episode_000000.mp4"
            assert p.exists(), f"视频不存在: {p}"
            assert p.stat().st_size > 0, f"视频为空: {p}"

    def test_ep1_videos_exist(self, converted_out):
        _, out = converted_out
        for cam in CAMS:
            p = out / "videos" / "chunk-000" / f"observation.images.{cam}" / "episode_000001.mp4"
            assert p.exists(), f"视频不存在: {p}"
            assert p.stat().st_size > 0, f"视频为空: {p}"


# ──────────────────────────────────────────────────────────────────────────────
# Test: CLI subprocess 退出 0
# ──────────────────────────────────────────────────────────────────────────────

def test_cli_subprocess_exit_zero(tmp_path):
    """CLI subprocess 调用退出码 == 0，产物 info.json 存在。"""
    in_dir = tmp_path / "cli_in"
    out_dir = tmp_path / "cli_out"
    _make_two_episodes(in_dir, cams=CAMS)

    result = subprocess.run(
        [
            _PY, str(_SCRIPT),
            "--in-dir", str(in_dir),
            "--out", str(out_dir),
            "--fps", "30",
            "--task", "pick",
            "--robot-type", "franka",
            "--state-layout", "native",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"CLI 退出码 {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert (out_dir / "meta" / "info.json").exists(), "CLI 后 info.json 不存在"


# ──────────────────────────────────────────────────────────────────────────────
# Test: argparse 默认路径来自 core.paths（FRANKA_JHLI_ROOT env override）
# ──────────────────────────────────────────────────────────────────────────────

def test_cli_default_paths_from_core_paths():
    """用 FRANKA_JHLI_ROOT=/tmp/xx_test_jhli override，--help 输出含该路径衍生值。"""
    fake_root = "/tmp/xx_test_jhli"
    env = os.environ.copy()
    env["FRANKA_JHLI_ROOT"] = fake_root

    result = subprocess.run(
        [_PY, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    # --help 退出码为 0
    assert result.returncode == 0, f"--help 失败: {result.stderr}"
    help_text = result.stdout

    # 默认 --in-dir 应含 fake_root（来自 core.paths.HDF5_EPISODES_DIR）
    assert fake_root in help_text, (
        f"--help 输出未含 FRANKA_JHLI_ROOT={fake_root} 衍生路径\n{help_text}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test: 空目录抛 ValueError
# ──────────────────────────────────────────────────────────────────────────────

def test_convert_empty_dir_raises(tmp_path):
    """in_dir 中无 .h5 文件时，convert() 抛 ValueError。"""
    in_dir = tmp_path / "empty"
    in_dir.mkdir()
    out_dir = tmp_path / "out_empty"
    with pytest.raises(ValueError, match="no hdf5"):
        convert(in_dir=in_dir, out=out_dir, fps=30, task="pick")
