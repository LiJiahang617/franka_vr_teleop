"""tests/test_v21_video.py — Task3 TDD：episode_to_videos 视频导出测试。

合成 hdf5 含 2 相机 vlen jpeg bytes，验证 ffmpeg libx264 yuv420p 编码正确性。
小图 16×16 验边界（规避 SVT-AV1<32px 坑）；另组 240×424 近真机尺寸。
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

# 自举：确保 <repo>/scripts 在 sys.path 中
_REPO_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_REPO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_REPO_SCRIPTS))

from tools.hdf5_to_lerobot_v21 import episode_to_videos

# ──────────────────────────────────────────────────────────────────────────────
# 合成 hdf5 生成工具
# ──────────────────────────────────────────────────────────────────────────────

def _encode_jpeg(bgr_img: np.ndarray) -> bytes:
    """将 BGR ndarray 编码为 jpeg bytes。"""
    ok, buf = cv2.imencode(".jpg", bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


def _make_synthetic_hdf5(h5_path: Path, cam_specs: dict, n_frames: int) -> None:
    """生成合成 franka-hdf5-v1 文件，含多相机 vlen jpeg bytes。

    Args:
        h5_path: 输出 hdf5 路径
        cam_specs: {cam_name: (H, W)} 各相机尺寸
        n_frames: 帧数
    """
    # 与真实 hdf5 一致：vlen uint8 数组，非 vlen bytes 字符串
    vlen_u8 = h5py.vlen_dtype(np.uint8)
    with h5py.File(h5_path, "w") as h5:
        # 骨架数据（episode_to_videos 只用图像，但 hdf5 需有基础结构）
        h5.create_dataset("observations/arm/joints", data=np.zeros((n_frames, 7), dtype=np.float32))
        h5.create_dataset("observations/arm/gripper", data=np.zeros((n_frames, 1), dtype=np.float32))
        h5.create_dataset("observations/arm/ee_pose", data=np.zeros((n_frames, 6), dtype=np.float32))
        h5.create_dataset("action", data=np.zeros((n_frames, 7), dtype=np.float32))

        for cam_name, (H, W) in cam_specs.items():
            ds = h5.create_dataset(
                f"observations/camera/rgb/{cam_name}/images",
                shape=(n_frames,),
                dtype=vlen_u8,
            )
            for i in range(n_frames):
                # 用不同颜色区分帧（便于肉眼检查）
                color = int(i * 255 / max(n_frames - 1, 1))
                bgr = np.full((H, W, 3), [color, 128, 255 - color], dtype=np.uint8)
                # 存为 uint8 ndarray（与真实 hdf5 dtype 一致）
                ds[i] = np.frombuffer(_encode_jpeg(bgr), dtype=np.uint8)


def _ffprobe_video(mp4_path: Path) -> dict:
    """用 ffprobe 读取视频流属性，返回 dict。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames",
        "-count_frames",
        "-of", "json",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert data.get("streams"), f"ffprobe 无流信息: {result.stdout}"
    return data["streams"][0]


# ──────────────────────────────────────────────────────────────────────────────
# 测试用例
# ──────────────────────────────────────────────────────────────────────────────

class TestEpisodeToVideosSmall:
    """小图 16×16：验 libx264 可处理极小帧，规避 SVT-AV1<32px 崩溃。"""

    N = 3
    CAM_SPECS = {"cam_left": (16, 16), "cam_right": (16, 24)}

    @pytest.fixture
    def h5_and_outdir(self, tmp_path):
        h5_path = tmp_path / "ep_small.h5"
        _make_synthetic_hdf5(h5_path, self.CAM_SPECS, self.N)
        out_dir = tmp_path / "out_small"
        return h5_path, out_dir

    def test_mp4_files_created_nonempty(self, h5_and_outdir):
        """每相机输出 mp4 路径存在且非空。"""
        h5_path, out_dir = h5_and_outdir
        result = episode_to_videos(
            h5_path=str(h5_path),
            out_dir=str(out_dir),
            episode_chunk=0,
            episode_index=0,
            cam_names=list(self.CAM_SPECS.keys()),
            fps=30,
        )
        for cam_name in self.CAM_SPECS:
            expected = (
                out_dir / "videos" / "chunk-000"
                / f"observation.images.{cam_name}" / "episode_000000.mp4"
            )
            assert expected.exists(), f"mp4 未生成: {expected}"
            assert expected.stat().st_size > 0, f"mp4 为空: {expected}"
            # 返回 dict 中路径一致
            assert result[cam_name] == expected

    def test_ffprobe_codec_pixfmt_fps(self, h5_and_outdir):
        """ffprobe 确认 h264 / yuv420p / 30fps。"""
        h5_path, out_dir = h5_and_outdir
        episode_to_videos(
            h5_path=str(h5_path),
            out_dir=str(out_dir),
            episode_chunk=0,
            episode_index=0,
            cam_names=list(self.CAM_SPECS.keys()),
            fps=30,
        )
        for cam_name in self.CAM_SPECS:
            mp4 = (
                out_dir / "videos" / "chunk-000"
                / f"observation.images.{cam_name}" / "episode_000000.mp4"
            )
            info = _ffprobe_video(mp4)
            assert info["codec_name"] == "h264", f"{cam_name}: codec={info['codec_name']}"
            assert info["pix_fmt"] == "yuv420p", f"{cam_name}: pix_fmt={info['pix_fmt']}"
            assert info["r_frame_rate"] == "30/1", f"{cam_name}: fps={info['r_frame_rate']}"

    def test_ffprobe_width_height_match_hdf5(self, h5_and_outdir):
        """视频宽高与 hdf5 jpeg 解码尺寸一致（H→height, W→width）。"""
        h5_path, out_dir = h5_and_outdir
        episode_to_videos(
            h5_path=str(h5_path),
            out_dir=str(out_dir),
            episode_chunk=0,
            episode_index=0,
            cam_names=list(self.CAM_SPECS.keys()),
            fps=30,
        )
        for cam_name, (H, W) in self.CAM_SPECS.items():
            mp4 = (
                out_dir / "videos" / "chunk-000"
                / f"observation.images.{cam_name}" / "episode_000000.mp4"
            )
            info = _ffprobe_video(mp4)
            assert int(info["width"]) == W, f"{cam_name}: width={info['width']} expect {W}"
            assert int(info["height"]) == H, f"{cam_name}: height={info['height']} expect {H}"

    def test_ffprobe_frame_count(self, h5_and_outdir):
        """视频帧数 == hdf5 N=3。"""
        h5_path, out_dir = h5_and_outdir
        episode_to_videos(
            h5_path=str(h5_path),
            out_dir=str(out_dir),
            episode_chunk=0,
            episode_index=0,
            cam_names=list(self.CAM_SPECS.keys()),
            fps=30,
        )
        for cam_name in self.CAM_SPECS:
            mp4 = (
                out_dir / "videos" / "chunk-000"
                / f"observation.images.{cam_name}" / "episode_000000.mp4"
            )
            info = _ffprobe_video(mp4)
            assert int(info["nb_read_frames"]) == self.N, (
                f"{cam_name}: nb_read_frames={info['nb_read_frames']} expect {self.N}"
            )


class TestEpisodeToVideosFull:
    """近真机尺寸 240×424（H×W）+ chunk/episode 索引参数化。"""

    N = 3
    CAM_SPECS = {"exterior_image_1": (240, 424), "wrist_image_1": (240, 424)}

    @pytest.fixture
    def h5_and_outdir(self, tmp_path):
        h5_path = tmp_path / "ep_full.h5"
        _make_synthetic_hdf5(h5_path, self.CAM_SPECS, self.N)
        out_dir = tmp_path / "out_full"
        return h5_path, out_dir

    def test_path_template_chunk_episode(self, h5_and_outdir):
        """chunk=2, episode=5 时路径模板正确（chunk-002, episode_000005）。"""
        h5_path, out_dir = h5_and_outdir
        episode_to_videos(
            h5_path=str(h5_path),
            out_dir=str(out_dir),
            episode_chunk=2,
            episode_index=5,
            cam_names=list(self.CAM_SPECS.keys()),
            fps=30,
        )
        for cam_name in self.CAM_SPECS:
            expected = (
                out_dir / "videos" / "chunk-002"
                / f"observation.images.{cam_name}" / "episode_000005.mp4"
            )
            assert expected.exists(), f"路径模板错误: {expected}"

    def test_full_size_h264_yuv420p(self, h5_and_outdir):
        """近真机尺寸 240×424 确认 h264/yuv420p/30fps。"""
        h5_path, out_dir = h5_and_outdir
        episode_to_videos(
            h5_path=str(h5_path),
            out_dir=str(out_dir),
            episode_chunk=0,
            episode_index=0,
            cam_names=list(self.CAM_SPECS.keys()),
            fps=30,
        )
        for cam_name, (H, W) in self.CAM_SPECS.items():
            mp4 = (
                out_dir / "videos" / "chunk-000"
                / f"observation.images.{cam_name}" / "episode_000000.mp4"
            )
            info = _ffprobe_video(mp4)
            assert info["codec_name"] == "h264"
            assert info["pix_fmt"] == "yuv420p"
            assert info["r_frame_rate"] == "30/1"
            assert int(info["width"]) == W
            assert int(info["height"]) == H


# ──────────────────────────────────────────────────────────────────────────────
# 新增测试（Codex review-fix：坏 jpeg / 帧尺寸不一致 / 失败无残留）
# ──────────────────────────────────────────────────────────────────────────────

def _make_bad_jpeg_hdf5(h5_path: Path, cam_name: str, n_frames: int, bad_frame_idx: int) -> None:
    """生成含坏 jpeg 的合成 hdf5（bad_frame_idx 帧为非法 bytes）。"""
    H, W = 16, 16
    vlen_u8 = h5py.vlen_dtype(np.uint8)
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("observations/arm/joints", data=np.zeros((n_frames, 7), dtype=np.float32))
        h5.create_dataset("observations/arm/gripper", data=np.zeros((n_frames, 1), dtype=np.float32))
        h5.create_dataset("observations/arm/ee_pose", data=np.zeros((n_frames, 6), dtype=np.float32))
        h5.create_dataset("action", data=np.zeros((n_frames, 7), dtype=np.float32))
        ds = h5.create_dataset(
            f"observations/camera/rgb/{cam_name}/images",
            shape=(n_frames,),
            dtype=vlen_u8,
        )
        for i in range(n_frames):
            if i == bad_frame_idx:
                # 非法 bytes，cv2.imdecode 会返回 None
                ds[i] = np.frombuffer(b"\xff\xd8\xff\x00invalid_not_jpeg", dtype=np.uint8)
            else:
                bgr = np.full((H, W, 3), [100, 128, 200], dtype=np.uint8)
                ds[i] = np.frombuffer(_encode_jpeg(bgr), dtype=np.uint8)


def _make_mismatched_size_hdf5(h5_path: Path, cam_name: str, n_frames: int, bad_frame_idx: int) -> None:
    """生成帧尺寸不一致的合成 hdf5（bad_frame_idx 帧用不同 HxW）。"""
    H, W = 16, 16
    H_bad, W_bad = 32, 32  # 与首帧不同的尺寸
    vlen_u8 = h5py.vlen_dtype(np.uint8)
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("observations/arm/joints", data=np.zeros((n_frames, 7), dtype=np.float32))
        h5.create_dataset("observations/arm/gripper", data=np.zeros((n_frames, 1), dtype=np.float32))
        h5.create_dataset("observations/arm/ee_pose", data=np.zeros((n_frames, 6), dtype=np.float32))
        h5.create_dataset("action", data=np.zeros((n_frames, 7), dtype=np.float32))
        ds = h5.create_dataset(
            f"observations/camera/rgb/{cam_name}/images",
            shape=(n_frames,),
            dtype=vlen_u8,
        )
        for i in range(n_frames):
            if i == bad_frame_idx:
                bgr = np.full((H_bad, W_bad, 3), [100, 128, 200], dtype=np.uint8)
            else:
                bgr = np.full((H, W, 3), [100, 128, 200], dtype=np.uint8)
            ds[i] = np.frombuffer(_encode_jpeg(bgr), dtype=np.uint8)


class TestEpisodeToVideosErrorHandling:
    """错误处理测试：坏 jpeg / 帧尺寸不一致 / 失败无残留。"""

    CAM_NAME = "cam_test"

    def test_bad_jpeg_raises_value_error(self, tmp_path):
        """中间帧包含非法 jpeg bytes 时，应抛出包含"解码失败"的 ValueError。"""
        h5_path = tmp_path / "bad_jpeg.h5"
        # 第 1 帧（非首帧）为坏 jpeg，确保首帧正常可以建立 H/W
        _make_bad_jpeg_hdf5(h5_path, self.CAM_NAME, n_frames=3, bad_frame_idx=1)
        out_dir = tmp_path / "out_bad_jpeg"
        with pytest.raises(ValueError, match="解码失败"):
            episode_to_videos(
                h5_path=str(h5_path),
                out_dir=str(out_dir),
                episode_chunk=0,
                episode_index=0,
                cam_names=[self.CAM_NAME],
                fps=30,
            )

    def test_mismatched_frame_size_raises_value_error(self, tmp_path):
        """某帧尺寸与首帧不一致时，应抛出包含"尺寸"的 ValueError。"""
        h5_path = tmp_path / "mismatch.h5"
        # 第 2 帧（非首帧）尺寸不同
        _make_mismatched_size_hdf5(h5_path, self.CAM_NAME, n_frames=3, bad_frame_idx=2)
        out_dir = tmp_path / "out_mismatch"
        with pytest.raises(ValueError, match="尺寸"):
            episode_to_videos(
                h5_path=str(h5_path),
                out_dir=str(out_dir),
                episode_chunk=0,
                episode_index=0,
                cam_names=[self.CAM_NAME],
                fps=30,
            )

    def test_failure_leaves_no_residual_files(self, tmp_path):
        """坏 jpeg 触发异常后，out_mp4 和 tmp 文件均不存在（原子写保证无残留）。"""
        h5_path = tmp_path / "bad_jpeg2.h5"
        _make_bad_jpeg_hdf5(h5_path, self.CAM_NAME, n_frames=3, bad_frame_idx=1)
        out_dir = tmp_path / "out_no_residual"

        # 确认异常被抛出
        with pytest.raises(ValueError):
            episode_to_videos(
                h5_path=str(h5_path),
                out_dir=str(out_dir),
                episode_chunk=0,
                episode_index=0,
                cam_names=[self.CAM_NAME],
                fps=30,
            )

        # 验证 mp4 和 tmp 均不存在
        vid_dir = out_dir / "videos" / "chunk-000" / f"observation.images.{self.CAM_NAME}"
        out_mp4 = vid_dir / "episode_000000.mp4"
        tmp_mp4 = vid_dir / "episode_000000.mp4.tmp"
        assert not out_mp4.exists(), f"失败后 mp4 仍残留: {out_mp4}"
        assert not tmp_mp4.exists(), f"失败后 tmp 仍残留: {tmp_mp4}"
