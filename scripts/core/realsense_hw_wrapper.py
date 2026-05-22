"""RealSense 硬件时间戳包装器（Task 8-A，spike-b 分支 A）。

设计要点：
- 独立维护 pyrealsense2 pipeline（不依赖 lerobot RealSenseCamera，不改 lerobot 包）。
- connect() 时设 rs.option.global_time_enabled=1，让 get_timestamp() 返回硬件全局时间戳。
- read() 返回 (rgb_ndarray, hw_timestamp_ms)：
    rgb_ndarray  — uint8 (H, W, 3) RGB 格式（与 lerobot RealSenseCamera.read() 一致）
    hw_timestamp_ms — float，cf.get_timestamp() 毫秒值（硬件戳）
- disconnect() 停止 pipeline，幂等。
- 不引 ROS2；纯 pyrealsense2 + numpy。

使用方式：
    wrapper = RealsenseHwWrapper(serial="419622073931", width=640, height=480, fps=30)
    wrapper.connect()
    rgb, hw_ts = wrapper.read()   # hw_ts 单位：毫秒
    wrapper.disconnect()

集成路径（run_record_hdf5.py）：
    _make_camera_read_fn(cam) 检测 cam 是否为 RealsenseHwWrapper 实例；
    若是，cam.read() 返回 (rgb, hw_ts_ms) 元组；
    否则（普通 lerobot 相机），返回 (cam.read(), None) 元组。
    record_episode 解包 data：if isinstance(data, tuple) → (img, hw_ts)；else → (data, None)。
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class RealsenseHwWrapper:
    """用 pyrealsense2 直接管理 RealSense pipeline，暴露硬件时间戳。

    Args:
        serial: 相机串号（纯数字字符串）。
        width: 颜色图像宽度（像素）。
        height: 颜色图像高度（像素）。
        fps: 目标帧率（Hz）。
        timeout_ms: try_wait_for_frames 超时毫秒数，默认 200ms。

    类属性 HW_WRAPPER = True 供 _make_camera_read_fn duck-type 检测，
    区分本 wrapper（read() 返回 (rgb, hw_ts) 元组）与普通相机（read() 返回 ndarray）。
    """

    # duck-type 标记：run_record_hdf5._make_camera_read_fn 靠此区分 wrapper vs 普通相机
    HW_WRAPPER: bool = True

    def __init__(self, serial: str, width: int, height: int, fps: int,
                 timeout_ms: int = 200):
        self._serial = str(serial)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._timeout_ms = int(timeout_ms)
        self._pipeline = None
        self._profile = None

    @property
    def serial(self) -> str:
        """相机串号。"""
        return self._serial

    @property
    def is_connected(self) -> bool:
        """pipeline 是否已启动。"""
        return self._pipeline is not None and self._profile is not None

    def connect(self) -> None:
        """启动 pipeline 并设 global_time_enabled=1。

        已连接时幂等（不重复 start）。
        """
        if self.is_connected:
            return

        import pyrealsense2 as rs  # 延迟 import，避免测试加载时依赖真实硬件

        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs.config.enable_device(rs_config, self._serial)
        rs_config.enable_stream(
            rs.stream.color,
            self._width,
            self._height,
            rs.format.rgb8,
            self._fps,
        )

        try:
            profile = pipeline.start(rs_config)
        except RuntimeError as e:
            raise ConnectionError(
                f"RealsenseHwWrapper({self._serial}) pipeline.start 失败: {e}"
            ) from e

        # 设置 global_time_enabled=1，令 get_timestamp() 返回硬件全局时间戳
        try:
            color_sensor = profile.get_device().first_color_sensor()
            color_sensor.set_option(rs.option.global_time_enabled, 1.0)
            gte_val = color_sensor.get_option(rs.option.global_time_enabled)
            logger.info(
                "RealsenseHwWrapper(%s): global_time_enabled 已设为 1.0（回读: %s）",
                self._serial,
                gte_val,
            )
        except Exception as e:  # noqa: BLE001 — option 可能在旧固件不支持，log 后继续
            logger.warning(
                "RealsenseHwWrapper(%s): 设置 global_time_enabled 失败（%s），"
                "hw_timestamp 精度可能降低",
                self._serial,
                e,
            )

        # 预热：丢弃前几帧，等曝光稳定
        warmup_frames = 10
        for _ in range(warmup_frames):
            try:
                pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                break

        self._pipeline = pipeline
        self._profile = profile
        logger.info("RealsenseHwWrapper(%s) 已连接，%dx%d @%dHz",
                    self._serial, self._width, self._height, self._fps)

    def read(self) -> tuple[np.ndarray, float]:
        """从 pipeline 读一帧，返回 (rgb_ndarray, hw_timestamp_ms)。

        Returns:
            rgb: (H, W, 3) uint8 ndarray，RGB 格式。
            hw_ts_ms: float，cf.get_timestamp() 返回的毫秒级硬件时间戳。

        Raises:
            RuntimeError: 未连接或读帧失败。
        """
        if not self.is_connected:
            raise RuntimeError(
                f"RealsenseHwWrapper({self._serial}) 未连接，请先调用 connect()"
            )

        ret, frames = self._pipeline.try_wait_for_frames(timeout_ms=self._timeout_ms)
        if not ret or frames is None:
            raise RuntimeError(
                f"RealsenseHwWrapper({self._serial}) try_wait_for_frames 超时/失败"
            )

        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(
                f"RealsenseHwWrapper({self._serial}) get_color_frame 返回空帧"
            )

        hw_ts_ms = color_frame.get_timestamp()           # 毫秒，硬件全局戳
        rgb = np.array(color_frame.get_data())           # (H, W, 3) uint8 RGB，独立副本（不依赖 RealSense buffer）

        return rgb, hw_ts_ms

    def disconnect(self) -> None:
        """停止 pipeline，幂等。"""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("RealsenseHwWrapper(%s) stop 时异常（忽略）: %s",
                               self._serial, e)
            finally:
                self._pipeline = None
                self._profile = None
            logger.info("RealsenseHwWrapper(%s) 已断连", self._serial)
