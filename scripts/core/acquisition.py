"""多模态独立采集模块：SensorThread + AcquisitionHub。

设计要点：
- 每个传感器模态各自运行一个 SensorThread，独立 read_fn 调用后立即打软件戳。
- AcquisitionHub 管理多个 SensorThread，提供 snapshot(now) 接口：
  取各模态最新 (data, ts, stale) 元组（stale = now-ts > 2/rate）。
- 优雅退出：stop() 置 stop_event + join timeout=2s，不阻塞主线程。
- 不引 ROS2：纯 Python threading + queue + threading.Event。
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class SensorReading:
    """单次传感器读数，含软件时间戳。"""
    data: Any       # read_fn 的返回值
    ts: float       # time.monotonic() 采样后立即打戳


class SensorThread:
    """独立采集线程：调 read_fn → 立即打软件戳 → 放入 queue。

    Args:
        name: 模态名称（调试用）。
        read_fn: 无参可调用，返回本次读数数据（应尽快返回）。
        target_rate: 目标采集频率（Hz）。
        queue_maxsize: 内部 queue 最大深度（0 = 无界）。
        stop_event: 外部共享的停止事件（可多个 SensorThread 共享一个）。
    """

    def __init__(
        self,
        name: str,
        read_fn: Callable[[], Any],
        target_rate: float,
        queue_maxsize: int = 1,
        stop_event: Optional[threading.Event] = None,
    ):
        self.name = name
        self._read_fn = read_fn
        self._target_rate = float(target_rate)
        self._stop_event = stop_event if stop_event is not None else threading.Event()
        # maxsize=1 默认只保留最新帧（旧帧被丢弃），消费方总能拿到最新数据
        self._q: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._thread = threading.Thread(target=self._loop, name=f"SensorThread-{name}", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # 公有接口
    # ------------------------------------------------------------------

    def latest(self) -> Optional[SensorReading]:
        """返回 queue 中最新的读数（非阻塞），没有则返回 None。

        将 queue 中所有旧帧排空，返回最后一帧（即最新一帧）。
        """
        reading = None
        while True:
            try:
                reading = self._q.get_nowait()
            except queue.Empty:
                break
        return reading

    def stop(self, timeout: float = 2.0) -> None:
        """置停止事件并 join 线程（超时后不再等待，防僵尸阻塞）。"""
        self._stop_event.set()
        self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """采集主循环：按 target_rate 节拍 read → 立即打戳 → 入队。"""
        interval = 1.0 / self._target_rate
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            data = self._read_fn()
            ts = time.monotonic()   # read_fn 返回后立即打软件戳
            reading = SensorReading(data=data, ts=ts)
            # 非阻塞入队：若 queue 已满则丢弃旧帧（保留最新）
            try:
                self._q.put_nowait(reading)
            except queue.Full:
                try:
                    self._q.get_nowait()   # 丢弃一帧旧数据
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait(reading)
                except queue.Full:
                    pass

            # 精确节拍：补偿 read_fn 耗时
            next_tick += interval
            now = time.monotonic()
            sleep_dur = next_tick - now
            if sleep_dur > 0:
                time.sleep(sleep_dur)
            else:
                # 超时：跳过积压的周期，对齐到下一个未来周期
                cycles_missed = int(-sleep_dur / interval) + 1
                next_tick += cycles_missed * interval


class AcquisitionHub:
    """管理多个 SensorThread，提供 snapshot(now) 接口。

    Args:
        sensors: dict[name, SensorThread]，每个模态一个。
    """

    def __init__(self, sensors: dict[str, SensorThread]):
        self._sensors = dict(sensors)

    def snapshot(self, now: float) -> dict[str, tuple[Any, float, bool]]:
        """取每模态最新读数，返回 {name: (data, ts, stale)}。

        stale 判据：now - ts > 2 / target_rate（超过两帧周期未更新）。

        Args:
            now: 当前时刻（time.monotonic()），由调用方传入。

        Returns:
            dict，键为模态名，值为 (data, ts, stale) 三元组。
            若某模态尚无数据，该模态缺席（不含该键）。
        """
        result: dict[str, tuple[Any, float, bool]] = {}
        for name, sensor in self._sensors.items():
            reading = sensor.latest()
            if reading is None:
                continue
            stale_threshold = 2.0 / sensor._target_rate
            stale = (now - reading.ts) > stale_threshold
            result[name] = (reading.data, reading.ts, stale)
        return result

    def stop(self, timeout: float = 2.0) -> None:
        """停止所有传感器线程（各自 join timeout=2s）。"""
        for sensor in self._sensors.values():
            sensor.stop(timeout=timeout)

    def __len__(self) -> int:
        return len(self._sensors)
