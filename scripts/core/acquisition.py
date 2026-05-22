"""多模态独立采集模块：SensorThread + AcquisitionHub。

设计要点：
- 每个传感器模态各自运行一个 SensorThread，独立 read_fn 调用后立即打软件戳。
- AcquisitionHub 管理多个 SensorThread，提供 snapshot(now) 接口：
  取各模态最新 (data, ts, stale) 元组（stale = now-ts > 2/rate）。
- 优雅退出：stop() 置 stop_event + join timeout，不阻塞主线程。
- 不引 ROS2：纯 Python threading + threading.Event + threading.Lock。
- latest() 始终返回最后已知读数（Lock 保护的单槽，不清空），
  保证 snapshot 在有过至少一帧后总能拿到上一帧并由 stale 标记是否陈旧。
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """单次传感器读数，含软件时间戳。"""
    data: Any       # read_fn 的返回值
    ts: float       # time.monotonic() 采样后立即打戳


class SensorThread:
    """独立采集线程：调 read_fn → 立即打软件戳 → 存入 Lock 保护的单槽。

    使用 threading.Lock 保护的单槽 `_latest` 代替 queue，保证：
    - latest() 始终返回最后已知读数（不清空），从未读到过才返回 None。
    - 高频 snapshot 不会因排空 queue 而拿到 None，模态不会从 snapshot 中消失。

    Args:
        name: 模态名称（调试用）。
        read_fn: 无参可调用，返回本次读数数据（应尽快返回）。
        target_rate: 目标采集频率（Hz），必须 > 0。
        stop_event: 外部共享的停止事件（可多个 SensorThread 共享一个）。
    """

    def __init__(
        self,
        name: str,
        read_fn: Callable[[], Any],
        target_rate: float,
        stop_event: Optional[threading.Event] = None,
    ):
        # 校验 target_rate（在启动线程之前）
        if target_rate <= 0:
            raise ValueError(f"target_rate 必须 > 0，收到 {target_rate}")

        self.name = name
        self._read_fn = read_fn
        self._target_rate = float(target_rate)
        self._stop_event = stop_event if stop_event is not None else threading.Event()

        # Lock 保护的单槽：始终保存最后一次成功读数
        self._lock = threading.Lock()
        self._latest: Optional[SensorReading] = None

        # 最后一次 read_fn 异常（None 表示无异常）
        self._last_error: Optional[Exception] = None

        self._thread = threading.Thread(target=self._loop, name=f"SensorThread-{name}", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # 公有接口
    # ------------------------------------------------------------------

    @property
    def target_rate(self) -> float:
        """目标采集频率（Hz）。"""
        return self._target_rate

    @property
    def last_error(self) -> Optional[Exception]:
        """最后一次 read_fn 抛出的异常（None 表示无异常）。"""
        return self._last_error

    def latest(self) -> Optional[SensorReading]:
        """返回最后已知读数（非阻塞）。

        从未读到过任何帧时返回 None；有过至少一帧后始终返回最后那帧，
        多次调用不清空、不会退化为 None。
        """
        with self._lock:
            return self._latest

    def stop(self, timeout: float = 2.0) -> bool:
        """置停止事件并 join 线程。

        Args:
            timeout: 等待线程退出的最长时间（秒）。

        Returns:
            True 表示线程已成功停止，False 表示 join 超时后线程仍存活。
        """
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning(
                "SensorThread '%s' 在 %.1fs 内未能停止，线程仍存活", self.name, timeout
            )
            return False
        return True

    def is_alive(self) -> bool:
        """返回线程是否仍在运行。"""
        return self._thread.is_alive()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """采集主循环：按 target_rate 节拍 read → 立即打戳 → 存入单槽。

        节拍语义：
        - 正常：next_tick += interval，然后 stop_event.wait(sleep_dur)。
        - 超时（read_fn 耗时超过 interval）：不睡，把 next_tick 重置为当前时刻，
          立即进入下一轮读，避免积压积压积压地多跳周期。
        """
        interval = 1.0 / self._target_rate
        next_tick = time.monotonic()

        while not self._stop_event.is_set():
            # --- 读取并打戳 ---
            try:
                data = self._read_fn()
                ts = time.monotonic()   # read_fn 返回后立即打软件戳
                reading = SensorReading(data=data, ts=ts)
                with self._lock:
                    self._latest = reading
            except Exception as exc:
                # 记录异常，线程继续运行（不静默崩溃）
                self._last_error = exc
                logger.warning(
                    "SensorThread '%s' read_fn 抛出异常（将在下一拍重试）: %s", self.name, exc
                )

            # --- 精确节拍：补偿 read_fn 耗时 ---
            next_tick += interval
            now = time.monotonic()
            sleep_dur = next_tick - now
            if sleep_dur > 0:
                # 用 stop_event.wait 替代 time.sleep，stop 时可立即响应
                self._stop_event.wait(sleep_dur)
            else:
                # 超时：不睡，重置 next_tick 到当前时刻，直接进入下一轮
                # （避免用 cycles_missed 跳过多个周期，导致节拍计算错乱）
                next_tick = time.monotonic()


class AcquisitionHub:
    """管理多个 SensorThread，提供 snapshot(now) 接口。

    Args:
        sensors: dict[name, SensorThread]，每个模态一个。
    """

    def __init__(self, sensors: dict[str, "SensorThread"]):
        self._sensors = dict(sensors)

    def snapshot(self, now: float) -> dict[str, tuple[Any, float, bool]]:
        """取每模态最新读数，返回 {name: (data, ts, stale)}。

        stale 判据：now - ts > 2 / target_rate（超过两帧周期未更新）。

        Args:
            now: 当前时刻（time.monotonic()），由调用方传入。

        Returns:
            dict，键为模态名，值为 (data, ts, stale) 三元组。
            若某模态尚无数据（从未读到过帧），该模态缺席（不含该键）。
            有过至少一帧的模态始终出现，数据陈旧时 stale=True。
        """
        result: dict[str, tuple[Any, float, bool]] = {}
        for name, sensor in self._sensors.items():
            reading = sensor.latest()
            if reading is None:
                continue
            stale_threshold = 2.0 / sensor.target_rate
            stale = (now - reading.ts) > stale_threshold
            result[name] = (reading.data, reading.ts, stale)
        return result

    def stop(self, timeout: float = 2.0) -> None:
        """停止所有传感器线程（各自 join timeout）。"""
        for sensor in self._sensors.values():
            sensor.stop(timeout=timeout)

    def __len__(self) -> int:
        return len(self._sensors)
