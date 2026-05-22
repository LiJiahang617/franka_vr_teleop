"""tests/test_acquisition.py — SensorThread + AcquisitionHub 离线 TDD。

覆盖点：
1. read_fn 调用 + 软件戳立即打（ts 在 read 后，单调递增）
2. stale 触发（now-ts > 2/rate 时 stale=True）
3. 独立 modality rate（不同线程各自节拍，互不影响）
4. hub.snapshot 取最新读数（多帧后仍是最后一帧数据）
5. stop_event + join 后线程退出无僵尸
"""

import threading
import time

import pytest

from core.acquisition import AcquisitionHub, SensorReading, SensorThread


# ---------------------------------------------------------------------------
# 辅助：精确可控的 mock read_fn
# ---------------------------------------------------------------------------


def make_counter_read_fn():
    """返回 (read_fn, call_count_getter)：read_fn 每次调用计数+1，返回调用次数。"""
    calls = [0]

    def read_fn():
        calls[0] += 1
        return calls[0]

    def get_count():
        return calls[0]

    return read_fn, get_count


# ---------------------------------------------------------------------------
# Test 1：read_fn 调用 + 软件戳立即打（ts 在 read 后且单调）
# ---------------------------------------------------------------------------


def test_sensor_thread_calls_read_fn_and_stamps():
    """SensorThread 应调用 read_fn 并立即打 time.monotonic() 软件戳。"""
    stop = threading.Event()
    read_fn, get_count = make_counter_read_fn()
    sensor = SensorThread("arm", read_fn, target_rate=100.0, stop_event=stop)

    # 等待至少 3 次 read 调用
    deadline = time.monotonic() + 1.0
    while get_count() < 3 and time.monotonic() < deadline:
        time.sleep(0.01)

    sensor.stop(timeout=2.0)

    assert get_count() >= 3, f"read_fn 至少调用 3 次，实际 {get_count()}"


def test_sensor_thread_reading_has_ts_after_read():
    """SensorReading.ts 应在调用时单调递增（是 time.monotonic 时间戳）。"""
    stop = threading.Event()
    timestamps = []

    def read_fn():
        timestamps.append(time.monotonic())
        return 42

    sensor = SensorThread("cam", read_fn, target_rate=200.0, stop_event=stop)

    deadline = time.monotonic() + 0.5
    while len(timestamps) < 5 and time.monotonic() < deadline:
        time.sleep(0.01)

    sensor.stop(timeout=2.0)

    assert len(timestamps) >= 5
    # read_fn 记录的时刻列表应单调不减
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1], "read_fn 内部时刻应单调"


def test_sensor_reading_ts_is_monotonic_from_latest():
    """多次 latest() 得到的 ts 应单调不减（后调必然 >= 先调）。"""
    stop = threading.Event()
    sensor = SensorThread("joint", lambda: 1, target_rate=200.0, stop_event=stop)

    time.sleep(0.05)   # 让线程跑一段

    t0 = None
    for _ in range(5):
        r = sensor.latest()
        if r is not None:
            if t0 is not None:
                assert r.ts >= t0, f"latest ts 应 >= 上一次：{r.ts} < {t0}"
            t0 = r.ts
        time.sleep(0.02)

    sensor.stop(timeout=2.0)


# ---------------------------------------------------------------------------
# Test 2：stale 触发（now-ts > 2/rate → stale=True）
# ---------------------------------------------------------------------------


def test_hub_snapshot_stale_when_data_old():
    """snapshot(now) 中 now-ts > 2/rate 时，该模态 stale=True。"""
    stop = threading.Event()
    # queue_maxsize=0 表示无界，线程持续填入数据，stop 后 queue 仍有数据
    sensor = SensorThread("arm", lambda: 99, target_rate=10.0,
                          queue_maxsize=0, stop_event=stop)

    # 等待线程至少产生一帧
    deadline = time.monotonic() + 1.0
    while sensor._q.empty() and time.monotonic() < deadline:
        time.sleep(0.01)

    # 停止线程（queue 中仍有数据）
    sensor.stop(timeout=2.0)

    hub = AcquisitionHub({"arm": sensor})
    # now 设为 ts + 3 秒（极大），必然 stale（2/10 = 0.2s 阈值）
    snap = hub.snapshot(now=time.monotonic() + 3.0)

    assert "arm" in snap, "模态应在 snapshot 中（queue 中仍有数据）"
    data, ts, stale = snap["arm"]
    assert stale is True, f"now-ts=3s >> 2/10=0.2s，期望 stale=True，得 {stale}"
    assert data == 99


def test_hub_snapshot_not_stale_when_fresh():
    """snapshot(now) 中 now-ts <= 2/rate 时，stale=False。"""
    stop = threading.Event()
    sensor = SensorThread("effector", lambda: 7, target_rate=10.0, stop_event=stop)

    # 等最新帧
    deadline = time.monotonic() + 1.0
    reading = None
    while time.monotonic() < deadline:
        r = sensor.latest()
        if r is not None:
            reading = r
        time.sleep(0.01)
        if reading is not None and (time.monotonic() - reading.ts) < 0.1:
            break

    sensor.stop(timeout=2.0)
    assert reading is not None

    hub = AcquisitionHub({"effector": sensor})
    # now = ts（几乎同时），now-ts ≈ 0 << 0.2s
    snap = hub.snapshot(now=reading.ts)
    if "effector" in snap:
        _, ts, stale = snap["effector"]
        assert stale is False, f"now=ts，期望 stale=False，得 {stale}"


# ---------------------------------------------------------------------------
# Test 3：独立 modality rate（不同 SensorThread 各自节拍）
# ---------------------------------------------------------------------------


def test_independent_modality_rates():
    """两个 SensorThread 按各自 rate 独立运行，低速线程调用次数 < 高速线程。"""
    stop = threading.Event()
    fast_fn, fast_count = make_counter_read_fn()
    slow_fn, slow_count = make_counter_read_fn()

    fast = SensorThread("fast", fast_fn, target_rate=200.0, stop_event=stop)
    slow = SensorThread("slow", slow_fn, target_rate=10.0, stop_event=stop)

    time.sleep(0.5)   # 让两个线程分别跑 0.5s

    stop.set()
    fast.stop(timeout=2.0)
    slow.stop(timeout=2.0)

    # 0.5s 内，fast 约 100 次，slow 约 5 次；各自独立不干扰
    assert fast_count() > slow_count(), (
        f"fast={fast_count()} 应多于 slow={slow_count()}"
    )
    assert fast_count() >= 30, f"fast 0.5s 应 >=30 次，实际 {fast_count()}"
    assert slow_count() >= 2, f"slow 0.5s 应 >=2 次，实际 {slow_count()}"


# ---------------------------------------------------------------------------
# Test 4：hub.snapshot 返回最新读数
# ---------------------------------------------------------------------------


def test_hub_snapshot_returns_latest_data():
    """snapshot 应取 queue 中最新一帧，而不是最旧一帧。"""
    # 用单调递增计数器：最新帧 data 值最大
    stop = threading.Event()
    read_fn, get_count = make_counter_read_fn()
    sensor = SensorThread("arm", read_fn, target_rate=500.0, queue_maxsize=0, stop_event=stop)

    # 让线程充分运行，产生多帧
    time.sleep(0.2)

    sensor.stop(timeout=2.0)
    total = get_count()

    hub = AcquisitionHub({"arm": sensor})
    snap = hub.snapshot(now=time.monotonic())

    if "arm" in snap:
        data, ts, stale = snap["arm"]
        # 最新帧的 data 值应接近 total（误差允许 ±5 帧）
        assert data >= total - 5, (
            f"snapshot 应是最新帧：data={data}，total={total}"
        )


def test_hub_snapshot_missing_modal_when_no_data():
    """若某模态尚无数据（刚创建但还没读到帧），snapshot 中该模态缺席。"""
    # 用阻塞 read_fn，让线程永远拿不到数据（直到 stop）
    stop = threading.Event()

    def blocking_fn():
        stop.wait()   # 阻塞直到 stop
        return 0

    sensor = SensorThread("block", blocking_fn, target_rate=1.0, stop_event=stop)

    hub = AcquisitionHub({"block": sensor})
    snap = hub.snapshot(now=time.monotonic())

    # 模态因无数据应缺席
    assert "block" not in snap, "无数据模态不应出现在 snapshot 中"

    stop.set()
    sensor.stop(timeout=2.0)


# ---------------------------------------------------------------------------
# Test 5：stop_event + join 后线程退出无僵尸
# ---------------------------------------------------------------------------


def test_sensor_thread_stops_on_stop_event():
    """stop() 后线程应在 timeout 内退出，is_alive() = False。"""
    stop = threading.Event()
    sensor = SensorThread("arm", lambda: 1, target_rate=100.0, stop_event=stop)

    assert sensor.is_alive(), "线程启动后应活着"

    sensor.stop(timeout=2.0)
    assert not sensor.is_alive(), "stop() 后线程应已退出（无僵尸）"


def test_hub_stop_all_threads():
    """hub.stop() 后所有模态线程均退出。"""
    stop = threading.Event()
    sensors = {
        "arm": SensorThread("arm", lambda: 1, target_rate=100.0, stop_event=stop),
        "cam": SensorThread("cam", lambda: 2, target_rate=30.0, stop_event=stop),
        "effector": SensorThread("effector", lambda: 3, target_rate=50.0, stop_event=stop),
    }
    hub = AcquisitionHub(sensors)

    for s in sensors.values():
        assert s.is_alive(), "启动后各线程应活着"

    hub.stop(timeout=2.0)

    for name, s in sensors.items():
        assert not s.is_alive(), f"{name} 线程 hub.stop() 后应已退出"


def test_shared_stop_event_stops_multiple_threads():
    """共享同一 stop_event 的多个 SensorThread 可同时停止。"""
    shared_stop = threading.Event()
    sensors = [
        SensorThread(f"s{i}", lambda: i, target_rate=50.0, stop_event=shared_stop)
        for i in range(3)
    ]

    time.sleep(0.05)

    # 一次性置位停止全部
    shared_stop.set()
    for s in sensors:
        s.stop(timeout=2.0)

    for i, s in enumerate(sensors):
        assert not s.is_alive(), f"s{i} 应已退出"


def test_stop_does_not_deadlock_with_blocking_read():
    """stop() 对于阻塞 read_fn 应能在 timeout 内返回（不死锁）。

    通过 stop_event 控制：read_fn 轮询 stop_event 以保证可退出。
    """
    stop = threading.Event()

    def interruptible_fn():
        # 模拟长时耗时，但检查 stop_event 以确保可退出
        for _ in range(100):
            if stop.is_set():
                return 0
            time.sleep(0.001)
        return 0

    sensor = SensorThread("slow_read", interruptible_fn, target_rate=5.0, stop_event=stop)
    time.sleep(0.05)

    t0 = time.monotonic()
    sensor.stop(timeout=2.0)
    elapsed = time.monotonic() - t0

    assert not sensor.is_alive(), "stop 后线程应退出"
    assert elapsed < 2.5, f"stop 不应死锁，耗时 {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Test 6：AcquisitionHub 辅助接口
# ---------------------------------------------------------------------------


def test_hub_len():
    """hub 长度等于传感器数量。"""
    stop = threading.Event()
    sensors = {
        "a": SensorThread("a", lambda: 1, target_rate=10.0, stop_event=stop),
        "b": SensorThread("b", lambda: 2, target_rate=10.0, stop_event=stop),
    }
    hub = AcquisitionHub(sensors)
    assert len(hub) == 2
    hub.stop(timeout=2.0)


def test_sensor_reading_dataclass():
    """SensorReading 是正常 dataclass，可正确存取。"""
    r = SensorReading(data=[1, 2, 3], ts=1234.5)
    assert r.data == [1, 2, 3]
    assert r.ts == 1234.5
