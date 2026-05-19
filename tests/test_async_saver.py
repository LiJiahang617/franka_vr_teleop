import importlib.util, os, threading, time
import pytest

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_s = importlib.util.spec_from_file_location(
    "async_saver", os.path.join(_P, "scripts/core/async_saver.py"))
asv = importlib.util.module_from_spec(_s); _s.loader.exec_module(asv)


def test_submit_runs_sink_in_background_and_join_drains():
    saved = []
    lock = threading.Lock()

    def sink(path, payload):
        time.sleep(0.05)              # 模拟写盘+validate 耗时
        with lock:
            saved.append((path, payload["v"]))

    s = asv.AsyncEpisodeSaver(sink=sink, maxsize=5)
    t0 = time.monotonic()
    s.submit("/tmp/a.h5", {"v": 1})
    s.submit("/tmp/b.h5", {"v": 2})
    # submit 不阻塞: 两次 submit 远快于 2*0.05s 串行写盘
    assert time.monotonic() - t0 < 0.05
    s.close()                         # join 排空
    assert sorted(saved) == [("/tmp/a.h5", 1), ("/tmp/b.h5", 2)]


def test_queue_full_raises_not_silent_drop():
    started = threading.Event()
    release = threading.Event()

    def sink(path, payload):
        started.set()
        release.wait(timeout=5)       # 卡住后台线程, 撑满队列

    s = asv.AsyncEpisodeSaver(sink=sink, maxsize=2)
    s.submit("/p0", {})               # 进后台线程(卡住)
    started.wait(timeout=2)
    s.submit("/p1", {})               # 填队列
    s.submit("/p2", {})               # 填满 (maxsize=2)
    with pytest.raises(asv.QueueFullError):
        s.submit("/p3", {})           # 满 -> 报错不静默丢
    release.set()
    s.close()


def test_sink_exception_surfaces_not_swallowed():
    def sink(path, payload):
        raise RuntimeError("validate failed in bg")

    s = asv.AsyncEpisodeSaver(sink=sink, maxsize=5)
    s.submit("/x", {})
    with pytest.raises(RuntimeError, match="validate failed in bg"):
        s.close()                     # 后台异常须在 close/下次 submit 浮现


def test_context_manager_joins_on_exit():
    saved = []
    with asv.AsyncEpisodeSaver(sink=lambda p, d: saved.append(p), maxsize=5) as s:
        s.submit("/c", {})
    assert saved == ["/c"]            # __exit__ 已 join 排空


# ── 新增 3 个 review-fix 测试 ──────────────────────────────────────────────

def test_submit_after_close_raises():
    """close() 后再 submit 必须抛 SaverClosedError，不得静默丢数据。"""
    s = asv.AsyncEpisodeSaver(sink=lambda p, d: None, maxsize=5)
    s.close()
    with pytest.raises(asv.SaverClosedError):
        s.submit("/after_close", {})


def test_close_is_idempotent():
    """连续两次 close() 不抛、不挂（第二次立即返回）。"""
    s = asv.AsyncEpisodeSaver(sink=lambda p, d: None, maxsize=5)
    s.close()
    s.close()                         # 幂等，不应挂或抛


def test_maxsize_must_be_positive():
    """maxsize=0 或负数均须抛 ValueError（防无界队列破坏背压设计）。"""
    sink = lambda p, d: None
    with pytest.raises(ValueError):
        asv.AsyncEpisodeSaver(sink=sink, maxsize=0)
    with pytest.raises(ValueError):
        asv.AsyncEpisodeSaver(sink=sink, maxsize=-1)
