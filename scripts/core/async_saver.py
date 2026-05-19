"""异步 episode 存盘器。

有界队列 + 后台单线程：submit() 入队即返回不阻塞录制循环；队列满抛
QueueFullError 不静默丢；后台 _save_loop 串行调 sink(path,payload)=写 hdf5
+validate_episode；close()/__exit__ join 排空保证进程退出前全落盘。

注意：payload 必须是调用方已 deepcopy 的快照，本类不拷贝（deepcopy 时序由
run_record_hdf5.run_episodes 在 buffer 复用前保证）。
close() 为**有意的无超时阻塞关闭**：必须等队列全部落盘才返回以保证零数据
丢失（数据安全优先于活性）；若 sink 卡死 close() 会一直阻塞——属设计取舍，
sink 卡死应由上层/运维另行检测，本类不引入超时以免提前退出丢数据。
"""

import queue
import threading
from typing import Callable


class QueueFullError(RuntimeError):
    """队列已满，submit 拒绝接受新 episode（不静默丢弃）。"""


class SaverClosedError(RuntimeError):
    """saver 已关闭后仍调用 submit（拒绝，防"提交成功但不落盘"静默丢数据）。"""


class AsyncEpisodeSaver:
    """有界队列 + 后台单线程异步存盘器。

    Args:
        sink: Callable[[str, dict], None]，后台线程串行调用（写 hdf5+validate）。
        maxsize: 队列最大深度（默认 5，必须 > 0；<=0 会使 Queue 无界，违背
                 有界背压设计=潜在无界内存/静默积压，故拒绝）。
    """

    def __init__(self, sink: Callable[[str, dict], None], maxsize: int = 5):
        if maxsize <= 0:
            raise ValueError(
                f"maxsize 必须 > 0（got {maxsize}）；<=0 会令 queue.Queue 无界，"
                "违背有界队列背压设计"
            )
        self._sink = sink
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()      # 保护 _err / _closed 跨线程读写
        self._err: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._save_loop, daemon=True)
        self._thread.start()

    def submit(self, path: str, payload: dict) -> None:
        """入队一条 episode，不阻塞。

        Raises:
            SaverClosedError: 已 close() 后再 submit（防静默丢数据）。
            QueueFullError: 队列已满（不静默丢弃）。
            后台 sink 已记录的异常: 透传。
        """
        with self._lock:
            if self._closed:
                raise SaverClosedError("AsyncEpisodeSaver 已关闭，拒绝新 submit")
            if self._err is not None:
                raise self._err
        try:
            self._q.put_nowait((path, payload))
        except queue.Full:
            raise QueueFullError(
                f"AsyncEpisodeSaver 队列已满（maxsize={self._q.maxsize}），"
                "请等待后台写盘完成或减少连录速度"
            ) from None

    def close(self) -> None:
        """放哨兵、等后台排空后返回。幂等（重复 close 安全）。

        无超时阻塞（见模块 docstring）。Raises 后台 sink 首个异常。
        """
        with self._lock:
            if self._closed:
                return                 # 幂等
            self._closed = True
        self._q.put(None)              # 哨兵（队列满则阻塞至腾空——保证排空语义）
        self._thread.join()            # 等排空
        self._raise_if_failed()        # 后台异常浮现

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False                   # 不吞调用方异常

    def _save_loop(self) -> None:
        """后台线程主循环：串行 pop + sink。

        关闭依赖 thread.join()（非 queue.join()）；task_done() 保留仅为将来
        可能引入 queue.join() 预留，当前不参与关闭逻辑。
        """
        while True:
            item = self._q.get()
            try:
                if item is None:       # 哨兵：退出
                    break
                path, payload = item
                try:
                    self._sink(path, payload)
                except Exception as e:  # noqa: BLE001 — 仅 sink 业务异常(写/校验失败)
                    # 记录首个异常，继续排空避免死锁（close join 能正常返回）；
                    # 致命/退出类(BaseException 非 Exception)不在此吞，正常传播。
                    with self._lock:
                        if self._err is None:
                            self._err = e
            finally:
                self._q.task_done()

    def _raise_if_failed(self) -> None:
        with self._lock:
            err = self._err
        if err is not None:
            raise err
