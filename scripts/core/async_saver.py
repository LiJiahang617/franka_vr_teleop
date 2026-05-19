"""异步 episode 存盘器。

将有界队列 + 后台单线程组合成生产级异步存盘组件：
- submit(path, payload) 入队即返回（O(1)），不阻塞录制循环。
- 队列满时抛 QueueFullError（不静默丢弃）。
- 后台 _save_loop 串行调用注入的 sink(path, payload)。
- close() / __exit__ join 排空队列，保证进程退出前所有 episode 落盘。

注意：payload 必须是调用方已 deepcopy 的快照，本类不负责拷贝；
deepcopy 时序由 run_record_hdf5.run_episodes 保证（在 buffer 复用前 deepcopy）。

sink 由调用方注入，默认接 write_episode（见 hdf5_writer），也可注入 fake_sink 做离线单测。
"""

import queue
import threading


class QueueFullError(RuntimeError):
    """队列已满，submit 拒绝接受新 episode（不静默丢弃）。"""


class AsyncEpisodeSaver:
    """有界队列 + 后台单线程异步存盘器。

    Args:
        sink: callable(path: str, payload: dict) -> None，由调用方注入。
              在后台线程串行调用，含写 hdf5 + validate_episode。
        maxsize: 队列最大深度（默认 5）。超出时 submit 抛 QueueFullError。
    """

    def __init__(self, sink, maxsize: int = 5):
        self._sink = sink
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._err: BaseException | None = None
        # daemon=True：主线程退出时后台线程随之结束（close/join 保证正常排空）
        self._thread = threading.Thread(target=self._save_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def submit(self, path: str, payload: dict) -> None:
        """入队一条 episode，不阻塞。

        Args:
            path: 目标 .h5 文件路径。
            payload: 已 deepcopy 的帧快照字典（由调用方保证）。

        Raises:
            QueueFullError: 队列已满时抛出，不静默丢弃。
            后台已记录的异常（BaseException）: 透传给调用方。
        """
        # 先检查后台是否已报错，避免调用方无感知地持续入队
        self._raise_if_failed()
        try:
            self._q.put_nowait((path, payload))
        except queue.Full:
            raise QueueFullError(
                f"AsyncEpisodeSaver 队列已满（maxsize={self._q.maxsize}），"
                "请等待后台写盘完成或减少连录速度"
            )

    def close(self) -> None:
        """放哨兵、等待后台线程排空后返回。

        Raises:
            后台 sink 抛出的首个异常（在此浮现）。
        """
        self._q.put(None)          # 哨兵通知 _save_loop 退出
        self._thread.join()        # 等排空
        self._raise_if_failed()    # 后台异常在此浮现

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False               # 不吞调用方异常

    # ------------------------------------------------------------------
    # 后台线程
    # ------------------------------------------------------------------

    def _save_loop(self) -> None:
        """后台线程主循环：串行 pop + sink。"""
        while True:
            item = self._q.get()
            try:
                if item is None:       # 哨兵：退出
                    break
                path, payload = item
                try:
                    self._sink(path, payload)
                except BaseException as e:
                    # 记录首个异常，继续排空避免死锁（close join 能正常返回）
                    if self._err is None:
                        self._err = e
            finally:
                self._q.task_done()

    def _raise_if_failed(self) -> None:
        """若后台线程已捕获异常，在调用方线程重新抛出。"""
        if self._err is not None:
            raise self._err
