"""终端键盘起止/丢弃控制 — EpisodeDecider。

复用 run_record.py 既有模式：
  lerobot ``init_keyboard_listener()`` → ``(listener, events)``

键位（与 run_record 语义一致，零学习成本）：
  →（exit_early）     = 结束当前 ep 并**保存**（keep）
  ←（rerecord_episode）= 结束当前 ep 并**丢弃**（discard）
  Esc（stop_recording）= **停止**录制（stop）

headless 降级：
  ``is_headless()`` 时 ``init_keyboard_listener`` 返回 listener=None，
  events 全 False → ``episode_finished()`` 返回 False（不提前结束），
  ``decide_after_episode()`` 返回 "keep"（按 episode_sec 计时后自动保存）。
  无键盘时不崩、不卡死，CI/headless 场景安全。

使用方式（在 run_record_hdf5.main 内）：
    from lerobot.utils.control_utils import init_keyboard_listener
    from core.episode_keyboard import EpisodeDecider

    listener, events = init_keyboard_listener()
    dec = EpisodeDecider(events)

    # 录制循环
    stop_flag = dec.episode_stop_flag()      # 传给 record_episode 提前结束
    ...
    action = dec.decide_after_episode()      # 录完后拿决策
    dec.reset_episode_flags()               # 下一条前重置
"""


class EpisodeDecider:
    """把 lerobot events dict 翻译为 decide(ep) 决策器。

    Args:
        events: lerobot ``init_keyboard_listener()`` 返回的 events dict，
                包含键 "exit_early"、"rerecord_episode"、"stop_recording"。
                headless 时三者均为 False（安全降级）。
    """

    def __init__(self, events: dict) -> None:
        self._ev = events

    def episode_stop_flag(self):
        """返回 callable()->bool，供 record_episode 的节拍循环查询提前结束。

        返回 True 当且仅当 exit_early 或 stop_recording 置位。
        headless 时 events 全 False → 恒返回 False（纯计时控制，行为等价旧版）。
        """
        ev = self._ev
        return lambda: bool(ev["exit_early"] or ev["stop_recording"])

    def episode_finished(self) -> bool:
        """当前 ep 的采集是否应提前结束（任意退出键置位则 True）。"""
        return bool(self._ev["exit_early"] or self._ev["stop_recording"])

    def decide_after_episode(self) -> str:
        """读取 events 状态，返回 "keep"/"discard"/"stop"。

        优先级：stop_recording > rerecord_episode > keep（含 headless 全 False）。
        """
        ev = self._ev
        if ev["stop_recording"]:
            return "stop"
        if ev["rerecord_episode"]:
            return "discard"
        return "keep"

    def reset_episode_flags(self) -> None:
        """下一条 ep 开始前重置逐 ep 标志。

        仿 run_record.py 既有模式：
          events["rerecord_episode"] = False
          events["exit_early"] = False
        保留 stop_recording（全局停止标志，不在 ep 间重置）。
        """
        self._ev["exit_early"] = False
        self._ev["rerecord_episode"] = False
