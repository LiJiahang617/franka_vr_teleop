import importlib.util, os
_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_s = importlib.util.spec_from_file_location(
    "episode_keyboard", os.path.join(_P, "scripts/core/episode_keyboard.py"))
ek = importlib.util.module_from_spec(_s); _s.loader.exec_module(ek)


def test_decide_keep_when_only_exit_early():
    ev = {"exit_early": True, "rerecord_episode": False, "stop_recording": False}
    d = ek.EpisodeDecider(ev)
    assert d.decide_after_episode() == "keep"
    assert d.episode_finished() is True       # exit_early -> 该 ep 采集应提前结束
    d.reset_episode_flags()
    assert ev["exit_early"] is False


def test_decide_discard_when_rerecord():
    ev = {"exit_early": True, "rerecord_episode": True, "stop_recording": False}
    d = ek.EpisodeDecider(ev)
    assert d.decide_after_episode() == "discard"


def test_decide_stop_when_stop_recording():
    ev = {"exit_early": True, "rerecord_episode": False, "stop_recording": True}
    d = ek.EpisodeDecider(ev)
    assert d.decide_after_episode() == "stop"


def test_headless_safe_default_keep():
    # headless: events 全 False -> 按计时结束自动保存(不丢/不停)
    ev = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    d = ek.EpisodeDecider(ev)
    assert d.episode_finished() is False       # 无键 -> 由 episode_sec 计时控制
    assert d.decide_after_episode() == "keep"


def test_stop_flag_callable_reflects_exit_early():
    ev = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    d = ek.EpisodeDecider(ev)
    sf = d.episode_stop_flag()
    assert sf() is False
    ev["exit_early"] = True
    assert sf() is True                        # 录制循环可据此提前结束当前 ep
