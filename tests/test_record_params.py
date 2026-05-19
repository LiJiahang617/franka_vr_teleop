import importlib.util, os
_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_s = importlib.util.spec_from_file_location(
    "record_params", os.path.join(_P, "scripts/core/record_params.py"))
rp = importlib.util.module_from_spec(_s); _s.loader.exec_module(rp)


def test_resolve_fps_cfg_when_cli_none():
    # CLI 未给 → 用 cfg(单一来源)
    assert rp.resolve_record_fps(None, 30) == 30.0
    assert rp.resolve_record_fps(None, "15") == 15.0


def test_resolve_fps_cli_overrides_cfg():
    # CLI 给了 → 覆盖 cfg(临时覆盖)
    assert rp.resolve_record_fps(60.0, 30) == 60.0


def test_resolve_fps_rejects_nonpositive():
    import pytest
    with pytest.raises(ValueError):
        rp.resolve_record_fps(None, 0)
    with pytest.raises(ValueError):
        rp.resolve_record_fps(-1.0, 30)


def test_resolve_fps_rejects_non_finite():
    # nan/inf 会穿过 fps<=0(nan<=0 为 False; inf 致 1.0/fps=0 忙循环) → 必须显式拒
    import pytest
    with pytest.raises(ValueError):
        rp.resolve_record_fps(None, float("nan"))
    with pytest.raises(ValueError):
        rp.resolve_record_fps(float("inf"), 30)


def test_extract_joint_vel_from_obs():
    import numpy as np
    obs = {f"joint_{i + 1}.vel": float(i) for i in range(7)}
    obs.update({f"joint_{i + 1}.pos": 0.0 for i in range(7)})
    v = rp.extract_joint_vel(obs)
    assert v.shape == (7,) and np.allclose(v, [0, 1, 2, 3, 4, 5, 6])


def test_extract_joint_vel_missing_falls_back_zeros():
    import numpy as np
    v = rp.extract_joint_vel({'joint_1.pos': 0.0})
    assert v.shape == (7,) and np.allclose(v, 0.0)


def test_realsense_fps_returns_int():
    # Task1 起 resolve_record_fps 恒 float; 相机需 int 否则 pyrealsense2 报 TypeError
    v = rp.realsense_fps(30.0)
    assert v == 30 and isinstance(v, int)
    assert rp.realsense_fps(15) == 15 and isinstance(rp.realsense_fps(15), int)


def test_realsense_fps_rounds_near_integer():
    assert rp.realsense_fps(29.999999) == 30
    assert rp.realsense_fps(60.4) == 60


# ================================================================
# parse_bool helpers 单测
# ================================================================

def test_parse_bool_true_variants():
    """True/字符串 true 变体 → True。"""
    import pytest
    for v in (True, "true", "True", "1", "yes", "on"):
        assert rp.parse_bool(v, key_name="k") is True, f"期望 True, got {rp.parse_bool(v, key_name='k')!r} for {v!r}"


def test_parse_bool_false_variants():
    """False/字符串 false 变体 → False。"""
    for v in (False, "false", "False", "0", "no", "off"):
        assert rp.parse_bool(v, key_name="k") is False, f"期望 False, got {rp.parse_bool(v, key_name='k')!r} for {v!r}"


def test_parse_bool_none_uses_default():
    """None → 取 default 参数。"""
    assert rp.parse_bool(None, default=True, key_name="k") is True
    assert rp.parse_bool(None, default=False, key_name="k") is False


def test_parse_bool_invalid_string_raises():
    """非法字符串 → ValueError。"""
    import pytest
    with pytest.raises(ValueError, match="非法 bool 字符串"):
        rp.parse_bool("maybe", key_name="test.key")


def test_parse_bool_non_bool_non_str_raises():
    """int/list/dict 等非 bool 非 str → ValueError。"""
    import pytest
    for bad in (1, 0, [], {}):
        with pytest.raises(ValueError):
            rp.parse_bool(bad, key_name="test.key")


# ================================================================
# parse_section_dict helpers 单测
# ================================================================

def test_parse_section_dict_none_to_empty():
    """None → 空 dict。"""
    assert rp.parse_section_dict(None, key_name="k") == {}


def test_parse_section_dict_dict_passthrough():
    """dict → 原样返回。"""
    d = {"a": 1}
    assert rp.parse_section_dict(d, key_name="k") is d


def test_parse_section_dict_non_dict_raises():
    """str/list/int → ValueError。"""
    import pytest
    for bad in ("bad", [1, 2], 42):
        with pytest.raises(ValueError):
            rp.parse_section_dict(bad, key_name="test.key")


# ================================================================
# parse_positive_int helpers 单测
# ================================================================

def test_parse_positive_int_none_default():
    """None → default。"""
    assert rp.parse_positive_int(None, default=240, key_name="k") == 240


def test_parse_positive_int_positive_ok():
    """正整数 → 自身。"""
    assert rp.parse_positive_int(500, default=240, key_name="k") == 500
    assert rp.parse_positive_int("100", default=240, key_name="k") == 100


def test_parse_positive_int_zero_raises():
    """0 → ValueError。"""
    import pytest
    with pytest.raises(ValueError, match="必须 > 0"):
        rp.parse_positive_int(0, default=240, key_name="k")


def test_parse_positive_int_negative_raises():
    """-1 → ValueError。"""
    import pytest
    with pytest.raises(ValueError, match="必须 > 0"):
        rp.parse_positive_int(-1, default=240, key_name="k")


def test_parse_positive_int_non_numeric_raises():
    """非数字字符串 → ValueError。"""
    import pytest
    with pytest.raises(ValueError):
        rp.parse_positive_int("abc", default=240, key_name="k")


def test_parse_positive_int_bool_rejected():
    """bool 值（True/False）→ ValueError（防 bool 子类 int 误用）。"""
    import pytest
    with pytest.raises(ValueError, match="bool"):
        rp.parse_positive_int(True, default=240, key_name="k")
    with pytest.raises(ValueError, match="bool"):
        rp.parse_positive_int(False, default=240, key_name="k")


# ================================================================
# parse_axis_gain helpers 单测
# ================================================================

def test_parse_axis_gain_none_default():
    """None → default [1.0, 1.0, 1.0]。"""
    result = rp.parse_axis_gain(None, key_name="k")
    assert result == [1.0, 1.0, 1.0]


def test_parse_axis_gain_valid_unit():
    """合法 [1,1,1] → [1.0,1.0,1.0]。"""
    assert rp.parse_axis_gain([1, 1, 1], key_name="k") == [1.0, 1.0, 1.0]


def test_parse_axis_gain_valid_custom():
    """合法 [2.0, 3.0, 0.5] → 原样 float 列表。"""
    assert rp.parse_axis_gain([2.0, 3.0, 0.5], key_name="k") == [2.0, 3.0, 0.5]


def test_parse_axis_gain_wrong_len_raises():
    """长度 != 3 → ValueError。"""
    import pytest
    with pytest.raises(ValueError, match="len==3"):
        rp.parse_axis_gain([1.0, 2.0], key_name="k")
    with pytest.raises(ValueError, match="len==3"):
        rp.parse_axis_gain([1.0, 2.0, 3.0, 4.0], key_name="k")


def test_parse_axis_gain_non_list_raises():
    """非 list/tuple → ValueError。"""
    import pytest
    with pytest.raises(ValueError, match="list/tuple"):
        rp.parse_axis_gain("1,2,3", key_name="k")
    with pytest.raises(ValueError, match="list/tuple"):
        rp.parse_axis_gain(1.0, key_name="k")


def test_parse_axis_gain_nan_raises():
    """含 nan → ValueError（有限性检查）。"""
    import pytest
    with pytest.raises(ValueError, match="有限"):
        rp.parse_axis_gain([1.0, float("nan"), 1.0], key_name="k")


def test_parse_axis_gain_inf_raises():
    """含 inf → ValueError（有限性检查）。"""
    import pytest
    with pytest.raises(ValueError, match="有限"):
        rp.parse_axis_gain([1.0, float("inf"), 1.0], key_name="k")


def test_parse_axis_gain_non_numeric_raises():
    """元素非数字 → ValueError。"""
    import pytest
    with pytest.raises(ValueError):
        rp.parse_axis_gain([1.0, "bad", 1.0], key_name="k")


def test_parse_axis_gain_bool_element_rejected():
    """元素为 bool → ValueError（防 True→1.0 误用）。"""
    import pytest
    with pytest.raises(ValueError, match="bool"):
        rp.parse_axis_gain([True, 1.0, 1.0], key_name="k")
"""追加到 test_record_params.py 的 resolve_record_overrides 测试。"""
import importlib.util, os
from types import SimpleNamespace

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_s = importlib.util.spec_from_file_location(
    "record_params", os.path.join(_P, "scripts/core/record_params.py"))
rp = importlib.util.module_from_spec(_s); _s.loader.exec_module(rp)


def _make_cfg(*, episodes=10, episode_sec=120.0, out_dir="/data/recs",
              task_description="pick apple", oc2base_path="/cal/R.npy"):
    """构造假 RecordConfig-like 对象（纯 SimpleNamespace，不依赖硬件）。"""
    return SimpleNamespace(
        num_episodes=episodes,
        episode_time_sec=episode_sec,
        out_dir=out_dir,
        task_description=task_description,
        oc2base_path=oc2base_path,
    )


_FALLBACK_DIR = "/default/fallback"


# ================================================================
# resolve_record_overrides 纯函数测
# ================================================================

def test_resolve_overrides_all_cli_none_takes_cfg():
    """CLI 全 None → 取 cfg 各字段。"""
    cfg = _make_cfg()
    res = rp.resolve_record_overrides(
        cli_episodes=None, cli_episode_sec=None, cli_out_dir=None,
        cli_task_name=None, cli_oc2base=None,
        record_cfg=cfg, out_dir_fallback=_FALLBACK_DIR,
    )
    assert res["episodes"] == cfg.num_episodes
    assert res["episode_sec"] == cfg.episode_time_sec
    assert res["out_dir"] == cfg.out_dir
    assert res["task_name"] == cfg.task_description
    assert res["oc2base_path"] == cfg.oc2base_path


def test_resolve_overrides_cli_values_override_cfg():
    """CLI 给值 → 覆盖 cfg 对应字段。"""
    cfg = _make_cfg()
    res = rp.resolve_record_overrides(
        cli_episodes=3, cli_episode_sec=30.0, cli_out_dir="/tmp/out",
        cli_task_name="custom_task", cli_oc2base="/cal/custom_R.npy",
        record_cfg=cfg, out_dir_fallback=_FALLBACK_DIR,
    )
    assert res["episodes"] == 3
    assert res["episode_sec"] == 30.0
    assert res["out_dir"] == "/tmp/out"
    assert res["task_name"] == "custom_task"
    assert res["oc2base_path"] == "/cal/custom_R.npy"


def test_resolve_overrides_partial_cli_mixed():
    """部分 CLI 给值，部分 None → 给了的覆盖，None 的回退 cfg。"""
    cfg = _make_cfg()
    res = rp.resolve_record_overrides(
        cli_episodes=5, cli_episode_sec=None, cli_out_dir=None,
        cli_task_name="cli_task", cli_oc2base=None,
        record_cfg=cfg, out_dir_fallback=_FALLBACK_DIR,
    )
    assert res["episodes"] == 5                         # CLI 覆盖
    assert res["episode_sec"] == cfg.episode_time_sec   # cfg 回退
    assert res["out_dir"] == cfg.out_dir                # cfg 回退
    assert res["task_name"] == "cli_task"               # CLI 覆盖
    assert res["oc2base_path"] == cfg.oc2base_path      # cfg 回退


def test_resolve_overrides_out_dir_two_level_fallback():
    """out_dir: CLI None 且 cfg.out_dir None → 回退 out_dir_fallback 常量。"""
    cfg = _make_cfg(out_dir=None)
    res = rp.resolve_record_overrides(
        cli_episodes=None, cli_episode_sec=None, cli_out_dir=None,
        cli_task_name=None, cli_oc2base=None,
        record_cfg=cfg, out_dir_fallback=_FALLBACK_DIR,
    )
    assert res["out_dir"] == _FALLBACK_DIR


def test_resolve_overrides_out_dir_cli_wins_over_both():
    """CLI 给 out_dir → 覆盖 cfg.out_dir 和 fallback。"""
    cfg = _make_cfg(out_dir=None)
    res = rp.resolve_record_overrides(
        cli_episodes=None, cli_episode_sec=None, cli_out_dir="/cli/dir",
        cli_task_name=None, cli_oc2base=None,
        record_cfg=cfg, out_dir_fallback=_FALLBACK_DIR,
    )
    assert res["out_dir"] == "/cli/dir"


def test_resolve_overrides_task_name_falls_back_to_description():
    """task_name CLI None → 回退 cfg.task_description（非 cli or cfg falsy 误判）。"""
    cfg = _make_cfg(task_description="detailed task desc")
    res = rp.resolve_record_overrides(
        cli_episodes=None, cli_episode_sec=None, cli_out_dir=None,
        cli_task_name=None, cli_oc2base=None,
        record_cfg=cfg, out_dir_fallback=_FALLBACK_DIR,
    )
    assert res["task_name"] == "detailed task desc"


def test_resolve_overrides_strict_is_none_episodes_zero_not_falsy():
    """严格 is None: CLI episodes=0 (falsy 但非 None) → 应覆盖，不被误当未给。"""
    cfg = _make_cfg(episodes=10)
    res = rp.resolve_record_overrides(
        cli_episodes=0, cli_episode_sec=None, cli_out_dir=None,
        cli_task_name=None, cli_oc2base=None,
        record_cfg=cfg, out_dir_fallback=_FALLBACK_DIR,
    )
    assert res["episodes"] == 0   # 0 是显式 CLI，不能被 or 误当 None


def test_resolve_overrides_strict_is_none_episode_sec_zero():
    """严格 is None: CLI episode_sec=0.0 → 覆盖，不回退 cfg。"""
    cfg = _make_cfg(episode_sec=120.0)
    res = rp.resolve_record_overrides(
        cli_episodes=None, cli_episode_sec=0.0, cli_out_dir=None,
        cli_task_name=None, cli_oc2base=None,
        record_cfg=cfg, out_dir_fallback=_FALLBACK_DIR,
    )
    assert res["episode_sec"] == 0.0


def test_resolve_overrides_strict_is_none_empty_string_task_name():
    """严格 is None: CLI task_name='' (falsy 但非 None) → 覆盖，不回退 cfg.task_description。"""
    cfg = _make_cfg(task_description="from cfg")
    res = rp.resolve_record_overrides(
        cli_episodes=None, cli_episode_sec=None, cli_out_dir=None,
        cli_task_name="", cli_oc2base=None,
        record_cfg=cfg, out_dir_fallback=_FALLBACK_DIR,
    )
    assert res["task_name"] == ""   # "" 是 CLI 显式给定，不应回退
