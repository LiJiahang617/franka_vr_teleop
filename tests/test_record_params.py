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
