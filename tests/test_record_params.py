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
