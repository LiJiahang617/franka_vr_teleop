import numpy as np
from scipy.spatial.transform import Rotation
import lerobot_teleoperator_franka.vr_align as vr_align


def test_solve_rotation_recovers_known_R_two_pairs():
    R_true = Rotation.random(random_state=42).as_matrix()
    d_oc = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    d_arm = (R_true @ d_oc.T).T
    R = vr_align.solve_rotation(d_oc, d_arm)
    assert np.allclose(R, R_true, atol=1e-6)


def test_solve_rotation_ignores_per_pair_scale():
    R_true = Rotation.random(random_state=7).as_matrix()
    d_oc = np.array([[2.0, 0.1, 0.0], [0.0, 0.05, 3.0]])
    d_arm = (R_true @ (d_oc * 0.01).T).T
    R = vr_align.solve_rotation(d_oc * 5.0, d_arm)
    assert np.allclose(R, R_true, atol=1e-6)


def test_solve_rotation_robust_to_small_noise():
    R_true = Rotation.random(random_state=3).as_matrix()
    rng = np.random.default_rng(2)
    d_oc = rng.normal(size=(4, 3))
    d_arm = (R_true @ d_oc.T).T + rng.normal(scale=1e-3, size=(4, 3))
    R = vr_align.solve_rotation(d_oc, d_arm)
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(R.T @ R_true).as_rotvec()))
    assert ang < 1.0


def test_validate_rotation_accepts_proper_rotation():
    R = Rotation.random(random_state=11).as_matrix()
    ok, ortho_err, det = vr_align.validate_rotation(R)
    assert ok and ortho_err < 1e-6 and abs(det - 1.0) < 1e-3


def test_validate_rotation_rejects_non_orthonormal():
    ok, _, _ = vr_align.validate_rotation(np.diag([1.0, 1.0, 2.0]))
    assert not ok


def test_validate_rotation_rejects_reflection():
    ok, _, det = vr_align.validate_rotation(np.diag([1.0, 1.0, -1.0]))
    assert not ok and det < 0


def test_gesture_quality_clean_data():
    R_true = Rotation.random(random_state=5).as_matrix()
    d_arm = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    d_oc = [R_true.T @ d_arm[0], R_true.T @ d_arm[1]]
    oc_int, arm_int, recon = vr_align.gesture_pair_quality(d_oc, d_arm)
    assert abs(arm_int - 90.0) < 1e-6
    assert abs(oc_int - 90.0) < 1e-6
    assert recon < 1e-6


def test_gesture_quality_flags_near_parallel():
    d_arm = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    d_oc = [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.05])]
    oc_int, arm_int, recon = vr_align.gesture_pair_quality(d_oc, d_arm)
    assert abs(oc_int - arm_int) > 15.0


def test_gesture_quality_requires_exactly_2():
    import pytest
    with pytest.raises(ValueError):
        vr_align.gesture_pair_quality(np.eye(3), np.eye(3))


def test_save_then_load_roundtrip(tmp_path):
    R = Rotation.random(random_state=21).as_matrix()
    p = str(tmp_path / 'R.npy')
    q = {'oc_inter_deg': 88.0, 'angle_err_deg': 2.0, 'recon_max_deg': 0.5}
    vr_align.save_rotation(p, R, q, oc_ref_rotvec=[0.1, 0.2, 0.3])
    loaded = vr_align.load_rotation(p)
    assert loaded is not None
    R2, meta = loaded
    assert np.allclose(R2, R, atol=1e-12)
    assert abs(meta['quality']['angle_err_deg'] - 2.0) < 1e-9
    assert np.allclose(meta['oc_ref_rotvec'], [0.1, 0.2, 0.3])


def test_load_missing_returns_none(tmp_path):
    assert vr_align.load_rotation(str(tmp_path / 'nope.npy')) is None


def test_resolve_mapping_prefers_calibrated():
    R = Rotation.random(random_state=31).as_matrix()
    pm, rm, ps, rs, mode = vr_align.resolve_mapping(
        R, np.eye(3), np.eye(3), np.array([1.0, -1.0, 1.0]), np.array([-1.0, -1.0, 1.0]))
    assert mode == 'calibrated'
    assert np.allclose(pm, R) and np.allclose(rm, R)
    assert np.allclose(ps, [1, 1, 1]) and np.allclose(rs, [1, 1, 1])


def test_resolve_mapping_falls_back_to_legacy():
    legacy_p = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
    pm, rm, ps, rs, mode = vr_align.resolve_mapping(
        None, legacy_p, np.eye(3), np.array([1.0, 1.0, 1.0]), np.array([-1.0, -1.0, 1.0]))
    assert mode == 'legacy'
    assert np.allclose(pm, legacy_p) and np.allclose(rs, [-1, -1, 1])
