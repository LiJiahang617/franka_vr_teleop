import csv
import numpy as np
import analyze_oc_tilt


FIELDNAMES = ['rg', 'rel_vr_x', 'rel_vr_y', 'rel_vr_z', 'rel_arm_x', 'rel_arm_y', 'rel_arm_z']


def _write_rows(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _make_csv(path, tilt_deg):
    a = np.radians(tilt_deg)
    rows = [{
        'rg': 0,
        'rel_vr_x': 0,
        'rel_vr_y': 0,
        'rel_vr_z': 0,
        'rel_arm_x': 0,
        'rel_arm_y': 0,
        'rel_arm_z': 0,
    } for _ in range(5)]
    for k in range(20):
        f = (k + 1) / 20.0
        rows.append({
            'rg': 1,
            'rel_vr_x': 0.2 * np.sin(a) * f,
            'rel_vr_y': 0.2 * np.cos(a) * f,
            'rel_vr_z': 0.0,
            'rel_arm_x': 0.01 * f,
            'rel_arm_y': 0.0,
            'rel_arm_z': 0.2 * f,
        })
    _write_rows(path, rows)


def test_tilt_recovered(tmp_path):
    p = str(tmp_path / 's.csv')
    _make_csv(p, 18.0)
    mean_deg, n = analyze_oc_tilt.oc_y_tilt_deg_from_csv(p)
    assert n == 1 and abs(mean_deg - 18.0) < 0.5


def test_no_vertical_segment_returns_none(tmp_path):
    p = str(tmp_path / 'e.csv')
    _write_rows(p, [{
        'rg': 0,
        'rel_vr_x': 0,
        'rel_vr_y': 0,
        'rel_vr_z': 0,
        'rel_arm_x': 0,
        'rel_arm_y': 0,
        'rel_arm_z': 0,
    }])
    assert analyze_oc_tilt.oc_y_tilt_deg_from_csv(p) is None
