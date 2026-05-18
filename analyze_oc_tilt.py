"""离线读旧 _stage3_*.csv，量化旧映射下 oc_y 相对真实竖直的倾角。\n\n启发式：取 rg=1 段，段末映射后 rel_arm 主导轴为 +Z 的段，算净 rel_vr 与 oc_y [0,1,0] 的夹角。\n"""
import csv
import sys

import numpy as np


def _as_float(row, key):
    return float(row[key])


def _read_rows(csv_path):
    with open(csv_path, newline='') as f:
        return list(csv.DictReader(f))


def _rg_segments(rows):
    cur = []
    prev_rg = 0
    for row in rows:
        rg = int(float(row['rg']))
        if rg == 1:
            cur.append(row)
        elif prev_rg == 1 and cur:
            yield cur
            cur = []
        prev_rg = rg
    if cur:
        yield cur


def oc_y_tilt_deg_from_csv(csv_path, min_disp_m=0.05):
    rows = _read_rows(csv_path)
    tilts = []
    for sg in _rg_segments(rows):
        if len(sg) < 10:
            continue
        last = sg[-1]
        rv = np.array([
            _as_float(last, 'rel_vr_x'),
            _as_float(last, 'rel_vr_y'),
            _as_float(last, 'rel_vr_z'),
        ])
        ra = np.array([
            _as_float(last, 'rel_arm_x'),
            _as_float(last, 'rel_arm_y'),
            _as_float(last, 'rel_arm_z'),
        ])
        if np.linalg.norm(rv) < min_disp_m:
            continue
        if int(np.argmax(np.abs(ra))) != 2 or ra[2] <= 0:
            continue
        u = rv / np.linalg.norm(rv)
        tilts.append(float(np.degrees(np.arccos(np.clip(abs(u[1]), -1.0, 1.0)))))
    if not tilts:
        return None
    return float(np.mean(tilts)), len(tilts)


if __name__ == '__main__':
    for path in sys.argv[1:]:
        r = oc_y_tilt_deg_from_csv(path)
        msg = '无竖直段' if r is None else f'倾角均值 {r[0]:.1f} deg (n={r[1]})'
        print(f'{path}: {msg}')
