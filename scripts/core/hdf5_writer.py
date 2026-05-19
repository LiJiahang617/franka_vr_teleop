"""HDF5EpisodeWriter：按 franka-hdf5-v1 写一 episode（缓冲→close 时整体 flush + 自检）。"""
import json
import sys

import h5py
import numpy as np

import importlib.util as _ilu, os as _os
if 'franka_hdf5_schema' in sys.modules:
    S = sys.modules['franka_hdf5_schema']  # 复用单一实例(还原旧 import 缓存语义)
else:
    _schema_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__)))), 'franka_hdf5_schema.py')
    _spec = _ilu.spec_from_file_location('franka_hdf5_schema', _schema_path)
    S = _ilu.module_from_spec(_spec)
    sys.modules['franka_hdf5_schema'] = S  # exec 前注册(importlib 规范, 防重复实例)
    _spec.loader.exec_module(S)  # noqa: E402

_VLEN_BYTES = h5py.special_dtype(vlen=np.dtype("uint8"))


class HDF5EpisodeWriter:
    def __init__(self, path, task_name, target_fps, oc2base_R, quality,
                 vr_source, cam_names):
        self._path = path
        self._task = str(task_name)
        self._target_fps = float(target_fps)
        self._R = np.asarray(oc2base_R, np.float64).reshape(3, 3)
        self._quality = dict(quality)
        self._vr_source = str(vr_source)
        self._cams = list(cam_names)
        self._buf = []

    def add(self, frame: dict):
        self._buf.append(frame)

    def close(self):
        n = len(self._buf)
        if n == 0:
            raise ValueError("空 episode（0 帧），拒绝写盘")
        b = self._buf
        ts = np.array([f["ts"] for f in b], np.float64)
        avg = float((n - 1) / (ts[-1] - ts[0])) if n > 1 and ts[-1] > ts[0] else self._target_fps
        with h5py.File(self._path, "w") as f:
            inf = f.create_group("infos")
            inf.create_dataset("schema_version", data=np.bytes_(S.SCHEMA_VERSION))
            ti = inf.create_group("task_info")
            ti.create_dataset("task_name", data=np.bytes_(self._task))
            ti.create_dataset("collection_frequency",
                              data=np.array([self._target_fps, avg], np.float64))
            ti.create_dataset("total_frames", data=np.int64(n))
            ti.create_dataset("robot", data=np.bytes_("franka_panda"))
            inf.create_group("camera_params")
            cal = inf.create_group("calibration")
            cal.create_dataset("oc2base_R", data=self._R)
            cal.create_dataset("quality", data=np.bytes_(json.dumps(self._quality)))
            cal.create_dataset("vr_source", data=np.bytes_(self._vr_source))

            obs = f.create_group("observations")
            obs.create_dataset("timestamp", data=ts.reshape(n, 1))
            arm = obs.create_group("arm")
            arm.create_dataset("joints", data=np.stack([np.asarray(x["joints"], np.float64) for x in b]))
            arm.create_dataset("joint_vel", data=np.stack([np.asarray(x["joint_vel"], np.float64) for x in b]))
            arm.create_dataset("pose", data=np.stack([np.asarray(x["ee_pose"], np.float64) for x in b]))
            arm.create_dataset("timestamp", data=ts.copy())
            eff = obs.create_group("effector")
            eff.create_dataset("position", data=np.array([[x["gripper_m"]] for x in b], np.float64))
            eff.create_dataset("position_norm", data=np.array([[x["gripper_norm"]] for x in b], np.float64))
            eff.create_dataset("type", data=np.array([b"gripper"] * n,
                               dtype=h5py.special_dtype(vlen=bytes)))
            eff.create_dataset("timestamp", data=ts.copy())
            cam = obs.create_group("camera")
            rgb = cam.create_group("rgb")
            for cn in self._cams:
                g = rgb.create_group(cn)
                imgs = g.create_dataset("images", (n,), dtype=_VLEN_BYTES)
                for i, x in enumerate(b):
                    imgs[i] = np.frombuffer(bytes(x["cams"][cn]), np.uint8)
                g.create_dataset("timestamp", data=ts.copy())
            hf = obs.create_group("state_hifreq")  # 占位空数组, SP-4 填
            hf.create_dataset("joints", data=np.zeros((0, 7), np.float64))
            hf.create_dataset("joint_vel", data=np.zeros((0, 7), np.float64))
            hf.create_dataset("pose", data=np.zeros((0, 6), np.float64))
            hf.create_dataset("timestamp", data=np.zeros((0,), np.float64))
            hf.create_dataset("poly_ts", data=np.zeros((0,), np.float64))

            act = f.create_group("action")
            act.create_dataset("delta_ee_pose",
                               data=np.stack([np.asarray(x["delta_ee_pose"], np.float64) for x in b]))
            act.create_dataset("gripper_cmd", data=np.array([[x["gripper_cmd"]] for x in b], np.float64))
            act.create_dataset("timestamp", data=ts.copy())

        violations = S.validate_episode(self._path)
        if violations:
            raise RuntimeError(f"写出的 episode 不符 franka-hdf5-v1: {violations}")
