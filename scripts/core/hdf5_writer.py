"""HDF5EpisodeWriter：按 franka-hdf5-v1 写一 episode（缓冲→close 时整体 flush + 自检）。

编码前提说明（中文）
--------------------
``frames[i]["cams"][cn]`` 进来时**必须已是编码后的 JPEG uint8 bytes**。
编码由上层 ``run_record_hdf5._encode_jpg`` 完成（``cvtColor(RGB->BGR)``->``imencode``），
须在 ``deepcopy`` 前执行。``write_episode`` 可安全在后台线程中调用（含 ``validate_episode``）。
"""
import json
import sys

import h5py
import numpy as np

from core.schema_loader import load_franka_hdf5_schema
S = load_franka_hdf5_schema()

_VLEN_BYTES = h5py.special_dtype(vlen=np.dtype("uint8"))


def write_episode(path, frames, *, task_name, target_fps, oc2base_R, quality,
                  vr_source, cam_names):
    """把一条 episode 的帧列表写入 HDF5 文件并验证合规性。

    参数说明（中文）：
        path: 输出 .h5 文件路径（字符串）。
        frames: list[dict]，每帧含 ts/joints/joint_vel/ee_pose/gripper_m/
                gripper_norm/gripper_cmd/delta_ee_pose/cams；
                ``cams[cn]`` 须已是 JPEG 编码后的 uint8 bytes（编码在上层完成）。
        task_name: 任务名称（字符串）。
        target_fps: 目标帧率（float）。
        oc2base_R: 3x3 旋转矩阵（ndarray）。
        quality: 质量参数字典（dict），JSON 序列化写入。
        vr_source: VR 来源标识（字符串）。
        cam_names: 相机名列表（list[str]）。

    可安全在后台线程中调用（含末尾 validate_episode）。
    """
    n = len(frames)
    if n == 0:
        raise ValueError("空 episode（0 帧），拒绝写盘")
    b = frames
    _task = str(task_name)
    _target_fps = float(target_fps)
    _R = np.asarray(oc2base_R, np.float64).reshape(3, 3)
    _quality = dict(quality)
    _vr_source = str(vr_source)
    _cams = list(cam_names)

    ts = np.array([f["ts"] for f in b], np.float64)
    avg = float((n - 1) / (ts[-1] - ts[0])) if n > 1 and ts[-1] > ts[0] else _target_fps
    with h5py.File(path, "w") as f:
        inf = f.create_group("infos")
        inf.create_dataset("schema_version", data=np.bytes_(S.SCHEMA_VERSION))
        ti = inf.create_group("task_info")
        ti.create_dataset("task_name", data=np.bytes_(_task))
        ti.create_dataset("collection_frequency",
                          data=np.array([_target_fps, avg], np.float64))
        ti.create_dataset("total_frames", data=np.int64(n))
        ti.create_dataset("robot", data=np.bytes_("franka_panda"))
        inf.create_group("camera_params")
        cal = inf.create_group("calibration")
        cal.create_dataset("oc2base_R", data=_R)
        cal.create_dataset("quality", data=np.bytes_(json.dumps(_quality)))
        cal.create_dataset("vr_source", data=np.bytes_(_vr_source))

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
        for cn in _cams:
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

    violations = S.validate_episode(path)
    if violations:
        raise RuntimeError(f"写出的 episode 不符 franka-hdf5-v1: {violations}")


class HDF5EpisodeWriter:
    """按 franka-hdf5-v1 schema 缓冲帧并在 close() 时整体写盘。

    同步路径（close()）与异步路径（后台线程调 write_episode）共用同一落盘实现，
    写出的 .h5 字节与重构前完全一致（行为零变化）。
    """

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
        """同步写盘并验证（行为与重构前零变化）。"""
        write_episode(
            self._path,
            self._buf,
            task_name=self._task,
            target_fps=self._target_fps,
            oc2base_R=self._R,
            quality=self._quality,
            vr_source=self._vr_source,
            cam_names=self._cams,
        )
