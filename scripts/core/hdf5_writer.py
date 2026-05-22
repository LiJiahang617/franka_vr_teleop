"""HDF5EpisodeWriter：按 franka-hdf5-v2 写一 episode（缓冲→close 时整体 flush + 自检）。

编码前提说明（中文）
--------------------
``frames[i]["cams"][cn]`` 进来时**必须已是编码后的 JPEG uint8 bytes**。
编码由上层 ``run_record_hdf5._encode_jpg`` 完成（``cvtColor(RGB->BGR)``->``imencode``），
须在 ``deepcopy`` 前执行。``write_episode`` 可安全在后台线程中调用（含 ``validate_episode``）。

v2 变更（相对 v1）：
  - 删除共用时间戳 observations/timestamp(N,1)
  - 每帧每模态带独立时间戳字段：arm_ts / effector_ts / cam_ts / cam_hw_ts（dict by cam）
  - 每模态写 stale(N,) bool：arm_stale / effector_stale / cam_stale（dict by cam）
  - camera 写 hw_timestamp(N,)（Task 8 实填真硬件戳；Task 4 软件戳占位）
  - state_hifreq 新增 wrench(M,6) 占位（Phase F 实填）
  - schema_version = franka-hdf5-v2
"""
import json
import sys

import h5py
import numpy as np

from core.schema_loader import load_franka_hdf5_schema
S = load_franka_hdf5_schema()

_VLEN_BYTES = h5py.special_dtype(vlen=np.dtype("uint8"))


def write_episode(path, frames, *, task_name, target_fps, oc2base_R, quality,
                  vr_source, cam_names, state_hifreq_block=None):
    """模块级落盘函数：写 franka-hdf5-v2 episode（有文件写入 + validate_episode 副作用）。

    参数说明（中文）：
        path: 输出 .h5 文件路径（字符串）。
        frames: list[dict]，每帧含以下字段：
            - ts: float，采集完成后的软件戳（monotonic，回退用）
            - joints: (7,) float64
            - joint_vel: (7,) float64
            - ee_pose: (6,) float64
            - gripper_m: float
            - gripper_norm: float
            - gripper_cmd: float
            - delta_ee_pose: (6,) float64
            - cams: dict[str, uint8 bytes]（已 JPEG 编码）
            # v2 扩展字段（缺失时用 ts 回退，stale=False，hw_ts=sw_ts）：
            - arm_ts: float（arm 模态独立软件戳）
            - effector_ts: float（effector 模态独立软件戳）
            - cam_ts: dict[str, float]（各相机软件戳）
            - cam_hw_ts: dict[str, float]（各相机硬件戳，Task 8 实填）
            - arm_stale: bool（arm 模态是否陈旧）
            - effector_stale: bool（effector 模态是否陈旧）
            - cam_stale: dict[str, bool]（各相机是否陈旧）
        state_hifreq_block: 可选 dict，含 240Hz 高频状态数据（Task 7 实填），格式：
            - joints: (M,7) float64
            - joint_vel: (M,7) float64
            - pose: (M,6) float64
            - timestamp: (M,) float64
            - poly_ts: (M,) float64
            - wrench: (M,6) float64（Phase F 实填，M=0 时合规）
            None 时写 M=0 占位。
        task_name/target_fps/oc2base_R/quality/vr_source/cam_names: 元信息。
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

    # --- 提取各模态时间戳 ---
    # 每帧若有 arm_ts/effector_ts/cam_ts 则用独立戳，否则回退到 ts（兼容旧格式）
    def _get_ts(frame, key):
        """从帧中取模态时间戳，缺失则回退到 frame['ts']。"""
        return float(frame.get(key) if frame.get(key) is not None else frame["ts"])

    arm_ts = np.array([_get_ts(x, "arm_ts") for x in b], np.float64)
    effector_ts = np.array([_get_ts(x, "effector_ts") for x in b], np.float64)
    action_ts = np.array([_get_ts(x, "ts") for x in b], np.float64)

    avg = float((n - 1) / (action_ts[-1] - action_ts[0])) if n > 1 and action_ts[-1] > action_ts[0] else _target_fps

    # --- 各模态 stale 数组 ---
    arm_stale = np.array([bool(x.get("arm_stale", False)) for x in b], dtype=bool)
    effector_stale = np.array([bool(x.get("effector_stale", False)) for x in b], dtype=bool)

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
        # v2：无共用 observations/timestamp；各模态独立时间戳
        arm = obs.create_group("arm")
        arm.create_dataset("joints", data=np.stack([np.asarray(x["joints"], np.float64) for x in b]))
        arm.create_dataset("joint_vel", data=np.stack([np.asarray(x["joint_vel"], np.float64) for x in b]))
        arm.create_dataset("pose", data=np.stack([np.asarray(x["ee_pose"], np.float64) for x in b]))
        arm.create_dataset("timestamp", data=arm_ts)
        arm.create_dataset("stale", data=arm_stale)

        eff = obs.create_group("effector")
        eff.create_dataset("position", data=np.array([[x["gripper_m"]] for x in b], np.float64))
        eff.create_dataset("position_norm", data=np.array([[x["gripper_norm"]] for x in b], np.float64))
        eff.create_dataset("type", data=np.array([b"gripper"] * n,
                           dtype=h5py.special_dtype(vlen=bytes)))
        eff.create_dataset("timestamp", data=effector_ts)
        eff.create_dataset("stale", data=effector_stale)

        cam = obs.create_group("camera")
        rgb = cam.create_group("rgb")
        for cn in _cams:
            g = rgb.create_group(cn)
            imgs = g.create_dataset("images", (n,), dtype=_VLEN_BYTES)
            for i, x in enumerate(b):
                imgs[i] = np.frombuffer(bytes(x["cams"][cn]), np.uint8)
            # 各相机独立软件戳
            cam_ts_arr = np.array(
                [_get_ts(x, None) if x.get("cam_ts") is None
                 else float(x["cam_ts"].get(cn, x["ts"]))
                 for x in b],
                np.float64,
            )
            g.create_dataset("timestamp", data=cam_ts_arr)
            # 各相机 stale
            cam_stale_arr = np.array(
                [bool((x.get("cam_stale") or {}).get(cn, False)) for x in b],
                dtype=bool,
            )
            g.create_dataset("stale", data=cam_stale_arr)
            # hw_timestamp（Task 8 实填真硬件戳；Task 4 用软件戳占位）
            cam_hw_ts_arr = np.array(
                [float((x.get("cam_hw_ts") or {}).get(cn, cam_ts_arr[i]))
                 for i, x in enumerate(b)],
                np.float64,
            )
            g.create_dataset("hw_timestamp", data=cam_hw_ts_arr)

        # state_hifreq（M=0 占位；Task 7 实填）
        hf = obs.create_group("state_hifreq")
        if state_hifreq_block is not None:
            hf.create_dataset("joints", data=np.asarray(state_hifreq_block["joints"], np.float64))
            hf.create_dataset("joint_vel", data=np.asarray(state_hifreq_block["joint_vel"], np.float64))
            hf.create_dataset("pose", data=np.asarray(state_hifreq_block["pose"], np.float64))
            hf.create_dataset("timestamp", data=np.asarray(state_hifreq_block["timestamp"], np.float64))
            hf.create_dataset("poly_ts", data=np.asarray(state_hifreq_block["poly_ts"], np.float64))
            hf.create_dataset("wrench", data=np.asarray(state_hifreq_block.get(
                "wrench", np.zeros((len(state_hifreq_block["joints"]), 6), np.float64)), np.float64))
        else:
            hf.create_dataset("joints", data=np.zeros((0, 7), np.float64))
            hf.create_dataset("joint_vel", data=np.zeros((0, 7), np.float64))
            hf.create_dataset("pose", data=np.zeros((0, 6), np.float64))
            hf.create_dataset("timestamp", data=np.zeros((0,), np.float64))
            hf.create_dataset("poly_ts", data=np.zeros((0,), np.float64))
            hf.create_dataset("wrench", data=np.zeros((0, 6), np.float64))

        act = f.create_group("action")
        act.create_dataset("delta_ee_pose",
                           data=np.stack([np.asarray(x["delta_ee_pose"], np.float64) for x in b]))
        act.create_dataset("gripper_cmd", data=np.array([[x["gripper_cmd"]] for x in b], np.float64))
        act.create_dataset("timestamp", data=action_ts)

    violations = S.validate_episode(path)
    if violations:
        raise RuntimeError(f"写出的 episode 不符 franka-hdf5-v2: {violations}")


class HDF5EpisodeWriter:
    """按 franka-hdf5-v2 schema 缓冲帧并在 close() 时整体写盘。

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
