import sys, numpy as np, h5py, cv2
sys.path.insert(0, "/home/ubuntu/Desktop/jhli")
sys.path.insert(0, "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop/scripts")
import franka_hdf5_schema as S
from tools.hdf5_lerobot_map import build_feature_specs, hdf5_frame_to_lerobot


def _mk(p, N=3, cams=("wrist",), img_hw=(8, 8)):
    """生成最小合规 franka-hdf5-v1 文件。img_hw=(H,W)。"""
    H, W = img_hw
    img = np.zeros((H, W, 3), np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    jb = np.frombuffer(enc.tobytes(), np.uint8)
    with h5py.File(p, "w") as f:
        inf = f.create_group("infos")
        inf.create_dataset("schema_version", data=np.bytes_(S.SCHEMA_VERSION))
        ti = inf.create_group("task_info")
        ti.create_dataset("task_name", data=np.bytes_("pick"))
        ti.create_dataset("collection_frequency", data=np.array([30.0, 30.0], np.float64))
        ti.create_dataset("total_frames", data=np.int64(N))
        ti.create_dataset("robot", data=np.bytes_("franka_panda"))
        inf.create_group("camera_params")
        cal = inf.create_group("calibration")
        cal.create_dataset("oc2base_R", data=np.eye(3, dtype=np.float64))
        cal.create_dataset("quality", data=np.bytes_("{}"))
        cal.create_dataset("vr_source", data=np.bytes_("unity"))
        obs = f.create_group("observations")
        obs.create_dataset("timestamp", data=(np.arange(N, dtype=np.float64) + 1).reshape(N, 1))
        arm = obs.create_group("arm")
        arm.create_dataset("joints", data=np.arange(N * 7, dtype=np.float64).reshape(N, 7))
        arm.create_dataset("joint_vel", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("pose", data=np.arange(N * 6, dtype=np.float64).reshape(N, 6))
        arm.create_dataset("timestamp", data=np.arange(N, dtype=np.float64))
        eff = obs.create_group("effector")
        eff.create_dataset("position", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("position_norm", data=(np.ones((N, 1)) * 0.5))
        eff.create_dataset("type", data=np.array([b"gripper"] * N, dtype=h5py.special_dtype(vlen=bytes)))
        eff.create_dataset("timestamp", data=np.arange(N, dtype=np.float64))
        cam = obs.create_group("camera"); rgb = cam.create_group("rgb")
        for c in cams:
            g = rgb.create_group(c)
            d = g.create_dataset("images", (N,), dtype=h5py.special_dtype(vlen=np.dtype("uint8")))
            for i in range(N):
                d[i] = jb
            g.create_dataset("timestamp", data=np.arange(N, dtype=np.float64))
        hf = obs.create_group("state_hifreq")
        for k, sh in [("joints",(0,7)),("joint_vel",(0,7)),("pose",(0,6)),("timestamp",(0,)),("poly_ts",(0,))]:
            hf.create_dataset(k, data=np.zeros(sh, np.float64))
        act = f.create_group("action")
        act.create_dataset("delta_ee_pose", data=np.arange(N * 6, dtype=np.float64).reshape(N, 6))
        act.create_dataset("gripper_cmd", data=np.ones((N, 1), np.float64))
        act.create_dataset("timestamp", data=np.arange(N, dtype=np.float64))


def test_feature_specs_keys(tmp_path):
    a, o = build_feature_specs(cam_names=["wrist"])
    # action hw: 7个 float 键
    assert "delta_ee_pose.x" in a and "gripper_cmd_bin" in a
    # obs hw: state float 键 + 相机 tuple 键
    assert "joint_1.pos" in o and "wrist" in o
    assert isinstance(o["wrist"], tuple) and len(o["wrist"]) == 3


def test_frame_mapping(tmp_path):
    p = str(tmp_path / "ep.h5")
    _mk(p, N=3)
    with h5py.File(p, "r") as f:
        fr = hdf5_frame_to_lerobot(f, 1, ["wrist"])
    # action 向量: delta_ee_pose row1=[6,7,8,9,10,11], gripper=1.0 → shape (7,)
    assert fr["action"].shape == (7,)
    assert abs(fr["action"][0] - 6.0) < 1e-6   # delta_ee_pose.x row1
    assert abs(fr["action"][6] - 1.0) < 1e-6   # gripper_cmd_bin
    # observation.state 向量: joints(7) + pose(6) + gripper_norm(1) = (14,)
    assert fr["observation.state"].shape == (14,)
    assert abs(fr["observation.state"][0] - 7.0) < 1e-6  # joint_1.pos row1
    # task
    assert fr["task"] == "task"
    # 图像
    img = fr["observation.images.wrist"]
    assert isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3
