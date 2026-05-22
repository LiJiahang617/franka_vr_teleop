"""tests/test_v21_parquet.py — Task 6 TDD：episode_to_parquet v2 接口测试。

合成 franka-hdf5-v2 episode（通过 align_offline 对齐），调 episode_to_parquet，
校验 parquet schema / 值 / index / realman 14D next-state action 语义。

v2 变更要点：
  - 无 state_layout 参数（统一 realman 14D）
  - action = next-state：14D，与 state 字段名完全一致
  - parquet schema: observation.state fixed_size_list[14], action fixed_size_list[14]
"""
import sys
import numpy as np
import h5py
import cv2
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# conftest 已把 <repo>/scripts 入 sys.path，故可直接 import
from tools.hdf5_to_lerobot_v21 import episode_to_parquet, compute_episode_stats


# ──────────────────────────────────────────────────────────────────────────────
# 合成 franka-hdf5-v2 生成器（仅含 episode_to_parquet 所需字段）
# ──────────────────────────────────────────────────────────────────────────────

def _mk_v2(p, N=4, cams=("wrist", "exterior"), img_hw=(8, 8)):
    """生成最小合规 franka-hdf5-v2 文件。"""
    import franka_hdf5_schema as S

    H, W = img_hw
    img = np.zeros((H, W, 3), np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    jb = np.frombuffer(enc.tobytes(), np.uint8)
    ts = np.arange(N, dtype=np.float64)

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

        arm = obs.create_group("arm")
        # joints 用有区分度的值，方便 state 断言
        arm.create_dataset("joints", data=np.arange(N * 7, dtype=np.float64).reshape(N, 7) * 0.1)
        arm.create_dataset("joint_vel", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("pose", data=np.arange(N * 6, dtype=np.float64).reshape(N, 6) * 0.2)
        arm.create_dataset("timestamp", data=ts.copy())
        arm.create_dataset("stale", data=np.zeros(N, dtype=bool))

        eff = obs.create_group("effector")
        eff.create_dataset("position", data=np.zeros((N, 1), np.float64))
        # gripper_norm：每帧不同值（0.5 + i*0.1），方便断言布局
        eff.create_dataset("position_norm",
                           data=(np.arange(N, dtype=np.float64) * 0.1 + 0.5).reshape(N, 1))
        eff.create_dataset("type", data=np.array([b"gripper"] * N, dtype=h5py.special_dtype(vlen=bytes)))
        eff.create_dataset("timestamp", data=ts.copy())
        eff.create_dataset("stale", data=np.zeros(N, dtype=bool))

        cam = obs.create_group("camera")
        rgb = cam.create_group("rgb")
        for c in cams:
            g = rgb.create_group(c)
            d = g.create_dataset("images", (N,), dtype=h5py.special_dtype(vlen=np.dtype("uint8")))
            for i in range(N):
                d[i] = jb
            g.create_dataset("timestamp", data=ts.copy())
            g.create_dataset("stale", data=np.zeros(N, dtype=bool))
            g.create_dataset("hw_timestamp", data=ts.copy())

        hf = obs.create_group("state_hifreq")
        for k, sh in [("joints", (0, 7)), ("joint_vel", (0, 7)), ("pose", (0, 6)),
                      ("timestamp", (0,)), ("poly_ts", (0,))]:
            hf.create_dataset(k, data=np.zeros(sh, np.float64))
        hf.create_dataset("wrench", data=np.zeros((0, 6), np.float64))

        act = f.create_group("action")
        act.create_dataset("delta_ee_pose",
                           data=np.arange(N * 6, dtype=np.float64).reshape(N, 6) * 0.05)
        act.create_dataset("gripper_cmd",
                           data=(np.arange(N, dtype=np.float64) * 0.25).reshape(N, 1))
        act.create_dataset("timestamp", data=ts.copy() + 0.001)  # 严格递增


# ──────────────────────────────────────────────────────────────────────────────
# 测试：列名与类型
# ──────────────────────────────────────────────────────────────────────────────

def test_parquet_columns_and_no_image(tmp_path):
    """parquet 列名必须精确，无图像列。"""
    h5p = str(tmp_path / "ep.h5")
    outp = str(tmp_path / "ep.parquet")
    _mk_v2(h5p, N=4, cams=("wrist", "exterior"))

    episode_to_parquet(
        h5_path=h5p, out_parquet=outp,
        episode_index=0, task_index=0,
        fps=30, cam_names=["wrist", "exterior"],
        task="pick", index_base=0,
    )

    tbl = pq.read_table(outp)
    expected_cols = [
        "observation.state", "action", "timestamp",
        "frame_index", "episode_index", "index", "task_index",
    ]
    assert list(tbl.schema.names) == expected_cols, f"列名不符: {list(tbl.schema.names)}"


def test_parquet_num_rows(tmp_path):
    """行数必须等于对齐后帧数。"""
    N = 5
    h5p = str(tmp_path / "ep.h5")
    outp = str(tmp_path / "ep.parquet")
    _mk_v2(h5p, N=N, cams=("wrist",))

    episode_to_parquet(
        h5_path=h5p, out_parquet=outp,
        episode_index=0, task_index=0,
        fps=15, cam_names=["wrist"],
        task="pick", index_base=0,
    )
    tbl = pq.read_table(outp)
    assert tbl.num_rows == N


def test_parquet_schema_types(tmp_path):
    """observation.state fixed_size_list<float32>[14]；action [14]；timestamp float32；index 列 int64。"""
    h5p = str(tmp_path / "ep.h5")
    outp = str(tmp_path / "ep.parquet")
    _mk_v2(h5p, N=3, cams=("wrist",))

    episode_to_parquet(
        h5_path=h5p, out_parquet=outp,
        episode_index=0, task_index=0,
        fps=30, cam_names=["wrist"],
        task="pick", index_base=0,
    )
    tbl = pq.read_table(outp)
    schema = tbl.schema

    # observation.state: fixed_size_list<float32>[14]
    state_field = schema.field("observation.state")
    assert pa.types.is_fixed_size_list(state_field.type), \
        f"observation.state 应为 fixed_size_list，实际: {state_field.type}"
    assert state_field.type.list_size == 14, \
        f"observation.state list_size 应为 14，实际: {state_field.type.list_size}"
    assert pa.types.is_float32(state_field.type.value_type)

    # action: fixed_size_list<float32>[14]（v2: action = next-state，14D）
    action_field = schema.field("action")
    assert pa.types.is_fixed_size_list(action_field.type), \
        f"action 应为 fixed_size_list，实际: {action_field.type}"
    assert action_field.type.list_size == 14, \
        f"action list_size 应为 14（next-state，realman 14D），实际: {action_field.type.list_size}"
    assert pa.types.is_float32(action_field.type.value_type)

    # timestamp: float32
    ts_field = schema.field("timestamp")
    assert pa.types.is_float32(ts_field.type)

    # index 列: int64
    for col in ("frame_index", "episode_index", "index", "task_index"):
        f = schema.field(col)
        assert pa.types.is_int64(f.type), f"{col} 应为 int64，实际: {f.type}"


# ──────────────────────────────────────────────────────────────────────────────
# 测试：timestamp / frame_index
# ──────────────────────────────────────────────────────────────────────────────

def test_parquet_timestamp_and_frame_index(tmp_path):
    """timestamp[i] == i/fps（row0==0.0）；frame_index[i] == i。"""
    N = 4
    fps = 20
    h5p = str(tmp_path / "ep.h5")
    outp = str(tmp_path / "ep.parquet")
    _mk_v2(h5p, N=N, cams=("wrist",))

    episode_to_parquet(
        h5_path=h5p, out_parquet=outp,
        episode_index=0, task_index=0,
        fps=fps, cam_names=["wrist"],
        task="pick", index_base=0,
    )
    tbl = pq.read_table(outp)
    ts = tbl["timestamp"].to_pylist()
    fi = tbl["frame_index"].to_pylist()

    for i in range(N):
        expected_ts = np.float32(i / fps)
        assert abs(ts[i] - float(expected_ts)) < 1e-6, \
            f"timestamp[{i}] 应为 {float(expected_ts)}，实际: {ts[i]}"
        assert fi[i] == i, f"frame_index[{i}] 应为 {i}，实际: {fi[i]}"


# ──────────────────────────────────────────────────────────────────────────────
# 测试：episode_index / global_index
# ──────────────────────────────────────────────────────────────────────────────

def test_parquet_episode_index_and_global_index(tmp_path):
    """episode_index 全等于传入值；index == index_base + i 连续。"""
    N = 3
    ep_idx = 2
    t_idx = 1
    base = 10
    h5p = str(tmp_path / "ep.h5")
    outp = str(tmp_path / "ep.parquet")
    _mk_v2(h5p, N=N, cams=("wrist",))

    episode_to_parquet(
        h5_path=h5p, out_parquet=outp,
        episode_index=ep_idx, task_index=t_idx,
        fps=30, cam_names=["wrist"],
        task="pick", index_base=base,
    )
    tbl = pq.read_table(outp)
    ep_col = tbl["episode_index"].to_pylist()
    idx_col = tbl["index"].to_pylist()
    ti_col = tbl["task_index"].to_pylist()

    assert all(v == ep_idx for v in ep_col), f"episode_index 列不全为 {ep_idx}: {ep_col}"
    assert all(v == t_idx for v in ti_col), f"task_index 列不全为 {t_idx}: {ti_col}"
    for i in range(N):
        assert idx_col[i] == base + i, f"index[{i}] 应为 {base + i}，实际: {idx_col[i]}"


# ──────────────────────────────────────────────────────────────────────────────
# 测试：next-state action 语义
# ──────────────────────────────────────────────────────────────────────────────

def test_action_equals_next_state(tmp_path):
    """action[i] == observation.state[i+1]（next-state 语义）。"""
    N = 5
    h5p = str(tmp_path / "ep.h5")
    outp = str(tmp_path / "ep.parquet")
    _mk_v2(h5p, N=N, cams=("wrist",))

    episode_to_parquet(
        h5_path=h5p, out_parquet=outp,
        episode_index=0, task_index=0,
        fps=30, cam_names=["wrist"],
        task="pick", index_base=0,
    )
    tbl = pq.read_table(outp)
    states = [tbl["observation.state"][i].as_py() for i in range(N)]
    actions = [tbl["action"][i].as_py() for i in range(N)]

    # action[i] == state[i+1]（i < N-1）
    for i in range(N - 1):
        for k in range(14):
            assert abs(actions[i][k] - states[i + 1][k]) < 1e-6, \
                f"action[{i}][{k}] = {actions[i][k]} 应等于 state[{i+1}][{k}] = {states[i+1][k]}"

    # 末帧：action[N-1] == state[N-1]（复制末帧）
    for k in range(14):
        assert abs(actions[-1][k] - states[-1][k]) < 1e-6, \
            f"末帧 action[{k}] = {actions[-1][k]} 应等于 state[-1][{k}] = {states[-1][k]}"


# ──────────────────────────────────────────────────────────────────────────────
# 测试：14D realman 布局验证
# ──────────────────────────────────────────────────────────────────────────────

def test_state_realman_layout(tmp_path):
    """验证 state 14D realman 布局：joint[0:7] | gripper[7] | eef_pos[8:11] | eef_rot[11:14]。

    注：eef_rot 经由 align_offline SLERP + unwrap 处理，数值可能与原始 hdf5 不同。
    通过 align_by_image_timestamp 获取对齐后的值与 parquet 比对（与转换器内部逻辑一致）。
    """
    from tools.align_offline import align_by_image_timestamp
    N = 4
    h5p = str(tmp_path / "ep.h5")
    outp = str(tmp_path / "ep.parquet")
    _mk_v2(h5p, N=N, cams=("wrist",))

    episode_to_parquet(
        h5_path=h5p, out_parquet=outp,
        episode_index=0, task_index=0,
        fps=30, cam_names=["wrist"],
        task="pick", index_base=0,
    )
    tbl = pq.read_table(outp)

    # 通过 align_by_image_timestamp 获取对齐值（与转换器内部一致）
    aligned = align_by_image_timestamp(h5p)
    joints = aligned["arm_joints"]              # (N, 7) 插值后
    gripper = aligned["gripper_position_norm"]  # (N, 1) 插值后
    pose = aligned["arm_pose"]                  # (N, 6) 位置线性+旋转SLERP后

    for i in range(N):
        row = tbl["observation.state"][i].as_py()

        # joint_1_rad..joint_7_rad（索引 0-6）来自 aligned arm_joints
        for k in range(7):
            assert abs(row[k] - joints[i, k]) < 1e-5, \
                f"row{i} state[{k}](joint) 应为 {joints[i,k]}，实际: {row[k]}"

        # gripper_open（索引 7）来自 aligned gripper_position_norm
        assert abs(row[7] - gripper[i, 0]) < 1e-5, \
            f"row{i} state[7](gripper) 应为 {gripper[i,0]}，实际: {row[7]}"

        # eef_pos_xyz（索引 8-10）来自 aligned arm_pose[:, 0:3]
        for k in range(3):
            assert abs(row[8 + k] - pose[i, k]) < 1e-5, \
                f"row{i} state[{8+k}](eef_pos[{k}]) 应为 {pose[i,k]}，实际: {row[8+k]}"

        # eef_rot_euler_xyz（索引 11-13）来自 aligned arm_pose[:, 3:6]（经 SLERP 处理）
        for k in range(3):
            assert abs(row[11 + k] - pose[i, 3 + k]) < 1e-5, \
                f"row{i} state[{11+k}](eef_rot[{k}]) 应为 {pose[i,3+k]}，实际: {row[11+k]}"


# ──────────────────────────────────────────────────────────────────────────────
# 测试：episode_to_parquet 返回 stat_arrays
# ──────────────────────────────────────────────────────────────────────────────

def test_episode_to_parquet_returns_stat_arrays(tmp_path):
    """episode_to_parquet 返回 (N, stat_arrays)，stat_arrays 含 7 键，observation.state (N,14)，action (N,14)。"""
    N = 3
    h5p = str(tmp_path / "ep.h5")
    outp = str(tmp_path / "ep.parquet")
    _mk_v2(h5p, N=N, cams=("wrist",))

    result = episode_to_parquet(
        h5_path=h5p, out_parquet=outp,
        episode_index=0, task_index=0,
        fps=30, cam_names=["wrist"],
        task="pick", index_base=0,
    )

    assert isinstance(result, tuple) and len(result) == 2, "应返回 (N, stat_arrays) 元组"
    n_frames, stat_arrays = result

    assert n_frames == N, f"n_frames 应为 {N}，实际: {n_frames}"

    expected_keys = {
        "observation.state", "action", "timestamp",
        "frame_index", "episode_index", "index", "task_index",
    }
    assert set(stat_arrays.keys()) == expected_keys, \
        f"stat_arrays 键不符: {set(stat_arrays.keys())}"

    # observation.state: (N, 14)
    st = stat_arrays["observation.state"]
    assert st.shape == (N, 14), f"observation.state shape 应为 ({N},14)，实际: {st.shape}"

    # action: (N, 14)（v2: next-state，14D）
    ac = stat_arrays["action"]
    assert ac.shape == (N, 14), f"action shape 应为 ({N},14)（next-state 14D），实际: {ac.shape}"

    # 1D 元列: (N, 1)
    for col in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        arr = stat_arrays[col]
        assert arr.shape == (N, 1), f"{col} shape 应为 ({N},1)，实际: {arr.shape}"
