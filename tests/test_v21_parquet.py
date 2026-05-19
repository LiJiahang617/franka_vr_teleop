"""Task 2 TDD：episode_to_parquet 帧写入测试。

合成 franka-hdf5-v1 → 调 episode_to_parquet → 校验 parquet schema/值/index/realman 重排。
"""
import sys
import numpy as np
import h5py
import cv2
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# conftest 已把 <repo>/scripts 入 sys.path，故可直接 import
from tools.hdf5_to_lerobot_v21 import episode_to_parquet
from tools.hdf5_lerobot_map import hdf5_frame_to_lerobot


# ──────────────────────────────────────────────────────────────────────────────
# 合成 franka-hdf5-v1 生成器（精简版，只含 episode_to_parquet 所需字段）
# ──────────────────────────────────────────────────────────────────────────────

def _mk_h5(p, N=4, cams=("wrist", "exterior"), img_hw=(8, 8)):
    """生成最小合规 franka-hdf5-v1 文件。"""
    import franka_hdf5_schema as S

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
        # joints 用有区分度的值，方便 state[0] 断言
        arm.create_dataset("joints", data=np.arange(N * 7, dtype=np.float64).reshape(N, 7) * 0.1)
        arm.create_dataset("joint_vel", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("pose", data=np.arange(N * 6, dtype=np.float64).reshape(N, 6) * 0.2)
        arm.create_dataset("timestamp", data=np.arange(N, dtype=np.float64))
        eff = obs.create_group("effector")
        eff.create_dataset("position", data=np.zeros((N, 1), np.float64))
        # gripper_norm：每帧不同值（0.5 + i*0.1），方便断言 realman 重排
        eff.create_dataset("position_norm", data=(np.arange(N, dtype=np.float64) * 0.1 + 0.5).reshape(N, 1))
        eff.create_dataset("type", data=np.array([b"gripper"] * N, dtype=h5py.special_dtype(vlen=bytes)))
        eff.create_dataset("timestamp", data=np.arange(N, dtype=np.float64))

        cam = obs.create_group("camera")
        rgb = cam.create_group("rgb")
        for c in cams:
            g = rgb.create_group(c)
            d = g.create_dataset("images", (N,), dtype=h5py.special_dtype(vlen=np.dtype("uint8")))
            for i in range(N):
                d[i] = jb
            g.create_dataset("timestamp", data=np.arange(N, dtype=np.float64))

        hf = obs.create_group("state_hifreq")
        for k, sh in [("joints", (0, 7)), ("joint_vel", (0, 7)), ("pose", (0, 6)),
                      ("timestamp", (0,)), ("poly_ts", (0,))]:
            hf.create_dataset(k, data=np.zeros(sh, np.float64))

        act = f.create_group("action")
        # delta_ee_pose：有区分度的值
        act.create_dataset("delta_ee_pose", data=np.arange(N * 6, dtype=np.float64).reshape(N, 6) * 0.05)
        # gripper_cmd：每帧 0.0..1.0
        act.create_dataset("gripper_cmd", data=(np.arange(N, dtype=np.float64) * 0.25).reshape(N, 1))
        act.create_dataset("timestamp", data=np.arange(N, dtype=np.float64))


# ──────────────────────────────────────────────────────────────────────────────
# 测试：native layout
# ──────────────────────────────────────────────────────────────────────────────

def test_parquet_columns_and_no_image(tmp_path):
    """parquet 列名必须精确，无图像列。"""
    h5p = str(tmp_path / "ep.h5")
    outp = str(tmp_path / "ep.parquet")
    _mk_h5(h5p, N=4, cams=("wrist", "exterior"))

    episode_to_parquet(
        h5_path=h5p, out_parquet=outp,
        episode_index=0, task_index=0,
        fps=30, cam_names=["wrist", "exterior"],
        task="pick", index_base=0,
        state_layout="native",
    )

    tbl = pq.read_table(outp)
    expected_cols = [
        "observation.state", "action", "timestamp",
        "frame_index", "episode_index", "index", "task_index",
    ]
    assert list(tbl.schema.names) == expected_cols, f"列名不符: {list(tbl.schema.names)}"


def test_parquet_num_rows(tmp_path):
    """行数必须等于 hdf5 帧数。"""
    N = 5
    h5p = str(tmp_path / "ep.h5")
    outp = str(tmp_path / "ep.parquet")
    _mk_h5(h5p, N=N, cams=("wrist",))

    episode_to_parquet(
        h5_path=h5p, out_parquet=outp,
        episode_index=0, task_index=0,
        fps=15, cam_names=["wrist"],
        task="pick", index_base=0,
    )
    tbl = pq.read_table(outp)
    assert tbl.num_rows == N


def test_parquet_schema_types(tmp_path):
    """observation.state fixed_size_list<float32>[14]；action [7]；timestamp float32；index 列 int64。"""
    h5p = str(tmp_path / "ep.h5")
    outp = str(tmp_path / "ep.parquet")
    _mk_h5(h5p, N=3, cams=("wrist",))

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
    assert state_field.type.list_size == 14, f"observation.state list_size 应为 14，实际: {state_field.type.list_size}"
    assert pa.types.is_float32(state_field.type.value_type), \
        f"observation.state 值类型应为 float32，实际: {state_field.type.value_type}"

    # action: fixed_size_list<float32>[7]
    action_field = schema.field("action")
    assert pa.types.is_fixed_size_list(action_field.type), \
        f"action 应为 fixed_size_list，实际: {action_field.type}"
    assert action_field.type.list_size == 7, f"action list_size 应为 7，实际: {action_field.type.list_size}"
    assert pa.types.is_float32(action_field.type.value_type), \
        f"action 值类型应为 float32，实际: {action_field.type.value_type}"

    # timestamp: float32
    ts_field = schema.field("timestamp")
    assert pa.types.is_float32(ts_field.type), f"timestamp 应为 float32，实际: {ts_field.type}"

    # index 列: int64
    for col in ("frame_index", "episode_index", "index", "task_index"):
        f = schema.field(col)
        assert pa.types.is_int64(f.type), f"{col} 应为 int64，实际: {f.type}"


def test_parquet_timestamp_and_frame_index(tmp_path):
    """timestamp[i] == i/fps（row0==0.0）；frame_index[i] == i。"""
    N = 4
    fps = 20
    h5p = str(tmp_path / "ep.h5")
    outp = str(tmp_path / "ep.parquet")
    _mk_h5(h5p, N=N, cams=("wrist",))

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


def test_parquet_episode_index_and_global_index(tmp_path):
    """episode_index 全等于传入值；index == index_base + i 连续。"""
    N = 3
    ep_idx = 2
    t_idx = 1
    base = 10
    h5p = str(tmp_path / "ep.h5")
    outp = str(tmp_path / "ep.parquet")
    _mk_h5(h5p, N=N, cams=("wrist",))

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


def test_parquet_values_match_hdf5_frame_to_lerobot(tmp_path):
    """parquet 帧值与 hdf5_frame_to_lerobot 逐字一致（state[0]==joints[0], action[6]==gripper）。"""
    N = 4
    h5p = str(tmp_path / "ep.h5")
    outp = str(tmp_path / "ep.parquet")
    _mk_h5(h5p, N=N, cams=("wrist",))

    episode_to_parquet(
        h5_path=h5p, out_parquet=outp,
        episode_index=0, task_index=0,
        fps=30, cam_names=["wrist"],
        task="pick", index_base=0,
    )
    tbl = pq.read_table(outp)

    with h5py.File(h5p, "r") as h5:
        for i in range(N):
            frame = hdf5_frame_to_lerobot(h5, i, cam_names=["wrist"], task="pick")
            state_row = tbl["observation.state"][i].as_py()
            action_row = tbl["action"][i].as_py()

            # state[0] == joints[0]（native layout 下 joint_1.pos 在 idx 0）
            assert abs(state_row[0] - float(frame["observation.state"][0])) < 1e-5, \
                f"行{i} state[0] 不匹配: parquet={state_row[0]}, map={frame['observation.state'][0]}"

            # 整个 state 向量逐元素比对
            for k in range(14):
                assert abs(state_row[k] - float(frame["observation.state"][k])) < 1e-5, \
                    f"行{i} state[{k}] 不匹配: parquet={state_row[k]}, map={frame['observation.state'][k]}"

            # action[6] == gripper_cmd（7D 最后一维）
            assert abs(action_row[6] - float(frame["action"][6])) < 1e-5, \
                f"行{i} action[6] 不匹配: parquet={action_row[6]}, map={frame['action'][6]}"

            # 整个 action 向量逐元素比对
            for k in range(7):
                assert abs(action_row[k] - float(frame["action"][k])) < 1e-5, \
                    f"行{i} action[{k}] 不匹配: parquet={action_row[k]}, map={frame['action'][k]}"


# ──────────────────────────────────────────────────────────────────────────────
# 测试：realman layout
# ──────────────────────────────────────────────────────────────────────────────

def test_realman_layout_state_reorder(tmp_path):
    """realman layout: state 重排为 [joint7, gripper_norm, ee_pose6]，长度仍 14。"""
    N = 3
    h5p = str(tmp_path / "ep.h5")
    out_native = str(tmp_path / "native.parquet")
    out_realman = str(tmp_path / "realman.parquet")
    _mk_h5(h5p, N=N, cams=("wrist",))

    episode_to_parquet(
        h5_path=h5p, out_parquet=out_native,
        episode_index=0, task_index=0,
        fps=30, cam_names=["wrist"],
        task="pick", index_base=0,
        state_layout="native",
    )
    episode_to_parquet(
        h5_path=h5p, out_parquet=out_realman,
        episode_index=0, task_index=0,
        fps=30, cam_names=["wrist"],
        task="pick", index_base=0,
        state_layout="realman",
    )

    tbl_n = pq.read_table(out_native)
    tbl_r = pq.read_table(out_realman)

    # realman state 长度仍 14
    r_field = tbl_r.schema.field("observation.state")
    assert r_field.type.list_size == 14, f"realman state list_size 应为 14，实际: {r_field.type.list_size}"

    # native layout: [joint0..6(idx0-6), ee_pose(idx7-12), gripper_norm(idx13)]
    # realman layout: [joint0..6(idx0-6), gripper_norm(was idx13), ee_pose(idx7-12)]
    # 即 realman[7] == native[13]（gripper_norm 移到 idx7）
    #    realman[8..13] == native[7..12]（ee_pose 移到 idx8-13）
    for i in range(N):
        n_state = tbl_n["observation.state"][i].as_py()
        r_state = tbl_r["observation.state"][i].as_py()

        # 前 7 维（joint）不变
        for k in range(7):
            assert abs(r_state[k] - n_state[k]) < 1e-6, \
                f"行{i} realman state[{k}](joint) 与 native 不符: r={r_state[k]}, n={n_state[k]}"

        # idx 7: realman gripper_norm == native[13]
        assert abs(r_state[7] - n_state[13]) < 1e-6, \
            f"行{i} realman state[7](gripper_norm) 应等于 native[13]={n_state[13]}，实际: {r_state[7]}"

        # idx 8-13: realman ee_pose == native[7-12]
        for k in range(6):
            assert abs(r_state[8 + k] - n_state[7 + k]) < 1e-6, \
                f"行{i} realman state[{8+k}](ee_pose[{k}]) 应等于 native[{7+k}]={n_state[7+k]}，实际: {r_state[8+k]}"


def test_realman_action_equals_native_action(tmp_path):
    """红线：realman layout 的 action 列必须与 native 完全相同（action 不受 layout 影响）。"""
    N = 4
    h5p = str(tmp_path / "ep.h5")
    out_native = str(tmp_path / "native.parquet")
    out_realman = str(tmp_path / "realman.parquet")
    _mk_h5(h5p, N=N, cams=("wrist",))

    episode_to_parquet(
        h5_path=h5p, out_parquet=out_native,
        episode_index=0, task_index=0,
        fps=30, cam_names=["wrist"],
        task="pick", index_base=0,
        state_layout="native",
    )
    episode_to_parquet(
        h5_path=h5p, out_parquet=out_realman,
        episode_index=0, task_index=0,
        fps=30, cam_names=["wrist"],
        task="pick", index_base=0,
        state_layout="realman",
    )

    tbl_n = pq.read_table(out_native)
    tbl_r = pq.read_table(out_realman)

    # action 逐行逐元素相等（红线断言）
    for i in range(N):
        n_action = tbl_n["action"][i].as_py()
        r_action = tbl_r["action"][i].as_py()
        assert len(n_action) == 7 and len(r_action) == 7, \
            f"action 长度应为 7: native={len(n_action)}, realman={len(r_action)}"
        for k in range(7):
            assert abs(n_action[k] - r_action[k]) < 1e-6, \
                f"行{i} action[{k}] native={n_action[k]} != realman={r_action[k]}（action 不应受 layout 影响）"


def test_episode_to_parquet_returns_stat_arrays(tmp_path):
    """episode_to_parquet 返回 (N, stat_arrays)，stat_arrays 含 7 键，observation.state shape (N,14)，action (N,7)。"""
    N = 3
    h5p = str(tmp_path / "ep.h5")
    outp = str(tmp_path / "ep.parquet")
    _mk_h5(h5p, N=N, cams=("wrist",))

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

    # action: (N, 7)
    ac = stat_arrays["action"]
    assert ac.shape == (N, 7), f"action shape 应为 ({N},7)，实际: {ac.shape}"

    # 1D 元列: (N, 1)
    for col in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        arr = stat_arrays[col]
        assert arr.shape == (N, 1), f"{col} shape 应为 ({N},1)，实际: {arr.shape}"
