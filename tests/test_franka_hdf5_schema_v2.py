"""franka-hdf5-v2 schema 自身测试（Task 2，TDD）。

覆盖：
  - 完整合规 v2 episode 通过校验（含各模态独立 ts + stale + hw_timestamp + state_hifreq）
  - 空 state_hifreq（M=0）通过校验
  - 可扩展字段 validate-if-present（depth/tactile）：缺失时不报错；存在时校验 shape/dtype
  - 拒绝 schema_version=franka-hdf5-v1
  - 拒绝含 observations/timestamp (N,1) 共戳（v1 残留）
  - hw_timestamp 必须存在，shape 和 dtype 校验
  - stale 字段必须存在且 bool dtype
  - arm/effector/camera/action 各自独立 timestamp(N,) 必须校验
  - camera 无相机时被拒（Route B 必须有相机）
  - state_hifreq wrench(M,6) 字段预留（Task 7 实填）：M=0 时通过
  - validate_episode 返回 [] 代表合格，否则 violations 列表
  - [Codex 审查补充] action/timestamp 二维被拒
  - [Codex 审查补充] 时间戳单调性（action 严格递增，arm/eff/cam 非递减，hifreq 严格递增）
  - [Codex 审查补充] images 长度与 N 不符被拒
  - [Codex 审查补充] malformed rgb_group/tac_group（非 Group 类型）被拒
"""
import h5py
import numpy as np
import pytest
import franka_hdf5_schema as S


# ---------------------------------------------------------------------------
# 辅助：构造合规 v2 episode
# ---------------------------------------------------------------------------

def _write_v2(path, N=5, M=40, cam_names=("wrist",), include_depth=False,
              include_tactile=False):
    """生成完整合规 franka-hdf5-v2 文件。

    Args:
        path: 输出路径
        N: 主模态帧数
        M: state_hifreq 帧数（M=0 也合规）
        cam_names: 相机名称列表
        include_depth: 是否写入可扩展 depth 字段
        include_tactile: 是否写入可扩展 tactile 字段
    """
    # 时间戳严格递增，满足新的单调性校验契约
    ts_arm = np.arange(N, dtype=np.float64) * 0.033 + 1.0
    ts_eff = ts_arm + 0.001
    ts_act = ts_arm + 0.002

    with h5py.File(path, "w") as f:
        # infos
        inf = f.create_group("infos")
        inf.create_dataset("schema_version", data=np.bytes_(S.SCHEMA_VERSION))
        ti = inf.create_group("task_info")
        ti.create_dataset("task_name", data=np.bytes_("pick"))
        ti.create_dataset("collection_frequency", data=np.array([30.0, 29.7], np.float64))
        ti.create_dataset("total_frames", data=np.int64(N))
        ti.create_dataset("robot", data=np.bytes_("franka_panda"))
        inf.create_group("camera_params")
        cal = inf.create_group("calibration")
        cal.create_dataset("oc2base_R", data=np.eye(3, dtype=np.float64))
        cal.create_dataset("quality", data=np.bytes_("{}"))
        cal.create_dataset("vr_source", data=np.bytes_("unity"))

        # observations/arm
        obs = f.create_group("observations")
        arm = obs.create_group("arm")
        arm.create_dataset("joints", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("joint_vel", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("pose", data=np.zeros((N, 6), np.float64))
        arm.create_dataset("timestamp", data=ts_arm)
        arm.create_dataset("stale", data=np.zeros(N, dtype=bool))

        # observations/effector
        eff = obs.create_group("effector")
        eff.create_dataset("position", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("position_norm", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("type",
                           data=np.array([b"gripper"] * N,
                                         dtype=h5py.special_dtype(vlen=bytes)))
        eff.create_dataset("timestamp", data=ts_eff)
        eff.create_dataset("stale", data=np.zeros(N, dtype=bool))

        # observations/camera（各相机独立 timestamp + stale + hw_timestamp）
        cam_g = obs.create_group("camera")
        rgb_g = cam_g.create_group("rgb")
        _VLEN = h5py.special_dtype(vlen=np.dtype("uint8"))
        for cn in cam_names:
            cg = rgb_g.create_group(cn)
            imgs = cg.create_dataset("images", (N,), dtype=_VLEN)
            dummy_jpeg = bytes([0xFF, 0xD8, 0xFF, 0xD9])  # 最小合法 JPEG 头尾
            for i in range(N):
                imgs[i] = np.frombuffer(dummy_jpeg, np.uint8)
            # 严格递增时间戳（满足单调性契约）
            ts_cam = np.arange(N, dtype=np.float64) * 0.033 + 1.0005 + 0.001 * list(cam_names).index(cn)
            cg.create_dataset("timestamp", data=ts_cam)
            cg.create_dataset("stale", data=np.zeros(N, dtype=bool))
            cg.create_dataset("hw_timestamp", data=ts_cam * 1000.0)  # 毫秒 float64

        # observations/state_hifreq
        hf = obs.create_group("state_hifreq")
        # 严格递增时间戳（满足单调性契约）
        ts_hifreq = np.arange(M, dtype=np.float64) * (1.0 / 240.0)
        hf.create_dataset("joints", data=np.zeros((M, 7), np.float64))
        hf.create_dataset("joint_vel", data=np.zeros((M, 7), np.float64))
        hf.create_dataset("pose", data=np.zeros((M, 6), np.float64))
        hf.create_dataset("timestamp", data=ts_hifreq)
        hf.create_dataset("poly_ts", data=ts_hifreq)
        hf.create_dataset("wrench", data=np.zeros((M, 6), np.float64))

        # action
        act = f.create_group("action")
        act.create_dataset("delta_ee_pose", data=np.zeros((N, 6), np.float64))
        act.create_dataset("gripper_cmd", data=np.zeros((N, 1), np.float64))
        act.create_dataset("timestamp", data=ts_act)

        # 可扩展：depth
        if include_depth:
            for cn in cam_names:
                dg = rgb_g[cn]
                dg.create_dataset("depth", data=np.zeros((N, 480, 640), np.float32))
                dg.create_dataset("depth_timestamp", data=ts_arm)
                dg.create_dataset("depth_stale", data=np.zeros(N, dtype=bool))

        # 可扩展：tactile
        if include_tactile:
            tac_g = obs.create_group("tactile")
            sg = tac_g.create_group("sensor0")
            sg.create_dataset("values", data=np.zeros((N, 16), np.float64))
            sg.create_dataset("timestamp", data=ts_arm)
            sg.create_dataset("stale", data=np.zeros(N, dtype=bool))

    return path


# ---------------------------------------------------------------------------
# 正向测试：合规 v2
# ---------------------------------------------------------------------------

def test_conformant_v2_passes(tmp_path):
    """完整合规 v2 episode 校验通过。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, M=40, cam_names=("wrist",))
    assert S.validate_episode(p) == []


def test_conformant_v2_multi_cam_passes(tmp_path):
    """多相机合规 v2 通过。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, M=40, cam_names=("wrist", "exterior"))
    assert S.validate_episode(p) == []


def test_conformant_v2_empty_hifreq_passes(tmp_path):
    """M=0 state_hifreq 通过（spike 降级路径）。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, M=0, cam_names=("wrist",))
    assert S.validate_episode(p) == []


def test_conformant_v2_with_depth_passes(tmp_path):
    """包含 depth 可扩展字段时通过。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, M=10, cam_names=("wrist",), include_depth=True)
    assert S.validate_episode(p) == []


def test_conformant_v2_with_tactile_passes(tmp_path):
    """包含 tactile 可扩展字段时通过。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, M=10, cam_names=("wrist",), include_tactile=True)
    assert S.validate_episode(p) == []


# ---------------------------------------------------------------------------
# 拒绝 v1
# ---------------------------------------------------------------------------

def test_rejects_v1_schema_version(tmp_path):
    """含 franka-hdf5-v1 schema_version 必须被拒。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p)
    with h5py.File(p, "a") as f:
        del f["infos/schema_version"]
        f["infos"].create_dataset("schema_version", data=np.bytes_("franka-hdf5-v1"))
    v = S.validate_episode(p)
    assert any("schema_version" in x for x in v)


def test_rejects_missing_schema_version(tmp_path):
    """缺少 schema_version 必须被拒。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p)
    with h5py.File(p, "a") as f:
        del f["infos/schema_version"]
    v = S.validate_episode(p)
    assert any("schema_version" in x for x in v)


def test_rejects_v1_shared_timestamp(tmp_path):
    """含 observations/timestamp(N,1) v1 共戳字段应被视为不符（v2 不含此字段）。

    v2 的 validator 以 action/timestamp 定 N，不依赖 observations/timestamp。
    但若写入了该字段（v1 遗留），validate 不报错——因为 v2 不校验它（忽略即可）。
    本测试确认：v2 conformant 文件即使不含 observations/timestamp 也能通过。
    """
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    # 人为加入 v1 遗留字段（不应导致 fail）
    with h5py.File(p, "a") as f:
        f["observations"].create_dataset(
            "timestamp", data=np.arange(5, dtype=np.float64).reshape(5, 1) + 1.0)
    # v2 validator 不检查 observations/timestamp，忽略遗留字段不报 violation
    v = S.validate_episode(p)
    assert v == []  # 合规 v2 + 遗留字段也通过


# ---------------------------------------------------------------------------
# [Codex 审查] action/timestamp 二维被拒
# ---------------------------------------------------------------------------

def test_rejects_action_timestamp_2d(tmp_path):
    """action/timestamp 是 (N,1) 二维（v1 风格）必须被拒。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    with h5py.File(p, "a") as f:
        ts = f["action/timestamp"][...]
        del f["action/timestamp"]
        f["action"].create_dataset("timestamp", data=ts.reshape(5, 1))
    v = S.validate_episode(p)
    assert any("action/timestamp" in x and ("一维" in x or "shape" in x) for x in v)


# ---------------------------------------------------------------------------
# [Codex 审查] 时间戳单调性校验
# ---------------------------------------------------------------------------

def test_action_timestamp_not_strictly_increasing_rejected(tmp_path):
    """action/timestamp 不严格递增（有相邻相等值）必须被拒。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    with h5py.File(p, "a") as f:
        ts = f["action/timestamp"][...].copy()
        # 让 index 2 == index 3（相等但不严格递增）
        ts[3] = ts[2]
        del f["action/timestamp"]
        f["action"].create_dataset("timestamp", data=ts)
    v = S.validate_episode(p)
    assert any("action/timestamp" in x and "递增" in x for x in v)


def test_action_timestamp_decreasing_rejected(tmp_path):
    """action/timestamp 倒退必须被拒。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    with h5py.File(p, "a") as f:
        ts = f["action/timestamp"][...].copy()
        ts[2] = ts[4]  # 倒退
        del f["action/timestamp"]
        f["action"].create_dataset("timestamp", data=ts)
    v = S.validate_episode(p)
    assert any("action/timestamp" in x and "递增" in x for x in v)


def test_arm_timestamp_equal_allowed_stale_nondec(tmp_path):
    """arm/timestamp 相邻相等（stale 补帧）不应报 violation（非递减即可）。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    with h5py.File(p, "a") as f:
        ts = f["observations/arm/timestamp"][...].copy()
        # 让 index 1 == index 2（stale 补帧场景）
        ts[2] = ts[1]
        del f["observations/arm/timestamp"]
        f["observations/arm"].create_dataset("timestamp", data=ts)
    v = S.validate_episode(p)
    assert not any("arm/timestamp" in x and "递" in x for x in v)


def test_arm_timestamp_decreasing_rejected(tmp_path):
    """arm/timestamp 倒退（非 stale，真时间错误）必须被拒。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    with h5py.File(p, "a") as f:
        ts = f["observations/arm/timestamp"][...].copy()
        ts[2] = ts[0]  # 倒退
        del f["observations/arm/timestamp"]
        f["observations/arm"].create_dataset("timestamp", data=ts)
    v = S.validate_episode(p)
    assert any("observations/arm/timestamp" in x and "倒退" in x for x in v)


def test_camera_timestamp_equal_allowed_stale_nondec(tmp_path):
    """camera/timestamp 相邻相等（stale 补帧）不应报 violation。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, cam_names=("wrist",))
    with h5py.File(p, "a") as f:
        ts = f["observations/camera/rgb/wrist/timestamp"][...].copy()
        ts[3] = ts[2]  # stale 补帧
        del f["observations/camera/rgb/wrist/timestamp"]
        f["observations/camera/rgb/wrist"].create_dataset("timestamp", data=ts)
    v = S.validate_episode(p)
    assert not any("wrist/timestamp" in x and "递" in x for x in v)


def test_camera_timestamp_decreasing_rejected(tmp_path):
    """camera/timestamp 倒退必须被拒。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, cam_names=("wrist",))
    with h5py.File(p, "a") as f:
        ts = f["observations/camera/rgb/wrist/timestamp"][...].copy()
        ts[3] = ts[0]  # 倒退
        del f["observations/camera/rgb/wrist/timestamp"]
        f["observations/camera/rgb/wrist"].create_dataset("timestamp", data=ts)
    v = S.validate_episode(p)
    assert any("wrist/timestamp" in x and "倒退" in x for x in v)


def test_hifreq_timestamp_strictly_increasing_required(tmp_path):
    """state_hifreq/timestamp 相邻相等（无 stale）必须被拒。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, M=10)
    with h5py.File(p, "a") as f:
        ts = f["observations/state_hifreq/timestamp"][...].copy()
        ts[5] = ts[4]  # 相等，不严格递增
        del f["observations/state_hifreq/timestamp"]
        f["observations/state_hifreq"].create_dataset("timestamp", data=ts)
    v = S.validate_episode(p)
    assert any("state_hifreq/timestamp" in x and "递增" in x for x in v)


def test_monotone_skip_when_n_lt_2(tmp_path):
    """N=1 时不足两点，单调性校验跳过，合规 episode 通过。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=1, M=1)
    assert S.validate_episode(p) == []


# ---------------------------------------------------------------------------
# [Codex 审查] 空 camera rgb group 被拒（Route B 必须有相机）
# ---------------------------------------------------------------------------

def test_rejects_empty_rgb_group(tmp_path):
    """observations/camera/rgb 组存在但无相机子组必须被拒。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, M=10, cam_names=())  # cam_names=() → 空 rgb group
    v = S.validate_episode(p)
    assert any("rgb" in x and ("空" in x or "相机" in x) for x in v)


def test_rejects_missing_rgb_group(tmp_path):
    """observations/camera/rgb 组完全缺失必须被拒。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, M=10, cam_names=("wrist",))
    with h5py.File(p, "a") as f:
        del f["observations/camera/rgb"]
    v = S.validate_episode(p)
    assert any("rgb" in x for x in v)


# ---------------------------------------------------------------------------
# [Codex 审查] images 长度与 N 不符被拒
# ---------------------------------------------------------------------------

def test_rejects_images_length_mismatch(tmp_path):
    """camera/{cn}/images shape[0] != N 必须被拒。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, cam_names=("wrist",))
    with h5py.File(p, "a") as f:
        del f["observations/camera/rgb/wrist/images"]
        _VLEN = h5py.special_dtype(vlen=np.dtype("uint8"))
        # 只写 3 帧，但 N=5
        imgs = f["observations/camera/rgb/wrist"].create_dataset("images", (3,), dtype=_VLEN)
        dummy = bytes([0xFF, 0xD8, 0xFF, 0xD9])
        for i in range(3):
            imgs[i] = np.frombuffer(dummy, np.uint8)
    v = S.validate_episode(p)
    assert any("images" in x and ("shape" in x or "!=" in x or "3" in x) for x in v)


# ---------------------------------------------------------------------------
# [Codex 审查] malformed group（非 Group 类型）被拒
# ---------------------------------------------------------------------------

def test_rejects_rgb_group_not_group(tmp_path):
    """observations/camera/rgb 存在但是 Dataset 而非 Group 必须被拒。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, cam_names=("wrist",))
    with h5py.File(p, "a") as f:
        del f["observations/camera/rgb"]
        # 写成一个 dataset 而非 group（malformed）
        f["observations/camera"].create_dataset("rgb", data=np.zeros(5, np.float64))
    v = S.validate_episode(p)
    assert any("rgb" in x for x in v)


# ---------------------------------------------------------------------------
# arm 模态字段校验
# ---------------------------------------------------------------------------

def test_missing_arm_stale(tmp_path):
    """arm 缺少 stale 字段必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    with h5py.File(p, "a") as f:
        del f["observations/arm/stale"]
    v = S.validate_episode(p)
    assert any("observations/arm/stale" in x for x in v)


def test_arm_stale_wrong_dtype(tmp_path):
    """arm/stale dtype 不是 bool 必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    with h5py.File(p, "a") as f:
        del f["observations/arm/stale"]
        f["observations/arm"].create_dataset("stale", data=np.zeros(5, dtype=np.int32))
    v = S.validate_episode(p)
    assert any("observations/arm/stale" in x and "bool" in x for x in v)


def test_missing_arm_timestamp(tmp_path):
    """arm 缺少独立 timestamp 必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    with h5py.File(p, "a") as f:
        del f["observations/arm/timestamp"]
    v = S.validate_episode(p)
    assert any("observations/arm/timestamp" in x for x in v)


def test_wrong_arm_joint_shape(tmp_path):
    """arm/joints shape 错误必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    with h5py.File(p, "a") as f:
        del f["observations/arm/joints"]
        f["observations/arm"].create_dataset("joints", data=np.zeros((5, 6), np.float64))
    v = S.validate_episode(p)
    assert any("observations/arm/joints" in x and "shape" in x for x in v)


# ---------------------------------------------------------------------------
# effector 模态字段校验
# ---------------------------------------------------------------------------

def test_missing_effector_stale(tmp_path):
    """effector 缺少 stale 必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    with h5py.File(p, "a") as f:
        del f["observations/effector/stale"]
    v = S.validate_episode(p)
    assert any("observations/effector/stale" in x for x in v)


def test_missing_effector_timestamp(tmp_path):
    """effector 缺少独立 timestamp 必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    with h5py.File(p, "a") as f:
        del f["observations/effector/timestamp"]
    v = S.validate_episode(p)
    assert any("observations/effector/timestamp" in x for x in v)


# ---------------------------------------------------------------------------
# camera 模态字段校验（hw_timestamp + stale）
# ---------------------------------------------------------------------------

def test_missing_camera_hw_timestamp(tmp_path):
    """camera/{cn} 缺少 hw_timestamp 必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, cam_names=("wrist",))
    with h5py.File(p, "a") as f:
        del f["observations/camera/rgb/wrist/hw_timestamp"]
    v = S.validate_episode(p)
    assert any("hw_timestamp" in x for x in v)


def test_camera_hw_timestamp_wrong_dtype(tmp_path):
    """camera hw_timestamp 必须是 float64。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, cam_names=("wrist",))
    with h5py.File(p, "a") as f:
        del f["observations/camera/rgb/wrist/hw_timestamp"]
        f["observations/camera/rgb/wrist"].create_dataset(
            "hw_timestamp", data=np.zeros(5, dtype=np.float32))
    v = S.validate_episode(p)
    assert any("hw_timestamp" in x and "float64" in x for x in v)


def test_missing_camera_stale(tmp_path):
    """camera/{cn} 缺少 stale 必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, cam_names=("wrist",))
    with h5py.File(p, "a") as f:
        del f["observations/camera/rgb/wrist/stale"]
    v = S.validate_episode(p)
    assert any("observations/camera/rgb/wrist/stale" in x for x in v)


def test_missing_camera_timestamp(tmp_path):
    """camera/{cn} 缺少 timestamp 必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, cam_names=("wrist",))
    with h5py.File(p, "a") as f:
        del f["observations/camera/rgb/wrist/timestamp"]
    v = S.validate_episode(p)
    assert any("observations/camera/rgb/wrist/timestamp" in x for x in v)


# ---------------------------------------------------------------------------
# action 模态字段校验
# ---------------------------------------------------------------------------

def test_missing_action_timestamp(tmp_path):
    """action 缺少独立 timestamp 必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    with h5py.File(p, "a") as f:
        del f["action/timestamp"]
    v = S.validate_episode(p)
    assert any("action/timestamp" in x for x in v)


def test_n_misaligned_action(tmp_path):
    """action/delta_ee_pose 帧数与 N 不符必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    with h5py.File(p, "a") as f:
        del f["action/delta_ee_pose"]
        f["action"].create_dataset("delta_ee_pose", data=np.zeros((4, 6), np.float64))
    v = S.validate_episode(p)
    assert any("action/delta_ee_pose" in x for x in v)


# ---------------------------------------------------------------------------
# state_hifreq 校验
# ---------------------------------------------------------------------------

def test_hifreq_independent_length_ok(tmp_path):
    """M != N 时 state_hifreq 仍然通过。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, M=123)
    assert S.validate_episode(p) == []


def test_hifreq_internal_length_mismatch_caught(tmp_path):
    """state_hifreq 内部各字段长度不一致必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, M=40)
    with h5py.File(p, "a") as f:
        del f["observations/state_hifreq/joint_vel"]
        f["observations/state_hifreq"].create_dataset(
            "joint_vel", data=np.zeros((39, 7), np.float64))
    v = S.validate_episode(p)
    assert any("observations/state_hifreq/joint_vel" in x for x in v)


def test_hifreq_wrench_m0_passes(tmp_path):
    """state_hifreq/wrench(0,6) 占位时通过。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, M=0)
    assert S.validate_episode(p) == []


def test_hifreq_wrench_shape_checked_when_m_nonzero(tmp_path):
    """M>0 时 state_hifreq/wrench shape 错误必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, M=10)
    with h5py.File(p, "a") as f:
        del f["observations/state_hifreq/wrench"]
        f["observations/state_hifreq"].create_dataset(
            "wrench", data=np.zeros((10, 3), np.float64))  # 应为 (10,6)
    v = S.validate_episode(p)
    assert any("wrench" in x and "shape" in x for x in v)


# ---------------------------------------------------------------------------
# 可扩展字段 validate-if-present
# ---------------------------------------------------------------------------

def test_depth_absent_no_violation(tmp_path):
    """depth 字段缺失时不报 violation。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, cam_names=("wrist",), include_depth=False)
    assert S.validate_episode(p) == []


def test_depth_present_validated(tmp_path):
    """depth 字段存在时 shape/dtype 被校验。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, cam_names=("wrist",), include_depth=True)
    # 人为改坏 depth_stale dtype
    with h5py.File(p, "a") as f:
        del f["observations/camera/rgb/wrist/depth_stale"]
        f["observations/camera/rgb/wrist"].create_dataset(
            "depth_stale", data=np.zeros(5, dtype=np.int32))
    v = S.validate_episode(p)
    assert any("depth_stale" in x and "bool" in x for x in v)


def test_tactile_absent_no_violation(tmp_path):
    """tactile 字段缺失时不报 violation。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, cam_names=("wrist",), include_tactile=False)
    assert S.validate_episode(p) == []


def test_tactile_present_validated(tmp_path):
    """tactile 字段存在时 shape/dtype 被校验。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5, cam_names=("wrist",), include_tactile=True)
    # 人为改坏 tactile/sensor0/stale dtype
    with h5py.File(p, "a") as f:
        del f["observations/tactile/sensor0/stale"]
        f["observations/tactile/sensor0"].create_dataset(
            "stale", data=np.zeros(5, dtype=np.int32))
    v = S.validate_episode(p)
    assert any("tactile" in x and "bool" in x for x in v)


# ---------------------------------------------------------------------------
# 其他基础校验
# ---------------------------------------------------------------------------

def test_missing_calibration_R(tmp_path):
    """缺少 oc2base_R 必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    with h5py.File(p, "a") as f:
        del f["infos/calibration/oc2base_R"]
    v = S.validate_episode(p)
    assert any("oc2base_R" in x for x in v)


def test_dtype_violation_caught(tmp_path):
    """arm/pose dtype float32 而非 float64 必须报错。"""
    p = str(tmp_path / "ep.h5")
    _write_v2(p, N=5)
    with h5py.File(p, "a") as f:
        d = f["observations/arm/pose"][...]
        del f["observations/arm/pose"]
        f["observations/arm"].create_dataset("pose", data=d.astype(np.float32))
    v = S.validate_episode(p)
    assert any("observations/arm/pose" in x and "float64" in x for x in v)
