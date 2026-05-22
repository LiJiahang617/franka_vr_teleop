"""test_state_hifreq_collector.py

Task 7-A 专项测试：state_hifreq 240Hz 累积采集机制（离线 mock 验证）。

验证内容：
1. HistoryCollectorThread 能累积所有历史采样（列表增长，不覆盖）
2. get_history() 返回正确的 HistorySample（data/ts/poly_ts 字段完整）
3. stop() 优雅停止，无线程泄漏
4. zerorpc_lock 约束：read_fn 持锁与 send_action 串行互斥（用计数器验证非并发）
5. record_episode(hifreq_rate=240.0) 采集到 M>0 帧，state_hifreq_block 结构正确
6. write_episode 写入 M>0 的 state_hifreq，schema validate_episode 通过
7. align_offline 对 M>0 的 state_hifreq 原样返回（不重采）
8. poly_ts：无 "poly_ts" 键时用 monotonic 占位（与 ts 相同）
"""
import importlib.util
import os
import sys
import threading
import time

import h5py
import numpy as np
import pytest

_P = "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop"
sys.path.insert(0, _P)
sys.path.insert(0, os.path.join(_P, "scripts"))

from core.acquisition import HistoryCollectorThread, HistorySample
from core.hdf5_writer import write_episode
import franka_hdf5_schema as S
from scripts.tools.align_offline import align_by_image_timestamp

# --- 加载 run_record_hdf5 模块（含硬件延迟 import）---
_rrh_spec = importlib.util.spec_from_file_location(
    "rrh", os.path.join(_P, "scripts/core/run_record_hdf5.py")
)


def _load_rrh():
    m = importlib.util.module_from_spec(_rrh_spec)
    _rrh_spec.loader.exec_module(m)
    return m


# ===========================================================================
# 测试工具类（FakeRobot/FakeCam/FakeTeleop）
# ===========================================================================

class FakeCam:
    """带 read() 方法的假相机。"""
    def __init__(self, shape=(8, 8, 3)):
        self._shape = shape

    def read(self):
        return np.zeros(self._shape, np.uint8)


class FakeRobot:
    """没有 _robot 属性（zerorpc 不可用），回退到 get_observation()。"""
    def __init__(self):
        self.cameras = {"wrist_image": FakeCam()}
        self._call_count = 0
        self._lock = threading.Lock()

    def get_observation(self):
        with self._lock:
            self._call_count += 1
        o = {f"joint_{i+1}.pos": float(self._call_count) * 0.01 for i in range(7)}
        o.update({f"joint_{i+1}.vel": 0.0 for i in range(7)})
        o.update({f"ee_pose.{a}": 0.0 for a in "x y z rx ry rz".split()})
        o["gripper_state_norm"] = 0.5
        o["wrist_image"] = np.zeros((8, 8, 3), np.uint8)
        return o

    def send_action(self, a):
        pass


class FakeTeleop:
    def get_action(self):
        a = {f"delta_ee_pose.{x}": 0.0 for x in "x y z rx ry rz".split()}
        a["gripper_cmd_bin"] = 0.0
        return a


# ===========================================================================
# HistoryCollectorThread 单元测试
# ===========================================================================

class TestHistoryCollectorThread:
    """HistoryCollectorThread 核心机制测试。"""

    def test_accumulates_all_samples(self):
        """累积采集：历史列表持续增长，不覆盖旧样本。"""
        stop_event = threading.Event()
        call_count = [0]

        def read_fn():
            call_count[0] += 1
            return {"joints": np.ones(7) * call_count[0], "joint_vel": np.zeros(7), "ee_pose": np.zeros(6)}

        collector = HistoryCollectorThread(
            name="test",
            read_fn=read_fn,
            target_rate=200.0,
            stop_event=stop_event,
        )

        time.sleep(0.05)  # 让它采 ~10 帧
        collector.stop(timeout=1.0)

        history = collector.get_history()
        assert len(history) >= 5, f"应累积 >=5 帧，实际 {len(history)}"
        # 验证是累积不是覆盖：所有样本的 joints[0] 应是连续递增
        vals = [h.data["joints"][0] for h in history]
        assert vals == sorted(vals), "历史列表顺序应与采集顺序一致"
        assert len(set(vals)) == len(vals), "累积采集不应有重复帧（单槽覆盖 bug）"

    def test_sample_has_ts_and_poly_ts(self):
        """每个 HistorySample 含 ts（monotonic 戳）和 poly_ts（同或不同）。"""
        stop_event = threading.Event()

        def read_fn():
            return {"joints": np.zeros(7), "joint_vel": np.zeros(7), "ee_pose": np.zeros(6)}

        collector = HistoryCollectorThread(
            name="test_ts",
            read_fn=read_fn,
            target_rate=100.0,
            stop_event=stop_event,
        )
        time.sleep(0.03)
        collector.stop(timeout=1.0)

        history = collector.get_history()
        assert len(history) > 0
        for s in history:
            assert isinstance(s, HistorySample)
            assert isinstance(s.ts, float) and s.ts > 0
            assert isinstance(s.poly_ts, float) and s.poly_ts > 0
            # 无 poly_ts 键时用 ts 占位，因此 poly_ts == ts
            assert s.poly_ts == s.ts, "无 poly_ts 键时应用 ts 占位"

    def test_poly_ts_from_dict_key(self):
        """若 read_fn 返回 dict 含 poly_ts 键，该值用作 poly_ts（非 ts 占位）。"""
        stop_event = threading.Event()
        fake_poly = [0.0]

        def read_fn():
            fake_poly[0] += 1.0
            return {
                "joints": np.zeros(7),
                "joint_vel": np.zeros(7),
                "ee_pose": np.zeros(6),
                "poly_ts": fake_poly[0],  # 显式 poly_ts
            }

        collector = HistoryCollectorThread(
            name="test_poly",
            read_fn=read_fn,
            target_rate=100.0,
            stop_event=stop_event,
        )
        time.sleep(0.03)
        collector.stop(timeout=1.0)

        history = collector.get_history()
        assert len(history) > 0
        for i, s in enumerate(history):
            # poly_ts 应是 fake_poly 的值（1.0, 2.0, ...），不是 ts
            assert s.poly_ts != s.ts, f"样本 {i}: poly_ts 应来自 dict key，不是 ts 占位"
            assert s.poly_ts >= 1.0

    def test_stop_cleanly_no_thread_leak(self):
        """stop() 优雅停止，线程数恢复到启动前水平。"""
        before = threading.active_count()
        stop_event = threading.Event()
        collector = HistoryCollectorThread(
            name="test_stop",
            read_fn=lambda: {"joints": np.zeros(7), "joint_vel": np.zeros(7), "ee_pose": np.zeros(6)},
            target_rate=100.0,
            stop_event=stop_event,
        )
        assert collector.is_alive()
        assert threading.active_count() > before
        ok = collector.stop(timeout=1.0)
        assert ok, "stop() 应返回 True（线程已 join）"
        assert not collector.is_alive()
        assert threading.active_count() <= before + 1  # ±1 波动允许

    def test_get_history_is_snapshot_copy(self):
        """get_history() 返回快照 copy，外部修改不影响内部列表。"""
        stop_event = threading.Event()
        collector = HistoryCollectorThread(
            name="test_copy",
            read_fn=lambda: {"joints": np.zeros(7), "joint_vel": np.zeros(7), "ee_pose": np.zeros(6)},
            target_rate=50.0,
            stop_event=stop_event,
        )
        time.sleep(0.05)
        collector.stop(timeout=1.0)

        h1 = collector.get_history()
        h1.clear()  # 外部修改
        h2 = collector.get_history()
        assert len(h2) > 0, "外部 clear() 不应影响内部历史列表（应是快照 copy）"

    def test_invalid_rate_raises(self):
        """target_rate <= 0 应在构造时抛 ValueError。"""
        with pytest.raises(ValueError):
            HistoryCollectorThread(name="bad", read_fn=lambda: {}, target_rate=0.0)
        with pytest.raises(ValueError):
            HistoryCollectorThread(name="bad", read_fn=lambda: {}, target_rate=-1.0)


# ===========================================================================
# zerorpc_lock 串行化测试
# ===========================================================================

class TestZerorpcLockSerialism:
    """验证 state_hifreq read_fn 与 send_action 在 zerorpc_lock 保护下串行执行。"""

    def test_concurrent_access_is_serialized(self):
        """模拟 state_hifreq 线程和 send_action 持同一把锁：不出现并发（计数器不乱）。

        设计：用一个共享的"正在锁内"布尔值，验证两个线程不同时持锁。
        """
        lock = threading.Lock()
        inside = [False]
        concurrency_violated = [False]

        def hifreq_read_fn():
            with lock:
                if inside[0]:
                    concurrency_violated[0] = True
                inside[0] = True
                time.sleep(0.001)  # 模拟 zerorpc 往返
                inside[0] = False
            return {"joints": np.zeros(7), "joint_vel": np.zeros(7), "ee_pose": np.zeros(6)}

        stop_event = threading.Event()
        collector = HistoryCollectorThread(
            name="lock_test",
            read_fn=hifreq_read_fn,
            target_rate=100.0,
            stop_event=stop_event,
        )

        # 主线程模拟 send_action（也持 lock）
        for _ in range(20):
            with lock:
                if inside[0]:
                    concurrency_violated[0] = True
                inside[0] = True
                time.sleep(0.0005)
                inside[0] = False
            time.sleep(0.002)

        collector.stop(timeout=1.0)

        assert not concurrency_violated[0], (
            "检测到并发访问！zerorpc_lock 串行化失效，state_hifreq 线程与 send_action 同时持锁"
        )


# ===========================================================================
# record_episode(hifreq_rate=240.0) 集成测试
# ===========================================================================

class TestRecordEpisodeWithHifreq:
    """验证 record_episode 接入 state_hifreq 累积采集。"""

    def test_record_episode_returns_tuple(self):
        """record_episode 返回 (buf, state_hifreq_block) tuple。"""
        m = _load_rrh()
        robot = FakeRobot()
        teleop = FakeTeleop()
        result = m.record_episode(
            robot, teleop, fps=30.0, max_sec=0.1,
            gripper_max_open=0.08, cam_names=["wrist_image"],
            hifreq_rate=240.0,
        )
        assert isinstance(result, tuple) and len(result) == 2, (
            f"record_episode 应返回 2 元素 tuple，实际 {type(result)}"
        )
        buf, block = result
        assert isinstance(buf, list) and len(buf) > 0

    def test_hifreq_block_not_none_when_rate_positive(self):
        """hifreq_rate>0 时，state_hifreq_block 不为 None 且含正确键。"""
        m = _load_rrh()
        robot = FakeRobot()
        teleop = FakeTeleop()
        buf, block = m.record_episode(
            robot, teleop, fps=30.0, max_sec=0.15,
            gripper_max_open=0.08, cam_names=["wrist_image"],
            hifreq_rate=240.0,
        )
        assert block is not None, "hifreq_rate=240.0 时 state_hifreq_block 不应为 None"
        for key in ("joints", "joint_vel", "pose", "timestamp", "poly_ts", "wrench"):
            assert key in block, f"state_hifreq_block 缺少键 {key!r}"

    def test_hifreq_block_shape(self):
        """state_hifreq_block 的数组 shape 正确：(M,7)/(M,7)/(M,6)/(M,)/(M,)/(M,6)。"""
        m = _load_rrh()
        robot = FakeRobot()
        teleop = FakeTeleop()
        buf, block = m.record_episode(
            robot, teleop, fps=30.0, max_sec=0.15,
            gripper_max_open=0.08, cam_names=["wrist_image"],
            hifreq_rate=240.0,
        )
        assert block is not None
        M = len(block["timestamp"])
        assert M > 0, "M 应 > 0（已采到样本）"
        assert block["joints"].shape == (M, 7), f"joints shape={block['joints'].shape}，期望 ({M},7)"
        assert block["joint_vel"].shape == (M, 7)
        assert block["pose"].shape == (M, 6)
        assert block["poly_ts"].shape == (M,)
        assert block["wrench"].shape == (M, 6)

    def test_hifreq_block_m_grows_with_time(self):
        """state_hifreq_block M 与录制时长正相关（hifreq_rate×max_sec）。"""
        m = _load_rrh()
        robot = FakeRobot()
        teleop = FakeTeleop()
        _, block = m.record_episode(
            robot, teleop, fps=30.0, max_sec=0.2,
            gripper_max_open=0.08, cam_names=["wrist_image"],
            hifreq_rate=100.0,  # 用 100Hz 更容易验证（测试环境 CPU 争抢）
        )
        assert block is not None
        M = len(block["timestamp"])
        # 0.2s × 100Hz = 20 帧，考虑测试环境 CPU 调度延迟，允许 M >= 5
        assert M >= 5, f"M={M}，期望 >=5（100Hz × 0.2s，测试环境宽容）"

    def test_hifreq_block_none_when_rate_zero(self):
        """hifreq_rate=0 时，state_hifreq_block 为 None（不起采集线程）。"""
        m = _load_rrh()
        robot = FakeRobot()
        teleop = FakeTeleop()
        buf, block = m.record_episode(
            robot, teleop, fps=30.0, max_sec=0.1,
            gripper_max_open=0.08, cam_names=["wrist_image"],
            hifreq_rate=0.0,
        )
        assert block is None, f"hifreq_rate=0 时 block 应为 None，实际 {block}"

    def test_hifreq_timestamp_strictly_increasing(self):
        """state_hifreq/timestamp 严格递增（累积采集时序正确）。"""
        m = _load_rrh()
        robot = FakeRobot()
        teleop = FakeTeleop()
        _, block = m.record_episode(
            robot, teleop, fps=30.0, max_sec=0.2,
            gripper_max_open=0.08, cam_names=["wrist_image"],
            hifreq_rate=100.0,
        )
        assert block is not None
        ts = block["timestamp"]
        if len(ts) >= 2:
            diffs = np.diff(ts)
            assert np.all(diffs > 0), (
                f"state_hifreq/timestamp 不严格递增：{diffs[diffs <= 0]}"
            )


# ===========================================================================
# write_episode + schema 校验（M>0 state_hifreq）
# ===========================================================================

def _make_frame(i: int) -> dict:
    """生成测试帧（v2 格式，含各模态独立时间戳）。"""
    ts = 1.0 + i * 0.033
    return dict(
        ts=ts,
        arm_ts=ts + 0.001,
        effector_ts=ts + 0.001,
        cam_ts={"wrist_image": ts + 0.003},
        cam_hw_ts={"wrist_image": ts + 0.003},
        arm_stale=False,
        effector_stale=False,
        cam_stale={"wrist_image": False},
        joints=np.zeros(7, np.float64),
        joint_vel=np.zeros(7, np.float64),
        ee_pose=np.array([0.1, 0.0, 0.3, 0.0, 0.0, 0.0], np.float64),
        gripper_m=0.04,
        gripper_norm=0.5,
        gripper_cmd=0.0,
        delta_ee_pose=np.zeros(6, np.float64),
        cams={"wrist_image": np.zeros((4,), np.uint8)},
    )


def _make_hifreq_block(M: int) -> dict:
    """生成 M>0 的 state_hifreq_block。"""
    ts_base = 0.5  # state_hifreq 从录制开始前就在采集
    return {
        "joints": np.zeros((M, 7), np.float64),
        "joint_vel": np.zeros((M, 7), np.float64),
        "pose": np.zeros((M, 6), np.float64),
        "timestamp": np.array([ts_base + i * (1.0 / 240.0) for i in range(M)], np.float64),
        "poly_ts": np.array([ts_base + i * (1.0 / 240.0) for i in range(M)], np.float64),
        "wrench": np.zeros((M, 6), np.float64),
    }


class TestWriteEpisodeWithHifreq:
    """write_episode 写入 M>0 state_hifreq 时 schema 校验通过。"""

    def test_write_m_gt_0_schema_valid(self, tmp_path):
        """M=50 的 state_hifreq_block 写入后 validate_episode 通过。"""
        path = str(tmp_path / "ep_hifreq.h5")
        frames = [_make_frame(i) for i in range(5)]
        block = _make_hifreq_block(50)

        write_episode(
            path, frames,
            task_name="hifreq_test",
            target_fps=30.0,
            oc2base_R=np.eye(3),
            quality={},
            vr_source="test",
            cam_names=["wrist_image"],
            state_hifreq_block=block,
        )

        violations = S.validate_episode(path)
        assert violations == [], f"schema 校验失败：{violations}"

    def test_write_m_gt_0_shape_in_hdf5(self, tmp_path):
        """M=50 时 hdf5 中 state_hifreq 各 dataset shape 正确。"""
        path = str(tmp_path / "ep_hifreq_shape.h5")
        frames = [_make_frame(i) for i in range(5)]
        M = 50
        block = _make_hifreq_block(M)

        write_episode(
            path, frames,
            task_name="hifreq_test",
            target_fps=30.0,
            oc2base_R=np.eye(3),
            quality={},
            vr_source="test",
            cam_names=["wrist_image"],
            state_hifreq_block=block,
        )

        with h5py.File(path, "r") as f:
            assert f["observations/state_hifreq/joints"].shape == (M, 7)
            assert f["observations/state_hifreq/joint_vel"].shape == (M, 7)
            assert f["observations/state_hifreq/pose"].shape == (M, 6)
            assert f["observations/state_hifreq/timestamp"].shape == (M,)
            assert f["observations/state_hifreq/poly_ts"].shape == (M,)
            assert f["observations/state_hifreq/wrench"].shape == (M, 6)

    def test_write_m_0_still_valid(self, tmp_path):
        """state_hifreq_block=None（M=0 占位）时仍 schema 校验通过。"""
        path = str(tmp_path / "ep_m0.h5")
        frames = [_make_frame(i) for i in range(5)]
        write_episode(
            path, frames,
            task_name="m0_test",
            target_fps=30.0,
            oc2base_R=np.eye(3),
            quality={},
            vr_source="test",
            cam_names=["wrist_image"],
            state_hifreq_block=None,
        )
        violations = S.validate_episode(path)
        assert violations == [], f"M=0 schema 校验失败：{violations}"

    def test_write_m_gt_0_timestamp_strictly_increasing(self, tmp_path):
        """写入的 state_hifreq/timestamp 应严格递增（schema 要求）。"""
        path = str(tmp_path / "ep_ts_mono.h5")
        frames = [_make_frame(i) for i in range(5)]
        M = 30
        block = _make_hifreq_block(M)

        write_episode(
            path, frames,
            task_name="ts_mono",
            target_fps=30.0,
            oc2base_R=np.eye(3),
            quality={},
            vr_source="test",
            cam_names=["wrist_image"],
            state_hifreq_block=block,
        )

        with h5py.File(path, "r") as f:
            ts = f["observations/state_hifreq/timestamp"][...]
        diffs = np.diff(ts)
        assert np.all(diffs > 0), f"state_hifreq/timestamp 不严格递增：{diffs[diffs <= 0]}"


# ===========================================================================
# align_offline：state_hifreq 原样传递（不重采）
# ===========================================================================

class TestAlignOfflineHifreq:
    """align_by_image_timestamp 对 M>0 state_hifreq 原样返回。"""

    def _write_test_hdf5(self, path, M_hifreq: int):
        """写入一个包含真实 state_hifreq(M>0) 的测试 hdf5。"""
        N = 5
        frames = [_make_frame(i) for i in range(N)]
        block = _make_hifreq_block(M_hifreq)
        write_episode(
            path, frames,
            task_name="align_test",
            target_fps=30.0,
            oc2base_R=np.eye(3),
            quality={},
            vr_source="test",
            cam_names=["wrist_image"],
            state_hifreq_block=block,
        )
        return block

    def test_align_preserves_hifreq_joints(self, tmp_path):
        """align_by_image_timestamp 返回的 state_hifreq_joints 与写入的完全一致。"""
        path = str(tmp_path / "ep_align.h5")
        M = 40
        block = self._write_test_hdf5(path, M)

        result = align_by_image_timestamp(path, on_stale="interpolate")

        assert "state_hifreq_joints" in result
        assert result["state_hifreq_joints"].shape == (M, 7)
        np.testing.assert_array_equal(result["state_hifreq_joints"], block["joints"])

    def test_align_preserves_hifreq_timestamp(self, tmp_path):
        """align 返回的 state_hifreq_timestamp 与原始完全一致（不重采）。"""
        path = str(tmp_path / "ep_align_ts.h5")
        M = 40
        block = self._write_test_hdf5(path, M)

        result = align_by_image_timestamp(path, on_stale="interpolate")

        np.testing.assert_array_equal(result["state_hifreq_timestamp"], block["timestamp"])

    def test_align_m0_hifreq_returns_empty_arrays(self, tmp_path):
        """M=0 的 state_hifreq，align 返回空数组（shape=(0,7) 等）。"""
        path = str(tmp_path / "ep_align_m0.h5")
        N = 5
        frames = [_make_frame(i) for i in range(N)]
        write_episode(
            path, frames,
            task_name="m0_align",
            target_fps=30.0,
            oc2base_R=np.eye(3),
            quality={},
            vr_source="test",
            cam_names=["wrist_image"],
            state_hifreq_block=None,
        )

        result = align_by_image_timestamp(path, on_stale="interpolate")

        assert result["state_hifreq_joints"].shape[0] == 0
        assert result["state_hifreq_timestamp"].shape[0] == 0


# ===========================================================================
# Codex 审查补充：Imp2/Imp3/Imp4 覆盖（新增集成+单元测试）
# ===========================================================================

class TestHistoryCollectorCounts:
    """Imp2：overrun_count / error_count 计数器单元测试。"""

    def test_error_count_increments_on_exception(self):
        """read_fn 持续抛异常时，error_count 累加。"""
        stop_event = threading.Event()

        def always_fail():
            raise RuntimeError("模拟 zerorpc 失败")

        collector = HistoryCollectorThread(
            name="err_count_test",
            read_fn=always_fail,
            target_rate=200.0,
            stop_event=stop_event,
        )
        time.sleep(0.03)  # 让它跑约 6 次节拍
        collector.stop(timeout=1.0)

        assert collector.error_count > 0, "error_count 应 > 0（read_fn 持续抛异常）"
        assert collector.last_error is not None, "last_error 应非 None"
        assert isinstance(collector.last_error, RuntimeError)

    def test_error_count_zero_when_no_error(self):
        """read_fn 正常时，error_count 应为 0。"""
        stop_event = threading.Event()

        collector = HistoryCollectorThread(
            name="no_err_test",
            read_fn=lambda: {"joints": np.zeros(7), "joint_vel": np.zeros(7), "ee_pose": np.zeros(6)},
            target_rate=100.0,
            stop_event=stop_event,
        )
        time.sleep(0.02)
        collector.stop(timeout=1.0)

        assert collector.error_count == 0, f"正常采集 error_count 应为 0，实际 {collector.error_count}"

    def test_overrun_count_property_accessible(self):
        """overrun_count property 可访问，类型为 int。"""
        stop_event = threading.Event()
        collector = HistoryCollectorThread(
            name="overrun_test",
            read_fn=lambda: {"joints": np.zeros(7), "joint_vel": np.zeros(7), "ee_pose": np.zeros(6)},
            target_rate=100.0,
            stop_event=stop_event,
        )
        time.sleep(0.01)
        collector.stop(timeout=1.0)

        assert isinstance(collector.overrun_count, int)
        assert collector.overrun_count >= 0


class TestRecordEpisodeEdgeCases:
    """Imp3：record_episode 空历史/异常路径的集成测试。"""

    def test_hifreq_rate_zero_returns_none_block(self):
        """hifreq_rate=0：不启动采集线程，block=None（已有覆盖，确认行为不变）。"""
        m = _load_rrh()
        robot = FakeRobot()
        teleop = FakeTeleop()
        buf, block = m.record_episode(
            robot, teleop, fps=30.0, max_sec=0.05,
            gripper_max_open=0.08, cam_names=["wrist_image"],
            hifreq_rate=0.0,
        )
        assert block is None, f"hifreq_rate=0 时 block 必须为 None，实际 {block}"
        assert isinstance(buf, list)

    def test_always_failing_read_fn_yields_none_block(self):
        """read_fn 持续抛异常 → history 空 → block=None，last_error 非 None，error_count>0。

        验证：np.stack([]) 不会被调用（无崩溃）。
        """
        m = _load_rrh()

        class FailingRobot:
            """_robot 属性存在（走真机路径），但所有 zerorpc 调用均失败。"""
            def __init__(self):
                self.cameras = {"wrist_image": FakeCam()}
                self._robot = self  # 使 zerorpc_client is not None
                self.config = type("C", (), {"gripper_max_open": 0.08})()

            # 模拟 zerorpc 方法——全部抛异常
            def robot_get_joint_positions(self):
                raise RuntimeError("zerorpc 连接断开（模拟）")

            def robot_get_joint_velocities(self):
                raise RuntimeError("zerorpc 连接断开（模拟）")

            def robot_get_ee_pose(self):
                raise RuntimeError("zerorpc 连接断开（模拟）")

            def get_observation(self):
                # 回退路径也失败（触发真机路径因有 _robot）
                raise RuntimeError("不应进入此路径")

            def send_action(self, a):
                pass

        robot = FailingRobot()
        teleop = FakeTeleop()

        # 录制很短时间，read_fn 全程失败 → history 为空
        buf, block = m.record_episode(
            robot, teleop, fps=30.0, max_sec=0.08,
            gripper_max_open=0.08, cam_names=["wrist_image"],
            hifreq_rate=240.0,
        )

        # 核心断言：空历史时 block=None，不崩溃
        assert block is None, (
            f"read_fn 持续失败时 history 空，block 应为 None，实际 {block}"
        )
        assert isinstance(buf, list)

    def test_normal_hifreq_block_m_gt_0(self):
        """正常采集（M>0）：block 结构正确，np.stack 组装无误。"""
        m = _load_rrh()
        robot = FakeRobot()
        teleop = FakeTeleop()
        buf, block = m.record_episode(
            robot, teleop, fps=30.0, max_sec=0.15,
            gripper_max_open=0.08, cam_names=["wrist_image"],
            hifreq_rate=100.0,
        )
        assert block is not None, "正常采集时 block 不应为 None"
        M = len(block["timestamp"])
        assert M > 0
        assert block["joints"].shape == (M, 7)
        assert block["joint_vel"].shape == (M, 7)
        assert block["pose"].shape == (M, 6)

    def test_empty_history_no_np_stack_crash(self):
        """空历史不触发 np.stack([])（用 HistoryCollectorThread 直接验证）。

        直接构造一个 stop 后 history 为空的 collector，验证 get_history() 返回 []，
        手动复现 record_episode 的组装逻辑——应走 if history: else 分支，不 np.stack。
        """
        stop_event = threading.Event()
        stop_event.set()  # 立刻置位，线程启动后立即退出，不采样

        # read_fn 每次都抛异常，确保 history 为空
        collector = HistoryCollectorThread(
            name="empty_hist_test",
            read_fn=lambda: (_ for _ in ()).throw(RuntimeError("立刻失败")),
            target_rate=1.0,  # 低频，stop_event 置位后不会进入第一拍
            stop_event=stop_event,
        )
        collector.stop(timeout=1.0)

        history = collector.get_history()
        assert len(history) == 0, f"预期 history 为空，实际 {len(history)}"

        # 复现 record_episode 组装逻辑：有 history 才 np.stack
        if history:
            # 不应进入此分支
            np.stack([s.data["joints"] for s in history])
            raise AssertionError("空 history 不应进入 np.stack 分支")
        # else 分支：block = None，无崩溃，测试通过
