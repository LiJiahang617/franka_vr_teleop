"""test_realsense_hw_wrapper.py — 离线 mock 测试 RealsenseHwWrapper 接口。

验证内容：
1. 接口契约：read() 返回 (ndarray, float) 元组，rgb shape/dtype 正确
2. connect() 幂等（重复调用不报错）
3. disconnect() 幂等（未连接时不报错，多次调用不报错）
4. 未连接时 read() 抛 RuntimeError
5. _make_camera_read_fn 对 wrapper 返回 (rgb, hw_ts) 元组
6. _make_camera_read_fn 对普通 cam 返回 (rgb, None) 元组
7. record_episode 帧中 cam_hw_ts 写入真 hw_ts（当相机是 wrapper 时）
8. record_episode 帧中 cam_hw_ts fallback 软件戳（当相机是普通 cam 时）
"""
import importlib.util
import os
import sys
import threading
import time
import types

import numpy as np
import pytest

_REPO = "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop"
sys.path.insert(0, os.path.join(_REPO, "scripts"))


# ---------------------------------------------------------------------------
# 辅助：加载被测模块（不触发硬件 import）
# ---------------------------------------------------------------------------

def _load_wrapper_mod():
    """加载 realsense_hw_wrapper 模块（核心类）。"""
    spec = importlib.util.spec_from_file_location(
        "realsense_hw_wrapper",
        os.path.join(_REPO, "scripts/core/realsense_hw_wrapper.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_rrh_mod():
    """加载 run_record_hdf5 模块（包含 _make_camera_read_fn + record_episode）。"""
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    spec = importlib.util.spec_from_file_location(
        "rrh",
        os.path.join(_REPO, "scripts/core/run_record_hdf5.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Mock：用 monkeypatch 替换 pyrealsense2（离线，不触碰真实硬件）
# ---------------------------------------------------------------------------

class _FakeColorFrame:
    """模拟 pyrealsense2 color_frame。"""
    def __init__(self, hw_ts_ms: float, shape=(480, 640, 3)):
        self._hw_ts = hw_ts_ms
        self._shape = shape

    def get_timestamp(self) -> float:
        return self._hw_ts

    def get_data(self) -> np.ndarray:
        return np.zeros(self._shape, dtype=np.uint8)

    def __bool__(self):
        return True


class _FakeFrames:
    def __init__(self, hw_ts_ms: float):
        self._hw_ts = hw_ts_ms

    def get_color_frame(self):
        return _FakeColorFrame(self._hw_ts)


class _FakeColorSensor:
    def __init__(self):
        self._options = {}

    def set_option(self, opt, val):
        self._options[opt] = val

    def get_option(self, opt):
        return self._options.get(opt, 0.0)


class _FakeDevice:
    def first_color_sensor(self):
        return _FakeColorSensor()


class _FakeProfile:
    def get_device(self):
        return _FakeDevice()


class _FakePipeline:
    """模拟 rs.pipeline，帧 hw_ts 从外部注入。"""
    def __init__(self, hw_ts_sequence=None, fail_start=False, fail_read=False):
        self._hw_ts_seq = hw_ts_sequence or [1000.0 + i * (1000.0 / 30) for i in range(300)]
        self._idx = 0
        self._fail_start = fail_start
        self._fail_read = fail_read
        self._started = False

    def start(self, config):
        if self._fail_start:
            raise RuntimeError("模拟 start 失败")
        self._started = True
        return _FakeProfile()

    def try_wait_for_frames(self, timeout_ms=200):
        if self._fail_read:
            return False, None
        if self._idx >= len(self._hw_ts_seq):
            # 重复最后一帧（循环）
            hw_ts = self._hw_ts_seq[-1]
        else:
            hw_ts = self._hw_ts_seq[self._idx]
            self._idx += 1
        return True, _FakeFrames(hw_ts)

    def wait_for_frames(self, timeout_ms=1000):
        _, frames = self.try_wait_for_frames(timeout_ms)
        return frames

    def stop(self):
        self._started = False


def _make_fake_rs_module(pipeline_factory=None, fail_start=False, fail_read=False):
    """构造 fake pyrealsense2 模块，供 monkeypatch 注入。"""
    rs = types.ModuleType("pyrealsense2")

    class FakeStream:
        color = "color"

    class FakeFormat:
        rgb8 = "rgb8"

    class FakeOption:
        global_time_enabled = "global_time_enabled"

    class FakeConfig:
        def __init__(self):
            pass
        def enable_stream(self, *args, **kwargs):
            pass

    @staticmethod
    def enable_device(config, serial):
        pass

    FakeConfig.enable_device = enable_device

    class FakeTimestampDomain:
        global_time = "global_time"

    rs.stream = FakeStream()
    rs.format = FakeFormat()
    rs.option = FakeOption()
    rs.config = FakeConfig
    rs.timestamp_domain = FakeTimestampDomain()

    def _make_pipeline(*args, **kwargs):
        if pipeline_factory is not None:
            return pipeline_factory()
        return _FakePipeline(fail_start=fail_start, fail_read=fail_read)

    rs.pipeline = _make_pipeline
    return rs


# ---------------------------------------------------------------------------
# 测试 1：基本接口 connect/read/disconnect
# ---------------------------------------------------------------------------

def test_wrapper_connect_read_returns_rgb_and_hw_ts(monkeypatch):
    """connect/read() 返回 (rgb ndarray, hw_ts float) 且形状正确。"""
    hw_ts_seq = [1000.0 + i * 33.3 for i in range(50)]
    fake_rs = _make_fake_rs_module(
        pipeline_factory=lambda: _FakePipeline(hw_ts_sequence=hw_ts_seq)
    )

    wrapper_mod = _load_wrapper_mod()
    # 在 wrapper 模块的 connect() 内动态 import pyrealsense2，patch sys.modules
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)

    wrapper = wrapper_mod.RealsenseHwWrapper(
        serial="419622073931", width=640, height=480, fps=30
    )
    assert not wrapper.is_connected

    wrapper.connect()
    assert wrapper.is_connected

    rgb, hw_ts = wrapper.read()
    assert isinstance(rgb, np.ndarray), f"rgb 应为 ndarray，得到 {type(rgb)}"
    assert rgb.dtype == np.uint8, f"rgb dtype 应为 uint8，得到 {rgb.dtype}"
    assert rgb.shape == (480, 640, 3), f"rgb shape 错误: {rgb.shape}"
    assert isinstance(hw_ts, float), f"hw_ts 应为 float，得到 {type(hw_ts)}"
    # 预热 10 帧消耗了序列前 10 个，hw_ts 应在 hw_ts_seq 中
    assert hw_ts in hw_ts_seq, f"hw_ts={hw_ts} 不在期望的 hw_ts_seq 中"
    assert hw_ts >= hw_ts_seq[0], f"hw_ts={hw_ts} 应 >= 序列起点 {hw_ts_seq[0]}"

    wrapper.disconnect()
    assert not wrapper.is_connected


def test_wrapper_connect_idempotent(monkeypatch):
    """connect() 幂等：重复调用不报错，不重建 pipeline。"""
    calls = []

    def factory():
        calls.append(1)
        return _FakePipeline()

    fake_rs = _make_fake_rs_module(pipeline_factory=factory)
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)

    wrapper_mod = _load_wrapper_mod()
    wrapper = wrapper_mod.RealsenseHwWrapper("SN1", 640, 480, 30)
    wrapper.connect()
    wrapper.connect()  # 第二次调用应幂等

    assert len(calls) == 1, f"pipeline 应只创建一次，实际 {len(calls)} 次"
    wrapper.disconnect()


def test_wrapper_disconnect_idempotent(monkeypatch):
    """disconnect() 幂等：未连接时调用不报错，多次调用不报错。"""
    fake_rs = _make_fake_rs_module()
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)

    wrapper_mod = _load_wrapper_mod()
    wrapper = wrapper_mod.RealsenseHwWrapper("SN1", 640, 480, 30)

    # 未 connect 时 disconnect 不报错
    wrapper.disconnect()
    wrapper.disconnect()

    # connect 后 disconnect 两次
    wrapper.connect()
    wrapper.disconnect()
    wrapper.disconnect()
    assert not wrapper.is_connected


def test_wrapper_read_before_connect_raises(monkeypatch):
    """未 connect 时 read() 抛 RuntimeError。"""
    fake_rs = _make_fake_rs_module()
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)

    wrapper_mod = _load_wrapper_mod()
    wrapper = wrapper_mod.RealsenseHwWrapper("SN1", 640, 480, 30)

    with pytest.raises(RuntimeError, match="未连接"):
        wrapper.read()


def test_wrapper_read_fail_raises(monkeypatch):
    """try_wait_for_frames 失败时 read() 抛 RuntimeError。"""
    fake_rs = _make_fake_rs_module(fail_read=True)
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)

    wrapper_mod = _load_wrapper_mod()
    wrapper = wrapper_mod.RealsenseHwWrapper("SN1", 640, 480, 30)
    wrapper.connect()

    with pytest.raises(RuntimeError, match="超时|失败"):
        wrapper.read()
    wrapper.disconnect()


# ---------------------------------------------------------------------------
# 测试 2：_make_camera_read_fn 对 wrapper 返回 (rgb, hw_ts) 元组
# ---------------------------------------------------------------------------

def test_make_camera_read_fn_wrapper_returns_tuple(monkeypatch):
    """_make_camera_read_fn 对 RealsenseHwWrapper 实例返回 (rgb, hw_ts) 元组。"""
    hw_ts_seq = [5000.0 + i * 33.0 for i in range(10)]
    fake_rs = _make_fake_rs_module(
        pipeline_factory=lambda: _FakePipeline(hw_ts_sequence=hw_ts_seq)
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)

    wrapper_mod = _load_wrapper_mod()
    rrh_mod = _load_rrh_mod()

    wrapper = wrapper_mod.RealsenseHwWrapper("SN1", 640, 480, 30)
    wrapper.connect()

    read_fn = rrh_mod._make_camera_read_fn(wrapper)
    result = read_fn()

    assert isinstance(result, tuple), f"wrapper 对应 read_fn 应返回 tuple，得到 {type(result)}"
    rgb, hw_ts = result
    assert isinstance(rgb, np.ndarray)
    assert isinstance(hw_ts, float)
    # 预热消耗了部分序列帧，hw_ts 在 hw_ts_seq 中
    assert hw_ts in hw_ts_seq, f"hw_ts={hw_ts} 不在期望序列中"

    wrapper.disconnect()


def test_make_camera_read_fn_plain_cam_returns_tuple_with_none_hw_ts():
    """_make_camera_read_fn 对普通 FakeCam 返回 (rgb, None) 元组。"""
    rrh_mod = _load_rrh_mod()

    class FakeCam:
        def read(self):
            return np.zeros((8, 8, 3), np.uint8)

    cam = FakeCam()
    read_fn = rrh_mod._make_camera_read_fn(cam)
    result = read_fn()

    assert isinstance(result, tuple), f"普通 cam 对应 read_fn 应返回 tuple，得到 {type(result)}"
    rgb, hw_ts = result
    assert isinstance(rgb, np.ndarray)
    assert hw_ts is None, f"普通 cam 的 hw_ts 应为 None，得到 {hw_ts}"


# ---------------------------------------------------------------------------
# 测试 3：record_episode 帧中 cam_hw_ts 写入真 hw_ts（wrapper）
# ---------------------------------------------------------------------------

class _FakeCamWithHwTs:
    """模拟 RealsenseHwWrapper：read() 返回 (rgb, hw_ts_ms) 元组。

    设置 HW_WRAPPER=True 使 _make_camera_read_fn 识别为 hw wrapper。
    """
    HW_WRAPPER = True  # duck-type 标记，对应 RealsenseHwWrapper.HW_WRAPPER

    def __init__(self, hw_ts_start_ms=5000.0, interval_ms=33.33):
        self._next_ts = hw_ts_start_ms
        self._interval = interval_ms
        self._calls = 0

    def read(self):
        hw_ts = self._next_ts
        self._next_ts += self._interval
        self._calls += 1
        return np.zeros((8, 8, 3), np.uint8), hw_ts


class _FakeCamPlain:
    """普通 cam：read() 返回 ndarray（无 hw_ts）。"""
    def read(self):
        return np.zeros((8, 8, 3), np.uint8)


class _FakeRobotWithHwCam:
    """带硬件时间戳相机的假机器人。"""
    def __init__(self):
        self._hw_cam = _FakeCamWithHwTs(hw_ts_start_ms=5000.0)
        self.cameras = {"wrist_image": self._hw_cam}

    def get_observation(self):
        o = {f"joint_{i+1}.pos": 0.0 for i in range(7)}
        o.update({f"joint_{i+1}.vel": 0.0 for i in range(7)})
        o.update({f"ee_pose.{a}": 0.0 for a in "x y z rx ry rz".split()})
        o["gripper_state_norm"] = 0.5
        return o

    def send_action(self, a):
        pass


class _FakeRobotWithPlainCam:
    """带普通（无 hw_ts）相机的假机器人。"""
    def __init__(self):
        self._cam = _FakeCamPlain()
        self.cameras = {"wrist_image": self._cam}

    def get_observation(self):
        o = {f"joint_{i+1}.pos": 0.0 for i in range(7)}
        o.update({f"joint_{i+1}.vel": 0.0 for i in range(7)})
        o.update({f"ee_pose.{a}": 0.0 for a in "x y z rx ry rz".split()})
        o["gripper_state_norm"] = 0.5
        return o

    def send_action(self, a):
        pass


class _FakeTeleop:
    def get_action(self):
        a = {f"delta_ee_pose.{x}": 0.0 for x in "x y z rx ry rz".split()}
        a["gripper_cmd_bin"] = 0.0
        return a


def test_record_episode_hw_cam_writes_true_hw_ts():
    """record_episode 使用 wrapper 相机时，cam_hw_ts 写入真硬件时间戳（非软件戳）。"""
    rrh_mod = _load_rrh_mod()
    # 手动 patch：让 _FakeCamWithHwTs 被 isinstance 识别为 RealsenseHwWrapper
    # 实际做法：在 run_record_hdf5 里检测 cam 是否有 read() 且返回 tuple
    # 这里测试端到端行为：cam_hw_ts != cam_ts（软件戳）

    robot = _FakeRobotWithHwCam()
    teleop = _FakeTeleop()

    buf, _ = rrh_mod.record_episode(
        robot, teleop, fps=30.0, max_sec=0.15,
        gripper_max_open=0.08, cam_names=["wrist_image"],
    )

    assert len(buf) > 0, "record_episode 应返回非空帧列表"

    # 每帧的 cam_hw_ts 应包含 wrist_image 的 hw_ts
    for i, frame in enumerate(buf):
        assert "cam_hw_ts" in frame, f"帧 {i} 缺少 cam_hw_ts"
        hw_ts = frame["cam_hw_ts"].get("wrist_image")
        assert hw_ts is not None, f"帧 {i} cam_hw_ts['wrist_image'] 为 None"
        # 真 hw_ts 是毫秒（~5000+），软件戳是 monotonic 秒（~几千秒或几十秒）
        # 两者数量级不同（hw_ts > 1000ms，sw_ts < 100s=100000ms for monotonic<100s）
        # 更直接的验证：hw_ts >= 5000（从 5000ms 开始）且与软件戳不同
        assert hw_ts >= 5000.0, f"帧 {i} hw_ts={hw_ts} 不是期望的硬件毫秒戳（>=5000）"
        sw_ts = frame["cam_ts"]["wrist_image"]  # 软件戳（单位：秒）
        # hw_ts 单位毫秒（>=5000），sw_ts 单位秒（通常 <1000s）
        # 若 hw_ts（毫秒）和 sw_ts（秒）混淆，值差异明显
        assert hw_ts != sw_ts, f"帧 {i} cam_hw_ts == cam_ts（应为独立硬件戳）"


def test_record_episode_plain_cam_hw_ts_fallback_sw_ts():
    """record_episode 使用普通 cam（无 hw_ts）时，cam_hw_ts fallback 到软件戳。"""
    rrh_mod = _load_rrh_mod()

    robot = _FakeRobotWithPlainCam()
    teleop = _FakeTeleop()

    buf, _ = rrh_mod.record_episode(
        robot, teleop, fps=30.0, max_sec=0.1,
        gripper_max_open=0.08, cam_names=["wrist_image"],
    )

    assert len(buf) > 0

    for i, frame in enumerate(buf):
        hw_ts = frame["cam_hw_ts"].get("wrist_image")
        sw_ts = frame["cam_ts"]["wrist_image"]
        # 普通 cam 无 hw_ts → fallback 软件戳 → 两者相等
        assert hw_ts == sw_ts, (
            f"帧 {i} 普通 cam 的 cam_hw_ts({hw_ts}) 应等于 cam_ts({sw_ts})"
        )


# ---------------------------------------------------------------------------
# 测试 4：align_offline 优先用 hw_timestamp 锚
# ---------------------------------------------------------------------------

def test_align_offline_uses_hw_timestamp_when_valid(tmp_path):
    """align_by_image_timestamp 在 hw_timestamp 有效时用 hw_ts 作为锚时间轴。"""
    import h5py

    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    from tools.align_offline import align_by_image_timestamp

    # 构造合成 v2 hdf5
    ep_path = str(tmp_path / "ep.h5")
    N = 20
    sw_ts = np.linspace(0.0, 0.633, N)         # 软件戳（秒）
    # hw_ts：与 sw_ts 线性相关，slope≈1.0，单位毫秒（乘以1000）
    hw_ts_ms = sw_ts * 1000.0 + 12345.0        # 完美线性，R²=1.0

    arm_ts = sw_ts + 0.001                     # arm 略有偏移
    eff_ts = sw_ts + 0.002

    with h5py.File(ep_path, "w") as f:
        f.attrs["schema_version"] = "franka-hdf5-v2"
        obs = f.create_group("observations")
        arm = obs.create_group("arm")
        arm.create_dataset("joints", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("joint_vel", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("pose", data=np.zeros((N, 6), np.float64))
        arm.create_dataset("timestamp", data=arm_ts)
        arm.create_dataset("stale", data=np.zeros(N, dtype=bool))

        eff = obs.create_group("effector")
        eff.create_dataset("position", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("position_norm", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("timestamp", data=eff_ts)

        cam = obs.create_group("camera")
        rgb = cam.create_group("rgb")
        cn_grp = rgb.create_group("wrist_image")
        # 存入 vlen bytes images（空）
        _VLEN_BYTES = h5py.special_dtype(vlen=np.dtype("uint8"))
        imgs = cn_grp.create_dataset("images", (N,), dtype=_VLEN_BYTES)
        for i in range(N):
            imgs[i] = np.array([], dtype=np.uint8)
        cn_grp.create_dataset("timestamp", data=sw_ts)      # 软件戳（秒）
        cn_grp.create_dataset("hw_timestamp", data=hw_ts_ms)  # 硬件戳（毫秒）
        cn_grp.create_dataset("stale", data=np.zeros(N, dtype=bool))

        hifreq = obs.create_group("state_hifreq")
        hifreq.create_dataset("joints", data=np.zeros((0, 7), np.float64))
        hifreq.create_dataset("joint_vel", data=np.zeros((0, 7), np.float64))
        hifreq.create_dataset("pose", data=np.zeros((0, 6), np.float64))
        hifreq.create_dataset("timestamp", data=np.zeros((0,), np.float64))

        act = f.create_group("action")
        act.create_dataset("delta_ee_pose", data=np.zeros((N, 6), np.float64))
        act.create_dataset("gripper_cmd", data=np.zeros((N, 1), np.float64))
        act.create_dataset("timestamp", data=sw_ts)

    aligned = align_by_image_timestamp(ep_path, on_stale="interpolate")

    # hw_timestamp 有效 → anchor_ts 应等于 hw_ts_ms（毫秒值）
    assert "anchor_ts" in aligned
    anchor_ts = aligned["anchor_ts"]
    assert len(anchor_ts) == N
    # 验证 anchor_ts 是 hw_ts_ms（毫秒量级）而非 sw_ts（秒量级）
    assert anchor_ts[0] >= 1000.0, (
        f"anchor_ts[0]={anchor_ts[0]} 看起来不是毫秒硬件戳（应 >= 1000ms）"
    )
    np.testing.assert_allclose(anchor_ts, hw_ts_ms, rtol=1e-9,
                               err_msg="anchor_ts 应等于 hw_timestamp（毫秒）")


def test_align_offline_fallback_sw_ts_when_hw_invalid(tmp_path):
    """align_by_image_timestamp 在 hw_timestamp 无效时 fallback 到 sw_ts（秒）。"""
    import h5py

    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    from tools.align_offline import align_by_image_timestamp

    N = 20
    sw_ts = np.linspace(0.0, 0.633, N)
    # hw_ts 无效：随机噪声，R² << 0.9999
    rng = np.random.default_rng(42)
    hw_ts_invalid = rng.uniform(0, 1e6, N)  # 随机，线性相关极差

    arm_ts = sw_ts + 0.001
    eff_ts = sw_ts + 0.002

    ep_path = str(tmp_path / "ep_invalid_hw.h5")
    with h5py.File(ep_path, "w") as f:
        obs = f.create_group("observations")
        arm = obs.create_group("arm")
        arm.create_dataset("joints", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("joint_vel", data=np.zeros((N, 7), np.float64))
        arm.create_dataset("pose", data=np.zeros((N, 6), np.float64))
        arm.create_dataset("timestamp", data=arm_ts)
        arm.create_dataset("stale", data=np.zeros(N, dtype=bool))

        eff = obs.create_group("effector")
        eff.create_dataset("position", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("position_norm", data=np.zeros((N, 1), np.float64))
        eff.create_dataset("timestamp", data=eff_ts)

        cam = obs.create_group("camera")
        rgb = cam.create_group("rgb")
        cn_grp = rgb.create_group("wrist_image")
        _VLEN_BYTES = h5py.special_dtype(vlen=np.dtype("uint8"))
        imgs = cn_grp.create_dataset("images", (N,), dtype=_VLEN_BYTES)
        for i in range(N):
            imgs[i] = np.array([], dtype=np.uint8)
        cn_grp.create_dataset("timestamp", data=sw_ts)
        cn_grp.create_dataset("hw_timestamp", data=hw_ts_invalid)  # 随机无效戳
        cn_grp.create_dataset("stale", data=np.zeros(N, dtype=bool))

        hifreq = obs.create_group("state_hifreq")
        hifreq.create_dataset("joints", data=np.zeros((0, 7), np.float64))
        hifreq.create_dataset("joint_vel", data=np.zeros((0, 7), np.float64))
        hifreq.create_dataset("pose", data=np.zeros((0, 6), np.float64))
        hifreq.create_dataset("timestamp", data=np.zeros((0,), np.float64))

        act = f.create_group("action")
        act.create_dataset("delta_ee_pose", data=np.zeros((N, 6), np.float64))
        act.create_dataset("gripper_cmd", data=np.zeros((N, 1), np.float64))
        act.create_dataset("timestamp", data=sw_ts)

    aligned = align_by_image_timestamp(ep_path, on_stale="interpolate")

    # hw_ts 无效 → fallback sw_ts（秒量级，<1.0）
    anchor_ts = aligned["anchor_ts"]
    assert anchor_ts[0] < 100.0, (
        f"anchor_ts[0]={anchor_ts[0]} 不像是 sw_ts（秒量级，应 <100s）"
    )
    np.testing.assert_allclose(anchor_ts, sw_ts, rtol=1e-9,
                               err_msg="hw_ts 无效时 anchor_ts 应等于 sw_ts（秒）")
