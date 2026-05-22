import copy, importlib.util, os, sys, time
import numpy as np

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P = "/home/ubuntu/Desktop/jhli/lerobot_franka_teleop"
_s = importlib.util.spec_from_file_location(
    "rrh", os.path.join(_P, "scripts/core/run_record_hdf5.py"))
# 注: 模块级 import 真包较重; 用 spec 但只测纯编排函数, 见下


class FakeRobot:
    def __init__(self): self.cameras = {"wrist_image": object()}
    def get_observation(self):
        o = {f"joint_{i+1}.pos": 0.0 for i in range(7)}
        o.update({f"joint_{i+1}.vel": 0.0 for i in range(7)})
        o.update({f"ee_pose.{a}": 0.0 for a in "x y z rx ry rz".split()})
        o["gripper_state_norm"] = 0.5
        o["wrist_image"] = np.zeros((4, 4, 3), np.uint8)
        return o
    def send_action(self, a): pass


class FakeTeleop:
    def get_action(self):
        a = {f"delta_ee_pose.{x}": 0.0 for x in "x y z rx ry rz".split()}
        a["gripper_cmd_bin"] = 0.0
        return a


class FakeSaver:
    def __init__(self): self.submitted = []
    def submit(self, path, payload): self.submitted.append((path, payload))


def _load():
    sys.path.insert(0, os.path.join(_P, "scripts"))
    m = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(m)
    return m


def test_run_episodes_submits_deepcopy_and_resets_buffer(tmp_path):
    m = _load()
    saver = FakeSaver()
    # 极短 ep (0.05s) x 2 条, save=keep
    m.run_episodes(FakeRobot(), FakeTeleop(), saver,
                   fps=50.0, episode_sec=0.05, gripper_max_open=0.08,
                   cam_names=["wrist_image"], out_dir=str(tmp_path),
                   task_name="t", oc2base_R=np.eye(3), vr_source="u",
                   episodes=2, decide=lambda i: "keep")
    assert len(saver.submitted) == 2
    # 提交的是 deepcopy 快照, 互不别名
    f0 = saver.submitted[0][1]["frames"]
    f1 = saver.submitted[1][1]["frames"]
    assert f0 is not f1 and len(f0) > 0


def test_discard_does_not_submit(tmp_path):
    m = _load()
    saver = FakeSaver()
    m.run_episodes(FakeRobot(), FakeTeleop(), saver,
                   fps=50.0, episode_sec=0.05, gripper_max_open=0.08,
                   cam_names=["wrist_image"], out_dir=str(tmp_path),
                   task_name="t", oc2base_R=np.eye(3), vr_source="u",
                   episodes=1, decide=lambda i: "discard")
    assert saver.submitted == []          # 丢弃 = 不入队不产文件


def test_loop_does_not_block_on_save(tmp_path):
    m = _load()

    class SlowSaver:
        def __init__(self): self.n = 0
        def submit(self, p, d): time.sleep(0.2); self.n += 1  # 模拟若误同步

    # run_episodes 自身不得在 submit 上做写盘(submit 应是入队即返回);
    # 这里 SlowSaver.submit 慢 -> 用真 AsyncEpisodeSaver 时 submit 是入队不慢,
    # 本测试断言 run_episodes 不在循环内 close/join (解耦正确)
    saver = SlowSaver()
    t0 = time.monotonic()
    m.run_episodes(FakeRobot(), FakeTeleop(), saver,
                   fps=50.0, episode_sec=0.02, gripper_max_open=0.08,
                   cam_names=["wrist_image"], out_dir=str(tmp_path),
                   task_name="t", oc2base_R=np.eye(3), vr_source="u",
                   episodes=2, decide=lambda i: "keep")
    # 2 条 ep 各 0.02s 采集 + 2*0.2s submit(本 fake 慢) = 串行约 0.44s;
    # 真实现用 AsyncEpisodeSaver submit O(1); 这里只断言 run_episodes
    # 不额外 join/阻塞 (无 close 调用), 上界宽松
    assert time.monotonic() - t0 < 1.0


# ============================================================
# 以下 4 个为 Codex review-fix 新增（Imp#1-4 + Minor 全采纳）
# ============================================================

def test_payload_is_deep_isolated_from_buffer_mutation(tmp_path):
    """Imp#3: deepcopy 真正深隔离嵌套 ndarray/dict，非仅顶层。

    record_episode 返回含嵌套结构的 buffer；submit 后就地修改原始对象的
    嵌套 ndarray/dict，断言 FakeSaver 捕获的 payload 对应值未变。
    """
    m = _load()

    # 保存原始 record_episode，供后续恢复
    orig_record = m.record_episode

    # 记录 record_episode 最后一次返回的原始 buf 引用
    last_buf_ref = []

    def fake_record_episode(robot, teleop, fps, max_sec, gripper_max_open, cam_names, *, stop_flag=None, frame_observer=None):
        buf = [
            {
                "ts": 1.0,
                "joints": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
                "joint_vel": np.zeros(7, dtype=np.float64),
                "ee_pose": np.zeros(6, dtype=np.float64),
                "gripper_m": 0.0,
                "gripper_norm": 0.5,
                "gripper_cmd": 0.0,
                "delta_ee_pose": np.zeros(6, dtype=np.float64),
                "cams": {"wrist_image": np.array([10, 20, 30], dtype=np.uint8)},
            }
        ]
        last_buf_ref.append(buf)
        return buf

    # monkeypatch：替换模块命名空间里的 record_episode
    m.record_episode = fake_record_episode

    saver = FakeSaver()
    try:
        m.run_episodes(FakeRobot(), FakeTeleop(), saver,
                       fps=50.0, episode_sec=0.05, gripper_max_open=0.08,
                       cam_names=["wrist_image"], out_dir=str(tmp_path),
                       task_name="t", oc2base_R=np.eye(3), vr_source="u",
                       episodes=1, decide=lambda i: "keep")
    finally:
        m.record_episode = orig_record

    assert len(saver.submitted) == 1
    submitted_payload = saver.submitted[0][1]

    # 就地修改原始 buf（submit 之后）——模拟后续 ep 复用场景
    orig_buf = last_buf_ref[0]
    orig_buf[0]["cams"]["wrist_image"][0] = 99          # 嵌套 ndarray 改值
    orig_buf[0]["joints"][0] = 99.0                     # 顶层 ndarray 改值

    # 断言 payload 深拷贝后不受影响
    frames = submitted_payload["frames"]
    assert frames[0]["cams"]["wrist_image"][0] == 10, (
        f"嵌套 ndarray 被别名：期望 10，实际 {frames[0]['cams']['wrist_image'][0]}；"
        "deepcopy 未深隔离嵌套 dict/ndarray"
    )
    assert frames[0]["joints"][0] == 1.0, (
        f"顶层 ndarray 被别名：期望 1.0，实际 {frames[0]['joints'][0]}"
    )


def test_run_episodes_real_saver_drains_all_on_exit(tmp_path):
    """Imp#1: 端到端零丢失——真 AsyncEpisodeSaver with 退出 join 排空所有已提交 ep。

    录 3 条全 keep；with AsyncEpisodeSaver 退出后断言 sink 收到 3 条且每条 frames>0。
    """
    sys.path.insert(0, os.path.join(_P, "scripts"))
    from core.async_saver import AsyncEpisodeSaver

    m = _load()
    saved = []

    def sink(path, payload):
        saved.append((path, len(payload["frames"])))

    with AsyncEpisodeSaver(sink=sink, maxsize=5) as saver:
        m.run_episodes(FakeRobot(), FakeTeleop(), saver,
                       fps=50.0, episode_sec=0.05, gripper_max_open=0.08,
                       cam_names=["wrist_image"], out_dir=str(tmp_path),
                       task_name="t", oc2base_R=np.eye(3), vr_source="u",
                       episodes=3, decide=lambda i: "keep")
    # with 退出后 join 已排空
    assert len(saved) == 3, f"期望 3 条，实际 {len(saved)} 条（零丢失保证失效）"
    for path, n_frames in saved:
        assert n_frames > 0, f"{path} 帧数为 0"


def test_reset_fn_call_timing(tmp_path):
    """Minor reset 语义: keep/discard 后（非末条/非 stop）均 reset；stop 后不 reset。

    用例 A：decide=keep/discard/keep（3 条），reset 应调用 2 次（ep0 keep 后、ep1 discard 后；
            ep2 末条不 reset）。
    用例 B：decide=keep/stop（ep0 keep, ep1 stop），reset 应调用 1 次（仅 ep0 后）。
    """
    m = _load()

    # ---- 用例 A ----
    reset_count_a = [0]
    decides_a = ["keep", "discard", "keep"]
    saver_a = FakeSaver()
    m.run_episodes(FakeRobot(), FakeTeleop(), saver_a,
                   fps=50.0, episode_sec=0.02, gripper_max_open=0.08,
                   cam_names=["wrist_image"], out_dir=str(tmp_path),
                   task_name="t", oc2base_R=np.eye(3), vr_source="u",
                   episodes=3, decide=lambda i: decides_a[i],
                   reset_fn=lambda: reset_count_a.__setitem__(0, reset_count_a[0] + 1))
    assert reset_count_a[0] == 2, (
        f"用例A: 期望 reset 2 次（ep0 keep 后 + ep1 discard 后），实际 {reset_count_a[0]}"
    )

    # ---- 用例 B ----
    reset_count_b = [0]
    saver_b = FakeSaver()
    m.run_episodes(FakeRobot(), FakeTeleop(), saver_b,
                   fps=50.0, episode_sec=0.02, gripper_max_open=0.08,
                   cam_names=["wrist_image"], out_dir=str(tmp_path),
                   task_name="t", oc2base_R=np.eye(3), vr_source="u",
                   episodes=3, decide=lambda i: "keep" if i == 0 else "stop",
                   reset_fn=lambda: reset_count_b.__setitem__(0, reset_count_b[0] + 1))
    assert reset_count_b[0] == 1, (
        f"用例B: 期望 reset 1 次（仅 ep0 keep 后），stop 后不应 reset，实际 {reset_count_b[0]}"
    )


def test_meta_is_deep_isolated_from_external_mutation(tmp_path):
    """Minor: meta 字段（oc2base_R/cam_names）也随整体 deepcopy 隔离，不受外部修改影响。"""
    m = _load()
    saver = FakeSaver()

    R = np.eye(3)
    cam_names = ["wrist_image"]

    m.run_episodes(FakeRobot(), FakeTeleop(), saver,
                   fps=50.0, episode_sec=0.05, gripper_max_open=0.08,
                   cam_names=cam_names, out_dir=str(tmp_path),
                   task_name="t", oc2base_R=R, vr_source="u",
                   episodes=1, decide=lambda i: "keep")

    assert len(saver.submitted) == 1
    submitted_meta = saver.submitted[0][1]["meta"]

    # 外部改写原始引用
    R[0, 0] = 999.0
    cam_names.append("extra_cam")

    # payload 内 meta 不受影响
    assert submitted_meta["oc2base_R"][0, 0] == 1.0, (
        "oc2base_R 被别名：外部修改后 payload meta 内值变了，整体 deepcopy 未覆盖 meta"
    )
    assert "extra_cam" not in submitted_meta["cam_names"], (
        "cam_names 被别名：外部 append 后 payload meta 内 cam_names 也变了"
    )
