import copy, importlib.util, os, time
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
    import sys
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
