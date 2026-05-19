import importlib.util, os, copy
import numpy as np

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 指向远端仓库（本文件 scp 到远端后 __file__ 变为远端路径，_P 自动正确）
_s = importlib.util.spec_from_file_location(
    "rrh", os.path.join(_P, "scripts/core/run_record_hdf5.py"))


def _load():
    import sys; sys.path.insert(0, os.path.join(_P, "scripts"))
    m = importlib.util.module_from_spec(_s); _s.loader.exec_module(m); return m


class FakeRobot:
    def __init__(self): self.cameras = {"wrist_image": object()}; self.resets = 0
    def get_observation(self):
        o = {f"joint_{i+1}.pos": 0.0 for i in range(7)}
        o.update({f"joint_{i+1}.vel": 0.0 for i in range(7)})
        o.update({f"ee_pose.{a}": 0.0 for a in "x y z rx ry rz".split()})
        o["gripper_state_norm"] = 0.5
        o["wrist_image"] = np.zeros((4, 4, 3), np.uint8)
        return o
    def send_action(self, a): pass
    def reset(self): self.resets += 1


class FakeTeleop:
    def get_action(self):
        a = {f"delta_ee_pose.{x}": 0.0 for x in "x y z rx ry rz".split()}
        a["gripper_cmd_bin"] = 0.0; return a


class FakeSaver:
    def __init__(self): self.submitted = []
    def submit(self, p, d): self.submitted.append((p, d))


def test_reset_called_between_episodes_not_after_last(tmp_path):
    m = _load(); r = FakeRobot()
    m.run_episodes(r, FakeTeleop(), FakeSaver(),
                   fps=50.0, episode_sec=0.02, gripper_max_open=0.08,
                   cam_names=["wrist_image"], out_dir=str(tmp_path),
                   task_name="t", oc2base_R=np.eye(3), vr_source="u",
                   episodes=3, decide=lambda i: "keep",
                   reset_fn=r.reset, reset_wait=0.0)
    assert r.resets == 2          # 3 条 ep -> 条间 reset 2 次, 最后一条后不 reset


def test_reset_skipped_when_disabled(tmp_path):
    m = _load(); r = FakeRobot()
    m.run_episodes(r, FakeTeleop(), FakeSaver(),
                   fps=50.0, episode_sec=0.02, gripper_max_open=0.08,
                   cam_names=["wrist_image"], out_dir=str(tmp_path),
                   task_name="t", oc2base_R=np.eye(3), vr_source="u",
                   episodes=2, decide=lambda i: "keep",
                   reset_fn=None, reset_wait=0.0)   # 关闭 = 不传 reset_fn
    assert r.resets == 0


def test_reset_not_called_after_stop(tmp_path):
    m = _load(); r = FakeRobot()
    seq = iter(["keep", "stop"])
    m.run_episodes(r, FakeTeleop(), FakeSaver(),
                   fps=50.0, episode_sec=0.02, gripper_max_open=0.08,
                   cam_names=["wrist_image"], out_dir=str(tmp_path),
                   task_name="t", oc2base_R=np.eye(3), vr_source="u",
                   episodes=5, decide=lambda i: next(seq),
                   reset_fn=r.reset, reset_wait=0.0)
    assert r.resets == 1          # ep0 keep 后 reset 1 次, ep1 stop 后不 reset
