"""Realman 世界系 Unity app 的 adb logcat 读取器。

接口与 OculusReader.get_transformations_and_buttons() 一致，可 drop-in 替换。
"""
import re
import subprocess
import threading

import numpy as np
from scipy.spatial.transform import Rotation as _R

_NUM = r"([-+]?\d*\.?\d+)"
_RE_RIGHT = re.compile(
    r"VRDeviceData:\s+RIGHT_POSE\s+"
    r"t=\(" + _NUM + r",\s*" + _NUM + r",\s*" + _NUM + r"\)\s+"
    r"r=\(" + _NUM + r",\s*" + _NUM + r",\s*" + _NUM + r",\s*" + _NUM + r"\)\s+"
    r"RIGHT_CONTROLLER\s+grip=" + _NUM + r"\s+A=(\d+)\s+B=(\d+)\s+"
    r"Joy_stick_button=(\d+)\s+trigger=" + _NUM + r"\s+joystick="
)


def parse_right_pose(line):
    """解析一行 logcat；非 RIGHT_POSE 行返回 None。"""
    m = _RE_RIGHT.search(line)
    if not m:
        return None
    g = m.groups()
    return {
        "pos": (float(g[0]), float(g[1]), float(g[2])),
        "quat": (float(g[3]), float(g[4]), float(g[5]), float(g[6])),
        "grip": float(g[7]),
        "A": int(g[8]),
        "B": int(g[9]),
        "trigger": float(g[11]),
    }


def _is_sentinel(p):
    return (all(abs(v) < 1e-9 for v in p["pos"])
            and abs(p["quat"][0]) < 1e-9 and abs(p["quat"][1]) < 1e-9
            and abs(p["quat"][2]) < 1e-9 and abs(p["quat"][3] - 1.0) < 1e-9)


def to_transform(p):
    """parsed -> 4x4。Unity 是左手系，用 S=diag(1,1,-1) 相似变换转右手系
    （Kabsch det=+1 只能吸收旋转、吸收不了手性翻转，否则未标定的 Y 轴会反）。
    未追踪 sentinel 返回 None。"""
    if p is None or _is_sentinel(p):
        return None
    S = np.diag([1.0, 1.0, -1.0])
    Rm = _R.from_quat(list(p["quat"])).as_matrix()
    T = np.eye(4)
    T[:3, :3] = S @ Rm @ S
    T[:3, 3] = S @ np.array(p["pos"], dtype=float)
    return T


def to_buttons(p):
    """parsed -> stage3 消费的按键 dict（RG/rightTrig/A/B）。"""
    if p is None:
        return {"RG": False, "rightTrig": (0.0,), "A": 0, "B": 0}
    return {
        "RG": p["grip"] > 0.5,
        "rightTrig": (p["trigger"],),
        "A": p["A"],
        "B": p["B"],
    }


class UnityVRReader:
    """读 Realman 世界系 Unity app 的 adb logcat，接口同 OculusReader。"""

    def __init__(self,
                 adb_path="/home/ubuntu/Desktop/jhli/platform-tools/adb",
                 package="com.UnityTechnologies.com.unity.template.urpblank",
                 logcat_tag="Unity"):
        self.adb = adb_path
        self.package = package
        self.tag = logcat_tag
        self._lock = threading.Lock()
        self._last_T = None
        self._last_btn = {"RG": False, "rightTrig": (0.0,), "A": 0, "B": 0}
        self.running = False
        self._proc = None

        subprocess.run([self.adb, "start-server"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([self.adb, "shell", "monkey", "-p", self.package,
                        "-c", "android.intent.category.LAUNCHER", "1"],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([self.adb, "logcat", "-c"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        self._proc = subprocess.Popen(
            [self.adb, "logcat", "-s", f"{self.tag}:I"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1)
        for line in self._proc.stdout:
            if not self.running:
                break
            p = parse_right_pose(line)
            if p is None:
                continue
            T = to_transform(p)
            btn = to_buttons(p)
            with self._lock:
                if T is not None:
                    self._last_T = T
                self._last_btn = btn

    def get_transformations_and_buttons(self):
        with self._lock:
            tr = {"r": self._last_T.copy()} if self._last_T is not None else {}
            return tr, dict(self._last_btn)

    def stop(self):
        self.running = False
        if self._proc is not None:
            self._proc.terminate()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
