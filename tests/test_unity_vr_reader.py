import numpy as np
import lerobot_teleoperator_franka.unity_vr_reader as u

VALID = ("05-18 15:45:35.369 10887 10920 I Unity   : VRDeviceData: RIGHT_POSE "
         "t=(0.135, -0.050, -0.175) r=(0.0, 0.0, 0.0, 1.0) RIGHT_CONTROLLER "
         "grip=0.900 A=1 B=0 Joy_stick_button=0 trigger=0.250 joystick=(0.0,0.0)")
SENTINEL = ("I Unity : VRDeviceData: RIGHT_POSE t=(0.000, 0.000, 0.000) "
            "r=(0.000, 0.000, 0.000, 1.000) RIGHT_CONTROLLER grip=0.000 A=0 B=0 "
            "Joy_stick_button=0 trigger=0.000 joystick=(0.0,0.0)")
HEAD = ("I Unity : VRDeviceData: HEAD_POSE t=(0,0.8,0) r=(0,0,0,1) "
        "HEADSET_MOUNTED=1 IPD=0.064 EYE_HEIGHT=0.8")


def test_parse_valid_right():
    p = u.parse_right_pose(VALID)
    assert p is not None
    assert np.allclose(p["pos"], [0.135, -0.050, -0.175])
    assert np.allclose(p["quat"], [0.0, 0.0, 0.0, 1.0])
    assert abs(p["grip"] - 0.9) < 1e-9
    assert p["A"] == 1 and p["B"] == 0
    assert abs(p["trigger"] - 0.25) < 1e-9


def test_parse_ignores_head_and_garbage():
    assert u.parse_right_pose(HEAD) is None
    assert u.parse_right_pose("random noise") is None


def test_to_transform_valid_is_proper_rotation():
    T = u.to_transform(u.parse_right_pose(VALID))
    assert T.shape == (4, 4)
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-6
    assert np.allclose(T[:3, 3], [0.135, -0.050, 0.175])  # Unity z 取负转右手系


def test_sentinel_is_invalid():
    assert u.to_transform(u.parse_right_pose(SENTINEL)) is None


def test_buttons_mapping():
    b = u.to_buttons(u.parse_right_pose(VALID))
    assert b["RG"] is True and b["A"] == 1 and b["B"] == 0
    assert isinstance(b["rightTrig"], tuple) and abs(b["rightTrig"][0] - 0.25) < 1e-9
    b2 = u.to_buttons(u.parse_right_pose(SENTINEL))
    assert b2["RG"] is False


def test_to_transform_flips_handedness_z_negated():
    line = ("I Unity : VRDeviceData: RIGHT_POSE t=(1.0, 2.0, 3.0) "
            "r=(0.0, 0.0, 0.0, 1.0) RIGHT_CONTROLLER grip=0.0 A=0 B=0 "
            "Joy_stick_button=0 trigger=0.0 joystick=(0.0,0.0)")
    T = u.to_transform(u.parse_right_pose(line))
    assert np.allclose(T[:3, 3], [1.0, 2.0, -3.0])          # z 取负
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-6       # 仍是真旋转
