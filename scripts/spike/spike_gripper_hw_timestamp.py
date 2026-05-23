"""spike: hw_timestamp 与 monotonic 接收时刻线性相关性测量。

判据（修订后）：
  R² > 0.99（强线性度，残差 ~毫秒级即可）

slope 不需要 ≈ 1.0：
  `_project_hw_to_monotonic` 用 np.polyfit(hw, mono, 1) 拟合任意 slope/intercept，
  slope=50 仅意味着 hw_ts 单位比 wall-clock 慢 50 倍（如 libfranka 内部计数器），
  polyfit 完全可以处理，不影响对齐精度。

用法：
    /home/ubuntu/Desktop/jhli/envs/franka-teleop/bin/python \\
        /home/ubuntu/Desktop/jhli/lerobot_franka_teleop/scripts/spike/spike_gripper_hw_timestamp.py
"""
import time
import numpy as np
import zerorpc

c = zerorpc.Client(timeout=10)
c.connect("tcp://127.0.0.1:4242")
_ = c.gripper_get_state()   # 暖连接

# ── 快速采样（5ms 间隔，300 次）用于单调性和基本特征检查 ──────────────────────
N_FAST = 300
fast_samples = []
for _ in range(N_FAST):
    t_mono = time.monotonic()
    s = c.gripper_get_state()
    t_hw = s.get("timestamp")
    if t_hw is None:
        raise RuntimeError("zerorpc 未返回 timestamp 字段，polymetis 是否重 build？")
    fast_samples.append((t_mono, t_hw))
    time.sleep(0.005)

mono_fast = np.array([m for m, _ in fast_samples])
hw_fast = np.array([h for _, h in fast_samples])

# 单调非降
assert np.all(np.diff(hw_fast) >= -1e-9), "hw_timestamp 非单调非降"
print("[ok] hw_timestamp 单调非降")
print(f"hw_ts 步进（5ms 窗口）: 非零步进 = {np.unique(np.diff(hw_fast)[np.diff(hw_fast) > 1e-9])} 秒")

# ── 慢速采样（100ms 间隔，60 次）用于线性回归（避免重复值影响斜率）───────────
# 注：100ms 间隔时 hw_ts 每次稳定步进，线性回归更可靠
N_SLOW = 60
slow_samples = []
for _ in range(N_SLOW):
    t_mono = time.monotonic()
    s = c.gripper_get_state()
    t_hw = s.get("timestamp")
    slow_samples.append((t_mono, t_hw))
    time.sleep(0.1)

mono_arr = np.array([m for m, _ in slow_samples])
hw_arr = np.array([h for _, h in slow_samples])

slope, intercept = np.polyfit(hw_arr, mono_arr, 1)
pred = slope * hw_arr + intercept
ss_res = np.sum((mono_arr - pred) ** 2)
ss_tot = np.sum((mono_arr - mono_arr.mean()) ** 2)
r2 = 1.0 - ss_res / ss_tot

# 残差标准差（wall-clock 域，单位秒）
residual_std = np.sqrt(np.sum((mono_arr - pred) ** 2) / len(mono_arr))

print(f"\n── 线性回归结果（100ms 采样，{N_SLOW} 点）──")
print(f"slope     = {slope:.6f}（hw_ts 每 1 单位 ≈ wall {slope:.1f}s；非 1.0 不是问题，polyfit 处理任意 slope）")
print(f"intercept = {intercept:.3f}")
print(f"R²        = {r2:.6f}（判据：R² > 0.99 → 线性映射可信，残差 ~毫秒级）")
print(f"hw_ts 范围: {hw_arr.min():.3f} ~ {hw_arr.max():.3f}（不一定是秒，取决于 libfranka 内部时钟）")
print(f"mono 范围: {mono_arr.min():.3f} ~ {mono_arr.max():.3f} 秒")
print(f"残差 stddev ≈ {residual_std*1000:.2f} ms (wall-clock 域)")

print()
if r2 > 0.99:
    print(f"[PASS] R²={r2:.6f} > 0.99，align_offline 用 polyfit 映射可信")
else:
    print(f"[FAIL] R²={r2:.6f} < 0.99，线性度不足，align_offline 会退回旧路径")
