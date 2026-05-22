"""Phase D Task 1 spike 脚本 —— 探测 RealSense 硬件时间戳可用性

用途：
    测 RealSense frame.get_timestamp() + rs.option.global_time_enabled
    能否提供可对齐的硬件戳，决定图像时间戳路径。

如何跑：
    1. 将 wrist cam（串号 419622073931）插好 USB
    2. 激活 venv：source /home/ubuntu/Desktop/jhli/envs/franka-teleop/bin/activate
    3. 运行：python scripts/spike/spike_realsense_hw_timestamp.py

如何解读结论：
    脚本末尾打印 hw 戳单调性、线性回归 slope/intercept/R²，并对照决策树给出建议分支：
      A) hw 戳毫秒级严格单调、slope ≈ 1.0、R² 高、domain 为 global_time
         → 硬件戳可用，启用 global_time + 读 hw 戳
      B) hw 戳不单调 / slope 偏离 / R² 低 / domain 异常
         → 硬件戳不可用，退化为软件戳
"""

import time

import numpy as np
import pyrealsense2 as rs

# wrist cam 串号（用户可按需替换为其他相机串号）
WRIST_CAM_SERIAL = "419622073931"

# 采集帧数
N_FRAMES = 200

# 640×480 为 D435 标准 color 模式；分辨率不影响时间戳探测结论
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
COLOR_FPS = 30


def main():
    pipeline = rs.pipeline()
    rs_config = rs.config()

    # 指定相机串号，避免多相机场景混用
    rs.config.enable_device(rs_config, WRIST_CAM_SERIAL)
    rs_config.enable_stream(
        rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.rgb8, COLOR_FPS
    )

    print(f"[spike] 启动 RealSense pipeline，串号={WRIST_CAM_SERIAL} ...")

    # started 标记：用于在 finally 里判断是否需要 stop，
    # 避免 pipeline.start() 失败时 pipeline.stop() 抛新异常掩盖原始错误
    started = False
    try:
        profile = pipeline.start(rs_config)
        started = True

        # 对 color sensor 启用 global_time_enabled，
        # 让 get_timestamp() 返回相机硬件同步时间（现有 RealSense 类未设此选项）
        color_sensor = profile.get_device().first_color_sensor()
        color_sensor.set_option(rs.option.global_time_enabled, 1)
        # 回读确认 option 已生效
        gte_value = color_sensor.get_option(rs.option.global_time_enabled)
        print(f"[spike] global_time_enabled 已设为 1（回读值: {gte_value}）")

        # 预热：丢弃前几帧，等曝光稳定
        WARMUP_FRAMES = 10
        for _ in range(WARMUP_FRAMES):
            pipeline.wait_for_frames()
        print(f"[spike] 预热 {WARMUP_FRAMES} 帧完成，开始采集 {N_FRAMES} 帧 ...")

        # 采集：记录 (hw_timestamp_ms, monotonic_s, timestamp_domain)
        hw_ts_list = []      # 单位：毫秒（get_timestamp() 返回毫秒）
        mono_list = []       # 单位：秒
        domain_set = set()   # 采集到的 timestamp domain 集合

        for i in range(N_FRAMES):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print(f"  [警告] 第 {i} 帧 color_frame 为空，跳过")
                continue
            hw_ts_list.append(color_frame.get_timestamp())          # 毫秒
            mono_list.append(time.monotonic())                      # 秒
            domain_set.add(color_frame.get_frame_timestamp_domain())  # timestamp domain

            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{N_FRAMES}")

    finally:
        # 仅在 pipeline 已成功 start 时才 stop，避免掩盖 start 异常
        if started:
            pipeline.stop()
            print("[spike] pipeline 已关闭")

    # ── 统计 ──────────────────────────────────────────
    hw_ts = np.array(hw_ts_list)  # 毫秒
    mono = np.array(mono_list)    # 秒

    n = len(hw_ts)

    # 有效帧数不足门槛：要求 >= 90% 的请求帧才进入分析
    if n < int(0.9 * N_FRAMES):
        missing = N_FRAMES - n
        print(
            f"[spike] inconclusive，有效帧数不足（{n}/{N_FRAMES}，缺 {missing} 帧），"
            "建议重跑或退化为软件戳。"
        )
        return

    # 1) 严格单调性检查
    diffs = np.diff(hw_ts)
    is_monotone = bool(np.all(diffs > 0))
    n_non_monotone = int(np.sum(diffs <= 0))

    # 2) 线性回归：验证 hw 戳与 monotonic 的线性对齐程度
    #    把两者统一换成秒（hw_ts_s = hw_ts / 1000）再做回归，
    #    slope 理想值 ≈ 1.0，intercept 为固定偏移，R² 理想值 ≈ 1.0
    hw_ts_s = hw_ts / 1000.0  # 毫秒 → 秒
    # 中心化减少数值精度误差
    mono_c = mono - mono[0]
    hw_s_c = hw_ts_s - hw_ts_s[0]
    # 最小二乘：hw_s_c ≈ slope * mono_c + intercept
    A_mat = np.vstack([mono_c, np.ones(n)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, hw_s_c, rcond=None)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    # R² 计算
    hw_pred = slope * mono_c + intercept
    ss_res = float(np.sum((hw_s_c - hw_pred) ** 2))
    ss_tot = float(np.sum((hw_s_c - hw_s_c.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # 3) timestamp domain 检查：需全部帧均为 global_time
    domain_ok = domain_set == {rs.timestamp_domain.global_time}

    # ── 打印结论块 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  spike_realsense_hw_timestamp 结论")
    print("=" * 60)
    print(f"  采集帧数           : {n}/{N_FRAMES}")
    print(f"  timestamp domain   : {domain_set}")
    print(f"  hw 戳严格单调递增   : {is_monotone}（非单调次数: {n_non_monotone}）")
    print(f"  线性回归 slope      : {slope:.6f}（理想值 ≈ 1.000000）")
    print(f"  线性回归 intercept  : {intercept:.6f} s")
    print(f"  R²                 : {r2:.8f}（理想值 ≈ 1.0）")
    print(f"  hw 戳范围           : {hw_ts[0]:.1f} ~ {hw_ts[-1]:.1f} ms")
    print(f"  总时长(mono)        : {mono[-1] - mono[0]:.3f} s")
    print("-" * 60)

    # 决策树：slope 容差 ±0.01，R² 阈值 0.9999，domain 须全为 global_time
    slope_ok = abs(slope - 1.0) < 0.01
    r2_ok = r2 > 0.9999
    if is_monotone and slope_ok and r2_ok and domain_ok:
        branch = "A"
        desc = "硬件戳可用，启用 global_time + 读 hw 戳"
    else:
        branch = "B"
        reasons = []
        if not is_monotone:
            reasons.append(f"hw 戳不单调（{n_non_monotone} 次）")
        if not slope_ok:
            reasons.append(f"slope={slope:.4f} 偏离 1.0")
        if not r2_ok:
            reasons.append(f"R²={r2:.6f} 偏低")
        if not domain_ok:
            reasons.append(
                f"timestamp domain 异常（期望 global_time，实际 {domain_set}）；"
                "可能 global_time_enabled 未生效或相机固件不支持"
            )
        desc = "hw 戳不可用（" + "；".join(reasons) + "），hw_timestamp 字段退化为软件戳"

    print(f"  建议分支: {branch} — {desc}")
    print("=" * 60)


if __name__ == "__main__":
    main()
