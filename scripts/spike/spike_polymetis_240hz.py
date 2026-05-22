"""Phase D Task 1 spike 脚本 —— 探测 polymetis/zerorpc 能否撑 240Hz 本体读

用途：
    测单 Python 进程经 zerorpc 往返调 robot_get_joint_positions() 的真实周期，
    决定 state_hifreq（240Hz 稠密本体）的实现路径。

如何跑：
    1. 先在另一终端起 zerorpc 服务端（franka_interface_server / start_server.py）
    2. 激活 venv：source /home/ubuntu/Desktop/jhli/envs/franka-teleop/bin/activate
    3. 运行：python scripts/spike/spike_polymetis_240hz.py

如何解读结论：
    脚本末尾打印往返耗时统计（mean/std/p50/p95/p99/max，毫秒）、
    脚本端实测吞吐频率（Hz）、掉帧数，并对照决策树给出建议分支：
      A) mean ≤ 4.2ms 且 p95 ≤ 5ms  → 240Hz 稳，走主进程独立线程方案
      B) mean 4.2~8ms               → 需 polymetis server 补 batch 接口
      C) mean > 8ms                 → state_hifreq 降级为后续子阶

    两个掉帧指标的区别：
      period_overrun(>4.17ms)：超过单个 240Hz 周期预算（4.17ms）的次数，反映偶发毛刺；
      drops(>8.33ms)：超过两倍 240Hz 周期（8.33ms）的次数，属更严重的掉帧，是计划指定口径。
"""

import sys
import time

import numpy as np

# lerobot_robot_franka 已 pip install 到 venv，直接 import 即可
from lerobot_robot_franka import FrankaInterfaceClient

# ── 配置常量 ──────────────────────────────────────────
SERVER_IP = "127.0.0.1"
SERVER_PORT = 4242

# 总测试次数
N_CALLS = 10000

# 单周期阈值：240Hz 单帧预算 ≈ 4.17ms，超过即算 period_overrun
PERIOD_THRESHOLD_MS = 1000.0 / 240.0  # ≈ 4.17ms

# 掉帧判定阈值：240Hz 单帧预算 4.17ms，超过 2 倍（8.33ms）视为掉帧
DROP_THRESHOLD_MS = 1000.0 / 240.0 * 2  # ≈ 8.33ms


def main():
    print(f"[spike] 连接 zerorpc server {SERVER_IP}:{SERVER_PORT} ...")
    client = FrankaInterfaceClient(ip=SERVER_IP, port=SERVER_PORT)

    # FrankaInterfaceClient.__init__ 用 except: pass 吞掉连接异常，
    # 必须在这里显式探测，连不上就清晰报错退出
    try:
        client.robot_get_joint_positions()
    except Exception as e:
        print(
            f"[spike] 连接失败：{e}\n"
            "请先启动 zerorpc server（franka_server / start_server.py），再运行本脚本。"
        )
        sys.exit(1)

    print(f"[spike] 连接成功，开始 {N_CALLS} 次往返调用计时 ...")

    # 记录每次调用的单次往返耗时（秒）
    durations_s = []
    t_start_wall = time.perf_counter()

    for i in range(N_CALLS):
        t0 = time.perf_counter()
        client.robot_get_joint_positions()
        t1 = time.perf_counter()
        durations_s.append(t1 - t0)

        # 每 1000 次打印一次进度
        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{N_CALLS}")

    t_total_s = time.perf_counter() - t_start_wall
    client.close()

    # ── 统计 ──────────────────────────────────────────
    dur_ms = np.array(durations_s) * 1000.0  # 转为毫秒
    mean_ms = float(np.mean(dur_ms))
    std_ms = float(np.std(dur_ms))
    p50_ms = float(np.percentile(dur_ms, 50))
    p95_ms = float(np.percentile(dur_ms, 95))
    p99_ms = float(np.percentile(dur_ms, 99))
    max_ms = float(np.max(dur_ms))
    avg_hz = N_CALLS / t_total_s
    # 超过单 240Hz 周期（4.17ms）的次数，反映偶发毛刺
    period_overrun = int(np.sum(dur_ms > PERIOD_THRESHOLD_MS))
    # 超过两倍 240Hz 周期（8.33ms）的掉帧次数（计划指定口径）
    drops = int(np.sum(dur_ms > DROP_THRESHOLD_MS))

    # ── 打印结论块 ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  spike_polymetis_240hz 结论")
    print("=" * 60)
    print(f"  总调用次数                : {N_CALLS}")
    print(f"  总耗时                    : {t_total_s:.2f} s")
    print(f"  脚本端实测吞吐频率        : {avg_hz:.1f} Hz")
    print(f"  单次往返 mean             : {mean_ms:.3f} ms")
    print(f"  单次往返 std              : {std_ms:.3f} ms")
    print(f"  p50                       : {p50_ms:.3f} ms")
    print(f"  p95                       : {p95_ms:.3f} ms")
    print(f"  p99                       : {p99_ms:.3f} ms")
    print(f"  max                       : {max_ms:.3f} ms")
    print(f"  period_overrun(>{PERIOD_THRESHOLD_MS:.2f}ms) : {period_overrun} 次 ({period_overrun / N_CALLS * 100:.2f}%)")
    print(f"  drops(>{DROP_THRESHOLD_MS:.2f}ms)       : {drops} 次 ({drops / N_CALLS * 100:.2f}%)")
    print("-" * 60)

    # 决策树（阈值为计划红线，不得修改）
    if mean_ms <= 4.2 and p95_ms <= 5.0:
        branch = "A"
        desc = "240Hz 稳，走主进程独立线程方案"
    elif mean_ms <= 8.0:
        branch = "B"
        desc = "均值尚可但未达 A 条件（4.2~8ms），需 polymetis server 补 batch 接口；请人工核对 std/p95 抖动"
    else:
        branch = "C"
        desc = "state_hifreq 降级为后续子阶（mean > 8ms，实测 <120Hz）"

    print(f"  建议分支: {branch} — {desc}")
    print("=" * 60)


if __name__ == "__main__":
    main()
