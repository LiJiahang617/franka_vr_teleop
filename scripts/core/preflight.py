"""§11.2 数据预检门：夹爪健康/homing 预检 + 图像色彩通道序预检。

设计原则：
- 纯判据函数（`gripper_goto_span_ok`、`gripper_health_verdict`、`image_color_verdict`）
  只做逻辑运算，零 IO、零硬件依赖，可全离线单测。
- 硬件薄壳（`run_gripper_preflight`、`run_color_preflight`、`default_proc_probe`、
  `default_log_probe`）封装所有 IO，由调用方注入 probe 回调实现可测可替换。
- 不连真机/不发真控制——zerorpc client 由调用方传入（录制入口在连接后传入）。

§10.5(B) 正解判据：
  - 子进程强存活：`pgrep -f franka_hand_client`（**非**端口 :50052 LISTEN，端口 LISTEN
    不等于子进程存活）。
  - 连接就绪：`_gripper_live.log` 出现 `Connected.`（LISTEN 后还需 ~6-7s，等真日志再验）。
  - width 真变：`gripper_goto` 后轮询 `is_moving` → False 或超时（settle），再量
    多目标 `span = max - min > 0.02`（**禁** 0.5s 采样 + 相邻差 > 0.01 假阴性判据）。
"""
import subprocess
import time
from collections import namedtuple

Verdict = namedtuple("Verdict", ["ok", "reason"])


# ---------------------------------------------------------------------------
# 纯判据函数（零 IO，零硬件依赖，全离线可单测）
# ---------------------------------------------------------------------------

def gripper_goto_span_ok(width_samples, min_span: float = 0.02) -> bool:
    """判断 goto 后采到的多个 settled width 样本跨度是否足够大。

    Args:
        width_samples: 各目标 goto 后 settle 完毕的实测 width 值列表（至少 2 个）。
        min_span: 最小跨度阈值，默认 0.02 m（§10.5B 正解判据）。

    Returns:
        True 表示跨度 > min_span，夹爪真实运动；False 表示跨度不足，疑似丢 homing。

    §10.5(B) 正解：用 max-min 整体跨度，**禁** 相邻样本差 > 0.01 的假阴性判据。
    例：[0.0001, 0.07, 0.04] → span=0.0699 > 0.02 → True。
        [0.04, 0.04, 0.04]   → span=0       → False（丢 homing 真阳性）。
    """
    if len(width_samples) < 2:
        return False
    return (max(width_samples) - min(width_samples)) > min_span


def gripper_health_verdict(state: dict, proc_alive: bool, connected: bool) -> Verdict:
    """判断夹爪初始健康状态（get_state + 进程 + 日志 Connected）。

    Args:
        state: `gripper_get_state()` 返回的 dict（至少含 `error_code`）。
        proc_alive: True 表示 `pgrep -f franka_hand_client` 找到进程（强存活判据）。
        connected: True 表示 `_gripper_live.log` 已出现 `Connected.`。

    Returns:
        Verdict(ok=True/False, reason=可行动说明)。

    §10.5(B)：proc_alive 来自 pgrep（**非** 端口 :50052 LISTEN）；connected 来自日志。
    """
    if not proc_alive:
        return Verdict(
            ok=False,
            reason=(
                "夹爪子进程(franka_hand_client)未存活 "
                "→ 重起夹爪服务 `scripts/services/_run_gripper.sh`"
            ),
        )
    if not connected:
        return Verdict(
            ok=False,
            reason=(
                "夹爪 zerorpc 未就绪（日志无 Connected.）"
                "→ 等待 ~10s 或重起夹爪服务"
            ),
        )
    error_code = state.get("error_code", 0)
    if error_code != 0:
        return Verdict(
            ok=False,
            reason=f"夹爪 error_code={error_code}，请检查夹爪状态或重起服务",
        )
    return Verdict(ok=True, reason="夹爪初始状态正常")


def image_color_verdict(decoded_rgb_frames: list, rb_gap_thresh: float = 60.0) -> Verdict:
    """弱判据：检测解码帧是否整体存在 RGB/BGR 反转（宁漏勿误报正常画面）。

    判据：若帧均值 B - R > rb_gap_thresh 且帧中几乎无暖色像素，则疑似 RGB/BGR 反转。
    纯统计无法 100% 区分内容偏蓝与 RGB/BGR 反——默认弱判据粗筛，优先避免误报正常画面。

    Args:
        decoded_rgb_frames: 经 `hdf5_lerobot_map._decode` 解码的 RGB ndarray 列表。
        rb_gap_thresh: B 均值 - R 均值超过此阈值时疑似通道反（默认 60）。

    Returns:
        Verdict(ok=True/False, reason=...)。
    """
    import numpy as np

    for frame in decoded_rgb_frames:
        r_mean = float(frame[..., 0].mean())
        b_mean = float(frame[..., 2].mean())
        # 暖色像素比例（R > B + 30）
        warm_ratio = float((frame[..., 0].astype(int) > frame[..., 2].astype(int) + 30).mean())
        if (b_mean - r_mean) > rb_gap_thresh and warm_ratio < 0.05:
            return Verdict(
                ok=False,
                reason=(
                    f"色彩通道序疑似 RGB/BGR 反（B_mean={b_mean:.1f} R_mean={r_mean:.1f} "
                    f"gap={b_mean - r_mean:.1f}>{rb_gap_thresh}，暖色像素比={warm_ratio:.3f}<0.05）"
                    " → 检查相机图像通道序约定（RGB/BGR）"
                ),
            )
    return Verdict(ok=True, reason="色彩通道序预检通过")


# ---------------------------------------------------------------------------
# 硬件薄壳（IO 与判据分离；proc_probe/log_probe 由调用方注入，测试时注入 fake）
# ---------------------------------------------------------------------------

def run_gripper_preflight(
    client,
    proc_probe,
    log_probe,
    targets=(0.0, 0.07, 0.04),
    settle_timeout: float = 8.0,
    poll: float = 0.3,
    min_span: float = 0.02,
) -> Verdict:
    """运行夹爪健康/homing 预检门。

    Args:
        client: zerorpc FrankaInterfaceClient（调用方已连接，本函数不新建连接）。
        proc_probe: `() -> bool`，True 表示 franka_hand_client 进程存活（注入 pgrep）。
        log_probe: `() -> bool`，True 表示日志出现 `Connected.`（注入读文件）。
        targets: goto 目标 width 序列（默认 (0.0, 0.07, 0.04)，闭→开→中，覆盖行程）。
        settle_timeout: 每次 goto 后等待 is_moving→False 的最大秒数（默认 8s）。
        poll: settle 轮询间隔（秒），测试时传 0.0 以跳过实际睡眠。
        min_span: 跨度阈值，见 `gripper_goto_span_ok`。

    Returns:
        Verdict(ok=True/False, reason=可行动说明)。

    §10.5(B) 正解：
    1. proc_alive = pgrep（**非** 端口 LISTEN）。
    2. connected = 日志 Connected.（等真日志再验，不在连接窗口过早验）。
    3. gripper_initialize() + gripper_get_state() 初始健康门。
    4. 对每个 target：goto → 轮询 is_moving→False/超时（给足行程时间，默认 8s）→ 记 width。
    5. span = max-min > min_span（**禁** 0.5s+相邻差>0.01 假阴性）。
    6. 不过 → 清晰可行动报错（重起夹爪/去 Desk Homing），开录前拦截。

    zerorpc 单线程约定：所有 gripper_goto / gripper_get_state 串行顺序调用，
    本函数不并发（预检在录制循环开始前一次性完成）。
    """
    # 步骤 1：进程 + 连接 + get_state 初始门
    proc_alive = proc_probe()
    connected = log_probe()
    client.gripper_initialize()
    st = client.gripper_get_state()
    v = gripper_health_verdict(st, proc_alive, connected)
    if not v.ok:
        return v

    # 步骤 2：多目标 goto → settle（轮询 is_moving→False/超时）→ 记 settled width
    measured = []
    for target in targets:
        client.gripper_goto(target, 0.05, 20.0, -1.0, -1.0, True)
        # settle：轮询直到 is_moving=False 或超时（给足行程时间，**非**固定 0.5s）
        t0 = time.monotonic()
        while True:
            s = client.gripper_get_state()
            if not s.get("is_moving", False):
                break
            if time.monotonic() - t0 >= settle_timeout:
                break
            if poll > 0:
                time.sleep(poll)
        measured.append(client.gripper_get_state()["width"])

    # 步骤 3：span 判据（§10.5B 正解：整体跨度 > min_span，非相邻差）
    if not gripper_goto_span_ok(measured, min_span):
        return Verdict(
            ok=False,
            reason=(
                f"夹爪 goto 后 width 未真变（行程跨度 {max(measured) - min(measured):.4f} "
                f"< {min_span:.3f}）→ 疑似丢 homing，请在 Franka Desk 执行 Homing 后重试"
            ),
        )

    return Verdict(ok=True, reason="夹爪预检通过（进程存活/连接就绪/width 真变）")


def run_color_preflight(decode_fn, encoded_frames: list) -> Verdict:
    """运行图像色彩通道序预检门。

    Args:
        decode_fn: 解码函数，约定 = `hdf5_lerobot_map._decode`（输出 RGB ndarray）。
        encoded_frames: 编码后 jpeg bytes 列表（与录制管道同款编码）。

    Returns:
        Verdict(ok=True/False, reason=...)。
    """
    decoded = [decode_fn(b) for b in encoded_frames]
    return image_color_verdict(decoded)


# ---------------------------------------------------------------------------
# 默认硬件 probe（供 run_record_hdf5.main 直接用；测试时注入 fake lambda 替换）
# ---------------------------------------------------------------------------

def default_proc_probe() -> bool:
    """强存活判据：pgrep -f franka_hand_client（**非** 端口 :50052 LISTEN）。

    §10.5(B)：端口 LISTEN 不足以证明子进程存活（外壳进程可崩而端口暂留），
    必须用 pgrep 直接探测进程名。
    """
    result = subprocess.run(
        ["pgrep", "-f", "franka_hand_client"],
        capture_output=True,
    )
    return result.returncode == 0


def default_log_probe(log_path: str, marker: str = "Connected.") -> bool:
    """连接就绪判据：读日志文件，检查是否出现 Connected. 标记。

    §10.5(B)：LISTEN 后还需 ~6-7s 才出现 Connected.，必须等真日志再验，
    不在连接窗口内过早验证。
    """
    try:
        with open(log_path) as f:
            return marker in f.read()
    except OSError:
        return False
