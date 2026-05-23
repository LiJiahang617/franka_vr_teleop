"""§11.2 数据预检门：夹爪健康/homing 预检 + 图像色彩通道序预检。

设计原则：
- 纯判据函数（`gripper_goto_span_ok`、`gripper_health_verdict`、
  `gripper_state_fields_ok`、`image_color_verdict`）
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
import os
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


def gripper_state_fields_ok(state: dict) -> Verdict:
    """校验 gripper_get_state 返回 dict 的必需字段是否齐全且无 error_code。

    纯函数，零 IO，可全离线单测。

    Args:
        state: `gripper_get_state()` 返回的 dict。

    Returns:
        Verdict(ok=False, reason=可行动说明) 若缺字段或 error_code!=0；
        Verdict(ok=True, ...) 若字段齐全且 error_code==0。
    """
    required = ("error_code", "is_moving", "width")
    missing = [k for k in required if k not in state]
    if missing:
        return Verdict(
            ok=False,
            reason=(
                f"夹爪 get_state 缺字段 {missing}"
                " → 夹爪客户端异常，重起 _run_gripper.sh"
            ),
        )
    if state["error_code"] != 0:
        return Verdict(
            ok=False,
            reason=f"夹爪 error_code={state['error_code']}，请检查夹爪状态或重起服务",
        )
    return Verdict(ok=True, reason="夹爪状态字段完整")


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
# 硬件薄壳（IO 与判据分离；proc_probe/log_probe 由调用方注入，测试时注入 fake lambda 替换）
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
        targets: goto 目标 width 序列（默认 (0.0, 0.07, 0.04)，闭→开→中，覆盖行程）；
                 **必须 ≥ 2 个目标**才能计算跨度，否则视为配置错误立即拦截。
        settle_timeout: 每次 goto 后等待 is_moving→False 的最大秒数（默认 8s）。
        poll: settle 轮询间隔（秒）；poll<=0 仅测试用（跳过 sleep，仍受 settle_timeout 界）；
              生产传 >0 避免忙等。
        min_span: 跨度阈值，见 `gripper_goto_span_ok`。

    Returns:
        Verdict(ok=True/False, reason=可行动说明)。

    §10.5(B) 正解：
    1. proc_alive = pgrep（**非** 端口 LISTEN）。
    2. connected = 日志 Connected.（等真日志再验，不在连接窗口过早验）。
    3. proc/connected 两门通过后才发起任何 client RPC（保证预检错误不被 zerorpc 抛/卡绕过）。
    4. gripper_initialize() + gripper_get_state() + 字段完整性校验。
    5. 对每个 target：goto → 轮询 is_moving→False/超时（给足行程时间，默认 8s）→ 记 width。
    6. span = max-min > min_span（**禁** 0.5s+相邻差>0.01 假阴性）。
    7. 不过 → 清晰可行动报错（重起夹爪/去 Desk Homing），开录前拦截。

    zerorpc 单线程约定：所有 gripper_goto / gripper_get_state 串行顺序调用，
    本函数不并发（预检在录制循环开始前一次性完成）。
    """
    # 步骤 1：进程存活门（先于任何 client RPC）
    proc_alive = proc_probe()
    if not proc_alive:
        return Verdict(
            ok=False,
            reason=(
                "夹爪子进程(franka_hand_client)未存活"
                " → 重起 scripts/services/_run_gripper.sh"
            ),
        )

    # 步骤 2：连接就绪门（先于任何 client RPC）
    connected = log_probe()
    if not connected:
        return Verdict(
            ok=False,
            reason=(
                "夹爪 zerorpc 未就绪（日志无 Connected.）"
                " → 等待 ~10s 或重起夹爪服务"
            ),
        )

    # 步骤 3：targets 配置校验（≥2 才能算跨度）
    if len(targets) < 2:
        return Verdict(
            ok=False,
            reason=(
                f"预检配置错误: targets 需≥2 个目标以测行程跨度, got {targets!r}"
            ),
        )

    # 步骤 4：初始 get_state + 字段完整性 + error_code 门（此时 proc/connected 已通过）
    client.gripper_initialize()
    st = client.gripper_get_state()
    v = gripper_state_fields_ok(st)
    if not v.ok:
        return v

    # 步骤 5：多目标 goto → settle（轮询 is_moving→False/超时）→ 记 settled width
    measured = []
    for target in targets:
        # gripper_goto(width, speed, force, epsilon_inner, epsilon_outer, blocking)
        # — zerorpc FrankaInterfaceClient 既有签名（与 Franka.reset 同款用法）
        client.gripper_goto(target, 0.05, 20.0, -1.0, -1.0, True)
        t0 = time.monotonic()
        # Phase 1：等 is_moving 先变 True（zerorpc goto 异步返回，
        # 命令到 franka_hand_client 真正开始执行有 ~0.1-0.3s 延迟；
        # 否则首次 poll 就读到 is_moving=False 误判已 settle）。
        # 最多等 1.5s；超时则视为夹爪压根没动（不算 settle，记当前 width 即可）。
        moving_start_deadline = t0 + 1.5
        while time.monotonic() < moving_start_deadline:
            s = client.gripper_get_state()
            if "is_moving" not in s:
                return Verdict(
                    ok=False,
                    reason="夹爪 get_state 缺 is_moving → 客户端异常，重起 _run_gripper.sh",
                )
            if s["is_moving"]:
                break
            if poll > 0:
                time.sleep(poll)
        # Phase 2：等 is_moving=False（动作完成）或超时
        while True:
            s = client.gripper_get_state()
            if "is_moving" not in s:
                return Verdict(
                    ok=False,
                    reason="夹爪 get_state 缺 is_moving → 客户端异常，重起 _run_gripper.sh",
                )
            if not s["is_moving"]:
                break
            if time.monotonic() - t0 >= settle_timeout:
                return Verdict(
                    ok=False,
                    reason=(
                        f"夹爪 goto target={target} 超时 {settle_timeout}s 仍未稳定"
                        "（is_moving 恒真）→ 检查夹爪/Desk Homing/重起夹爪客户端"
                    ),
                )
            if poll > 0:
                time.sleep(poll)
        # settle 完毕，取最终 width（校验字段存在）
        s_final = client.gripper_get_state()
        if "width" not in s_final:
            return Verdict(
                ok=False,
                reason="夹爪 get_state 缺 width → 客户端异常，重起 _run_gripper.sh",
            )
        measured.append(s_final["width"])

    # 步骤 6：span 判据（§10.5B 正解：整体跨度 > min_span，非相邻差）
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

    **陈旧日志风险**：依赖夹爪 (重)启动脚本（`scripts/services/_run_gripper.sh` /
    debug `franka_start_gripper.sh` 已 `: > _gripper_live.log` 截断，§11.1-T4 验）
    清空旧日志，否则上轮残留 Connected. 致假阳性。
    """
    try:
        with open(log_path) as f:
            return marker in f.read()
    except OSError:
        return False


# ---------------------------------------------------------------------------
# controller preflight（录制前幂等启动 cartesian impedance controller）
# ---------------------------------------------------------------------------

def run_controller_preflight(
    client,
    Kx=None,
    Kxd=None,
    *,
    settle_timeout: float = 8.0,
    poll: float = 0.1,
    polymetis_python: str = "/home/ubuntu/Desktop/jhli/envs/polymetis-local/bin/python",
    polymetis_conda_prefix: str = "/home/ubuntu/Desktop/jhli/envs/polymetis-local",
    starter=None,  # Callable[[list, list], None]，None → 默认 subprocess 启动；测试可注入 mock
) -> Verdict:
    """录制前幂等启动 cartesian impedance controller。

    实现：用 subprocess 调 polymetis-local Python 直接 import RobotInterface 启动
    (与 scripts/services/_run_polymetis_rw.sh:88-104 同款路径)，绕开 zerorpc
    server.robot_start_cartesian_impedance_control 未经使用/有 bug 的代码路径。

    背景：_run_polymetis_rw.sh 后台异步启 controller 偶发性失败；主循环
    send_action 调 update_desired_ee_pose 时若 controller 未跑会抛
    grpc.RpcError "no controller running"。本函数是兜底防御。

    Args:
        client: zerorpc FrankaInterfaceClient(即 robot._robot)，用于 get_ee_pose 验证
        Kx: 笛卡尔刚度(list of 6 floats)，默认 [100,100,100,40,40,40]
        Kxd: 笛卡尔阻尼(list of 6 floats)，默认 [1,1,1,0.2,0.2,0.2]
        settle_timeout: 启动后等待 controller register 的最长时间(秒)
        poll: 轮询间隔(秒)
        polymetis_python: polymetis-local venv python 绝对路径
        polymetis_conda_prefix: polymetis-local CONDA_PREFIX 环境变量值
        starter: Callable[[list, list], None]，接受 (Kx, Kxd) 参数；
                 None → 默认 subprocess 启动；测试可注入 mock 函数避免真 subprocess

    Returns:
        Verdict(ok=True) 控制器就绪；Verdict(ok=False, reason=...) 失败有可行动指引。
    """
    if Kx is None:
        Kx = [100.0, 100.0, 100.0, 40.0, 40.0, 40.0]
    if Kxd is None:
        Kxd = [1.0, 1.0, 1.0, 0.2, 0.2, 0.2]

    if starter is None:
        # subprocess 调 polymetis-local Python(与 _run_polymetis_rw.sh 同款路径)
        # 先 terminate_current_policy 干掉 Franka.connect 启动的 joint impedance（若有），
        # 否则 start_cartesian 会与现有 policy 冲突卡死
        def starter(Kx_arg, Kxd_arg):
            """默认 starter：subprocess 调 polymetis-local Python 启动 controller。"""
            code = (
                "import torch\n"
                "from polymetis import RobotInterface\n"
                "r = RobotInterface(ip_address='localhost', enforce_version=False)\n"
                "try:\n"
                "    r.terminate_current_policy()\n"
                "except Exception:\n"
                "    pass\n"   # 无 policy 时 terminate 会抛，忽略
                f"r.start_cartesian_impedance(Kx=torch.Tensor({Kx_arg}), Kxd=torch.Tensor({Kxd_arg}))\n"
            )
            env = os.environ.copy()
            env["CONDA_PREFIX"] = polymetis_conda_prefix
            result = subprocess.run(
                [polymetis_python, "-c", code],
                capture_output=True,
                text=True,
                timeout=15.0,
                env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"start_cartesian_impedance subprocess failed (rc={result.returncode}): "
                    f"{result.stderr[-400:]}"
                )

    # 启动 controller（幂等：starter 内部处理 terminate + start_cartesian）
    try:
        starter(Kx, Kxd)
    except subprocess.TimeoutExpired:
        return Verdict(
            ok=False,
            reason=(
                "启动 cartesian impedance controller 超时 15s；"
                "检查 polymetis 服务(端口 50051 / launch_robot.py)是否健康"
            ),
        )
    except FileNotFoundError as e:
        return Verdict(
            ok=False,
            reason=(
                f"polymetis-local Python 不存在: {polymetis_python}({e})；"
                f"检查 venv 路径或 yaml record.controller_preflight.polymetis_python"
            ),
        )
    except Exception as e:
        return Verdict(
            ok=False,
            reason=(
                f"启动 cartesian impedance controller 失败: {e}；"
                f"检查 polymetis 服务是否健康"
            ),
        )

    # 等 controller register(用 get_ee_pose 探测——controller 跑了才能拿到稳定 pose)
    # 接受 list / tuple / ndarray 三种类型(Franka.connect 后 zerorpc 可能返 ndarray)
    t0 = time.monotonic()
    last_ee = None
    last_exc = None
    while time.monotonic() - t0 < settle_timeout:
        try:
            ee = client.robot_get_ee_pose()
            last_ee = ee
            if hasattr(ee, "__len__") and len(ee) == 6:
                return Verdict(ok=True, reason="cartesian impedance controller 就绪")
        except Exception as e:
            last_exc = e
        if poll > 0:
            time.sleep(poll)

    return Verdict(
        ok=False,
        reason=(
            f"controller 启动后 {settle_timeout}s 内 robot_get_ee_pose 未返回有效 6D 位姿；"
            f"最后一次返回: {last_ee!r} (type={type(last_ee).__name__}); "
            f"最后异常: {last_exc!r}; "
            f"check polymetis launch_robot.py 日志"
        ),
    )
