# 开发说明

> Franka 数采系统的环境搭建、代码组织、测试、扩展规范与调试陷阱。

## 目录

- [1. 环境搭建](#1-环境搭建)
- [2. 代码组织](#2-代码组织)
- [3. 运行测试](#3-运行测试)
- [4. 新增功能模块示例](#4-新增功能模块示例)
- [5. schema 修改规范](#5-schema-修改规范)
- [6. 服务启停与运维](#6-服务启停与运维)
- [7. 打包与 console_scripts](#7-打包与-console_scripts)
- [8. 调试陷阱](#8-调试陷阱)
- [9. 代码规范](#9-代码规范)

---

## 1. 环境搭建

系统涉及两套 Python 环境：

| 环境 | 路径 | 用途 |
|---|---|---|
| `franka-teleop` | `/home/ubuntu/Desktop/jhli/envs/franka-teleop`（venv） | 采集侧：录制、转换、测试 |
| `polymetis-local` | `/home/ubuntu/Desktop/jhli/envs/polymetis-local`（conda） | Polymetis 三进程服务（Python<3.10） |

采集侧 `franka-teleop` 内安装三个 editable 包：

```bash
source /home/ubuntu/Desktop/jhli/envs/franka-teleop/bin/activate
cd /home/ubuntu/Desktop/jhli/lerobot_franka_teleop
pip install -e .                          # 主包 lerobot_franka_teleop（含 scripts/）
pip install -e lerobot_robot_franka        # 子包①
pip install -e lerobot_teleoperator_franka # 子包②
```

> `setup.py` 的 `install_requires` 已用 `file://` 声明两子包依赖，但子包以 editable 安装更便于开发。LeRobot 本体为 franka2 本机已装版本（v3.0）。

---

## 2. 代码组织

```
franka_hdf5_schema.py         # schema 冻结契约（仓库根 loose 模块）
scripts/
  core/      录制/回放/可视化/复位入口 + 录制核心模块（见 architecture.md §5）
  tools/     hdf5→lerobot 转换器、数据集检查
  services/  三进程服务启动包装脚本
  config/    record_cfg_unityvr.yaml（Route B 录制配置）
  utils/     数据集辅助工具
  help/      franka-help
  ui/        数采 Web UI Flask app（控制面板、状态机、预览编码、HTML 模板）
lerobot_robot_franka/         子包①：Franka Robot 接口 + zerorpc server/client
lerobot_teleoperator_franka/  子包②：遥操作设备 + Route B 映射
tests/                        pytest 测试（纯逻辑离线可跑）
debug/                        诊断与运维脚本
```

**分层原则**：

- **纯逻辑模块**（`record_params.py`、`unityvr_mapping.py`、`vr_align.py`、`preflight.py` 判据函数、`hdf5_lerobot_map.py`）：零硬件依赖、零 IO，模块顶层可 import，全离线单测。
- **硬件依赖**（`franka.py`、teleop 实体、相机）：在录制入口里**延迟 import**（函数内 import），避免测试加载时因缺硬件包而崩溃。`run_record_hdf5.py` 顶部只 import 纯逻辑模块。

---

## 3. 运行测试

测试为纯逻辑离线测试，无需硬件、无需服务：

```bash
source /home/ubuntu/Desktop/jhli/envs/franka-teleop/bin/activate
cd /home/ubuntu/Desktop/jhli/lerobot_franka_teleop
python -m pytest tests/ -q
```

当前共 **397 个用例**。覆盖范围：schema 校验、hdf5 writer（同步/异步）、async saver、preflight（夹爪/色彩）、record params、record loop、episode keyboard、unityvr mapping、vr align、unity vr reader、v21 转换器（meta/parquet/video/structure-diff/cli）、路径一致性、import 顺序守门、record_cfg yaml 解析等。

新增/修改纯逻辑代码时**必须**补充对应 `tests/test_*.py`，并保证 397（或新增后总数）全绿。

---

## 4. 新增功能模块示例

以新增一个录制超参为例（遵循"纯逻辑可单测 + CLI None 仅覆盖"范式）：

1. **纯函数实现** —— 在 `scripts/core/record_params.py` 加解析/校验函数，零 IO、零硬件依赖：
   ```python
   def resolve_xxx(cli_xxx, cfg_xxx):
       """单一来源: CLI 给了用 CLI(临时覆盖), 否则用 cfg(唯一真值)。"""
       v = cli_xxx if cli_xxx is not None else cfg_xxx  # 严格 is None，禁 cli or cfg
       if 非法:
           raise ValueError(f"xxx 非法: {v!r}")
       return v
   ```
2. **接线入口** —— 在 `run_record_hdf5.py` 的 `main()` 里调用，CLI 参数 `default=None`。
3. **补测试** —— 新建/扩充 `tests/test_record_params.py`，覆盖合法/非法/边界（含 yaml 引号字符串 `"false"` 等陷阱输入）。
4. **更新 yaml 注释** —— `record_cfg_unityvr.yaml` 内补字段 + 中文注释（注明"单一来源"/"CLI 仅覆盖"）。
5. **跑全量测试**确认无回归。

> 新增硬件依赖代码时，import 必须放在函数内（延迟 import），不放模块顶层。

---

## 5. schema 修改规范

`franka-hdf5-v1` 是**冻结契约**，多方并行依赖。修改 schema 时：

1. **bump 版本** —— 改 `franka_hdf5_schema.py` 的 `SCHEMA_VERSION`（如 `franka-hdf5-v2`）。
2. **同步消费方** —— 至少同步以下三处，缺一即产/读不一致：
   - `scripts/core/hdf5_writer.py::write_episode`（写出方）；
   - `franka_hdf5_schema.py::validate_episode`（校验方）；
   - `scripts/tools/hdf5_to_lerobot.py` 与 `hdf5_to_lerobot_v21.py`（两个转换器）+ `hdf5_lerobot_map.py`（映射）。
3. **同步 tests** —— 更新 `tests/test_franka_hdf5_schema.py`、`test_hdf5_writer*.py`、`test_hdf5_lerobot_map.py`、`test_v21_*.py`。
4. 校验 shape 用**精确匹配**、dtype 强制 `float64`（除图像）。新增 dataset 同时加入 `validate_episode` 必查项。

> `schema_loader.py` 复用 `sys.modules` 单实例缓存，所有消费方经它加载 schema，勿各自 `import franka_hdf5_schema`。

---

## 6. 服务启停与运维

三进程服务（臂 50051 / zerorpc 4242 / 夹爪 50052）的启停包装脚本在 `scripts/services/`，运维/诊断脚本在 `debug/`。

**一键有序重起**（推荐）：

```bash
bash debug/franka_clean_restart.sh   # 清理 → 起臂 → 验 → 起 zerorpc → 验
```

**分步**：

| 操作 | 脚本 |
|---|---|
| 清净残留 + 删锁 + 验空 | `debug/franka_cleanup.sh` |
| 起臂栈 / 轮询 Connected | `debug/franka_start_arm.sh` / `franka_poll_arm.sh` |
| 起 zerorpc 4242 / 5×RPC 验 | `debug/franka_start_zerorpc.sh` / `franka_verify_zerorpc.sh` |
| 起夹爪 / 判 FCI refused | `debug/franka_start_gripper.sh` / `franka_poll_gripper.sh` |
| 夹爪受控实测 | `debug/franka_gripper_verify.sh` / `_verify2.sh` / `_diag.sh` |
| 臂只读健康/快照 | `debug/franka_arm_health.sh` / `franka_arm_snapshot.sh` |

**启动时序**：先臂（50051）→ 再 zerorpc（4242）→ 再夹爪（50052）。`_run_polymetis_rw.sh` / `_run_zerorpc_iface.sh` 用 `flock` 防重复启动、`setsid` 独立进程组便于 `kill -- -PGID` 整组清理。

> 真机相关脚本中"仅观测、不发控制指令"的已在注释标注。涉及真机务必先做连接/读取验证。
> debug 脚本内绝对路径已按整合（jhli → jhli/lerobot_franka_teleop）回归修正（指向 `scripts/services/`），但仍请核对服务实际状态。

---

## 7. 打包与 console_scripts

`setup.py` 注册的 `console_scripts`（pip install 后可直接调用）：

| 命令 | 入口 |
|---|---|
| `franka-replay` | `scripts.core.run_replay:main` |
| `franka-visualize` | `scripts.core.run_visualize:main` |
| `franka-reset` | `scripts.core.reset_robot:main` |
| `franka-help` | `scripts.help.help_info:main` |
| `tools-check-dataset` | `scripts.tools.check_dataset_info:main` |
| `tools-check-rs` | `scripts.tools.rs_devices:main` |

> **注意**：Route B 录制入口（`run_record_hdf5.py` / `run_record_hdf5_ui.py`）与转换器（`hdf5_to_lerobot*.py`）未注册为 `console_scripts`，需用 `python scripts/core/run_record_hdf5.py ...` 直接调用。新增入口时在 `setup.py` 的 `entry_points` 里补一行并重装。

---

## 8. 调试陷阱

`docs/lessons/` 记录的踩坑教训，相关任务开始前**必读**：

| Lesson 文件 | 要点 |
|---|---|
| `2026-05-19-namespace-pkg-file-none-guard.md` | 命名空间包 `__file__=None` 是合法状态；§11.1 守门只断言子模块真文件 |
| `2026-05-19-rgb-bgr-encode-convention.md` | RGB/BGR 编码惯例不一致致录制图像反色（黄变青）。`_encode_jpg` 必须 RGB→BGR 再 `imencode`，下游 `_decode` 才 `imdecode`(BGR)→`cvtColor`(BGR2RGB) |
| `2026-05-20-phaseC-axis-gain-orthogonal-to-mapping.md` | per-axis 增益层与映射方向/手性正交，增益只缩放灵敏度，绝不改方向/手性（红线） |
| `2026-05-20-preflight-gripper-span-verdict.md` | 夹爪预检正解判据：用多目标 width 整体跨度 `max-min>0.02`，禁相邻差 `>0.01` 假阴性判据 |
| `2026-05-20-v21-loadability-franka2-probe.md` | v2.1 加载性 franka2 探测取舍 |
| `2026-05-22-flask-ui-no-cache-and-js-newline.md` | Flask 响应必须加 `Cache-Control: no-cache` 头（防 stale UI）；Python 三引号字符串内 JS `\n` 变真换行炸 SyntaxError，HTML 模板放外部文件规避 |

其他常见陷阱：

- **yaml 引号字符串误判**：`reset_between_episodes: "false"` 被 `bool()` 当 True → 用户以为关了仍执行真机 `robot.reset()`。`parse_reset_config` 已做严格 bool 解析，新增 bool 配置须照此处理。
- **CLI 覆盖用 `is None`**：禁 `cli or cfg`，否则 `0`/`""`/`False` 等 falsy 值被误判覆盖。
- **延迟 import**：硬件依赖（franka/lerobot 真包）放函数内 import，避免测试/CI 加载时崩。
- **deepcopy 时序**：异步保存的 payload 必须在 buffer 复用前 `deepcopy`，图像编码必须在 deepcopy 前完成。

---

## 9. 代码规范

- **语言**：注释、文档字符串、提交信息一律中文。
- **文档字符串**：每个模块/公开函数写中文 docstring，说明用途、参数、返回、副作用、线程模型与设计取舍。
- **fail-loud**：拒绝静默吞错。非法配置 `raise ValueError` 带可行动上下文；接线错让其在 import/调用处崩。
- **纯逻辑/硬件分层**：纯函数零 IO 零硬件、可单测；硬件依赖延迟 import。
- **单一真值**：fps、路径、reset 配置等只有一个来源（`resolve_record_fps` / `paths.py` / `RecordConfig`），杜绝多处不一致。
- **精确修改**：只触必须改的部分，不"顺手"重构无关代码。
- **测试先行**：纯逻辑改动配套 `tests/test_*.py`，全量 pytest 必须全绿才算完成。

---

*相关文档：[architecture.md](architecture.md)、[data-format.md](data-format.md)。*
