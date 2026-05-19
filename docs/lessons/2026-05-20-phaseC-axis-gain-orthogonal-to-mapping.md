# 教训：per-axis 增益层与映射方向/手性正交（§11.3 / Phase C）

日期：2026-05-20　场景：Franka 数采 Route B — §11.3 每轴运动灵敏度可调实现

## 核心结论

**per-axis 增益是纯标量逐轴缩放，与 `_POS_MAP`/`R_cal` 映射方向/手性完全正交（互不干扰）。**

增益不需要、也绝不应该进入任何映射矩阵/换基公式内部。若感觉"要改映射才能调灵敏度"，说明理解有误，应立即停下报告。

## 实现关键：增益只加在已验通映射输出之后

`compute_delta_action` 内部结构（`unityvr_mapping.py:15-44`）：

```
# §10.2(0) 红线区域（一字节不改）
d[:3] = _POS_MAP @ d_pos_oc          # :38  位置换基（极矢量）
d[3:] = R_cal @ d_rot_oc             # :39  旋转换基（赝矢量，刚体共轭）

# §11.3 增益层（在红线输出之后追加，与上方公式正交）
pg = np.asarray(pos_axis_gain, float)  # :42 前置
rg = np.asarray(rot_axis_gain, float)  # :42 前置
d[:3] = d[:3] * ps * pg * s[:3]        # :42  全局标量 * 每轴增益 * 手性符号
d[3:] = d[3:] * os_ * rg * s[3:]       # :43  全局标量 * 每轴增益 * 手性符号
```

增益 `pg/rg` 的位置：在 `_POS_MAP@`、`R_cal@` 之后，在 `channel_signs` 之前（与 signs 同行）。  
此顺序保证：方向/手性由 `§10.2(0)` 决定，灵敏度幅度由 `pose_scaler × axis_gain` 决定，两层责任清晰分离。

## 两层 fail-loud 防御

**第一层（运行时，`compute_delta_action` 内）**：  
- `shape` 校验：`pos_axis_gain`/`rot_axis_gain` 必须 broadcast 为 3 元素数组，否则 numpy 广播报错立即暴露。  
- `finite` 校验：增益含 NaN/Inf 将导致 delta action 全 NaN，下游控制器会拒绝或产生危险运动。  
- 建议在生产调用点断言 `np.isfinite(pg).all() and pg.shape == (3,)`。

**第二层（config-load 时，`parse_axis_gain` 纯函数）**：  
- `RecordConfig.__init__` 调用 `parse_axis_gain(uvr_cfg.get("pos_axis_gain", [1.0, 1.0, 1.0]))` 在程序启动时即拦截畸形 yaml（非 3 元素/含非数/含 0）。  
- fail-loud 在 config 加载阶段，远早于真机运动，防止 yaml 拼错的增益值喂到真机。

两层防御合力：yaml 畸形 → config-load 即拦；代码调用错 → 运行时即报；真机不会收到 NaN/异形增益动作。

## 向后兼容：无新键等价历史行为

`compute_delta_action` 新参为 **keyword-only 且带默认 `(1.,1.,1.)`**：
- 现有 5 位置参调用（`unityvr_robot.py:56-58`）与所有既有 7 测试均无感知。
- yaml 不写 `pos_axis_gain`/`rot_axis_gain` 时，`RecordConfig` 的 `cfg.get(键, [1.0,1.0,1.0])` 回退 → 增益全 1.0 → `d[:3]*1.0*[1,1,1]*signs` 逐字等价历史 `d[:3]*ps*signs`。
- **无新 cfg 键 = 行为零变化**，这是 Phase C 所有 RecordConfig 扩展的统一范式。

## pose_scaler 与 axis_gain 的关系

- `pose_scaler = [pos_scale, rot_scale]`：全局两标量（位置/姿态整体缩放），既有旋钮，yaml 已有。
- `axis_gain = [gx, gy, gz]`（3 元素）：每轴独立微调，叠在 pose_scaler 之后。
- 建议调参策略：固定 pose_scaler 不变，仅调 axis_gain 做每轴平衡；避免两者同时改导致语义混乱。
- yaml 注释已明确："全局[位置,姿态]标量; 每轴微调用 axis_gain"。

## 真机三轴调试 DEFERRED

真机灵敏度调参由用户现场执行，见 HANDOFF §6.5 的 6 条单行命令清单。  
调参原则：急停在手，churn≤2 次不行即停报告；任一轴出现方向反/镜像 = 增益层之外的问题（增益不改方向），立即停止。

## 回链

- [kabsch 不能吸收手性翻转的教训](2026-05-18-kabsch-cannot-absorb-handedness.md) — 说明为何 §10.2(0) 映射方向/手性是红线；增益层 ⟂ 映射方向正是吸取此教训的设计：灵敏度调整绝不触碰方向决策层。
- spec §10.2(0)：Route-B 映射方向/手性红线（`_POS_MAP`/`R_cal`/`d_rot_oc` 零改动）。
- spec §11.3：per-axis 增益需求与设计决策。

## 通用规则

1. **增益层 ≠ 映射层**：灵敏度调整（正标量缩放）与方向/手性决策（矩阵乘法/坐标系变换）在代码中应物理分离，不共行。
2. **fail-loud 两层**：config-load 时（parse_axis_gain）+ 运行时（shape/finite 校验）；不依赖"用户不会填错"。
3. **keyword-only 带默认 = 向后兼容的正确范式**：给已有函数追加可选参数时，keyword-only+默认值同时满足：现有调用无感、新调用可选择性使用。
4. **两参数叠乘语义要在 yaml 注释中明确**：防用户同时改两个旋钮方向相反互相抵消。
