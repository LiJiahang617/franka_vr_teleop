# Phase E 数采 Web UI 真机验收清单

> **状态**: DEFERRED — 由用户现场执行（急停在手，churn≤2 不行即停报告）。
>
> **前置条件**: 三进程服务已起（臂 50051 → zerorpc 4242 → 夹爪 50052），
> 每路服务日志均出现 `Connected.`。
> 以下命令均经 `ssh franka2` 在远端执行（或直接在 xlab-2 终端运行，去掉 `ssh franka2` 前缀）。

---

## 1. 配置启用 UI 模式

```bash
# 确认 cfg 中 ui.enabled 为 true（手动编辑或用下行命令）
ssh franka2 "grep -A5 'ui:' /home/ubuntu/Desktop/jhli/lerobot_franka_teleop/scripts/config/record_cfg_unityvr.yaml"
```

**期望结果**: 看到 `enabled: true`。若为 `false` 则手动编辑改为 `true`。

---

## 2. 启动 UI 入口（前台运行，便于观察日志）

```bash
ssh -t franka2 "cd /home/ubuntu/Desktop/jhli/lerobot_franka_teleop && /home/ubuntu/Desktop/jhli/envs/franka-teleop/bin/python scripts/core/run_record_hdf5_ui.py --config scripts/config/record_cfg_unityvr.yaml"
```

**期望结果**:
- 预检门通过（夹爪 span>0.02m，色彩预检通过）
- 终端出现 `Running on http://0.0.0.0:5055`（或配置的端口）
- 无 `sys.exit(2)` 中断

---

## 3. 浏览器访问控制面板

在用户本机浏览器输入：`http://<franka2-IP>:5055`（当前 IP: `172.16.204.235`）

**期望结果**:
- 页面加载，显示 5 个按钮：**开始录制、结束并保存、结束并丢弃、回 Home、停止**
- 两路相机（腕部 / 外部）预览图像可见（约 30Hz 刷新）
- 状态徽章显示当前状态（如 `waiting`）
- 日志区域滚动显示最新日志

---

## 4. 验证 Cache-Control 响应头（防 stale UI）

在浏览器开发者工具 → Network 面板，任选一个请求（如 `/api/status` 或 `/api/preview/wrist_image`）→ Response Headers 查看：

**期望结果**: 必须同时出现以下三行：
```
Cache-Control: no-cache, no-store, must-revalidate
Pragma: no-cache
Expires: 0
```

若缺少上述头，说明破红线，立即停止并报告。

---

## 5. happy path：一轮完整录制（保存分支）

按以下顺序操作：

```
1. 点「开始录制」
   → 期望：状态变为 recording，计时器/帧计数开始滚动
2. VR 控制器遥操 Franka 执行任务（几秒）
3. 点「结束并保存」
   → 期望：状态变为 confirming → saving → ready → waiting
            日志出现 ep0000 写盘完成，_hdf5_episodes/ 下多一个 .h5 文件
```

验证落盘：

```bash
ssh franka2 "ls -lh /path/to/out_dir/_hdf5_episodes/"
```

**期望结果**: 可见 `ep0000_<timestamp>.h5`，大小 > 0。

---

## 6. 丢弃分支

```
1. 点「开始录制」→ 状态变 recording
2. 操作几秒
3. 点「结束并丢弃」
   → 期望：状态回 waiting，当前 ep 不产生 .h5 文件
```

验证：

```bash
ssh franka2 "ls /path/to/out_dir/_hdf5_episodes/ | wc -l"
```

**期望结果**: 文件数量与上一步保存后相同（未增加）。

---

## 7. 回 Home

在 waiting 状态下点「回 Home」：

**期望结果**: 机械臂回到 HOME_JOINT_POSITION，夹爪打开，日志出现 `home cmd done`。

---

## 8. 多条 episode 连续录制

重复步骤 5（保存分支）3 次：

**期望结果**: `_hdf5_episodes/` 下有 ep0000/ep0001/ep0002 三个 .h5 文件；
每条录制期间相机预览持续显示，状态徽章正确变色。

---

## 9. 优雅退出

在终端按 `Ctrl-C`（或点「停止」按钮后再 Ctrl-C）：

**期望结果**:
- 后台录制线程 stop + join（日志出现相关信息）
- AsyncEpisodeSaver 排空（`saver.close()` 完成）
- robot / teleop disconnect
- 进程正常退出（无悬挂线程、无残留 partial .h5 文件）

---

## 10. 不通过时的处理

任一步骤不通过，**立即停止，不靠重启硬试**，记录：

- 现象：哪一步、哪个按钮、终端报了什么错
- 日志：`ssh franka2 "journalctl --user -n 50"` 或终端直接复制
- 特别关注：
  - Cache-Control 头丢失 → Flask `@after_request` 未生效
  - 预览空白 → `frame_observer` hook 未接通或编码报错
  - 录制不响应按钮 → events dict 写入失败或状态机非法转移
  - 进程不退出 → 后台线程未 join（`wait_until_done` 超时）

---

*生成时间: 2026-05-22 | Phase E 离线完成 (T1-T8，398 passed)*
*真机操作 DEFERRED 交用户现场，急停在手，churn≤2 不行即停报告。*
