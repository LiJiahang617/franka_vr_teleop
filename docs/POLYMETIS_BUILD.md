# polymetis-local 编译指引

本项目依赖 fairo-franka 的定制版 polymetis（含 hw_timestamp + 其他修改），需要从源码编译，
约 10 分钟（C++ + Python）。polymetis 在独立的 conda env (`polymetis-local`) 中运行，
通过 zerorpc 与本项目 (`franka-teleop` env) 通信，因此两边 env 分开。

## 前置依赖

- Ubuntu 22.04（推荐 PREEMPT_RT 实时内核，普通内核也能跑）
- gcc 12+ / g++ 12+
- cmake 3.20+
- conda (miniconda3 或 anaconda3)
- libfranka 0.20.x（系统级 .deb 安装；Franka FR3 + libfranka 0.20 组合验证过）
- Franka 机器人 FCI 已激活、用户已 `unlock joints` 并切到 `Programming` 模式

## 编译步骤

### 1. clone fairo-franka 源码

```bash
cd ~
git clone https://github.com/<your-fork>/fairo-franka.git
cd fairo-franka/polymetis/polymetis
```

> 用 fork 是因为 upstream Meta fairo-franka 已停止维护，本项目用的版本含
> `hw_timestamp` 硬件时间戳 + `setLoad` 暴露等定制。

### 2. 创建独立 conda env

```bash
conda create -n polymetis-local python=3.8 -y
conda activate polymetis-local
pip install -r requirements.txt
```

Python 3.8 是 polymetis upstream 要求，不要升级。

### 3. 编译 C++ 部分

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DBUILD_FRANKA=ON
make -j8
```

输出 `franka_panda_client` / `franka_hand_client` 等二进制位于 `build/`，
Python 绑定 `.so` 会自动 link 到下一步的 Python 包里。

### 4. 安装 Python 包

```bash
cd ../python
pip install -e .
```

### 5. 验证安装

```bash
python -c "from polymetis import RobotInterface, GripperInterface; print('OK')"
```

输出 `OK` 表示编译成功。

## 启动 polymetis server（每次开机后）

```bash
conda activate polymetis-local
cd ~/fairo-franka/polymetis/polymetis

# 启动 Panda arm server
launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=<FRANKA_IP>

# 另开 terminal 启动 hand server
launch_gripper.py gripper=franka_hand gripper.executable_cfg.robot_ip=<FRANKA_IP>
```

本项目的录制脚本 (`scripts/core/run_record_hdf5_ui.py`) 通过 zerorpc 连接 server。

## 常见故障

- **`libfranka not found` / `Cannot find libfranka.so`**
  ```bash
  sudo apt install libfranka  # 或手动安装 libfranka_0.20.3_jammy_amd64.deb
  ```

- **`cpu_dma_latency permission denied`**
  ```bash
  sudo chmod a+rw /dev/cpu_dma_latency
  ```

- **gcc 版本过老 (报 C++17/20 feature 缺失)**
  ```bash
  conda install -c conda-forge gxx_linux-64=12
  ```

- **`enforce_version` mismatch (libfranka 0.20 vs polymetis 默认期望 0.9)**
  本项目用 `enforce_version=False` 默认 (在 `robot_interface.py` 已定制)，
  如果 upstream RobotInterface 报版本错，确认你用的是本项目对应的 fork。

- **homing 后夹爪不动 / RPC refused**
  见 `docs/lessons/franka-gripper-fci-refused-vs-arm-ok.md`：
  断电后必须在 Desk 网页里手动 `homing` 一次，FCI profile 迁移过的话需重新绑定。

## 修改点（vs upstream Meta fairo-franka）

- `polymetis/polymetis/src/clients/franka_hand_client.cpp`：
  - 新增 `ENABLE_GRIPPER_HW_TIMESTAMP` 宏，开启后夹爪 state 带硬件时间戳
  - 暴露 `setLoad` 接口（默认 upstream 没绑 Python）

- `polymetis/polymetis/python/polymetis/robot_interface.py`：
  - `enforce_version=False` 改为默认值，兼容 libfranka 0.20.x
  - 增加 `get_robot_state_with_timestamp()` 辅助方法

具体 diff 见 fork 仓库的 `vs-upstream` 分支。
