def main():
    print("""
==================================================
 Franka Teleoperation Utilities - Command Reference
==================================================

录制命令（直接调用脚本）：
  python run_record_hdf5.py     终端模式录制（hdf5-v1 格式）
  python run_record_hdf5_ui.py  Web UI 模式录制

数据工具：
  franka-replay           回放已录制的数据集
  franka-visualize        可视化已录制的数据集
  franka-reset            将机械臂复位到初始姿态

辅助工具：
  tools-check-dataset     查看本地数据集信息
  tools-check-rs          获取已连接 RealSense 相机序列号

--------------------------------------------------
 提示：随时使用 'franka-help' 查看本摘要。
==================================================
""")
