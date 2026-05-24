
## 自跑测试后必清 (2026-05-24 lesson)

主控/agent 自跑 main (run_record_hdf5_ui.py) 测试完, **必须 pkill 进程释放相机/zerorpc 资源**.
否则用户启 main 会撞 RealSense 'Device or resource busy'.

自测模板:
```bash
# 1. 启 main (后台)
setsid <python> run_record_hdf5_ui.py ... > /tmp/ui_selftest.log 2>&1 < /dev/null & MAIN_PID=$!

# 2. 测试 (curl /api/status, /api/vr_enable 等)
...

# 3. 必须清理 (放在 trap 或测试末尾)
curl -s -X POST http://localhost:5055/api/stop > /dev/null ; sleep 5
pgrep -af run_record_hdf5_ui && pkill -f run_record_hdf5_ui ; sleep 2
```
