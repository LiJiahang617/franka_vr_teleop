"""
Flask app 工厂函数。

红线（spec §3.4 + lessons）：
1. 所有响应必须经 @app.after_request 加 Cache-Control: no-cache, no-store, must-revalidate
   + Pragma: no-cache + Expires: 0
   （lesson 2026-05-04-flask-no-cache-stale-ui: 浏览器 heuristic cache 导致 stale UI，
   单路由忘加=破红线）
2. HTML 模板内 JS 字符串中的换行必须用 \\\\n，禁止 Python 三引号字面量内的真换行流入 JS
   （lesson 2026-05-04-python-triple-quote-js-newline-trap）
"""
import os
import importlib.util

from flask import Flask, jsonify

# 负载标定占位路由的引导文案（修订 B）
_PAYLOAD_CALIB_GUIDANCE = (
    "当前版本负载标定为扩展位。"
    "请在 Franka Desk 的负载标定向导中完成末端负载（质量/质心/惯量）设置。"
    "后续若 franka_interface_server 暴露 set_load 接口，将接入真实功能。"
)

# 动态加载同包 preview 模块（与其他模块保持一致的加载方式）
_UI_DIR = os.path.dirname(os.path.abspath(__file__))
_preview_spec = importlib.util.spec_from_file_location(
    "ui_preview", os.path.join(_UI_DIR, "preview.py")
)
_preview_mod = importlib.util.module_from_spec(_preview_spec)
_preview_spec.loader.exec_module(_preview_mod)
encode_preview_base64 = _preview_mod.encode_preview_base64


def build_app(controller=None) -> Flask:
    """创建并返回配置好的 Flask 实例。

    Args:
        controller: RecorderController 实例（后续 Task 接入），
                    None 时控制类路由退化为 503，便于 TDD 离线测试。

    Returns:
        配置好 after_request cache 头和基础路由的 Flask app。
    """
    app = Flask(__name__)

    def _require_controller():
        """controller=None 时返回 503 JSON，否则返回 None（表示 controller 可用）。"""
        if controller is None:
            return jsonify({"ok": False, "error": "controller unavailable"}), 503
        return None

    @app.after_request
    def _no_cache(resp):
        """所有响应加 no-cache 头，防止浏览器/中间代理缓存 stale UI（红线）。"""
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    @app.route("/api/ping")
    def _ping():
        """烟测路由，验证 Flask 实例正常工作。"""
        return jsonify({"ok": True})

    # ---------- Task 2：录制控制路由 ----------

    @app.route("/api/start", methods=["POST"])
    def _api_start():
        """请求开始录制。命令入队 'start'，不直接调机器人（守坑 7）。"""
        err = _require_controller()
        if err is not None:
            return err
        ok = controller.start_recording()
        if not ok:
            return jsonify({"ok": False, "error": "command queue full"}), 503
        return jsonify({"ok": True})

    @app.route("/api/save", methods=["POST"])
    def _api_save():
        """保存当前 episode（等价键盘 → keep）。写 exit_early=True。"""
        err = _require_controller()
        if err is not None:
            return err
        controller.save_episode()
        return jsonify({"ok": True})

    @app.route("/api/discard", methods=["POST"])
    def _api_discard():
        """丢弃当前 episode（等价键盘 ← discard）。写 rerecord+exit_early=True。"""
        err = _require_controller()
        if err is not None:
            return err
        controller.discard_episode()
        return jsonify({"ok": True})

    @app.route("/api/stop", methods=["POST"])
    def _api_stop():
        """停止整个录制会话（等价键盘 Esc stop）。写 stop_recording+exit_early=True。"""
        err = _require_controller()
        if err is not None:
            return err
        controller.stop_recording()
        return jsonify({"ok": True})

    @app.route("/api/home", methods=["POST"])
    def _api_home():
        """请求机械臂回 Home。命令入队 'home'，不直接调机器人（守坑 7）。"""
        err = _require_controller()
        if err is not None:
            return err
        ok = controller.go_home()
        if not ok:
            return jsonify({"ok": False, "error": "command queue full"}), 503
        return jsonify({"ok": True})

    @app.route("/api/status", methods=["GET"])
    def _api_status():
        """返回录制器当前状态快照（JSON）。"""
        err = _require_controller()
        if err is not None:
            return err
        return jsonify(controller.status_snapshot())

    # ---------- 修订 B：负载标定占位路由 ----------

    @app.route("/api/payload-calib", methods=["POST"])
    def _api_payload_calib():
        """负载标定占位路由（扩展位，无副作用）。

        当前不调任何机器人接口，直接返回引导文案。
        不写 events dict，不写命令队列。
        后续若 franka_interface_server 暴露 set_load 接口再接入真实功能。
        """
        return jsonify({
            "ok": True,
            "supported": False,
            "guidance": _PAYLOAD_CALIB_GUIDANCE,
        })

    # ---------- Task 3：相机预览路由 ----------

    @app.route("/api/preview/<string:cam>", methods=["GET"])
    def _api_preview(cam):
        """返回指定相机最新帧的 base64 jpeg data-url（JSON）。

        从 controller.get_latest_frame(cam) 取帧，编码为 ≤320×240 jpeg q60。
        - controller=None → 503（_require_controller 统一处理）
        - 无帧（帧缓存为空）→ 404
        - 有帧 → 200 JSON {"cam": cam, "data_url": "data:image/jpeg;base64,..."}

        不直接调 robot.get_observation()，守坑 7（帧由录制器主循环 hook 写入缓存）。
        """
        err = _require_controller()
        if err is not None:
            return err
        arr = controller.get_latest_frame(cam)
        if arr is None:
            return jsonify({"cam": cam, "error": "no_frame"}), 404
        data_url = encode_preview_base64(arr, max_w=320, max_h=240, quality=60)
        return jsonify({"cam": cam, "data_url": data_url})

    return app
