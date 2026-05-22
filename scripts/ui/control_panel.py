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
from flask import Flask, jsonify


def build_app(controller=None) -> Flask:
    """创建并返回配置好的 Flask 实例。

    Args:
        controller: RecorderController 实例（后续 Task 接入），
                    None 时只做静态路由，便于 TDD 离线测试。

    Returns:
        配置好 after_request cache 头和基础路由的 Flask app。
    """
    app = Flask(__name__)
    # 保存 controller 引用，后续路由（Task 2-6）通过 app.controller 访问
    app.controller = controller  # type: ignore[attr-defined]

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

    return app
