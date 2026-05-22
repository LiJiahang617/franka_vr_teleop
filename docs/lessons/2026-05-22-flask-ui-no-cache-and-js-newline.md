# Flask UI 两条红线：no-cache 响应头 + Python 三引号内 JS `\n` 陷阱

**日期**: 2026-05-22
**阶段**: Phase E 数采 Web UI
**回链**: 计划 `docs/superpowers/plans/2026-05-20-phaseE-ui.md` §公共约定 2/3

---

## 红线 1：Flask 响应必须加 Cache-Control no-cache 头

### 症状

用户浏览器访问 UI 后，后端 HTML/JSON 已更新，但浏览器仍展示旧版状态。
换浏览器或无痕模式可能仍无效——因为 HTTP 中间代理或浏览器自身做了 heuristic caching。

### 根因

Flask 默认不加任何 `Cache-Control` 头。浏览器与代理按 RFC 7234 做 heuristic caching：
若响应无 `Cache-Control`/`Expires`/`Pragma`，浏览器自行估算 freshness lifetime（通常几小时）。
对频繁变化的 UI HTML、轮询接口 `/api/status`、相机预览 `/api/preview/*` 来说，这会造成 stale UI。

### 正解

在 Flask app 中用 `@app.after_request` 钩子统一给**所有响应**加三行头：

```python
@app.after_request
def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp
```

前端 `fetch` 也双保险加 `{cache: 'no-store'}`：

```js
fetch('/api/status', {cache: 'no-store'})
```

### 守门测试

每个 HTTP 响应路由都应有专测断言 `Cache-Control: no-cache, no-store, must-revalidate` 存在：

```python
r = client.get("/api/status")
assert "no-store" in r.headers.get("Cache-Control", "")
```

### 禁止

- 单路由单独加头（漏一个就破红线）——必须走 `after_request` 钩子统一加
- 仅前端 `no-store` 而忽略 HTTP 响应头（CDN/代理侧仍可能缓存）

---

## 红线 2：Python 三引号字符串里的 JS `\n` 陷阱

### 症状

页面加载完全无响应：状态徽章不刷新、按钮点击无任何反应、相机预览空白。
F12 Console 显示 `Uncaught SyntaxError: Invalid or unexpected token`，定位到 `<script>` 块。

### 根因

Python 三引号字符串（`"""..."""`）在渲染时将 `\n` 解释为真换行符（ASCII 0x0A）。
若 JS 代码中有多行字符串字面量：

```python
html = """
<script>
var msg = "line1\nline2";   // Python 把 \n 变成真换行！
</script>
"""
```

渲染结果变成：

```js
var msg = "line1
line2";  // JS 字符串不能含真换行，语法错误
```

整个 `<script>` 块解析失败，后续所有 JS 代码不执行。

### 正解（本项目方案）

**将 HTML 模板放外部文件**（`scripts/ui/templates/control_panel.html`），Flask 用 `render_template` 加载：

```python
app = Flask(__name__, template_folder=_TPL_DIR)

@app.route("/")
def _index():
    return render_template("control_panel.html", ...)
```

外部文件里的 JS 字面量按 JS 语法正常书写（无 Python 三引号干扰）。

### 若必须内嵌 HTML（备选）

将 JS 字符串中所有 `\n` 改为 `\\n`（4 字符：反斜杠+n），或拆成数组 join：

```python
# 改前（炸 JS）
js_str = "line1\nline2"
# 改后（安全）
js_str = "line1\\nline2"
```

### 守门测试

用 `test_client` 断言响应 HTML 含关键 JS 标志，确认模板被正确渲染且无语法错误：

```python
r = app.test_client().get("/")
body = r.data.decode()
assert "setInterval" in body  # 轮询脚本存在
assert "\\n" not in body.split("<script>")[1]  # 无裸真换行在 JS 字符串中
```

---

## 关联

- lesson `2026-05-04-flask-no-cache-stale-ui.md`（本地工作站同名文件，franka2 补落盘）
- lesson `2026-05-04-python-triple-quote-js-newline-trap.md`（本地工作站同名文件，franka2 补落盘）
- 计划 `docs/superpowers/plans/2026-05-20-phaseE-ui.md` §公共约定 2/3（本计划两条红线）
- `scripts/ui/control_panel.py`（`@app.after_request _no_cache` 实现）
- `scripts/ui/templates/control_panel.html`（外部模板，规避三引号陷阱）
