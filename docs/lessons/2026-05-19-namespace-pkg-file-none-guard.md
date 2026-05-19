# 命名空间包 __file__=None 是合法状态；§11.1 守门只断言子模块真文件

日期：2026-05-19　关联：§11.1 路径打包重构 Task2/Task3；
docs/lessons/2026-05-19-namespace-dir-shadows-editable-install.md

## 现象
§11.1 Task2 写守门测试时，从 repo 根 cwd（标准门控 `cd <repo> && pytest`，
`_probe` 子进程 `python -c` 继承 cwd 并把 repo 根注入 sys.path[0]）执行
`import lerobot_teleoperator_franka as p` → `p.__file__ is None`、
`p.__path__=['<repo>/lerobot_teleoperator_franka']`。对 `pathlib.Path(p.__file__)`
断言即 `TypeError: expected str ... not NoneType`，疑似 §11.1 遮蔽 bug 复现。

## 根因与正解（勿重走）
repo 根存在与 editable 包同名的**外层目录**（无 __init__.py）。repo 根上 path 时，
顶层 `import lerobot_teleoperator_franka` 解析为**命名空间包**——`__file__=None`
是命名空间包的**合法状态，不是生产 bug**。关键：命名空间包 `__path__` 是动态
`_NamespacePath`，**子模块导入仍正确解析到 editable 真包**（旧弱测试只 import
子模块 vr_align 并判 `a.__file__`，repo-根 cwd 下一直 63 passed 即证）。

§11.1 真正根因是子模块解析错位（`cannot import name FrankaConfig`），故守门要守
**子模块解析到双层嵌套 editable 真包内文件**，**不是**顶层命名空间包要有 __file__：
- 真包判别签名：子模块 __file__ 父目录名 == 其再上一级目录名 ==
  `lerobot_teleoperator_franka`（双层嵌套 `<repo>/lerobot_teleoperator_franka/
  lerobot_teleoperator_franka/`）；repo-根单层遮蔽副本父目录是 repo 根（名不为
  lerobot_teleoperator_franka），据此区分。
- 守门测试只 import/断言子模块（vr_align/unity_vr_reader/unityvr_robot）的真实
  __file__ 同目录 + 双层嵌套签名 + `m.vr_align.__name__`/`UnityVRReader.__module__`
  为包内模块；绝不断言顶层包 __file__。

实现：tests/test_import_order_guard.py 的
test_vr_modules_importable_from_teleop_package（Task2 守门③）/
test_editable_pkg_resolves_even_if_repo_root_on_path（Task3 守门②）即此口径。

## 一句话
顶层命名空间包 `__file__=None` 合法 ≠ 遮蔽 bug；§11.1 守门断言子模块真文件
+双层嵌套签名，不碰顶层 __file__；改这块前先读本篇。
