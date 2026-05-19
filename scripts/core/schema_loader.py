"""加载仓库根 loose 模块 franka_hdf5_schema 的单一入口。

整合期 franka_hdf5_schema.py 仍是仓库根散装模块(未包化, 由 validator/转换器/
writer 多方消费; 包化属后续 schema 工作, 不在 §11.1 范围)。本 helper 集中其
加载: 复用 sys.modules 单一实例(还原旧 `import franka_hdf5_schema` 缓存语义),
exec 前注册并在失败时回滚, 用 __file__ 相对定位避免 sys.path 污染/硬编码。
"""
import importlib.util
import sys
from pathlib import Path

_SCHEMA_NAME = "franka_hdf5_schema"
# 本文件在 <repo>/scripts/core/ ; 仓库根 = parents[2]
_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "franka_hdf5_schema.py"


def load_franka_hdf5_schema():
    """返回 franka_hdf5_schema 模块, 全进程单一实例(同 import 缓存语义)。"""
    cached = sys.modules.get(_SCHEMA_NAME)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(_SCHEMA_NAME, str(_SCHEMA_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_SCHEMA_NAME] = mod  # exec 前注册(importlib 规范, 防自引用/重复实例)
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(_SCHEMA_NAME, None)  # 回滚, 不留半初始化模块
        raise
    return mod
