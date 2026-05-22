# 文档索引

Franka 数采系统（Route B）文档体系。

## 文档清单

| 文档 | 用途 |
|---|---|
| [architecture.md](architecture.md) | **系统架构与原理**：系统概述、仓库目录结构、三进程服务架构（含拓扑图）、数据采集全链路数据流、`scripts/core` 模块详解、Route B 遥操方案、预检门与异步保存、配置系统、关键设计决策。 |
| [development-guide.md](development-guide.md) | **开发说明**：环境搭建、代码组织、运行测试（pytest 317 用例）、新增功能模块示例、schema 修改规范、服务启停运维、打包 console_scripts、调试陷阱、代码规范。 |
| [data-format.md](data-format.md) | **数据格式说明**：`franka-hdf5-v1` schema 完整契约（每个 group/dataset 的 shape/dtype/含义）、`validate_episode` 校验内容、State/Action 维度约定、hdf5→LeRobot v3.0/v2.1 转换流程与差异。 |
| [lessons/](lessons/) | **踩坑教训**：开发中沉淀的经验教训，相关任务开始前必读。 |

## 建议阅读顺序

1. **快速上手**：先看仓库根 [../README.md](../README.md)，跑通"启动服务 → 录一条 → 转 LeRobot → 可视化"。
2. **理解系统**：读 [architecture.md](architecture.md)，建立对三进程架构与全链路数据流的整体认识。
3. **理解数据**：读 [data-format.md](data-format.md)，掌握 `franka-hdf5-v1` schema 与 LeRobot 转换。
4. **动手开发**：读 [development-guide.md](development-guide.md)，按规范扩展功能、改 schema、跑测试。
5. **避坑**：动手前查阅 [lessons/](lessons/) 中与任务相关的教训。
