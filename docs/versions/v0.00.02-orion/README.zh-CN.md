# v0.00.02 Orion —— 基线记录

| | |
|---|---|
| 版本 | `v0.00.02` |
| 代号 | Orion |
| Stage | 73–100 |
| 主题 | 多系统记忆（工作 / 情景 / 语义 / 程序性） |
| 状态 | 已实现 |
| 模块 | `experiments/civilization_transformer_qwen3/analysis/stage73…stage99` |
| 合同测试 | 19 个测试文件 |

## 新增能力

- **记忆内核。** 四系统存储（工作、情景、语义、程序性），单元带重要度、置信度、过期时间与类型化链接（`stage73_orion_memory_kernel`）。
- **记忆感知推理。** 记忆服务在预测前检索并把选中的单元注入普通预测上下文，响应返回含检索与注入单元 id 的 Orion trace（`stage74_orion_memory_service`）；adapter 上下文桥把同样的单元送往结构化 adapter 路径（`stage75`）。
- **巩固与提升。** 基于 replay 的巩固与经批准的语义提升，使重复出现的情景可以转为语义记忆（`stage76`、`stage78`）。
- **全局记忆。** 跨会话检索、全局维护、可持久化的全局存储及其恢复、生命周期管理与存储医生（`stage79`、`stage82`、`stage91`–`stage96`）。
- **冲突处理。** 冲突感知检索、冲突裁决与裁决轨迹，使相互矛盾的单元可以被暴露和决策，而不是被静默覆盖（`stage83`–`stage85`）。
- **策略与评测。** 任务记忆策略、多会话评测、全局证据基准，以及带回归检查的保留策略（`stage86`、`stage89`、`stage90`、`stage98`、`stage99`）。

## 基线合同

- 记忆写入经过校验，每个单元只属于一个记忆系统，且来源可审计。
- 检索默认限定在会话内；跨会话读取只经全局路径，隔离会话之间不会泄漏。
- 巩固与提升需经批准：replay 得到的内容不会静默提升为语义记忆。
- 全局存储可保存、重载与恢复；保留策略不会静默删除被标记为必须保留的单元。

## 本地验证

```bash
python -m pip install '.[test]'
pytest experiments/civilization_transformer_qwen3/tests -q -k "stage7 or stage8 or stage9"
```

## 边界

- 本版本的记忆是显式的：单元通过 API 写入，不声称从自由文本自动抽取。
- 本版本不含向量检索；排序使用代码实现的词法与策略信号。
- 本版本不附带准确率主张 —— 它是合同基线。

## 测量

本版本无留档测量。Orion 的对照为合同级。

[English](README.md) · [版本索引](../README.zh-CN.md)
