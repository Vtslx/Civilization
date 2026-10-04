# v0.00.04 Lagoon —— 基线记录

| | |
|---|---|
| 版本 | `v0.00.04` |
| 代号 | Lagoon |
| Stage | 122–132 |
| 主题 | 跨情景 schema 巩固、来源链、冲突复核 |
| 状态 | 已实现 |
| 模块 | `civilization.engine.stages`（stage122 … stage132） |
| 合同测试 | 11 个测试文件 |

## 新增能力

- **巩固聚类。** 在提出任何 schema 之前先对情景聚类，使巩固作用于成组情景而非单条（`stage122_lagoon_consolidation_clusters`）。
- **schema 候选。** 从聚类构造候选抽象，候选在获批前保持候选状态（`stage123_lagoon_schema_candidates`）。
- **经批准的巩固。** 只有获批候选才成为语义 schema 记忆（`stage124_lagoon_approved_consolidation`）。
- **schema 检索与来源链。** 对已批准 schema 的语义检索，以及把 schema 追溯回其来源情景的来源查询（`stage125`、`stage126`）。
- **schema 冲突处理。** schema 冲突守卫、冲突感知召回、冲突复核与冲突裁决，使矛盾 schema 被暴露而不是被合并掉（`stage127`–`stage130`）。
- **按裁决召回。** 召回尊重已记录的冲突裁决（`stage131_lagoon_decision_aware_recall`）。
- **发布门。** 重放整个版本合同并输出具名布尔检查的门（`stage132_lagoon_release_gate`）。

## 基线合同

- 巩固出的 schema 始终有来源链：可追溯到产生它的情景。
- 提升需要批准；未经复核的候选不会进入语义记忆。
- 冲突 schema 被保留以供复核 —— 证据不明确时，系统不会静默挑一个赢家。
- 发布门可复现，并写出可审计摘要。

## 本地验证

```bash
python -m pip install '.[test]'
pytest -q -k "stage12 or stage13"
```

## 边界

- schema 从显式情景巩固而来；不声称从任意文本自动抽取事实。
- 冲突裁决是被记录的决策，不是"发现的真相"；本版本不声称自主事实裁决。
- 本版本不附带准确率主张 —— 它是合同基线。

## 测量

本版本无留档测量。Lagoon 的对照为合同级。

[English](README.md) · [版本索引](../README.zh-CN.md)
