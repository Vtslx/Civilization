# v0.00.01 Sun —— 基线记录

| | |
|---|---|
| 版本 | `v0.00.01` |
| 代号 | Sun |
| Stage | 44–72 |
| 主题 | 冻结底座推理、结构化 Adapter、可运行服务 |
| 状态 | 已实现 |
| 模块 | `experiments/civilization_transformer_qwen3/analysis/stage45…stage72` |
| 合同测试 | 32 个测试文件（位于 `experiments/civilization_transformer_qwen3/tests`） |

## 新增能力

- **结构化 Adapter。** 在冻结底座模型之上训练 Civilization Adapter，具备可审计的包格式（`stage45_adapter_package`）与执行该包的运行时（`stage46_runtime_inference`）。
- **批量与生产推理。** 质心批量推理与基于 JSONL 的生产推理路径（`stage47`、`stage48`）。
- **常驻服务。** 可持久化的 HTTP 推理服务，含运行时加载、就绪检查、指标与带版本的运行时重载（`stage49`–`stage58`），并有重载、并发与长时运行的压测。
- **访问控制与排队。** Bearer 令牌访问控制与审计日志，以及队列与限流（`stage59`、`stage60`）。
- **异步任务。** 任务提交、持久化、恢复、保留与列表，结果存放于任务记录之外（`stage61`–`stage64`）。
- **导出与分发。** 批量任务、导出创建、导出生命周期、包生成、流式分发、HTTP Range 分发、`If-Range` 续传，以及可续传的下载客户端（`stage65`–`stage72`）。

## 基线合同

- 推理只走经过验证的生产控制路径；请求携带显式控制模式，审计链记录该模式。
- 服务如实上报就绪状态：运行时未加载前 `/ready` 失败；指标暴露请求、失败与延迟计数。
- 任务在服务重启后仍然存在，可按策略列出、分页、保留或清理，不会静默丢数据。
- 导出包按内容校验：下载经 SHA-256 验证，中断的传输续传而不是重来。

## 本地验证

```bash
python -m pip install '.[test]'
pytest experiments/civilization_transformer_qwen3/tests -q -k "stage4 or stage5 or stage6 or stage7"
```

## 边界

- 底座模型始终冻结；本版本增加的是训练好的 adapter 与服务，不是新的基础模型。
- 本版本不声称服务进程之外的记忆持久性；多系统记忆在 `v0.00.02` 引入。
- 本版本不附带准确率主张 —— 它是合同基线。

## 测量

本版本无留档测量。Sun 的对照为合同级。

[English](README.md) · [版本索引](../README.zh-CN.md)
