# v0.00.05 Eagle —— 基线记录

| | |
|---|---|
| 版本 | `v0.00.05` |
| 代号 | Eagle |
| Stage | 133–138 |
| 主题 | 任务轨迹与经批准的程序性记忆 |
| 状态 | 已实现 |
| 模块 | `civilization.engine.stages`（stage133 … stage138） |
| 合同测试 | 6 个测试文件 |

## 新增能力

- **任务轨迹。** 执行轨迹被记录为一等公民（`stage133_eagle_task_trace`）。
- **技能候选。** 重复轨迹产生候选流程，而不是直接生成技能（`stage134_eagle_skill_candidates`）。
- **经批准的技能。** 候选只有获批后才成为程序性记忆（`stage135_eagle_approved_skill`）。
- **技能检索。** 已批准流程可被检索并用于后续任务（`stage136_eagle_skill_retrieval`）。
- **技能冲突守卫。** 相互矛盾的流程会被检出，而不是同时被套用（`stage137_eagle_skill_conflict_guard`）。
- **发布门。** 覆盖本版本合同的可复现门（`stage138_eagle_release_gate`）。

## 基线合同

- 技能始终可追溯到已记录的任务轨迹。
- 未获批候选不会进入程序性记忆。
- 冲突技能被上报，而不是静默解决。
- 发布门可复现，并写出可审计摘要。

## 本地验证

```bash
python -m pip install '.[test]'
pytest -q -k "stage13"
```

## 边界

- 流程来自本系统中的显式轨迹；本版本不声称从开放式交互中自主学会技能。
- 本版本不附带准确率主张 —— 它是合同基线。

## 测量

本版本无留档测量。Eagle 的对照为合同级。

[English](README.md) · [版本索引](../README.zh-CN.md)
