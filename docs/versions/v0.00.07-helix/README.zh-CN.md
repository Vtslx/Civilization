# v0.00.07 Helix —— 基线记录

| | |
|---|---|
| 版本 | `v0.00.07` |
| 代号 | Helix |
| Stage | 151–160 |
| 主题 | 可学习检索路径权重，含恢复与校准门 |
| 状态 | 已实现（当前线） |
| 模块 | `civilization.engine.stages`（stage151 … stage160） |
| 合同测试 | 10 个测试文件 |
| 实验 | [原生记忆配对 A/B](experiments/memory-on-off-ab.zh-CN.md) |

## 新增能力

- **路径权重模型。** 检索路径携带影响排序的权重（`stage151_helix_path_weight_model`）。
- **反馈信号。** 结果与反馈被收集为可支撑权重变更的信号（`stage152_helix_feedback_signal`）。
- **更新规则。** 规则从信号提出权重更新，而不是直接生效（`stage153_helix_weight_update_rule`）。
- **批准门控变更。** 拟议更新需批准后才改变检索行为（`stage154_helix_approval_gated_mutation`）。
- **加权检索。** 排序使用已批准的权重（`stage155_helix_weighted_retrieval`）。
- **冲突守卫、快照、恢复。** 权重冲突可检出，权重可快照，快照可恢复（`stage156`–`stage158`）。
- **校准门与发布门。** 校准检查与可复现的版本门（`stage159`、`stage160`）。

## 基线合同

- 学习到的权重在没有批准记录时绝不改变检索。
- 权重冲突被上报，并阻止静默覆盖。
- 快照恢复出与批准时完全一致的权重状态。
- 校准门与发布门可复现，并写出可审计摘要。

## 已记录的测量对照

本版本是整条线中唯一有留档测量的版本：在固定的 25 道合成题上做原生（Orion）记忆开/关配对 A/B，三轮，同模型同选项，只切换 `read_memory`。

| 指标 | 结果 |
|---|---|
| 可回答题，开启记忆 | 63/63（100%） |
| 可回答题，关闭记忆 | 17/63（27.0%） |
| 配对结果 | 46 对仅在开启记忆时答对；0 对仅在关闭时答对 |
| 接口错误 | 两种条件均为 0 |

完整设计、限制、支撑性基准背景与复现命令：[experiments/memory-on-off-ab.zh-CN.md](experiments/memory-on-off-ab.zh-CN.md)（另有[英文版](experiments/memory-on-off-ab.md)）。
留档数据：[data/memory-on-off-ab-recorded.json](experiments/data/memory-on-off-ab-recorded.json)。
复现脚本：[scripts/memory_ab.py](experiments/scripts/memory_ab.py)。

**引用该数字前请先读范围。** 它是 25 个唯一合成题在同一部署上与自身重复三轮的结果 —— 不是公共基准成绩，也不是普遍更优的主张。

## 本地验证

```bash
python -m pip install '.[test]'
pytest -q -k "stage15 or stage160"
```

## 边界

- 本次部署不含向量检索；排序使用代码实现的词法与策略信号。
- 检索权重在系统内学习并获批；不声称外部训练信号。
- 进程重启后的会话存储持久性、全局存储行为与自由对话质量不在已记录实验范围内。

[English](README.md) · [版本索引](../README.zh-CN.md)
