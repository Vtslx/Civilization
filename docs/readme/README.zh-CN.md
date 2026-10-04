# Civilization（简体中文）

Civilization 是一条研究线：把**记忆（memory）、状态（state）、规则（rule）**信号从
提示词里搬进语言模型的计算路径，并把结果沉淀为可审计的服务与稳定的 SDK。

v1 线把一个小型语言模型**冻结**，在其上训练 Civilization Adapter 与诊断读出。
记忆、状态、规则以结构化上下文进入模型，被投影进 hidden states，并且可以按路径做
消融与审计。在此机制之上，这条线叠加了多系统记忆架构、常驻推理服务、异步任务、
导出打包，以及面向应用的 SDK。

本仓库是 **v1 基线**：完整的第一代线，不掺杂后续研究线，也不包含内部开发文档。

> 其他语言：[English](../../README.md) · [繁體中文](README.zh-TW.md) ·
> [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) ·
> [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

## 仓库内容

| 路径 | 说明 |
|---|---|
| `src/civilization/research/prototype/` | 第一个可执行试验台：NumPy 实现的类 Transformer 核心，含 memory / state / rule 向量与融合的 CivilizationBlock。 |
| `src/civilization/research/torch_line/` | PyTorch 后端线：逻辑数据集、codebook、消融配置，以及后续所有阶段使用的训练与评测框架。 |
| `src/civilization/engine/` | 冻结底座线，Stage 44–160：hidden-state 基线、Civilization Adapter、memory/state/rule 路径训练，以及常驻服务链。 |
| `src/civilization/` | Python SDK：零依赖 HTTP 客户端、请求/预测/任务模型、与 provider 无关的 runtime 层，以及进程内服务宿主。 |
| `sdk/civilization-transformer/` | TypeScript SDK：零依赖客户端，覆盖决策、记忆、异步任务与导出包。 |
| `SDK.md`、`pyproject.toml` | `astreusn-civilization-v1` 的 Python 打包配置。 |

全仓库有 500 多个测试，覆盖从最早的规则门单元测试到服务、记忆与 SDK 合同。

## 架构概览

```
                 冻结的语言模型（权重从不更新）
                              │
   记忆 / 状态 / 规则 ──►  Civilization Adapter ──► hidden-state 读出
        信号                 （可训练部分）              │
                              │                        ▼
                              └──────────────►  决策读出（选项选择）
                                                     │
   ┌─────────────────────────────────────────────────┴──────────────────┐
   │  服务链：推理 → 访问控制 → 队列 → 异步任务 → 任务持久化 →           │
   │  结果存储 → 批量 → 导出 → 打包 → 流式分发 → Orion 多系统记忆        │
   └────────────────────────────────────────────────────────────────────┘
                              │
                  Python SDK / TypeScript SDK
```

两条设计原则贯穿全部实现：

1. **能力靠声明，不靠假设。** 服务通过 `GET /v1/capabilities` 声明其 runtime
   真正具备的能力。纯文本 provider 声明 `hidden_states: false` 与
   `adapter_execution: false`；只有经过审计的本地 adapter runtime 才会声明为
   `true`。
2. **决策后端是配置，不是代码。** 任意 OpenAI 兼容的 Chat Completions 端点、
   任意本地 Hugging Face 因果语言模型、以及本地经审计的 adapter 都可以承载生产流量。

| Runtime 类型 | 后端 | 与 provider 无关 | Hidden states | Adapter 执行 |
|---|---|---|---|---|
| `provider` | 任意 OpenAI 兼容 Chat Completions 端点 | 是 | 否 | 否 |
| `local_transformers` | 任意本地 Hugging Face 因果语言模型 | 是 | 否 | 否 |
| `local_adapter` | 固定本地权重上经审计的 Civilization Adapter | 是 | 是 | 是 |

## 版本线

| 版本 | 代号 | Stage | 主题 | 状态 |
|---|---|---|---|---|
| `v0.00.01` | Sun | 44–72 | 冻结底座推理、结构化 Adapter、可运行服务 | 已实现 |
| `v0.00.02` | Orion | 73–100 | 多系统记忆（工作 / 情景 / 语义 / 程序性） | 已实现 |
| `v0.00.03` | Trifid | 101–121 | 情景快速绑定、线索与时间消歧、受控 replay | 已实现 |
| `v0.00.04` | Lagoon | 122–132 | 跨情景 schema 巩固、来源链、冲突复核 | 已实现 |
| `v0.00.05` | Eagle | 133–138 | 任务轨迹与经批准的程序性记忆 | 已实现 |
| `v0.00.06` | Rosette | 139–150 | 类型化上下文包、冲突预算、多尺度蜂窝图谱 | 已实现 |
| `v0.00.07` | Helix | 151–160 | 可学习检索路径权重，含恢复与校准门 | 已实现 |
| `v0.00.08` | Crab | — | 记忆冲突检测、裁决与遗忘策略 | **未实现** |

发布标签不等于能力声明。v1 线实现到 `v0.00.07`；`v0.00.08`（Crab）只是计划，
本仓库没有任何代码实现它。

每个已实现版本都有一份**基线记录**（范围、新增能力、必须守住的合同不变量、验证命令与边界）：[docs/versions](../versions/README.zh-CN.md)。目前只有 `v0.00.07` Helix 有留档测量：[原生记忆配对 A/B](../versions/v0.00.07-helix/experiments/memory-on-off-ab.zh-CN.md)（同题同模型，开启/关闭记忆，46 对独赢、0 对独负）。

## 快速开始

### Python SDK

远程客户端只需要标准库：

```bash
python -m pip install .
```

```python
from civilization import CivilizationClient, CivilizationRequest

client = CivilizationClient("https://civilization.example.com", token_env="CIVILIZATION_API_TOKEN")
prediction = client.predict(
    CivilizationRequest(
        text="Choose the deployment action supported by the verified evidence.",
        answer_options=("approve", "reject"),
        session_id="deployment-42",
        task_name="deployment_decision",
        memory_items=("The deployment signature and health checks are valid.",),
        rule_items=("Reject only when a critical verification is unresolved.",),
        state_values=(0.9, 0.1, 0.8),
    )
)
print(prediction.option_id, prediction.option_text, prediction.memory_trace)
```

同一个客户端还提供 Orion 记忆操作、异步任务、导出、包分发、健康检查、就绪检查、
指标与 runtime 能力查询。

### 在进程内承载服务

```bash
python -m pip install '.[embedded]'
```

```python
from civilization import EmbeddedCivilization, EmbeddedConfig

service = EmbeddedCivilization(
    EmbeddedConfig(
        runtime="provider",
        provider_base_url="https://provider.example.com/v1",
        provider_model="any-chat-model",
        provider_api_key_env="PROVIDER_API_KEY",
    )
)
with service:
    client = service.start()
    print(client.ready(), service.capabilities.to_dict())
```

把 `runtime` 切换为 `local_transformers`（任意本地 Hugging Face 因果语言模型）或
`local_adapter`（经审计的 adapter 路径），服务链本身无需改动。用
`register_runtime(RuntimeKind(...))` 可以注册新的后端。

### TypeScript SDK

```bash
cd sdk/civilization-transformer
npm install && npm run build && npm test
```

```ts
import { CivilizationClient } from "@astreusn/civilization-transformer";

const client = new CivilizationClient("https://civilization.example.com", {
  tokenEnv: "CIVILIZATION_API_TOKEN",
});

const capabilities = await client.capabilities();
console.log(capabilities.capabilities?.kind); // "provider" | "local_transformers" | "local_adapter"

const prediction = await client.predict({
  text: "Choose the supported operation.",
  answerOptions: ["approve", "reject"],
  sessionId: "deployment-42",
});
```

## 目录结构

```text
.
├── src/civilization/                     已安装的包
│   ├── __init__.py                       公开 API
│   ├── client.py                         零依赖 HTTP 客户端
│   ├── models.py                         请求 / 预测 / 任务模型
│   ├── runtimes.py                       runtime 类型、能力声明、注册表
│   ├── embedded.py                       进程内服务宿主
│   ├── cli.py                            `civilization` 命令行
│   ├── engine/                           决策引擎（需要 engine extras）
│   │   ├── model_paths.py                可选本地 checkpoint 解析
│   │   ├── adapter/                      可训练的 Civilization Adapter
│   │   ├── backend/                      本地 / provider / adapter runtime
│   │   └── stages/                       版本化服务链，Stage 44–160
│   └── research/                         早期研究线，仅为溯源保留
│       ├── prototype/                    最早的 NumPy 试验台
│       └── torch_line/                   PyTorch 后端线
├── tests/                                engine / torch_line / prototype 测试
├── examples/                             可运行示例
├── docs/versions/                        每个版本的基线记录与实验
├── docs/readme/                          本 README 的八种语言版本
├── sdk/civilization-transformer/         TypeScript SDK
├── SDK.md                                Python SDK 指南
└── pyproject.toml                        astreusn-civilization-v1 打包配置
```

## 测试

```bash
python -m pip install '.[test]'
pytest
```

大部分测试不需要任何模型权重：服务链、记忆策略、任务与导出合同、两个 SDK 都可以
基于 fake runtime 或远程 runtime 运行。

需要**本地 Qwen3-0.6B checkpoint** 的测试在模型缺失时会自动跳过。该 checkpoint
不随仓库分发；请自行获取，放到 `Models/Qwen3-0.6B`，或用
`CIVILIZATION_MODEL_PATH` 指向它：

```bash
CIVILIZATION_MODEL_PATH=/path/to/Qwen3-0.6B pytest
```

各阶段运行脚本把输出写到 `experiments/*/artifacts/`，该目录已被 git 忽略；
仓库中没有任何代码依赖已提交的实验输出。

## 范围与声明边界

本仓库**是**：

- v1 线的一个版本化实现：冻结底座推理、结构化 adapter、多系统记忆、服务链与 SDK。
- 一个不变量经过测试的系统：fail-closed 输入校验、访问控制、任务持久化与恢复、
  导出完整性（下载经 SHA-256 校验）、按路径消融审计。

本仓库**不声称**：

- 它不是新的基础模型。v1 线中底座语言模型始终冻结；可训练部分是 Adapter 及其读出。
- 不声称终身学习、通用长期记忆、自主事实裁决或人类水平泛化。
- 不声称生物等价。记忆设计中的脑区名称只是工程标签。
- 不声称纯文本 provider 会执行 adapter 或暴露 hidden states；能力端点存在正是为了
  避免这种误称。
- 经审计的 adapter 路径只对固定 checkpoint 与记录的包产物有效；其他本地模型走纯文本路径。

## 许可证

Apache License 2.0，见 [LICENSE](../../LICENSE) 与 [NOTICE](../../NOTICE)。
