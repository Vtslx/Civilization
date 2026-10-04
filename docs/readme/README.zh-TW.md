# Civilization（繁體中文）

Civilization 是一條研究線：把**記憶（memory）、狀態（state）、規則（rule）**訊號
從提示詞搬進語言模型的運算路徑，並把成果沉澱為可審計的服務與穩定的 SDK。

v1 線把一個小型語言模型**凍結**，在其上訓練 Civilization Adapter 與診斷讀出。
記憶、狀態、規則以結構化上下文進入模型，被投影進 hidden states，並且可以按路徑
做消融與審計。在此機制之上，這條線疊加了多系統記憶架構、常駐推論服務、非同步
任務、匯出打包，以及面向應用的 SDK。

本倉庫是 **v1 基線**：完整的第一代線，不混入後續研究線，也不包含內部開發文件。

> 其他語言：[English](../../README.md) · [简体中文](README.zh-CN.md) ·
> [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) ·
> [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

## 倉庫內容

| 路徑 | 說明 |
|---|---|
| `src/civilization/research/prototype/` | 第一個可執行試驗台：NumPy 實作的類 Transformer 核心，含 memory / state / rule 向量與融合的 CivilizationBlock。 |
| `src/civilization/research/torch_line/` | PyTorch 後端線：邏輯資料集、codebook、消融設定，以及後續各階段使用的訓練與評測框架。 |
| `src/civilization/engine/` | 凍結底座線，Stage 44–160：hidden-state 基線、Civilization Adapter、memory/state/rule 路徑訓練，以及常駐服務鏈。 |
| `src/civilization/` | Python SDK：零依賴 HTTP 客戶端、請求/預測/任務模型、與 provider 無關的 runtime 層，以及行程內服務宿主。 |
| `sdk/civilization-transformer/` | TypeScript SDK：零依賴客戶端，涵蓋決策、記憶、非同步任務與匯出包。 |
| `SDK.md`、`pyproject.toml` | `astreusn-civilization-v1` 的 Python 打包設定。 |

全倉庫有 500 多個測試，涵蓋從最早的規則閘單元測試到服務、記憶與 SDK 合約。

## 架構概覽

```
                 凍結的語言模型（權重從不更新）
                              │
   記憶 / 狀態 / 規則 ──►  Civilization Adapter ──► hidden-state 讀出
        訊號                 （可訓練部分）              │
                              │                        ▼
                              └──────────────►  決策讀出（選項選擇）
                                                     │
   ┌─────────────────────────────────────────────────┴──────────────────┐
   │  服務鏈：推論 → 存取控制 → 佇列 → 非同步任務 → 任務持久化 →         │
   │  結果儲存 → 批次 → 匯出 → 打包 → 串流分發 → Orion 多系統記憶        │
   └────────────────────────────────────────────────────────────────────┘
                              │
                  Python SDK / TypeScript SDK
```

兩條設計原則貫穿全部實作：

1. **能力靠宣告，不靠假設。** 服務透過 `GET /v1/capabilities` 宣告其 runtime
   真正具備的能力。純文字 provider 宣告 `hidden_states: false` 與
   `adapter_execution: false`；只有經審計的本地 adapter runtime 才宣告為 `true`。
2. **決策後端是設定，不是程式碼。** 任意 OpenAI 相容的 Chat Completions 端點、
   任意本地 Hugging Face 因果語言模型、以及本地經審計的 adapter 都可承載生產流量。

| Runtime 類型 | 後端 | 與 provider 無關 | Hidden states | Adapter 執行 |
|---|---|---|---|---|
| `provider` | 任意 OpenAI 相容 Chat Completions 端點 | 是 | 否 | 否 |
| `local_transformers` | 任意本地 Hugging Face 因果語言模型 | 是 | 否 | 否 |
| `local_adapter` | 固定本地權重上經審計的 Civilization Adapter | 是 | 是 | 是 |

## 版本線

| 版本 | 代號 | Stage | 主題 | 狀態 |
|---|---|---|---|---|
| `v0.00.01` | Sun | 44–72 | 凍結底座推論、結構化 Adapter、可執行服務 | 已實作 |
| `v0.00.02` | Orion | 73–100 | 多系統記憶（工作 / 情景 / 語意 / 程序性） | 已實作 |
| `v0.00.03` | Trifid | 101–121 | 情景快速綁定、線索與時間消歧、受控 replay | 已實作 |
| `v0.00.04` | Lagoon | 122–132 | 跨情景 schema 鞏固、來源鏈、衝突複核 | 已實作 |
| `v0.00.05` | Eagle | 133–138 | 任務軌跡與經核准的程序性記憶 | 已實作 |
| `v0.00.06` | Rosette | 139–150 | 型別化上下文包、衝突預算、多尺度蜂窩圖譜 | 已實作 |
| `v0.00.07` | Helix | 151–160 | 可學習檢索路徑權重，含回復與校準閘 | 已實作 |
| `v0.00.08` | Crab | — | 記憶衝突偵測、裁決與遺忘策略 | **未實作** |

發佈標籤不等於能力宣告。v1 線實作到 `v0.00.07`；`v0.00.08`（Crab）只是計畫，
本倉庫沒有任何程式碼實作它。

每個已實作版本都有一份**基線記錄**（範圍、新增能力、必須守住的合約不變量、驗證指令與邊界）：[docs/versions](../versions/README.zh-CN.md)。目前只有 `v0.00.07` Helix 有留檔測量：[原生記憶配對 A/B](../versions/v0.00.07-helix/experiments/memory-on-off-ab.zh-CN.md)（同題同模型，開啟/關閉記憶，46 對獨贏、0 對獨負）。

## 快速開始

### Python SDK

遠端客戶端只需要標準庫：

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

同一個客戶端還提供 Orion 記憶操作、非同步任務、匯出、包分發、健康檢查、就緒檢查、
指標與 runtime 能力查詢。

### 在行程內承載服務

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

把 `runtime` 切換為 `local_transformers`（任意本地 Hugging Face 因果語言模型）或
`local_adapter`（經審計的 adapter 路徑），服務鏈本身無需改動。用
`register_runtime(RuntimeKind(...))` 可註冊新的後端。

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

## 目錄結構

```text
.
├── src/civilization/                     已安裝的套件
│   ├── __init__.py                       公開 API
│   ├── client.py                         零依賴 HTTP 客戶端
│   ├── models.py                         請求 / 預測 / 任務模型
│   ├── runtimes.py                       runtime 類型、能力宣告、註冊表
│   ├── embedded.py                       行程內服務宿主
│   ├── cli.py                            `civilization` 命令列
│   ├── engine/                           決策引擎（需要 engine extras）
│   │   ├── model_paths.py                可選本地 checkpoint 解析
│   │   ├── adapter/                      可訓練的 Civilization Adapter
│   │   ├── backend/                      本地 / provider / adapter runtime
│   │   └── stages/                       版本化服務鏈，Stage 44–160
│   └── research/                         早期研究線，僅為溯源保留
│       ├── prototype/                    最早的 NumPy 試驗台
│       └── torch_line/                   PyTorch 後端線
├── tests/                                engine / torch_line / prototype 測試
├── examples/                             可執行範例
├── docs/versions/                        每個版本的基線記錄與實驗
├── docs/readme/                          本 README 的八種語言版本
├── sdk/civilization-transformer/         TypeScript SDK
├── SDK.md                                Python SDK 指南
└── pyproject.toml                        astreusn-civilization-v1 打包設定
```

## 測試

```bash
python -m pip install '.[test]'
pytest
```

大部分測試不需要任何模型權重：服務鏈、記憶策略、任務與匯出合約、兩個 SDK 都可以
基於 fake runtime 或遠端 runtime 執行。

需要**本地 Qwen3-0.6B checkpoint** 的測試在模型缺失時會自動跳過。該 checkpoint
不隨倉庫分發；請自行取得，放到 `Models/Qwen3-0.6B`，或用
`CIVILIZATION_MODEL_PATH` 指向它：

```bash
CIVILIZATION_MODEL_PATH=/path/to/Qwen3-0.6B pytest
```

各階段執行腳本把輸出寫到 `experiments/*/artifacts/`，該目錄已被 git 忽略；
倉庫中沒有任何程式碼依賴已提交的實驗輸出。

## 範圍與宣告邊界

本倉庫**是**：

- v1 線的一個版本化實作：凍結底座推論、結構化 adapter、多系統記憶、服務鏈與 SDK。
- 一個不變量經過測試的系統：fail-closed 輸入驗證、存取控制、任務持久化與回復、
  匯出完整性（下載經 SHA-256 驗證）、按路徑消融審計。

本倉庫**不宣稱**：

- 它不是新的基礎模型。v1 線中底座語言模型始終凍結；可訓練部分是 Adapter 及其讀出。
- 不宣稱終身學習、通用長期記憶、自主事實裁決或人類水準泛化。
- 不宣稱生物等價。記憶設計中的腦區名稱只是工程標籤。
- 不宣稱純文字 provider 會執行 adapter 或暴露 hidden states；能力端點存在正是為了
  避免這種誤稱。
- 經審計的 adapter 路徑只對固定 checkpoint 與記錄的包產物有效；其他本地模型走純文字路徑。

## 授權條款

Apache License 2.0，見 [LICENSE](../../LICENSE) 與 [NOTICE](../../NOTICE)。
