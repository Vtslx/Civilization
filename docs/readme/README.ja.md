# Civilization（日本語）

Civilization は、**記憶（memory）・状態（state）・ルール（rule）**の信号をプロンプト
から取り出し、言語モデルの計算経路そのものへ移す研究ラインです。その結果を、監査可能な
サービスと安定した SDK としてバージョン管理します。

v1 ラインでは小さな言語モデルを**凍結**し、その上に Civilization Adapter と診断用の
読み出しを学習します。記憶・状態・ルールは構造化コンテキストとして入力され、hidden
states に射影され、パス単位でアブレーションと監査ができます。その機構の上に、マルチ
システム記憶、常駐推論サービス、非同期ジョブ、エクスポートパッケージ、アプリケーション
向け SDK を積み上げています。

このリポジトリは **v1 ベースライン**です。第一世代ラインの完全な実装であり、後続の
研究ラインや社内開発ドキュメントは含みません。

> 他の言語：[English](../../README.md) · [简体中文](README.zh-CN.md) ·
> [繁體中文](README.zh-TW.md) · [한국어](README.ko.md) · [Español](README.es.md) ·
> [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

## リポジトリの内容

| パス | 内容 |
|---|---|
| `src/civilization/research/prototype/` | 最初の実行可能なテストベンチ。NumPy による Transformer 風コアと、memory / state / rule ベクトル、統合された CivilizationBlock。 |
| `src/civilization/research/torch_line/` | PyTorch バックエンドライン。論理データセット、codebook、アブレーション設定、以降の全ステージが使う学習・評価ハーネス。 |
| `src/civilization/engine/` | 凍結ベースライン、Stage 44–160。hidden-state ベースライン、Civilization Adapter、memory/state/rule パス学習、常駐サービスチェーン。 |
| `src/civilization/` | Python SDK。依存ゼロの HTTP クライアント、リクエスト／予測／ジョブのモデル、provider 非依存の runtime 層、プロセス内サービスホスト。 |
| `sdk/civilization-transformer/` | TypeScript SDK。決定・記憶・ジョブ・エクスポートを扱う依存ゼロのクライアント。 |
| `SDK.md`、`pyproject.toml` | `astreusn-civilization-v1` の Python パッケージング。 |

リポジトリ全体で 500 を超えるテストがあり、初期のルールゲート単体テストからサービス・
記憶・SDK の契約までをカバーしています。

## アーキテクチャ概観

```
                 凍結された言語モデル（重みは更新しない）
                              │
   記憶 / 状態 / ルール ──►  Civilization Adapter ──► hidden-state 読み出し
        信号                  （学習される部分）           │
                              │                          ▼
                              └────────────►  決定読み出し（選択肢の選択）
                                                     │
   ┌─────────────────────────────────────────────────┴──────────────────┐
   │  サービスチェーン：推論 → アクセス制御 → キュー → 非同期ジョブ →    │
   │  ジョブ永続化 → 結果ストア → バッチ → エクスポート → パッケージ →   │
   │  ストリーミング配信 → Orion マルチシステム記憶                      │
   └────────────────────────────────────────────────────────────────────┘
                              │
                  Python SDK / TypeScript SDK
```

設計原則は 2 つです。

1. **能力は宣言し、仮定しない。** サービスは `GET /v1/capabilities` で runtime が
   実際にできることを宣言します。テキスト専用 provider は
   `hidden_states: false` と `adapter_execution: false` を返し、監査済みのローカル
   adapter runtime だけが `true` を返します。
2. **決定バックエンドは設定であり、コードではない。** OpenAI 互換の Chat
   Completions エンドポイント、任意のローカル Hugging Face 因果言語モデル、監査済み
   ローカル adapter のいずれでも本番トラフィックを処理できます。

| Runtime 種別 | バックエンド | provider 非依存 | Hidden states | Adapter 実行 |
|---|---|---|---|---|
| `provider` | OpenAI 互換 Chat Completions エンドポイント | はい | いいえ | いいえ |
| `local_transformers` | 任意のローカル Hugging Face 因果言語モデル | はい | いいえ | いいえ |
| `local_adapter` | 固定ローカル重み上の監査済み Civilization Adapter | はい | はい | はい |

## バージョンライン

| バージョン | コード名 | Stage | テーマ | 状態 |
|---|---|---|---|---|
| `v0.00.01` | Sun | 44–72 | 凍結ベース推論、構造化 Adapter、実行可能なサービス | 実装済み |
| `v0.00.02` | Orion | 73–100 | マルチシステム記憶（作業／エピソード／意味／手続き） | 実装済み |
| `v0.00.03` | Trifid | 101–121 | エピソード高速結合、手がかり・時間の曖昧性解消、統制された replay | 実装済み |
| `v0.00.04` | Lagoon | 122–132 | 横断エピソードの schema 統合、来歴、衝突レビュー | 実装済み |
| `v0.00.05` | Eagle | 133–138 | タスクトレースと承認済み手続き記憶 | 実装済み |
| `v0.00.06` | Rosette | 139–150 | 型付きコンテキストパケット、衝突予算、マルチスケールハニカムグラフ | 実装済み |
| `v0.00.07` | Helix | 151–160 | 学習可能な検索パス重み（回復・校正ゲート付き） | 実装済み |
| `v0.00.08` | Crab | — | 記憶衝突の検出・裁定・忘却ポリシー | **未実装** |

リリースラベルは能力の主張ではありません。v1 ラインは `v0.00.07` まで実装されており、
`v0.00.08`（Crab）は計画のみで、このリポジトリに実装コードはありません。

実装済みの各バージョンには**ベースライン記録**（範囲、追加された能力、守るべき契約不変条件、検証コマンド、境界）があります：[docs/versions](../versions/README.md)。記録された測定があるのは現在 `v0.00.07` Helix のみで、[ネイティブ記憶のペア A/B](../versions/v0.00.07-helix/experiments/memory-on-off-ab.md)（同一問題・同一モデル、46 ペアの勝利・0 ペアの敗北）です。

## クイックスタート

### Python SDK

リモートクライアントは標準ライブラリだけで動作します。

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

同じクライアントで Orion 記憶操作、非同期ジョブ、エクスポート、パッケージ配信、
ヘルスチェック、レディネス、メトリクス、runtime 能力の照会ができます。

### サービスをプロセス内でホストする

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

`runtime` を `local_transformers`（ローカルの Hugging Face 因果言語モデル）や
`local_adapter`（監査済み adapter パス）に切り替えても、サービスチェーンは変更不要です。
`register_runtime(RuntimeKind(...))` で新しいバックエンドを登録できます。

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

## リポジトリ構成

```text
.
├── src/civilization/                     インストールされるパッケージ
│   ├── __init__.py                       公開 API
│   ├── client.py                         依存ゼロ HTTP クライアント
│   ├── models.py                         リクエスト / 予測 / ジョブモデル
│   ├── runtimes.py                       runtime 種別・能力・レジストリ
│   ├── embedded.py                       プロセス内サービスホスト
│   ├── cli.py                            `civilization` コマンド
│   ├── engine/                           決定エンジン（engine extras が必要）
│   │   ├── model_paths.py                任意のローカルチェックポイント解決
│   │   ├── adapter/                      学習される Civilization Adapter
│   │   ├── backend/                      ローカル / provider / adapter runtime
│   │   └── stages/                       バージョン化されたサービスチェーン、Stage 44–160
│   └── research/                         以前の研究ライン（来歴として保持）
│       ├── prototype/                    最初の NumPy テストベンチ
│       └── torch_line/                   PyTorch バックエンドライン
├── tests/                                engine / torch_line / prototype のテスト
├── examples/                             実行可能な例
├── docs/versions/                        バージョンごとのベースライン記録と実験
├── docs/readme/                          この README の 8 言語版
├── sdk/civilization-transformer/         TypeScript SDK
├── SDK.md                                Python SDK ガイド
└── pyproject.toml                        astreusn-civilization-v1 のパッケージング
```

## テスト

```bash
python -m pip install '.[test]'
pytest
```

多くのテストはモデル重みを必要としません。サービスチェーン、記憶ポリシー、ジョブ／
エクスポート契約、両 SDK は fake runtime またはリモート runtime で実行できます。

**ローカル Qwen3-0.6B チェックポイント**が必要なテストは、モデルが無い場合に自動的に
スキップされます。チェックポイントはリポジトリに含まれません。各自で取得し、
`Models/Qwen3-0.6B` に置くか `CIVILIZATION_MODEL_PATH` で指定してください。

```bash
CIVILIZATION_MODEL_PATH=/path/to/Qwen3-0.6B pytest
```

ステージ実行スクリプトの出力は `experiments/*/artifacts/` に書き出され、git 管理外です。
リポジトリ内にコミット済みの実験出力に依存するコードはありません。

## 範囲と主張の境界

本リポジトリが**であるもの**：

- v1 ラインのバージョン管理された実装（凍結ベース推論、構造化 adapter、マルチシステム
  記憶、サービスチェーン、SDK）。
- 不変条件がテストされたシステム（fail-closed 入力検証、アクセス制御、ジョブ永続化と
  回復、SHA-256 で検証されるエクスポート、パス単位のアブレーション監査）。

本リポジトリが**主張しないもの**：

- 新しい基盤モデルではない。v1 ラインのベース言語モデルは凍結され、学習されるのは
  Adapter とその読み出しです。
- 生涯学習、汎用的な長期記憶、自律的な事実裁定、人間水準の汎化を主張しません。
- 生物学的等価性を主張しません。記憶設計の脳領域名は工学的ラベルです。
- テキスト専用 provider が adapter を実行したり hidden states を露出したりするとは
  主張しません。能力エンドポイントはまさにその誤称を避けるために存在します。
- 監査済み adapter パスは固定チェックポイントと記録されたパッケージ成果物に対して
  検証済みです。他のローカルモデルはテキスト専用パスを使います。

## ライセンス

Apache License 2.0。[LICENSE](../../LICENSE) と [NOTICE](../../NOTICE) を参照。
