# Civilization（한국어）

Civilization은 **기억(memory) · 상태(state) · 규칙(rule)** 신호를 프롬프트에서 꺼내
언어 모델의 계산 경로로 옮기는 연구 라인입니다. 그 결과를 감사 가능한 서비스와 안정적인
SDK로 버전 관리합니다.

v1 라인은 작은 언어 모델을 **동결**하고 그 위에 Civilization Adapter와 진단용 읽기를
학습합니다. 기억 · 상태 · 규칙은 구조화된 컨텍스트로 입력되어 hidden states로 투영되며,
경로별로 소거(ablation)와 감사가 가능합니다. 그 위에 다중 시스템 기억, 상주 추론 서비스,
비동기 작업, 내보내기 패키지, 애플리케이션 SDK를 쌓았습니다.

이 저장소는 **v1 베이스라인**입니다. 1세대 라인의 완전한 구현이며, 이후 연구 라인이나
내부 개발 문서는 포함하지 않습니다.

> 다른 언어: [English](../../README.md) · [简体中文](README.zh-CN.md) ·
> [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [Español](README.es.md) ·
> [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `src/civilization/research/prototype/` | 최초의 실행 가능한 테스트 벤치. NumPy 기반 Transformer 유사 코어, memory / state / rule 벡터, 결합된 CivilizationBlock. |
| `src/civilization/research/torch_line/` | PyTorch 백엔드 라인. 논리 데이터셋, codebook, 소거 설정, 이후 모든 단계가 쓰는 학습·평가 하네스. |
| `src/civilization/engine/` | 동결 베이스 라인, Stage 44–160. hidden-state 베이스라인, Civilization Adapter, memory/state/rule 경로 학습, 상주 서비스 체인. |
| `src/civilization/` | Python SDK. 의존성 없는 HTTP 클라이언트, 요청/예측/작업 모델, provider 비종속 runtime 계층, 프로세스 내 서비스 호스트. |
| `sdk/civilization-transformer/` | TypeScript SDK. 결정·기억·작업·내보내기를 다루는 의존성 없는 클라이언트. |
| `SDK.md`, `pyproject.toml` | `astreusn-civilization-v1` Python 패키징. |

저장소 전체에 500개가 넘는 테스트가 있어 초기 규칙 게이트 단위 테스트부터 서비스·기억·SDK
계약까지 포괄합니다.

## 아키텍처 개요

```
                 동결된 언어 모델 (가중치는 갱신되지 않음)
                              │
   기억 / 상태 / 규칙 ──►  Civilization Adapter ──► hidden-state 읽기
        신호                 (학습되는 부분)             │
                              │                        ▼
                              └────────────►  결정 읽기 (선택지 선택)
                                                     │
   ┌─────────────────────────────────────────────────┴──────────────────┐
   │  서비스 체인: 추론 → 접근 제어 → 큐 → 비동기 작업 → 작업 영속화 →   │
   │  결과 저장소 → 배치 → 내보내기 → 패키징 → 스트리밍 전달 →           │
   │  Orion 다중 시스템 기억                                            │
   └────────────────────────────────────────────────────────────────────┘
                              │
                  Python SDK / TypeScript SDK
```

두 가지 설계 원칙이 전체를 관통합니다.

1. **능력은 선언하고 가정하지 않는다.** 서비스는 `GET /v1/capabilities`로 runtime이
   실제로 할 수 있는 일을 선언합니다. 텍스트 전용 provider는 `hidden_states: false`와
   `adapter_execution: false`를 반환하고, 감사된 로컬 adapter runtime만 `true`를 반환합니다.
2. **결정 백엔드는 코드가 아니라 설정이다.** OpenAI 호환 Chat Completions 엔드포인트,
   임의의 로컬 Hugging Face 인과 언어 모델, 감사된 로컬 adapter 모두 프로덕션 트래픽을
   처리할 수 있습니다.

| Runtime 종류 | 백엔드 | provider 비종속 | Hidden states | Adapter 실행 |
|---|---|---|---|---|
| `provider` | OpenAI 호환 Chat Completions 엔드포인트 | 예 | 아니오 | 아니오 |
| `local_transformers` | 임의의 로컬 Hugging Face 인과 언어 모델 | 예 | 아니오 | 아니오 |
| `local_adapter` | 고정 로컬 가중치 위의 감사된 Civilization Adapter | 예 | 예 | 예 |

## 버전 라인

| 버전 | 코드명 | Stage | 주제 | 상태 |
|---|---|---|---|---|
| `v0.00.01` | Sun | 44–72 | 동결 베이스 추론, 구조화 Adapter, 실행 가능한 서비스 | 구현됨 |
| `v0.00.02` | Orion | 73–100 | 다중 시스템 기억(작업/일화/의미/절차) | 구현됨 |
| `v0.00.03` | Trifid | 101–121 | 일화 고속 결합, 단서·시간 중의성 해소, 통제된 replay | 구현됨 |
| `v0.00.04` | Lagoon | 122–132 | 일화 간 schema 통합, 출처, 충돌 검토 | 구현됨 |
| `v0.00.05` | Eagle | 133–138 | 작업 추적과 승인된 절차 기억 | 구현됨 |
| `v0.00.06` | Rosette | 139–150 | 타입화 컨텍스트 패킷, 충돌 예산, 다중 스케일 허니컴 그래프 | 구현됨 |
| `v0.00.07` | Helix | 151–160 | 학습 가능한 검색 경로 가중치(복구·보정 게이트 포함) | 구현됨 |
| `v0.00.08` | Crab | — | 기억 충돌 감지·중재·망각 정책 | **미구현** |

릴리스 라벨은 능력 주장이 아닙니다. v1 라인은 `v0.00.07`까지 구현되어 있고,
`v0.00.08`(Crab)은 계획일 뿐이며 이 저장소에 구현 코드가 없습니다.

구현된 각 버전에는 **베이스라인 기록**(범위, 추가된 능력, 지켜야 할 계약 불변식, 검증 명령, 경계)이 있습니다: [docs/versions](../versions/README.md). 현재 기록된 측정이 있는 버전은 `v0.00.07` Helix 하나이며, [네이티브 기억 페어 A/B](../versions/v0.00.07-helix/experiments/memory-on-off-ab.md)(동일 문제·동일 모델, 46쌍 승리·0쌍 패배)입니다.

## 빠른 시작

### Python SDK

원격 클라이언트는 표준 라이브러리만 필요합니다.

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

같은 클라이언트로 Orion 기억 연산, 비동기 작업, 내보내기, 패키지 전달, 헬스 체크,
레디니스, 메트릭, runtime 능력 조회를 할 수 있습니다.

### 서비스를 프로세스 내에서 호스팅

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

`runtime`을 `local_transformers`(로컬 Hugging Face 인과 언어 모델) 또는
`local_adapter`(감사된 adapter 경로)로 바꾸어도 서비스 체인은 수정할 필요가 없습니다.
`register_runtime(RuntimeKind(...))`로 새 백엔드를 등록할 수 있습니다.

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

## 저장소 레이아웃

```text
.
├── src/civilization/                     설치되는 패키지
│   ├── __init__.py                       공개 API
│   ├── client.py                         의존성 없는 HTTP 클라이언트
│   ├── models.py                         요청 / 예측 / 작업 모델
│   ├── runtimes.py                       runtime 종류, 능력, 레지스트리
│   ├── embedded.py                       프로세스 내 서비스 호스트
│   ├── cli.py                            `civilization` 명령줄
│   ├── engine/                           결정 엔진(engine extras 필요)
│   │   ├── model_paths.py                선택적 로컬 체크포인트 경로 해석
│   │   ├── adapter/                      학습되는 Civilization Adapter
│   │   ├── backend/                      로컬 / provider / adapter runtime
│   │   └── stages/                       버전 관리되는 서비스 체인, Stage 44–160
│   └── research/                         이전 연구 라인(계보 보존)
│       ├── prototype/                    최초의 NumPy 테스트 벤치
│       └── torch_line/                   PyTorch 백엔드 라인
├── tests/                                engine / torch_line / prototype 테스트
├── examples/                             실행 가능한 예제
├── docs/versions/                        버전별 베이스라인 기록과 실험
├── docs/readme/                          이 README의 8개 언어판
├── sdk/civilization-transformer/         TypeScript SDK
├── SDK.md                                Python SDK 가이드
└── pyproject.toml                        astreusn-civilization-v1 패키징
```

## 테스트

```bash
python -m pip install '.[test]'
pytest
```

대부분의 테스트는 모델 가중치가 필요 없습니다. 서비스 체인, 기억 정책, 작업·내보내기
계약, 두 SDK 모두 fake runtime 또는 원격 runtime으로 실행됩니다.

**로컬 Qwen3-0.6B 체크포인트**가 필요한 테스트는 모델이 없으면 자동으로 건너뜁니다.
체크포인트는 저장소에 포함되지 않습니다. 직접 준비해 `Models/Qwen3-0.6B`에 두거나
`CIVILIZATION_MODEL_PATH`로 지정하십시오.

```bash
CIVILIZATION_MODEL_PATH=/path/to/Qwen3-0.6B pytest
```

단계 실행 스크립트의 출력은 `experiments/*/artifacts/`에 기록되며 git에서 제외됩니다.
커밋된 실험 출력에 의존하는 코드는 저장소에 없습니다.

## 범위와 주장 경계

이 저장소가 **하는 것**:

- v1 라인의 버전 관리된 구현(동결 베이스 추론, 구조화 adapter, 다중 시스템 기억, 서비스
  체인, SDK).
- 불변량이 테스트된 시스템(fail-closed 입력 검증, 접근 제어, 작업 영속화와 복구,
  SHA-256으로 검증되는 내보내기, 경로별 소거 감사).

이 저장소가 **주장하지 않는 것**:

- 새로운 기반 모델이 아닙니다. v1 라인의 베이스 언어 모델은 동결되어 있고 학습되는 부분은
  Adapter와 그 읽기입니다.
- 평생 학습, 범용 장기 기억, 자율적 사실 중재, 인간 수준 일반화를 주장하지 않습니다.
- 생물학적 등가를 주장하지 않습니다. 기억 설계의 뇌 영역 이름은 공학적 라벨입니다.
- 텍스트 전용 provider가 adapter를 실행하거나 hidden states를 노출한다고 주장하지 않습니다.
  능력 엔드포인트는 바로 그 오칭을 피하기 위해 존재합니다.
- 감사된 adapter 경로는 고정 체크포인트와 기록된 패키지 산출물에 대해 검증되었습니다.
  다른 로컬 모델은 텍스트 전용 경로를 사용합니다.

## 라이선스

Apache License 2.0. [LICENSE](../../LICENSE) 및 [NOTICE](../../NOTICE) 참조.
