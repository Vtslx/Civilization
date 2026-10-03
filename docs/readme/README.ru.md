# Civilization (Русский)

Civilization — исследовательская линия, которая выносит сигналы **памяти (memory),
состояния (state) и правил (rule)** из промптов в сам вычислительный путь языковой
модели, а результат версионирует как аудируемый сервис со стабильными SDK.

Линия v1 держит небольшую языковую модель **замороженной** и обучает поверх неё
Civilization Adapter и диагностические считывания. Память, состояние и правила попадают
в модель как структурированный контекст, проецируются в hidden states и могут быть
аблатированы и аудированы по путям. Поверх этого механизма добавлены многосистемная
архитектура памяти, постоянный сервис инференса, асинхронные задачи, упаковка экспортов
и прикладные SDK.

Этот репозиторий — **база v1**: полное первое поколение, без последующих
исследовательских линий и без внутренних документов разработки.

> Другие языки: [English](../../README.md) · [简体中文](README.zh-CN.md) ·
> [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) ·
> [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md)

## Содержимое репозитория

| Путь | Описание |
|---|---|
| `experiments/civilization_transformer/` | Первый исполняемый стенд: ядро в стиле Transformer на NumPy с векторами памяти/состояния/правил и объединённым CivilizationBlock. |
| `experiments/civilization_transformer_torch/` | Линия бэкенда PyTorch: логические наборы данных, codebook, конфигурации аблаций и стенды обучения/оценки всех последующих стадий. |
| `experiments/civilization_transformer_qwen3/` | Линия с замороженной базой, Stages 44–160: базовые линии hidden states, Civilization Adapter, обучение путей memory/state/rule и постоянная сервисная цепочка. |
| `civilization_v1/` | Python SDK: HTTP-клиент без зависимостей, модели запроса/предсказания/задачи, независимый от провайдера слой runtime и внутрипроцессный хост сервиса. |
| `sdk/civilization-transformer/` | TypeScript SDK: клиент без зависимостей для решений, памяти, задач и пакетов экспорта. |
| `SDK.md`, `pyproject.toml` | Упаковка Python для `astreusn-civilization-v1`. |

Более 500 тестов покрывают линию — от первых модульных тестов правил до контрактов
сервиса, памяти и SDK.

## Архитектура

```
                 замороженная языковая модель (веса никогда не обновляются)
                              │
   память / состояние / ──►  Civilization Adapter ──► считывания hidden state
        правила               (обучаемая часть)              │
                              │                            ▼
                              └──────────────►  считывание решения (выбор варианта)
                                                     │
   ┌─────────────────────────────────────────────────┴──────────────────┐
   │  сервисная цепочка: инференс → контроль доступа → очередь →        │
   │  асинхронные задачи → сохранение задач → хранилище результатов →   │
   │  batch → экспорт → пакет → потоковая доставка → многосистемная     │
   │  память Orion                                                      │
   └────────────────────────────────────────────────────────────────────┘
                              │
                  Python SDK / TypeScript SDK
```

Два принципа проектирования проходят через всё:

1. **Возможности объявляются, а не предполагаются.** Сервис сообщает, что его runtime
   действительно умеет, через `GET /v1/capabilities`. Чисто текстовый провайдер
   объявляет `hidden_states: false` и `adapter_execution: false`; только аудированный
   локальный adapter runtime объявляет их как `true`.
2. **Бэкенд решений — это конфигурация, а не код.** Любой OpenAI-совместимый
   Chat Completions эндпоинт, любая локальная causal-модель Hugging Face или
   аудированный локальный adapter могут обслуживать продакшн-трафик.

| Тип runtime | Бэкенд | Независим от провайдера | Hidden states | Выполнение adapter |
|---|---|---|---|---|
| `provider` | любой OpenAI-совместимый Chat Completions эндпоинт | да | нет | нет |
| `local_transformers` | любая локальная causal-модель Hugging Face | да | нет | нет |
| `local_adapter` | аудированный Civilization Adapter на фиксированных локальных весах | да | да | да |

## Линия версий

| Версия | Кодовое имя | Stages | Тема | Статус |
|---|---|---|---|---|
| `v0.00.01` | Sun | 44–72 | Инференс с замороженной базой, структурный Adapter, работающий сервис | реализовано |
| `v0.00.02` | Orion | 73–100 | Многосистемная память (рабочая / эпизодическая / семантическая / процедурная) | реализовано |
| `v0.00.03` | Trifid | 101–121 | Быстрая эпизодическая привязка, разрешение неоднозначности по признаку и времени, контролируемый replay | реализовано |
| `v0.00.04` | Lagoon | 122–132 | Консолидация схем между эпизодами, происхождение, разбор конфликтов | реализовано |
| `v0.00.05` | Eagle | 133–138 | Трассы задач и одобренная процедурная память | реализовано |
| `v0.00.06` | Rosette | 139–150 | Типизированные контекстные пакеты, бюджеты конфликтов, многоуровневый граф-соты | реализовано |
| `v0.00.07` | Helix | 151–160 | Обучаемые веса путей извлечения с восстановлением и калибровочными шлюзами | реализовано |
| `v0.00.08` | Crab | — | Обнаружение, арбитраж и забывание конфликтов памяти | **не реализовано** |

Метки релизов не являются заявлениями о возможностях. Линия v1 реализована до
`v0.00.07`; `v0.00.08` (Crab) — это план, и ни один файл в этом репозитории его не
реализует.

У каждой реализованной версии есть **запись базовой линии** (охват, добавленные возможности, инварианты контракта, команда проверки, границы): [docs/versions](../versions/README.md). Записанное измерение пока есть только у `v0.00.07` Helix: [парный A/B нативной памяти](../versions/v0.00.07-helix/experiments/memory-on-off-ab.md) (46 выигранных пар, 0 проигранных).

## Быстрый старт

### Python SDK

Удалённому клиенту нужна только стандартная библиотека:

```bash
python -m pip install .
```

```python
from civilization_v1 import CivilizationClient, CivilizationRequest

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

Тот же клиент предоставляет операции памяти Orion, асинхронные задачи, экспорт, доставку
пакетов, health, readiness, метрики и запрос возможностей runtime.

### Хостинг сервиса внутри процесса

```bash
python -m pip install '.[embedded]'
```

```python
from civilization_v1 import EmbeddedCivilization, EmbeddedConfig

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

Переключите `runtime` на `local_transformers` (любая локальная модель Hugging Face) или
`local_adapter` (аудированный путь adapter) без изменения сервисной цепочки. Новый
бэкенд регистрируется через `register_runtime(RuntimeKind(...))`.

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

## Структура репозитория

```text
.
├── civilization_v1/                     Python SDK
│   ├── client.py                        HTTP-клиент без зависимостей
│   ├── models.py                        модели запроса / предсказания / задачи
│   ├── runtimes.py                      типы runtime, возможности, реестр
│   └── embedded.py                      внутрипроцессный хост сервиса
├── experiments/
│   ├── civilization_transformer/        первый стенд на NumPy
│   ├── civilization_transformer_torch/  линия бэкенда PyTorch
│   └── civilization_transformer_qwen3/  линия с замороженной базой, Stages 44–160
│       ├── adapter/                     Civilization Adapter
│       ├── backend/                     бэкенд Qwen3 и runtime провайдеров
│       ├── analysis/                    запускатели стадий и сервисная цепочка
│       ├── tests/                       контракты каждой стадии
│       └── model_paths.py               разрешение пути к чекпоинту (см. Тесты)
├── sdk/civilization-transformer/        TypeScript SDK
├── SDK.md                               руководство по Python SDK
└── pyproject.toml                       упаковка astreusn-civilization-v1
```

## Тесты

```bash
python -m pip install '.[test]'
pytest experiments -q
```

Большая часть набора не требует весов модели: сервисная цепочка, политики памяти,
контракты задач и экспорта, а также оба SDK работают с фиктивными или удалёнными
runtime.

Тесты, которым нужен **локальный чекпоинт Qwen3-0.6B**, пропускаются при его отсутствии.
Чекпоинт не распространяется вместе с репозиторием; получите его и разместите в
`Models/Qwen3-0.6B` либо укажите `CIVILIZATION_MODEL_PATH`:

```bash
CIVILIZATION_MODEL_PATH=/path/to/Qwen3-0.6B pytest experiments -q
```

Запускатели стадий пишут результаты в `experiments/*/artifacts/`, который игнорируется
git; ничто в репозитории не зависит от закоммиченных результатов экспериментов.

## Область и границы утверждений

Чем этот репозиторий **является**:

- Версионированной реализацией линии v1: инференс с замороженной базой, структурный
  adapter, многосистемная память, сервисная цепочка и SDK.
- Системой, инварианты которой протестированы: fail-closed валидация входа, контроль
  доступа, сохранение и восстановление задач, целостность экспорта (загрузки с проверкой
  SHA-256) и аудит аблаций по путям.

Чего этот репозиторий **не** утверждает:

- Это не новая фундаментальная модель. В линии v1 базовая языковая модель остаётся
  замороженной; обучаемая часть — Adapter и его считывания.
- Не утверждается непрерывное обучение, общая долговременная память, автономный
  арбитраж фактов или обобщение человеческого уровня.
- Не утверждается биологическая эквивалентность. Названия областей мозга в дизайне
  памяти — инженерные ярлыки.
- Не утверждается, что чисто текстовый провайдер выполняет adapter или раскрывает
  hidden states; эндпоинт возможностей существует именно чтобы избежать такого
  заявления.
- Аудированный путь adapter проверен для фиксированного чекпоинта и записанных
  артефактов пакета; другие локальные модели идут по чисто текстовому пути.

## Лицензия

Apache License 2.0. См. [LICENSE](../../LICENSE) и [NOTICE](../../NOTICE).
