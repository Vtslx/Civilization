# Civilization

Civilization is a research line that moves **memory, state, and rule** signals
out of prompts and into the computation path of a language model, and then
versions the result as an auditable service with stable SDKs.

The v1 line keeps a small language model **frozen** and trains a Civilization
Adapter plus diagnostic readouts on top of it. Memory, state, and rule signals
enter the model as structured context, are projected into hidden states, and can
be ablated and audited per path. On top of that mechanism the line adds a
multi-system memory architecture, a persistent inference service, asynchronous
jobs, export packaging, and application-facing SDKs.

This repository is the **v1 baseline**: the complete first-generation line, with
no later research lines and no internal development documents.

> Translations: [简体中文](docs/readme/README.zh-CN.md) ·
> [繁體中文](docs/readme/README.zh-TW.md) · [日本語](docs/readme/README.ja.md) ·
> [한국어](docs/readme/README.ko.md) · [Español](docs/readme/README.es.md) ·
> [Français](docs/readme/README.fr.md) · [Deutsch](docs/readme/README.de.md) ·
> [Русский](docs/readme/README.ru.md) · [index](docs/readme/README.md)

## What is in this repository

| Path | What it is |
|---|---|
| `experiments/civilization_transformer/` | The first executable test bench: a NumPy Transformer-like core with memory, state, rule vectors, and a fused CivilizationBlock. |
| `experiments/civilization_transformer_torch/` | The PyTorch backend line: logic datasets, codebooks, ablation configs, and the training/evaluation harnesses used by every later stage. |
| `experiments/civilization_transformer_qwen3/` | The frozen-base line, Stages 44–160: hidden-state baselines, the Civilization Adapter, memory/state/rule path training, and the persistent service chain. |
| `civilization_v1/` | The Python SDK: a dependency-free HTTP client, the request/prediction/job models, the provider-agnostic runtime layer, and an in-process service host. |
| `sdk/civilization-transformer/` | The TypeScript SDK: a zero-dependency client for decisions, memory, jobs, and export packages. |
| `SDK.md`, `pyproject.toml` | Python packaging for `astreusn-civilization-v1`. |
| `docs/versions/` | One baseline record per implemented version, plus recorded experiments. |
| `docs/readme/` | This README in eight additional languages. |

Over 500 tests cover the line, from the first rule-gate unit tests to the
service, memory, and SDK contracts.

## Architecture at a glance

```
                 frozen language model (weights never updated)
                              │
   memory / state / rule ──►  Civilization Adapter ──► hidden-state readouts
        signals               (the trainable part)          │
                              │                             ▼
                              └──────────────►  decision readout (option choice)
                                                     │
   ┌─────────────────────────────────────────────────┴──────────────────┐
   │  service chain: inference → access control → queue → async jobs    │
   │  → persistent jobs → result store → batch → export → package      │
   │  → streaming delivery → Orion multi-system memory                  │
   └────────────────────────────────────────────────────────────────────┘
                              │
                  Python SDK / TypeScript SDK
```

Two design rules shape everything:

1. **Capabilities are declared, never assumed.** The service reports what its
   runtime can actually do over `GET /v1/capabilities`. A text-only provider
   reports `hidden_states: false` and `adapter_execution: false`; only the
   audited local adapter runtime reports them as `true`.
2. **The decision backend is configuration, not code.** Any OpenAI-compatible
   Chat Completions endpoint, any local Hugging Face causal model, or the local
   audited adapter can serve production traffic.

| Runtime kind | Backend | Provider-agnostic | Hidden states | Adapter execution |
|---|---|---|---|---|
| `provider` | any OpenAI-compatible Chat Completions endpoint | yes | no | no |
| `local_transformers` | any local Hugging Face causal LM | yes | no | no |
| `local_adapter` | the audited Civilization Adapter over pinned local weights | yes | yes | yes |

## Version line

| Version | Codename | Stages | Theme | Status |
|---|---|---|---|---|
| `v0.00.01` | Sun | 44–72 | Frozen-base inference, structured Adapter, runnable service | implemented |
| `v0.00.02` | Orion | 73–100 | Multi-system memory (working / episodic / semantic / procedural) | implemented |
| `v0.00.03` | Trifid | 101–121 | Episodic fast binding, cue and time disambiguation, controlled replay | implemented |
| `v0.00.04` | Lagoon | 122–132 | Cross-episode schema consolidation, provenance, conflict review | implemented |
| `v0.00.05` | Eagle | 133–138 | Task traces and approved procedural memory | implemented |
| `v0.00.06` | Rosette | 139–150 | Typed context packets, conflict budgets, multi-scale honeycomb graph | implemented |
| `v0.00.07` | Helix | 151–160 | Learned retrieval-path weights with recovery and calibration gates | implemented |
| `v0.00.08` | Crab | — | Memory conflict detection, arbitration, and forgetting policy | **not implemented** |

Release labels are not capability claims. The v1 line implements through
`v0.00.07`; `v0.00.08` (Crab) is a plan, and no code in this repository
implements it.

Every implemented version has a **baseline record** — scope, capabilities added,
the contract invariants it must hold, a verification command, and its boundary:
[docs/versions](docs/versions/README.md). One version currently has a recorded
measurement: `v0.00.07` Helix, via the [native memory paired
A/B](docs/versions/v0.00.07-helix/experiments/memory-on-off-ab.md) — memory on
vs. off on the same questions and model, 46 paired wins and 0 paired losses.

## Quick start

### Python SDK

The remote client needs only the standard library:

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

The same client exposes Orion memory operations, asynchronous jobs, exports,
package delivery, health, readiness, metrics, and runtime capabilities.

### Hosting the service in-process

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

Switch `runtime` to `local_transformers` (any local Hugging Face causal LM) or
`local_adapter` (the audited adapter path) without touching the service chain.
Register a new backend with `register_runtime(RuntimeKind(...))`.

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

## Repository layout

```text
.
├── civilization_v1/                     Python SDK
│   ├── client.py                        dependency-free HTTP client
│   ├── models.py                        request / prediction / job models
│   ├── runtimes.py                      runtime kinds, capabilities, registry
│   └── embedded.py                      in-process service host
├── experiments/
│   ├── civilization_transformer/        first NumPy test bench
│   ├── civilization_transformer_torch/  PyTorch backend line
│   └── civilization_transformer_qwen3/  frozen-base line, Stages 44–160
│       ├── adapter/                     Civilization Adapter
│       ├── backend/                     Qwen3 backend, provider runtimes
│       ├── analysis/                    stage runners and service chain
│       ├── tests/                       contracts for every stage
│       └── model_paths.py               checkpoint resolution (see Testing)
├── sdk/civilization-transformer/        TypeScript SDK
├── SDK.md                               Python SDK guide
└── pyproject.toml                       packaging for astreusn-civilization-v1
```

## Testing

```bash
python -m pip install '.[test]'   # pytest plus the analysis dependencies
pytest experiments -q             # 508 tests; the model-dependent ones skip
```

Extras: `.[test]` is enough to run the suite, `.[embedded]` hosts the service
in-process, and `.[analysis]` adds the plotting, clustering, and projection
libraries used by the stage runners.

Most of the suite runs with no model weights at all: the service chain, memory
policies, job/export contracts, and both SDKs run against fake or remote
runtimes.

Tests that need the **local Qwen3-0.6B checkpoint** are skipped when it is
absent. The checkpoint is not distributed with this repository; obtain it
yourself and either place it at `Models/Qwen3-0.6B` or point
`CIVILIZATION_MODEL_PATH` at it:

```bash
CIVILIZATION_MODEL_PATH=/path/to/Qwen3-0.6B pytest experiments/civilization_transformer_qwen3/tests -q
```

Stage runners write their outputs under `experiments/*/artifacts/`, which is
git-ignored; nothing in the repository depends on committed experiment output.

## Scope and claims boundary

What this repository **is**:

- A versioned implementation of the v1 line: frozen-base inference, structured
  adapter, multi-system memory, service chain, and SDKs.
- A system whose invariants are tested: fail-closed input validation, access
  control, job persistence and recovery, export integrity (SHA-256 verified
  downloads), and per-path ablation auditing.

What this repository does **not** claim:

- It is not a new foundation model. The base language model stays frozen in the
  v1 line; the trainable part is the Adapter and its readouts.
- It does not claim lifelong learning, general long-term memory, autonomous
  fact arbitration, or human-level generalization.
- It does not claim biological equivalence. Region names in the memory design
  are engineering labels.
- It does not claim that a text-only provider executes the adapter or exposes
  hidden states; the capability endpoint exists precisely to avoid that claim.
- The audited adapter path is validated for the pinned checkpoint and recorded
  package artifacts. Other local models run through the text-only path.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
