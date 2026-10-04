# Civilization

Civilization is an **auditable decision service with a native memory system**.
It answers structured decision requests, records which memory it used, and
carries working, episodic, semantic, and procedural memory across sessions.

Two install tiers, so the client stays lightweight: the bare package is a
standard-library client, and the `embedded` extra adds the decision engine and
service chain. Install it, and it runs:

```bash
python -m pip install -e .
civilization demo          # the whole path, no model, no key, no network
civilization serve --provider-base-url <url> --provider-model <model>
```

The decision backend is **configuration, not code** — any OpenAI-compatible
endpoint, any local Hugging Face model, or the audited local adapter — and the
service **declares what that backend can actually do** over
`GET /v1/capabilities` instead of assuming it.

The v1 line keeps its base language model frozen and trains an adapter over it,
so memory, state, and rule signals enter the computation path as structured,
per-path auditable context. This repository is the **v1 baseline**: the complete
first-generation line, with no later research lines and no internal development
documents.

> Translations: [简体中文](docs/readme/README.zh-CN.md) ·
> [繁體中文](docs/readme/README.zh-TW.md) · [日本語](docs/readme/README.ja.md) ·
> [한국어](docs/readme/README.ko.md) · [Español](docs/readme/README.es.md) ·
> [Français](docs/readme/README.fr.md) · [Deutsch](docs/readme/README.de.md) ·
> [Русский](docs/readme/README.ru.md) · [index](docs/readme/README.md)

## What is in this repository

| Path | What it is |
|---|---|
| `src/civilization/research/prototype/` | The first executable test bench: a NumPy Transformer-like core with memory, state, rule vectors, and a fused CivilizationBlock. |
| `src/civilization/research/torch_line/` | The PyTorch backend line: logic datasets, codebooks, ablation configs, and the training/evaluation harnesses used by every later stage. |
| `src/civilization/engine/` | The frozen-base line, Stages 44–160: hidden-state baselines, the Civilization Adapter, memory/state/rule path training, and the persistent service chain. |
| `src/civilization/` | The Python SDK: a dependency-free HTTP client, the request/prediction/job models, the provider-agnostic runtime layer, and an in-process service host. |
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

### Run it with no configuration

```bash
python -m pip install -e '.[embedded]'    # client + CLI + engine
civilization demo
```

The demo runs the complete service path — request validation, memory write,
retrieval, context injection, and an auditable trace — with a deterministic
offline runtime. No model, no API key, no network. It is the fastest way to see
what the system actually does.

```bash
civilization doctor     # what can this machine run, and what is missing
```

### Host the service

```bash
python -m pip install -e '.[embedded]'      # engine, service chain, local models
export PROVIDER_API_KEY=...                 # your provider's key

civilization serve \
  --provider-base-url https://provider.example.com/v1 \
  --provider-model any-chat-model
```

The service prints its bound URL and declared capabilities, then serves
`/health`, `/ready`, `/metrics`, `/v1/capabilities`, `/v1/predict`, `/v1/batch`,
`/v1/jobs`, the export/package endpoints, and the Orion memory admin endpoints.

Backends are configuration, not code — the same command hosts a local model:

```bash
civilization serve --runtime local_transformers --local-model-path /path/to/model
```

### Send a decision

```bash
civilization predict --text "Choose the supported operation." --options approve,reject
civilization predict --text "..." --options approve,reject --json   # full trace
```

### Python SDK

The remote client needs only the standard library:

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

The same client exposes Orion memory operations, asynchronous jobs, exports,
package delivery, health, readiness, metrics, and runtime capabilities.

### Hosting the service in-process

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
├── src/civilization/                     the installed package
│   ├── __init__.py                       public API
│   ├── client.py                         dependency-free HTTP client
│   ├── models.py                         request / prediction / job models
│   ├── runtimes.py                       runtime kinds, capabilities, registry
│   ├── embedded.py                       in-process service host
│   ├── cli.py                            `civilization` command line
│   ├── engine/                           decision engine (needs the engine extras)
│   │   ├── model_paths.py                optional local checkpoint resolution
│   │   ├── adapter/                      the trainable Civilization Adapter
│   │   ├── backend/                      local, provider, and adapter runtimes
│   │   └── stages/                       the versioned service chain, Stages 44–160
│   └── research/                         earlier lines, kept for provenance
│       ├── prototype/                    first NumPy test bench
│       └── torch_line/                   PyTorch backend line
├── tests/                                engine, torch_line, prototype test suites
├── examples/                             runnable examples
├── docs/versions/                        baseline record per version + experiments
├── docs/readme/                          this README in eight languages
├── sdk/civilization-transformer/         TypeScript SDK
├── SDK.md                                Python SDK guide
└── pyproject.toml                        packaging for astreusn-civilization-v1
```

Installing the package is what makes `import civilization` and the `civilization`
command available; the `src/` layout keeps the importable package separate from
tests, examples, and documentation.

## Testing

```bash
python -m pip install -e '.[test]'   # pytest plus the engine and analysis deps
pytest                                # 508 tests; the model-dependent ones skip
```

Extras: `.[embedded]` hosts the service in-process, `.[test]` runs the suite,
and `.[research]` adds the dataset, plotting, and projection libraries used by
the research runners.

Most of the suite runs with no model weights at all: the service chain, memory
policies, job/export contracts, and both SDKs run against fake or remote
runtimes.

Tests that need the **local Qwen3-0.6B checkpoint** are skipped when it is
absent. The checkpoint is not distributed with this repository; obtain it
yourself and either place it at `Models/Qwen3-0.6B` or point
`CIVILIZATION_MODEL_PATH` at it:

```bash
CIVILIZATION_MODEL_PATH=/path/to/Qwen3-0.6B pytest src/civilization/engine/tests -q
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
