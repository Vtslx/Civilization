# AoNeb Civilization v1 Python SDK

`aoneb-civilization-v1` provides a stable application-facing API over the
Civilization v1 service chain. Production inference always uses the `full`
control mode. Offline ablation controls remain internal experiment tools and
are intentionally absent from this SDK.

The framework is provider-agnostic: the decision backend is selected by runtime
kind, and each runtime declares what it can actually do instead of the SDK
assuming a model family. `GET /v1/capabilities` reports the same declaration
over HTTP.

| runtime kind | backend | hidden states | adapter execution |
|---|---|---|---|
| `provider` | any OpenAI-compatible Chat Completions endpoint | no | no |
| `local_transformers` | any local Hugging Face causal LM | no | no |
| `local_adapter` | the audited Civilization Adapter over pinned local weights | yes | yes |

```python
from civilization_v1 import EmbeddedCivilization, EmbeddedConfig, ProviderRuntimeConfig

service = EmbeddedCivilization(
    EmbeddedConfig(
        runtime="provider",
        provider_base_url="https://provider.example.com/v1",
        provider_model="any-chat-model",
        provider_api_key_env="PROVIDER_API_KEY",
        provider_headers={"x-provider-session": "deployment-42"},
        provider_extra_body={"reasoning_effort": "none"},  # provider-specific knobs
    )
)
print(service.capabilities.to_dict())
```

Register an additional backend with `register_runtime(RuntimeKind(...))`; the
service chain itself is unchanged. TypeScript consumers use
`@astreusn/civilization-transformer` from the internal npm registry.

## Install

Remote HTTP clients need only the standard library:

```bash
python -m pip install .
```

Install the optional embedded runtime dependencies when the application will
host the Stage49-74 service in its own process:

```bash
python -m pip install '.[embedded]'
```

## Remote client

```python
from civilization_v1 import CivilizationClient, CivilizationRequest

client = CivilizationClient(
    "https://civilization.example.com",
    token_env="CIVILIZATION_API_TOKEN",
)
result = client.predict(
    CivilizationRequest(
        text="Choose the deployment action.",
        answer_options=("approve", "reject"),
        session_id="deployment-42",
        task_name="deployment_decision",
        memory_items=("The canary passed health checks.",),
        rule_items=("Reject when an unresolved critical alert exists.",),
        state_values=(0.9, 0.1, 0.8),
    )
)
print(result.option_id, result.option_text, result.memory_trace)
```

The same client exposes explicit Orion memory operations, asynchronous jobs,
exports, package delivery, health, readiness, and metrics. Pass secrets through
environment variables; do not commit provider keys or service bearer tokens.

## Embedded service

```python
from civilization_v1 import EmbeddedCivilization, EmbeddedConfig

config = EmbeddedConfig(
    provider_base_url="https://provider.example.com/v1",
    provider_model="provider-model-name",
    provider_api_key_env="PROVIDER_API_KEY",
    bearer_token_env="CIVILIZATION_API_TOKEN",
    state_dir="~/.my-application/civilization-v1",
)

with EmbeddedCivilization(config) as runtime:
    client = runtime.start()
    assert client.ready()["ready"] is True
```

Embedded mode hosts the validated Stage49-74 production path: request
validation, access control, queueing, async jobs, persistence, exports,
resumable-capable package endpoints, and Orion session memory. The configured
OpenAI-compatible provider supplies the decision backend. Such providers do
not expose Qwen hidden states, so this mode does not claim local adapter or
hidden-state execution. Trifid and later version modules remain independently
validated components and are not implicitly inserted into this online path.
