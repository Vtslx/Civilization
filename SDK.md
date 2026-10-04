# AstreusN Civilization v1

`astreusn-civilization-v1` is the Python package behind the Civilization
decision service: a dependency-free client, a provider-agnostic runtime layer,
an in-process service host, and a command line that starts a working service in
one command.

Production decisions always use the validated `full` control mode. Offline
ablation controls stay in the engine as internal tooling and are intentionally
absent from the public API.

## Install

| Goal | Command | What you get |
|---|---|---|
| Talk to a running service | `pip install astreusn-civilization-v1` | client + CLI, standard library only |
| Host the service | `pip install 'astreusn-civilization-v1[embedded]'` | engine, service chain, local model support |
| Run the test suite | `pip install -e '.[test]'` | engine plus the analysis and test dependencies |

From a checkout, install the package itself so `import civilization` and the
`civilization` command resolve to this tree:

```bash
python -m pip install -e '.[embedded]'
```

## Verify the install in one command

The demo hosts the real service chain, so install the engine tier first:

```bash
pip install 'astreusn-civilization-v1[embedded]'
civilization demo
```

The demo runs the complete path — request validation, memory write, retrieval,
context injection, and an auditable trace — against a deterministic offline
runtime. No model, no API key, no network. Follow it with:

```bash
civilization doctor     # what this machine can run, and what is missing
```

## Command line

```bash
# host the service (prints the bound URL and declared capabilities)
civilization serve \
  --provider-base-url https://provider.example.com/v1 \
  --provider-model any-chat-model

# host with a local model instead — backends are configuration, not code
civilization serve --runtime local_transformers --local-model-path /path/to/model

# send one decision to a running service
civilization predict --text "Choose the supported operation." --options approve,reject
civilization predict --text "..." --options approve,reject --json
```

`serve` exposes `/health`, `/ready`, `/metrics`, `/v1/capabilities`,
`/v1/predict`, `/v1/batch`, `/v1/jobs` (submit, poll, results, retention),
`/v1/jobs/export` (create, package, download with Range and `If-Range`), and the
`/admin/orion/*` memory endpoints.

## Remote client

```python
from civilization import CivilizationClient, CivilizationRequest

client = CivilizationClient(
    "https://civilization.example.com",
    token_env="CIVILIZATION_API_TOKEN",
)

print(client.ready())
print(client.capabilities())          # what the deployment can actually do

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

The same client exposes the Orion memory operations, asynchronous jobs, exports,
package delivery, health, readiness, and metrics.

## Embedded service

```python
from civilization import EmbeddedCivilization, EmbeddedConfig

config = EmbeddedConfig(
    runtime="provider",
    provider_base_url="https://provider.example.com/v1",
    provider_model="provider-model-name",
    provider_api_key_env="PROVIDER_API_KEY",
    provider_headers={"x-provider-session": "deployment-42"},
    provider_extra_body={"reasoning_effort": "none"},   # provider-specific knobs
    bearer_token_env="CIVILIZATION_API_TOKEN",
    state_dir="~/.my-application/civilization",
)

with EmbeddedCivilization(config) as runtime:
    client = runtime.start()
    assert client.ready()["ready"] is True
    print(runtime.capabilities.to_dict())
```

Embedded mode hosts the full production path in your process: request
validation, access control, queueing, async jobs with persistence, exports and
resumable package delivery, and Orion session memory. All mutable files live
under `state_dir`.

## Runtime kinds and capabilities

The backend is configuration. Each runtime declares what it can and cannot do,
and `GET /v1/capabilities` reports the same declaration over HTTP.

| Runtime kind | Backend | Hidden states | Adapter execution |
|---|---|---|---|
| `provider` | any OpenAI-compatible Chat Completions endpoint | no | no |
| `local_transformers` | any local Hugging Face causal LM | no | no |
| `local_adapter` | the audited Civilization Adapter over pinned local weights | yes | yes |

Register an additional backend with `register_runtime(RuntimeKind(...))`; the
service chain itself is unchanged.

## Operational notes

- Secrets come from the environment: provider keys through `provider_api_key_env`,
  service bearer tokens through `bearer_token_env`. Never commit them.
- `state_dir` holds jobs, results, exports, audit records, and memory. Back it up
  to keep them; delete it to start clean.
- Session and global memory are durable by default: mutations are journaled
  under `state_dir/sessions/` and replayed on restart. `persist_sessions=False` keeps
  memory in-process only; `fsync_policy="every_write"` trades write cost for the
  smallest possible loss window; `persistence_mode="sidecar"` moves the journal
  and the commit loop into a separate process. `civilization prune` (or
  `POST /admin/orion/memory/retention`) compacts journals and drops expired
  sessions without touching anything the service is currently serving. See
  `docs/operations/session-memory-durability.md`.
- `/v1/capabilities` is the contract. A text-only provider never claims hidden
  states or adapter execution, and the SDK never implies them.
- A completed job can still contain a failed row: check
  `job_result(...)["result"]["rows"][*]["status"]` before consuming results.

## Related

- Repository, examples, and version baselines: <https://github.com/Vtslx/Civilization>
- TypeScript SDK: `sdk/civilization-transformer` (`@astreusn/civilization-transformer`)
