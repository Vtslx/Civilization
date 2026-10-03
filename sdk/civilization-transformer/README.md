# @astreusn/civilization-transformer

Provider-agnostic TypeScript SDK for the **Civilization Transformer** decision,
memory, job, and export service.

The SDK makes no assumption about the model behind the endpoint. Any
OpenAI-compatible Chat Completions provider, a self-hosted gateway, or a local
model can be the decision backend; the service declares what its runtime can
actually do and the SDK reports it instead of guessing.

- Zero runtime dependencies (global `fetch`, Node built-ins only)
- ESM with TypeScript types, Node `>= 22`
- Typed client for decisions, Orion memory, async jobs, and export packages
- Runtime capability introspection, including honest `hidden_states` /
  `adapter_execution` flags

## Install

Build the package from this repository:

~~~bash
cd sdk/civilization-transformer
npm install        # TypeScript toolchain only; the SDK itself has no runtime dependencies
npm run build      # emits dist/ with ESM + .d.ts
npm pack           # produces the installable tarball
~~~

Consume the tarball from another project:

~~~bash
npm install /path/to/sdk/civilization-transformer/astreusn-civilization-transformer-0.00.08-crab.tgz
~~~

Releases are also published to a private npm registry for internal consumers;
that registry is not part of this repository, so configure it in your own
`.npmrc` rather than relying on a checked-in default.

## Quick start

~~~ts
import { CivilizationClient } from "@astreusn/civilization-transformer";

const client = new CivilizationClient("https://civilization.example.com", {
  tokenEnv: "CIVILIZATION_API_TOKEN",
});

const capabilities = await client.capabilities();
console.log(capabilities.capabilities?.kind); // "provider" | "local_transformers" | "local_adapter"

const prediction = await client.predict({
  text: "The deployment signature and health checks are valid. Choose the supported operation.",
  answerOptions: ["approve", "reject"],
  sessionId: "deployment-42",
  taskName: "deployment_decision",
  memoryItems: ["The deployment signature and health checks are valid."],
  ruleItems: ["Reject only when a critical verification is unresolved."],
  stateValues: [0.9, 0.1, 0.8],
});

console.log(prediction.optionId, prediction.optionText, prediction.memoryTrace);
~~~

Batch decisions, asynchronous jobs, explicit Orion memory operations, exports,
and resumable package downloads are all available on the same client:

~~~ts
const batch = await client.batch([
  { text: "Evidence A is verified. Choose.", answerOptions: ["approve", "reject"], requestId: "b1" },
  { text: "A critical check is unresolved. Choose.", answerOptions: ["approve", "reject"], requestId: "b2" },
]);

const job = await client.submitJob({
  text: "Choose the supported operation.",
  answerOptions: ["approve", "reject"],
  sessionId: "deployment-42",
});
const finished = await client.waitJob(job.jobId, { timeoutMs: 180_000 });

await client.writeMemory({
  sessionId: "deployment-42",
  memorySystem: "episodic",
  content: "The canary passed health checks.",
});

const download = await client.downloadPackage("export-1", "./export.tar.gz");
console.log(download.sha256, download.verified);
~~~

## Runtime capabilities

`GET /v1/capabilities` is how a deployment states its own limits. Text-only
runtimes report `adapter_execution: false` and `hidden_states: false`; only the
audited local adapter runtime reports both as `true`.

~~~json
{
  "kind": "provider",
  "label": "openai-compatible:some-model",
  "provider_agnostic": true,
  "local_weights": false,
  "chat_completions": true,
  "hidden_states": false,
  "adapter_execution": false,
  "ablation_controls": false,
  "production_full_mode_only": true,
  "notes": ["any OpenAI-compatible Chat Completions endpoint"]
}
~~~

Treat these flags as the contract: a text-only deployment never claims hidden
states, and the SDK never implies them.

## Scope

- Production requests always use the validated `full` control path; online
  ablation controls are not exposed by this SDK.
- A completed job can still contain a failed row. Inspect
  `jobResult(jobId).result.rows[].status` before consuming results.
- Provider quirks are the deployment's concern, not the caller's: empty
  completions caused by provider-side reasoning padding are retried by the
  service before a decision fails.

## Development

~~~bash
npm install
npm run build
npm test                      # offline: stub service, no network

CIVILIZATION_BASE_URL=http://127.0.0.1:8765 \
  CIVILIZATION_EXPECT_OPTION_ID=0 \
  npm run test:live           # live end-to-end against a running service
~~~
