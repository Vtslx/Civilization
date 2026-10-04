# v0.00.01 Sun — baseline record

| | |
|---|---|
| Version | `v0.00.01` |
| Codename | Sun |
| Stages | 44–72 |
| Theme | Frozen-base inference, structured Adapter, runnable service |
| Status | implemented |
| Modules | `civilization.engine.stages` (stage45 … stage72) |
| Contract tests | 32 test files under `tests/engine` |

## Capabilities added

- **Structured adapter.** A trainable Civilization Adapter over a frozen base
  model, with an auditable package format (`stage45_adapter_package`) and a
  runtime that executes it (`stage46_runtime_inference`).
- **Batch and production inference.** Centroid batch inference and a
  JSONL-based production inference path (`stage47`, `stage48`).
- **Persistent service.** A durable HTTP inference service with runtime
  loading, readiness, metrics, and versioned runtime reloads
  (`stage49`–`stage58`), including stress tests for reload, concurrency, and
  long-running operation.
- **Access control and queueing.** Bearer-token access control with an audit
  log, plus queue and rate-limit handling (`stage59`, `stage60`).
- **Asynchronous work.** Job submission, persistence, recovery, retention, and
  listing, with results stored outside the job record (`stage61`–`stage64`).
- **Export and delivery.** Batch jobs, export creation, export lifecycle,
  package creation, streaming delivery, HTTP Range delivery, `If-Range`
  resumption, and a resumable download client (`stage65`–`stage72`).

## Baseline contract

- Inference runs the validated production control path only; requests carry an
  explicit control mode and the audit trail records it.
- The service reports readiness honestly: `/ready` fails until a runtime is
  loaded, and metrics expose request, failure, and latency counters.
- Jobs survive a service restart and can be listed, paged, and retained or
  dropped by policy without silent data loss.
- Export packages are content-addressed: downloads are verified by SHA-256, and
  interrupted transfers resume instead of restarting.

## Verify locally

```bash
python -m pip install '.[test]'
pytest -q -k "stage4 or stage5 or stage6 or stage7"
```

## Boundary

- The base model stays frozen; this version adds a trained adapter and a
  service, not a new foundation model.
- No memory persistence beyond the service process is claimed here; multi-system
  memory arrives with `v0.00.02`.
- No accuracy claim is attached to this version — it is a contract baseline.

## Measurements

None recorded for this version. Baseline comparisons for Sun are contract-level
only.

[简体中文](README.zh-CN.md) · [version index](../README.md)
