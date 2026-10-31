# ADR-011: Audit stored-memory precision against source spans

## Status

Accepted, 2026-10-31.

## Context

LongMemEval measures whether the answerer produces the right answer from
retrieved evidence. It does not measure whether the cells a session stores are
supported by the transcript. A store can answer correctly while holding an
invented fact, and a store can be entirely true while the answerer reasons
poorly. The project needs a separate, reproducible metric for the second claim.

## Decision

Add `civilization memory-precision`, a fail-closed audit over persisted Orion
cells. A cell is supported only when every provenance quote is a normalized
substring of the cited transcript span with a compatible role, or when it is a
derived/consolidation cell whose parents are all supported. Missing provenance,
fabricated quotes, missing spans, assistant-only user claims, unsupported
parents, and cycles are findings. `precision = supported / total`; an empty store
is `1.0` with `empty=true` and is not production evidence.

Provenance is carried in cell `metadata`, which the write and consolidation APIs
already accept. No schema change is required.

## Consequences

- A 100% result means the audited session is provenance-complete, not that the
  production corpus was audited.
- Writers that do not attach provenance will fail the audit. That is the
  intended fail-closed behavior; opt in per deployment.
- The metric is cheap and model-free, so it can run in CI and on every release.

## Verification

`tests/engine/test_memory_precision.py` covers the positive path and every
failure mode, including CLI exit codes. Measured reports:
`docs/operations/data/memory-precision-controlled-session-2026-10-31.json`
(7/7) and
`docs/operations/data/memory-precision-controlled-multisession-2026-10-31.json`
(132/132).
