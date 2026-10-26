# ADR-001: Keep stochastic token reservation outside provider generation

- Status: Accepted for experimental stage only
- Date: 2026-10-26

## Context

The requested memory-synthesis experiment needs candidate-position prediction, a seeded token reservation, reconciliation against emitted output, and a subsequent memory-token reservation. Civilization supports both local transformer runtimes and OpenAI-compatible provider runtimes. Provider responses expose completed text but not the per-position logits or KV-cache state required for token-level speculation. Existing memory consolidation is evidence-backed and approval-gated; a speculative single token must not be written as durable memory.

## Decision

Implement the protocol as a pure stage component that accepts caller-provided logits. Use a SHA-256 counter stream keyed by transaction ID, seed, stream name, and counter. Require multiple candidate offsets and retain per-candidate token IDs and probabilities. Reconcile against emitted token IDs; if they differ, discard the draft and return the emitted sequence unchanged with the rollback offset. Reserve a memory token only after output is complete and bind it to the resulting output digest and source-cell IDs. The component performs no storage writes and is not connected to provider mode.

## Alternatives considered

1. Integrate directly into the provider runtime. Rejected because the API does not return next-token logits or incremental decode state; fabricating those would not implement speculative decoding.
2. Replace local `generate()` with a custom sampler immediately. Deferred because the current task does not expose a memory-synthesis lifecycle there, and stochastic sampling could change answer behavior before a benchmark-backed gate exists.
3. Build a pure staged protocol first. Chosen because it is replayable, testable with controlled logits, has no production side effects, and can later be adapted to a local runtime that exposes the required token data.

## Consequences

- Candidate choice, reserved output token, reconciliation, and memory-token reservation can be replayed from an audit record without global RNG state.
- Emitted tokens remain authoritative; mismatch handling cannot leak a speculative token into returned output.
- A reserved memory token remains a draft. Semantic validation and explicit approval are still required before any durable memory write.
- This experiment has no LongMemEval accuracy result and does not close the 97% release gate.
