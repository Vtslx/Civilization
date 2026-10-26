# Orion Stochastic Prediction and Regression Implementation Plan

> **For Codex:** Implement this plan task by task, preserving the user’s existing 97% LongMemEval release gate.

**Goal:** Add an auditable, deterministic stochastic reservation component that predicts one output token across several candidate positions, reconciles it against emitted output, and then reserves one memory token against the reconciled output.

**Architecture:** Keep the mechanism as a pure, model-agnostic stage component that consumes caller-provided token logits. Bind every reservation to a transaction ID, reproducible random seed, context fingerprint, and source-cell IDs. The emitted output remains authoritative: accept a reservation only on an exact token match, otherwise roll back the draft slot and regress to the emitted token sequence. Memory-token reservations remain drafts and cannot write to Orion storage.

**Tech Stack:** Python 3.10+, standard library, dataclasses, hashlib, pytest.

---

## Design decisions

1. **Provider boundary:** OpenAI-compatible providers return completed text and do not expose incremental logits or KV-cache state. Do not pretend token-level speculation is integrated into provider mode. The stage accepts logits from a future local-token runtime adapter.
2. **Candidate positions:** Positions are zero-based token offsets in the synthesized output draft. The planner predicts a token at every supplied candidate position, then uses a transaction-scoped deterministic random stream to reserve one candidate position/token pair.
3. **Random state:** Derive each draw from SHA-256 over the seed, transaction ID, stream name, and counter. This avoids global RNG state and supports deterministic replay after a process restart.
4. **Regression:** Compare the reserved token to the actual emitted token at that offset. On a match, accept it. On a mismatch, discard the speculative token and return the complete emitted sequence unchanged as the authoritative merged output; expose the rollback offset and output digest in the audit result.
5. **Memory safety:** Reserve the next memory token only after output reconciliation and bind it to the reconciled output digest and source-cell IDs. A one-token draft is never persisted as a memory cell; semantic validation and explicit approval remain future integration gates.
6. **Scope:** This is an experimental component, disabled by default and not connected to provider chat generation or production memory writes. It is not an accuracy claim or a substitute for a full 500-question benchmark.

## Task 1: Implement the pure reservation state machine

**Files:**
- Create: `src/civilization/engine/stages/stage166_orion_stochastic_memory.py`
- Test: `tests/engine/test_stage166_orion_stochastic_memory.py`

**Acceptance:** Validate dimensions and finite logits; predict candidates at multiple positions; reserve reproducibly from a seeded transaction; accept exact matches; roll back mismatches without changing emitted output; refuse memory reservation before reconciliation or without provenance; bind memory reservations to the reconciled output digest.

## Task 2: Add an auditable offline smoke runner and docs

**Files:**
- Modify: `src/civilization/engine/stages/stage166_orion_stochastic_memory.py`
- Modify: `docs/plans/2026-10-25-orion-memory-accuracy.md`
- Create: `docs/decisions/adr-001-stochastic-token-reservation.md`

**Acceptance:** Repeated smoke runs with the same transaction ID, seed, logits, and emitted output produce byte-for-byte equivalent summary payloads. The runner records token IDs, candidate offsets, acceptance/rollback status, source IDs, and digests, but no raw user text or logits. The memory store is unchanged.

## Task 3: Verify and commit the module

Run the targeted test module, the existing stage 160/161/163/164/165 regression tests, compile the new module, and run `git diff --check`. Commit this module independently with an English message.
