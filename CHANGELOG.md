# Changelog

Package releases use Semantic Versioning. The `v0.00.xx` identifiers in the
README and `docs/versions/` track internal research milestones separately.

## [Unreleased]

### Added

- `civilization memory-precision`: fail-closed audit of stored memory cells
  against exact cited transcript spans, with derived-cell support, role checks,
  JSON reports, and non-zero exit below the target.
- `civilization dr-drill`: backup, cross-filesystem replication, restore,
  rehydration, and cell-by-cell comparison as one acceptance gate.
- ADR-011 (stored-memory precision audit) and ADR-012 (executable disaster
  recovery drills).
- Stage312 LongMemEval evidence: GPT-5.6 Luna answerer on fixed reranked
  evidence, 466/500 (93.2%), the best verified full result.

### Validation status

- Latest complete Top-20 result: Stage312 scored 466/500 (93.2%), 19 below the
  485/500 gate.
- Stored-memory precision: 132/132 cells on a controlled multi-session corpus
  and 126/126 cells in the DR drill; a production corpus census is not measured.
- Disaster recovery: local cross-filesystem and local S3-compatible restore
  drills passed with 126/126 cells each; a live offsite restore is not verified.

## [1.0.0] - 2026-10-29

First formal release of the Civilization Python package, TypeScript SDK, and
service baseline.

### Included

- Dependency-free Python client, request and prediction models, runtime
  registry, and `civilization` command-line entry point.
- Provider, local Transformers, and audited local-adapter runtime options with
  declared capabilities.
- Auditable decision traces and multi-system memory across working, episodic,
  semantic, and procedural stores.
- Persistent session and global memory, asynchronous jobs, access controls,
  rate limits, export delivery, encrypted backups, restore, and optional
  S3-compatible backup replication.
- Python and TypeScript SDK surfaces.
- README embeds the v1.0.0 LongMemEval results figure and links to the
  accompanying research write-up.

### Validation status and known boundaries

- Latest complete Top-20 result: Stage311 scored 449/500 (89.8%). The best
  complete result remains Stage299 at 461/500 (92.2%); the 485/500 (97%) goal
  is not met.
- Stored-memory factual precision has not been measured at 100%.
- A live offsite disaster-recovery restore has not been verified.
- Stage166 stochastic token reservation remains an experimental synthetic
  protocol without production generation or durable-memory integration.
- The internal `v0.00.08` Crab milestone remains unimplemented.
