# ADR-012: Make disaster recovery an executable drill

## Status

Accepted, 2026-10-31.

## Context

Backup, replication, and restore each had unit tests, and an early synthetic
drill copied one session across local filesystems. Those checks can all pass
while the composed recovery path fails: a restored journal can hydrate to a
different cell set, a replica can be on the same device, or a bundle can be
verified but not restorable. Disaster-recovery claims need an executed,
repeatable acceptance gate.

## Decision

Add `civilization dr-drill`. It creates a fresh authenticated encrypted backup,
verifies it, replicates it through the production cross-filesystem path,
restores it, rehydrates every session, and compares every cell and link with the
source. It exits non-zero on any mismatch and records the mode, hashes, counts,
and duration. `--allow-same-filesystem` is test/CI only and is recorded as a
weaker mode.

Keep offsite recovery a separate gate: a local drill or a local S3-compatible
server is not an independent failure domain.

## Consequences

- Releases can run one command that proves the composed local recovery path.
- The drill is destructive only to its own new directories; it refuses existing
  paths and never merges or replaces state.
- Offsite claims still require an authorized bucket, credentials, schedule,
  alerting, and a restore into an isolated account.

## Verification

`tests/engine/test_memory_drill.py` covers the passing path, the same-filesystem
refusal, an empty state directory, and the CLI. Executed evidence:
`docs/operations/data/disaster-recovery-drill-2026-10-31.json` (126/126 cells,
cross-filesystem) and
`docs/operations/data/disaster-recovery-s3-drill-2026-10-31.json` (S3 version
read-back, 126/126 cells).
