# ADR-009: Replicate encrypted backups only to a distinct filesystem

## Status

Implemented for manual distinct-filesystem replicas; offsite operation remains unconfigured

## Context

Civilization can create authenticated encrypted bundles and restore them transactionally, but it has no replication command, schedule, retention policy, or restore monitoring. `/Volumes/Net` is an iSCSI filesystem. `/Volumes/Recovery` is a fixed, root-owned macOS recovery volume on the internal SSD and is not writable by the user. `/tmp` is writable on the internal data volume, which has a different device ID from `/Volumes/Net`, but is ephemeral and is suitable only for a synthetic cross-device drill. No writable offsite destination or credentials are configured.

Copying an encrypted bundle to another directory on the same filesystem does not protect against filesystem loss. A manual replication operation must verify the source bundle, write ciphertext to a caller-selected existing destination directory on a distinct filesystem, verify the copied bytes and bundle authentication, and publish without replacing an existing generation.

## Decision

- Add a `replicate-backup` operation that accepts an already encrypted v2 bundle and an explicit existing destination directory.
- Require a high-entropy passphrase from `CIVILIZATION_BACKUP_PASSPHRASE`; never accept it as a command-line argument or include it in output.
- Verify the source first and reject legacy plaintext bundles.
- Require source and destination to have different filesystem device IDs.
- Stream ciphertext into a mode-0600 temporary file on the destination filesystem, verify the digest and encrypted bundle, then atomically publish without overwriting an existing file.
- Do not delete prior backups. Retention and scheduling require a separately reviewed operational policy.
- Validate the operation with a synthetic source on `/Volumes/Net` and a temporary destination on `/tmp`; do not use the macOS Recovery volume as a destination.

## Consequences

### Positive

- Detects the common configuration error of placing the replica on the same filesystem.
- Verifies both cryptographic readability and byte-for-byte copy integrity before publication.
- Leaves source artifacts untouched and preserves existing destination generations.
- Uses the existing standard-library streaming and encrypted-bundle primitives without adding a cloud SDK.
- A synthetic cross-device copy and restore reproduced every payload byte exactly.

### Negative

- A different filesystem on the same computer is not offsite disaster recovery.
- The macOS `/Volumes/Recovery` mount is a root-owned, non-writable system recovery volume and cannot serve as a backup destination.
- Operators must choose a writable destination, run the operation, schedule it, and monitor failures.
- A filesystem device-ID check cannot prove geographic or administrative independence.

## Non-functional boundaries

- Data confidentiality: replica contains ciphertext only; passphrase remains process environment input.
- Integrity: source and destination are authenticated and SHA-256 checked before publication.
- RPO/RTO: not promised by this manual command. No schedule, retention, alerting, or production restore drill is configured.
- Capacity: destination must have room for another complete encrypted bundle.

## Alternatives considered

- **Copy into `/Volumes/Net/Recovery`:** rejected because it remains on the same iSCSI filesystem.
- **Use `/Volumes/Recovery`:** rejected because it is a non-writable macOS recovery volume, not a user backup target.
- **Upload directly to a cloud object store:** deferred because no destination account, endpoint, credentials, retention requirements, or network policy have been supplied.
- **Accept any directory without checking its device:** rejected because it can produce a false sense of protection while keeping both copies on one filesystem.
