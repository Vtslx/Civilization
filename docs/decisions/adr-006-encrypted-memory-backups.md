# ADR-006: Authenticated encryption for memory backup bundles

**Status:** Accepted
**Date:** 2026-10-26

## Context

Orion memory backups currently use a mode-0600 tar archive with SHA-256 checksums. Checksums detect accidental corruption but do not encrypt the personal memory data or authenticate a rewritten archive. The repository is on a filesystem with limited free space, and no independent offsite destination is configured. Backup creation must not imply that same-volume copies provide disaster recovery.

The core package intentionally has no runtime dependencies. Encryption must therefore be isolated behind a documented optional dependency while remaining the only mode for creating new backups.

## Decision

- Create new backup bundles in a versioned encrypted envelope using AES-256-GCM authenticated encryption, with independently authenticated 1 MiB chunks so archive size is not limited by the single-message AEAD API.
- Derive a per-backup key from a user-provided passphrase using Argon2id with fixed, validated parameters and a fresh random salt. Use a fresh random nonce prefix and a monotonic chunk counter; authenticate the envelope header, counter, chunk length, and final-frame marker.
- Require a high-entropy passphrase of at least 16 UTF-8 bytes for new backup creation. The Python API accepts it explicitly; the CLI reads it only from `CIVILIZATION_BACKUP_PASSPHRASE`. Never accept it as a command argument or write it into archives, logs, reports, or memory.
- Keep format-v1 plaintext bundles readable for verify and restore, but stop creating them. This permits recovery of existing backups without preserving an insecure creation path.
- Verify authenticated chunks and the inner manifest before reporting success. Restore into a private staging area and atomically promote only after every chunk and payload checksum validates. A wrong passphrase or damaged/truncated envelope must leave no promoted state.
- Put `cryptography>=50.0.2` in a `secure-backup` optional extra to preserve the standard-library-only core installation. This minimum is above the patched wheel version in the upstream [OpenSSL wheel advisory](https://github.com/pyca/cryptography/security/advisories/GHSA-537c-gmf6-5ccf).

## Consequences

- New backups protect confidentiality and archive integrity. Loss of the passphrase makes them unrecoverable; operators must store it separately in an approved secret manager.
- Existing v1 backups remain restorable but are plaintext and should be migrated to a new encrypted bundle.
- Encryption does not create an independent copy. Automated offsite replication, retention, alerting, and restore drills remain separate DR requirements and are not considered complete by this ADR.
- Chunk authentication adds small per-chunk overhead and bounds memory use. Verify streams through the authenticated reader without writing a plaintext archive temporary file. Restore requires enough space for the staged recovered state.

## Rejected alternatives

- **Use SHA-256 only:** detects accidental corruption but offers no confidentiality or protection against a malicious rewrite.
- **Use a single `AESGCM.encrypt()` call:** simple but its API limits each message to less than 2 GiB and requires the full archive in memory.
- **Use a home-grown stream cipher or XOR/HMAC construction:** rejected in favor of a maintained cryptographic library and a standard AEAD primitive.
- **Rely on the mounted volume's encryption:** does not make portable backup files independently protected and does not address file-level access after copying.
