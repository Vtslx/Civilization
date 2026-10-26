# Encrypted Orion Memory Backups Implementation Plan

> Execute this plan in the authorized Civilization checkout; keep offsite replication explicitly separate until a destination is configured.

**Goal:** Make all newly created Orion memory backup bundles encrypted and authenticated while preserving verification and restore support for existing plaintext bundles.

**Architecture:** Keep the backup API offline and versioned. Add a chunked AES-GCM envelope using a key derived from a passphrase with Argon2id; use a fresh salt and nonce prefix for every backup. Read old v1 tar archives for recovery, but create only encrypted v2 bundles. The CLI obtains secrets from `CIVILIZATION_BACKUP_PASSPHRASE`, never from command arguments.

**Tech Stack:** Python standard library for archive, framing, temporary files, and atomic promotion; optional `cryptography` extra for Argon2id and AESGCM.

---

## Requirements and invariants

- New backup creation fails closed when no passphrase or crypto dependency is available.
- Passphrases never appear in command-line arguments, archive manifests, return values, logs, or saved memory.
- New backups reject passphrases shorter than 16 UTF-8 bytes; operational documentation requires high-entropy secret-manager values.
- A chunk is authenticated before plaintext is used. The frame counter, size, header, and terminal marker are authenticated; missing, reordered, duplicated, truncated, or modified chunks fail.
- Verification and restoration must parse the encrypted stream directly; do not spool a complete decrypted tar to disk.
- Verify and restore continue to support v1 plaintext bundles. New v1 creation is prohibited.
- Restore never overwrites an existing directory and never promotes a partial, unauthenticated, or checksum-invalid state.
- The archive format is bounded and versioned; attacker-controlled KDF parameters are rejected before expensive derivation.
- No actual production memory state is backed up during development; tests use synthetic state under a temporary directory.

## Task 1: Implement the encrypted envelope

**Files:** `src/civilization/engine/persistence/backup.py`

1. Add a new envelope version with bounded JSON metadata, Argon2id salt/parameters, nonce prefix, and fixed 1 MiB authenticated frames.
2. Stream the tar writer into AES-GCM frames; append an authenticated terminal frame so truncation cannot appear complete.
3. Add an authenticated decrypting reader and stream the tar parser directly over it; verify-only must not write a plaintext tar temporary file.
4. Make `create_memory_backup` require an explicit passphrase and emit only v2 encrypted bundles. Add `encrypted` to reports without exposing key material.
5. Preserve v1 detection, verification, and restore. Reject unsupported versions and invalid parameters with `MemoryBackupError`.

## Task 2: Wire the secure CLI

**Files:** `src/civilization/cli.py`, `pyproject.toml`, `src/civilization/engine/persistence/__init__.py`

1. Add a `secure-backup` optional dependency extra for `cryptography`.
2. Read `CIVILIZATION_BACKUP_PASSPHRASE` inside backup/verify/restore handlers. Never add a passphrase CLI argument.
3. Return a direct actionable error when the variable or optional crypto dependency is missing.
4. Let legacy v1 verification and restore work without a passphrase even when one is present in the environment.

## Task 3: Add security and recovery tests

**Files:** `tests/engine/test_memory_backup.py`

1. Verify encrypted round trip, confidentiality (known memory text absent from the bundle), wrong-passphrase rejection, tamper detection, and truncated-frame rejection.
2. Verify v1 plaintext bundles remain readable and restorable, while the public creation path emits only encrypted v2.
3. Cover malformed headers/KDF parameters, invalid frame order/counter, missing terminal frame, existing destinations, and interrupted atomic promotion.
4. Cover CLI environment handling, including no passphrase in arguments/output and missing-secret failure.
5. Run the focused backup suite, package compilation, and `git diff --check`.

## Task 4: Document and commit

**Files:** `docs/operations/session-memory-durability.md`, `docs/operations/session-memory-durability.zh-CN.md`, `README.md`, `docs/plans/2026-10-25-orion-memory-accuracy.md`, and this plan.

Document install/setup, key-loss implications, legacy migration, and the remaining offsite/retention/restore-drill requirements. Record the test result and commit this module with the English message `feat: encrypt Orion memory backup bundles`.

## Scope boundary

This module addresses confidentiality and integrity of backup artifacts. It does not configure a remote destination, schedule recurring backups, guarantee offsite retention, or prove a production restore time. Disaster recovery remains incomplete until an independent target and successful restore drill exist.
