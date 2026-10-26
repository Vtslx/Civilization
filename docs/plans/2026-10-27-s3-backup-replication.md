# S3-Compatible Encrypted Backup Replication Implementation Plan

> **For Codex:** Implement and test locally with a fake S3 client. Do not upload real backup data until the user authorizes the endpoint and bucket.

**Goal:** Replicate completed encrypted Orion backups to a versioned S3-compatible target with resumable multipart transfer, independent read-back verification, and Object Lock-aware retention planning.

**Architecture:** Add a separate optional S3 module using boto3's low-level client. The bundle remains encrypted before upload. A caller-selected durable checkpoint directory stores atomic, mode-0600 upload state after each verified part. Resume reconciles remote parts using SHA-256; completed objects are downloaded by VersionId and verified with the existing encrypted-bundle reader before success. A separate retention planner is dry-run by default and never removes locked or legally-held versions.

**Tech Stack:** Python standard library for checkpointing and hashing; optional `boto3`; existing `verify_memory_backup` and encrypted v2 format; fake S3 client for tests.

---

## Task 1: Define client/config contracts and failure behavior

**Files:** `src/civilization/engine/persistence/s3_backup.py`, `tests/engine/test_s3_backup.py`

1. Define a small S3 client protocol covering versioning, object lock, multipart, head/get, version listing, and exact-version deletion.
2. Validate HTTPS endpoint URLs; allow HTTP only for loopback or an explicit test opt-in. Credentials come from boto3's standard provider chain, never CLI values.
3. Require encrypted v2 source, stable file signature, explicit bucket/prefix, an existing durable checkpoint directory, and versioning enabled.
4. Use a deterministic key derived from source ciphertext digest; reject an existing object with mismatched digest/size metadata.

## Task 2: Implement checkpointed multipart replication

**Files:** `src/civilization/engine/persistence/s3_backup.py`, `tests/engine/test_s3_backup.py`, `pyproject.toml`

1. Add an optional `backup-s3` extra; import boto3 only when constructing the remote client.
2. Start multipart upload with SHA-256 checksum support and metadata identifying encrypted format, ciphertext digest, and size.
3. Persist upload ID and confirmed part number, ETag, and checksum atomically after each part.
4. On resume, call `ListParts`, recompute expected part checksums from the immutable source, and skip only parts whose remote checksum matches; upload missing/mismatched parts.
5. Use SDK retry configuration and preserve checkpoint state on every transient or terminal failure. Do not abort completed/previous backups on error.
6. Handle a crash after remote completion by checking for an existing matching content-addressed object before creating another upload.

## Task 3: Verify remote objects and enforce version retention

**Files:** `src/civilization/engine/persistence/s3_backup.py`, `tests/engine/test_s3_backup.py`

1. Complete multipart upload with ordered ETags/checksums and record VersionId.
2. `HeadObject` the exact version; validate content length and metadata.
3. Stream the exact version into a private local temporary file; verify full SHA-256 and call `verify_memory_backup` with the passphrase. Preserve remote object and checkpoint if any check fails.
4. If retention lock is requested, require bucket versioning/Object Lock, set mode/retention at upload, and verify lock metadata.
5. Implement a retention planner over object versions that preserves the newest configured count, newer-than-cutoff generations, active retention locks, and legal holds. Default to dry-run; explicit apply deletes exact version IDs only after rechecking lock state.

## Task 4: Add CLI and fake-client tests

**Files:** `src/civilization/cli.py`, `tests/engine/test_s3_backup.py`

Add a command accepting endpoint URL, region, bucket, prefix, checkpoint directory, and optional Object Lock policy. Never accept an access key or backup passphrase as a CLI argument. Record checkpoint path, VersionId, source hash, bytes, checksum verification, and remote read-back verification without recording credentials.

Fake-client tests must cover successful multipart upload, checkpoint resume, remote-part reconciliation, retry state after a part failure, source mutation rejection, no overwrite, remote read-back corruption, Object Lock retention skips, retention dry-run, explicit version deletion, and credential-free reports.

Run focused tests, `compileall`, and `git diff --check`. Keep the optional dependency absent from the default install.

## Task 5: Authorized live recovery drill

After local implementation, request the user's authorized S3-compatible endpoint and bucket. Do not upload real or synthetic objects until the target is authorized. Then test an encrypted synthetic bundle end-to-end: multipart checkpoint/resume, remote checksum, VersionId, read-back AEAD verification, and restore. Record retention settings and remaining operational gaps. A mock-client pass is not a live DR claim.
