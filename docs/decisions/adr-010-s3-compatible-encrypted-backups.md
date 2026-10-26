# ADR-010: Add resumable S3-compatible encrypted backup replication

## Status

Implemented locally; live target validation pending a user-authorized endpoint and bucket

## Context

ADR-009 added a manual filesystem replica for authenticated encrypted v2 bundles. The current machine has no writable offsite target or S3 credentials. The core Civilization package intentionally has no runtime dependencies. A future remote target must support authenticated multipart upload, durable resume state, source/destination integrity verification, and version-aware retention without exposing plaintext or credentials.

The existing bundle is encrypted before replication. The S3 adapter therefore moves ciphertext only and does not create a second encryption format. The caller provides an S3-compatible client; the CLI constructs an optional boto3 client from an explicit endpoint/bucket and the standard credential-provider chain.

## Decision

- Add an optional `backup-s3` dependency extra; core installation remains dependency-free.
- Require encrypted v2 source bundles and a high-entropy passphrase for local source/remote read-back verification.
- Use multipart upload with per-part SHA-256 checksums and a local mode-0600 checkpoint containing only bucket/key, source digest/size, multipart upload ID, completed part ETags/checksums, and state. Never store credentials or the backup passphrase.
- Derive a content-addressed object key from the source ciphertext SHA-256; never overwrite a key whose remote metadata differs.
- On each run, reconcile the durable checkpoint with `ListParts`, retry missing or checksum-mismatched parts, and preserve the checkpoint on failure.
- After completion, require versioning, record the VersionId, check metadata and size, download the exact object version, recompute SHA-256, and run the existing authenticated backup verifier before marking the checkpoint verified.
- When Object Lock retention is requested, require bucket versioning and Object Lock to be enabled, set the requested mode/retention on upload, and verify it on read-back.
- Provide version-aware retention planning that preserves the newest configured generations, all versions inside the retention window, legal holds, and unexpired Object Lock versions. Deletion is dry-run by default and requires an explicit apply action.
- Do not upload any real backup until the user authorizes an endpoint and bucket. Do not configure a schedule or claim offsite recovery until an end-to-end test and retention policy are verified.

## Consequences

### Positive

- Multipart transfer resumes after process or network interruption without retransmitting verified parts.
- Destination integrity is checked by S3 per-part checksums and an independent full read-back hash plus authenticated decryption.
- Version IDs and Object Lock metadata are included in the replication audit; protected versions are excluded from retention deletion.
- Optional dependency isolation keeps the standard Civilization package lightweight.

### Negative

- A local checkpoint directory must be durable and protected; losing it may leave an incomplete remote multipart upload that requires operator cleanup.
- S3-compatible providers vary in checksum, versioning, and Object Lock support; unsupported behavior must fail closed.
- The implementation alone does not provide a geographic offsite copy, schedule, alerts, retention application, RPO/RTO, or a production restore drill.

## Non-functional requirements

- **Confidentiality:** upload encrypted v2 bytes only; credentials use the SDK provider chain; local checkpoint mode is `0600`; no secret appears in CLI arguments, logs, or results.
- **Integrity:** source verification, SHA-256 per part, SHA-256 of the completed remote read-back, and authenticated bundle verification are required before success.
- **Durability:** checkpoint updates are atomic and fsynced after each confirmed part; object completion is idempotently recovered by checking a matching content-addressed object.
- **Retention:** no deletion occurs in upload; pruning is an explicit dry-run/confirm workflow and never removes locked or held versions.

## Alternatives considered

- **Custom S3 Signature V4 over `urllib`:** rejected because it duplicates credential-provider, signing, retry, and endpoint compatibility logic.
- **Direct single-request PUT:** rejected because it cannot resume large transfers or persist per-part checkpoints.
- **Use ETag as a whole-object checksum:** rejected because multipart ETags are not necessarily a direct full-object digest.
- **Automatically prune after upload:** rejected because an upload failure or policy mistake must never delete older known-good generations.

## References

- [Amazon S3 multipart upload overview](https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html)
- [Amazon S3 checksum validation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity.html)
- [Amazon S3 Object Lock](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock.html)
- [Boto3 S3 client API](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)
