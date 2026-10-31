# Disaster recovery: backup, replication, restore, and drills

Civilization's memory is journaled and snapshotted, and the offline backup path
is authenticated encryption. This page is the operator-facing contract: what is
guaranteed per failure, how to run a restore, how to prove a restore actually
works, and which claims remain unverified.

## Failure model

| Failure | What survives | Recovery |
|---|---|---|
| Process crash (`kill -9`) | Every acknowledged write; replay stops at a torn tail | Restart; the journal replays idempotently |
| Power loss | Writes committed before the last `fsync`; at most the uncommitted tail with `--fsync interval` (default 0.25 s); nothing with `--fsync every_write` | Restart; replay the snapshot plus committed journal |
| Session journal corruption | The snapshot plus the valid journal prefix; a torn record is not applied | Restart; restore from the last verified backup if the snapshot is also damaged |
| Whole state directory lost | The last verified backup, and its replica | `civilization restore` into a new directory, then point the service at it |
| Host/filesystem lost | The replica on a different filesystem, and the S3 object version if replication completed | Restore from the replica or download the verified S3 version, then restore |
| Region/provider lost | Only an independent offsite copy | **Not verified by this repository**; see the offsite checklist |

RPO for acknowledged writes is zero. RPO for a backup is the interval between
backup runs; this repository does not schedule backups. RTO is the restore time;
the drill reports it but a local 126-cell drill is not a production RTO.

## Commands

```bash
# 1. Create an authenticated encrypted backup (service and sidecar stopped).
export CIVILIZATION_BACKUP_PASSPHRASE='...'   # inject from a secret manager
civilization backup --state-dir var/state --output /backups/civ-$(date +%F).civbackup

# 2. Verify before trusting it.
civilization verify-backup --backup /backups/civ-$(date +%F).civbackup

# 3. Copy to an existing directory on a different filesystem.
civilization replicate-backup \
  --backup /backups/civ-$(date +%F).civbackup \
  --destination-dir /mnt/offline-replica

# 4. For S3-compatible storage, use a versioned bucket.
civilization replicate-s3-backup \
  --backup /backups/civ-$(date +%F).civbackup \
  --bucket civilization-backups \
  --key-prefix prod/memory \
  --checkpoint-dir /var/lib/civilization/s3-checkpoints \
  --endpoint-url "$CIVILIZATION_BACKUP_S3_ENDPOINT_URL"

# 5. Restore into a new directory; existing data is never merged or replaced.
civilization restore \
  --backup /mnt/offline-replica/civ-$(date +%F).civbackup \
  --destination /srv/civilization-restored

# 6. Prove the path end to end.
civilization dr-drill \
  --state-dir var/state \
  --backup-dir /backups \
  --replica-dir /mnt/offline-replica \
  --output dr-report.json
```

`civilization dr-drill` is the acceptance gate, not a convenience wrapper. It
creates a fresh encrypted backup, verifies authentication, replicates it through
the production cross-filesystem path, restores it, rehydrates every session, and
compares every cell and link with the source. It exits non-zero on any mismatch.
`--allow-same-filesystem` is for tests and CI only; it records the weaker mode in
the report.

## Measured evidence

On 2026-10-31 a synthetic 3-session, 126-cell state directory on `/Volumes/Net`
was backed up, replicated to `/tmp` (a different device), restored, and
rehydrated. All eight checks passed: backup created, mode `0600`, authenticated
encryption, replica created through the cross-filesystem path, replica hash
match, restore completed, session set match, and all 126 cells identical. The
report is
[disaster-recovery-drill-2026-10-31.json](data/disaster-recovery-drill-2026-10-31.json);
the drill took 0.121 s and the backup SHA-256 was
`32e50acdf792f6a0f867b539f2b25543ed94ef810b71b1b567fd17ca7d18cee0`.

An earlier synthetic drill on 2026-10-29 copied a single session across local
filesystems and hydrated it through `SessionPersistence`; see
[civilization-synthetic-cross-volume-restore-2026-10-29.json](data/civilization-synthetic-cross-volume-restore-2026-10-29.json).

These are **local cross-filesystem drills on one host**. They prove the
software path, not survival of a host, disk, or region loss. The S3 replication
path has unit tests but this repository has not executed a live restore from an
offsite S3 bucket.

### S3 version round trip (local S3-compatible server)

On 2026-10-31 the same 3-session, 126-cell backup was replicated to a versioned
bucket on a local S3-compatible HTTP server (moto), then the exact version was
downloaded by `VersionId`, authenticated with `verify-backup`, restored, and
rehydrated. All checks passed: remote version read-back, downloaded SHA-256
match, authenticated bundle, restore, and 126/126 cells equal. The object key
was `drill/2026-10-31/7f/7fe1f79b...enc`, version
`0ebe6c7f-283b-46bf-95a0-42e464f05881`, backup SHA-256
`7fe1f79b6790e3a207003276c0fa47d3421d29863b13cb5493dfc1ab49803204`.
Report:
[disaster-recovery-s3-drill-2026-10-31.json](data/disaster-recovery-s3-drill-2026-10-31.json).

This exercises the production S3 client, multipart upload, versioning, remote
read-back, and restore path against a real HTTP S3 API. Moto is a local test
server, not an independent failure domain; the offsite checklist below remains
mandatory before claiming offsite recovery.

## Offsite readiness checklist

An authorized operator must complete every item before claiming offsite DR:

- [ ] A versioned bucket in a different region or provider account.
- [ ] Bucket versioning enabled; incomplete multipart uploads abort after a
      window longer than the expected retry/resume period.
- [ ] Object Lock (GOVERNANCE or COMPLIANCE) when immutability is required.
- [ ] Credentials from the SDK provider chain; the backup passphrase in a
      secret manager, never in the bucket or the checkpoint.
- [ ] A backup schedule and a replication schedule, with alerting on failure.
- [ ] A retention policy reviewed with `civilization s3-backup-retention`
      (dry-run by default; `--apply` only after review).
- [ ] A restore drill from the offsite copy into an isolated account, with cell
      counts and hashes recorded.
- [ ] Documented RPO/RTO for the deployment, not the local drill numbers.

## Limits

- The local drill does not test offsite latency, credential expiry, bucket
  policy drift, or an independent failure domain.
- The drill compares the source store as hydrated at drill time. A live service
  writing during the backup is outside the supported procedure; stop the service
  and sidecar first.
- Encryption protects the bundle at rest; it does not protect a compromised
  host that holds the passphrase.
- The drill does not schedule itself. Scheduling and alerting are deployment
  responsibilities.
