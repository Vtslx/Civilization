"""Resumable replication of authenticated encrypted backups to S3-compatible stores."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator
from urllib.parse import urlsplit

from .backup import MemoryBackupError, _fsync_directory, _sha256_file, verify_memory_backup


S3_CHECKPOINT_SCHEMA = 1
DEFAULT_S3_PART_SIZE = 8 * 1024 * 1024
MIN_S3_PART_SIZE = 5 * 1024 * 1024
MAX_S3_PART_SIZE = 5 * 1024 * 1024 * 1024
MAX_S3_PARTS = 10_000
_KEY_PREFIX_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=-]{0,127}$")
_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")


class S3BackupError(MemoryBackupError):
    """The S3 target, checkpoint, or remote backup could not be safely used."""


def validate_s3_endpoint_url(endpoint_url: str | None, *, allow_insecure_http: bool = False) -> str | None:
    """Validate a credential-free S3 endpoint URL before passing it to boto3."""

    if endpoint_url is None:
        return None
    if not isinstance(endpoint_url, str) or not endpoint_url.strip():
        raise ValueError("S3 endpoint URL must be a non-empty URL")
    parsed = urlsplit(endpoint_url.strip())
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("S3 endpoint URL must use HTTPS or HTTP with a hostname")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("S3 endpoint URL must not contain credentials, query, or fragment")
    if parsed.scheme == "http":
        loopback = parsed.hostname.casefold() in {"localhost", "127.0.0.1", "::1"}
        if not loopback and not allow_insecure_http:
            raise ValueError("non-loopback S3 endpoints require HTTPS")
    return endpoint_url.strip().rstrip("/")


def create_s3_client(
    *,
    endpoint_url: str | None = None,
    region_name: str = "us-east-1",
    allow_insecure_http: bool = False,
) -> Any:
    """Create an optional boto3 client using its standard credential chain."""

    endpoint = validate_s3_endpoint_url(endpoint_url, allow_insecure_http=allow_insecure_http)
    try:
        import boto3
        from botocore.config import Config
    except ImportError as error:
        raise MemoryBackupError("S3 replication requires the optional dependency; install with `pip install 'astreusn-civilization-v1[backup-s3]'`") from error
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region_name,
        config=Config(
            signature_version="s3v4",
            retries={"mode": "standard", "max_attempts": 5},
            s3={"addressing_style": "path" if endpoint else "auto"},
        ),
    )


def replicate_encrypted_backup_to_s3(
    backup_file: str | Path,
    *,
    client: Any,
    bucket: str,
    key_prefix: str,
    checkpoint_directory: str | Path,
    endpoint_identity: str,
    passphrase: str | None,
    part_size_bytes: int = DEFAULT_S3_PART_SIZE,
    object_lock_mode: str | None = None,
    retention_days: int | None = None,
) -> dict[str, Any]:
    """Resume a multipart upload, then read back and authenticate the exact S3 version.

    The local checkpoint contains an upload ID and checksums but never a credential
    or passphrase. Failed uploads remain resumable; prior objects are never deleted.
    """

    if not isinstance(bucket, str) or not bucket.strip() or "/" in bucket or "\\" in bucket:
        raise ValueError("bucket must be a non-empty bucket name")
    prefix = _validate_s3_key_prefix(key_prefix)
    if isinstance(part_size_bytes, bool) or not isinstance(part_size_bytes, int):
        raise ValueError("part_size_bytes must be an integer")
    if part_size_bytes < MIN_S3_PART_SIZE or part_size_bytes > MAX_S3_PART_SIZE:
        raise ValueError("part_size_bytes must be between 5 MiB and 5 GiB")
    if (object_lock_mode is None) != (retention_days is None):
        raise ValueError("object_lock_mode and retention_days must be provided together")
    if object_lock_mode is not None and object_lock_mode not in {"GOVERNANCE", "COMPLIANCE"}:
        raise ValueError("object_lock_mode must be GOVERNANCE or COMPLIANCE")
    if retention_days is not None and (type(retention_days) is not int or retention_days < 1):
        raise ValueError("retention_days must be a positive integer")
    if not isinstance(endpoint_identity, str) or not endpoint_identity.strip():
        raise ValueError("endpoint_identity must identify the S3 endpoint without credentials")

    source_input = Path(backup_file).expanduser()
    if source_input.is_symlink():
        raise MemoryBackupError("backup source must not be a symlink")
    source = source_input.resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    source_signature = _file_signature(source)
    source_report = verify_memory_backup(source, passphrase=passphrase)
    if source_report.get("encrypted") is not True:
        raise MemoryBackupError("only authenticated encrypted v2 backups can be replicated to S3")
    if _file_signature(source) != source_signature:
        raise MemoryBackupError("backup source changed while it was being verified")
    source_sha256 = _sha256_file(source)
    source_size = source_signature[2]
    if source_size <= 0:
        raise MemoryBackupError("encrypted backup source is empty")
    part_count = (source_size + part_size_bytes - 1) // part_size_bytes
    if part_count > MAX_S3_PARTS:
        raise ValueError(f"backup requires {part_count} parts; S3 multipart supports at most {MAX_S3_PARTS}")

    key = s3_backup_object_key(prefix, source_sha256)
    checkpoint_dir = _prepare_checkpoint_directory(checkpoint_directory)
    endpoint_fingerprint = hashlib.sha256(endpoint_identity.strip().encode("utf-8")).hexdigest()
    checkpoint_id = hashlib.sha256(f"{endpoint_fingerprint}\n{bucket}\n{key}".encode("utf-8")).hexdigest()
    checkpoint_path = checkpoint_dir / f"s3-backup-{checkpoint_id}.json"
    lock_path = checkpoint_dir / f".s3-backup-{checkpoint_id}.lock"

    with _checkpoint_lock(lock_path):
        versioning = client.get_bucket_versioning(Bucket=bucket)
        if versioning.get("Status") != "Enabled":
            raise MemoryBackupError("S3 backup bucket must have versioning enabled")
        lock_config = _validate_object_lock_config(
            client,
            bucket,
            object_lock_mode=object_lock_mode,
            retention_days=retention_days,
        )

        checkpoint = _load_s3_checkpoint(checkpoint_path)
        expected_identity = {
            "schema_version": S3_CHECKPOINT_SCHEMA,
            "endpoint_fingerprint": endpoint_fingerprint,
            "bucket": bucket,
            "key": key,
            "source_sha256": source_sha256,
            "source_size": source_size,
            "part_size_bytes": part_size_bytes,
            "object_lock_mode": object_lock_mode,
            "retention_days": retention_days,
            "checkpoint_path": str(checkpoint_path),
        }
        if checkpoint is None:
            checkpoint = {
                **expected_identity,
                "state": "new",
                "upload_id": None,
                "parts": {},
                "version_id": None,
                "retention_until": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        else:
            for field, expected in expected_identity.items():
                if checkpoint.get(field) != expected:
                    raise MemoryBackupError(f"S3 checkpoint {field} does not match the requested backup")
            if checkpoint.get("state") not in {"new", "uploading", "remote_validation", "verified"}:
                raise MemoryBackupError("S3 checkpoint state is unsupported")

        if retention_days is not None and checkpoint.get("retention_until") is None:
            if lock_config is None:
                raise MemoryBackupError("S3 Object Lock configuration was not validated")
            checkpoint["retention_until"] = lock_config["retain_until"]
        elif retention_days is not None:
            expected_lock = _retention_datetime(checkpoint["retention_until"])
            if expected_lock <= datetime.now(timezone.utc):
                raise MemoryBackupError("S3 checkpoint Object Lock retention has expired; use a new backup key")

        if checkpoint["state"] in {"remote_validation", "verified"}:
            report = _verify_remote_s3_version(
                client,
                bucket=bucket,
                key=key,
                version_id=checkpoint.get("version_id"),
                source_sha256=source_sha256,
                source_size=source_size,
                passphrase=passphrase,
                checkpoint_directory=checkpoint_dir,
                expected_lock_mode=object_lock_mode,
                expected_retain_until=checkpoint.get("retention_until"),
            )
            checkpoint.update({"state": "verified", "verified_at": datetime.now(timezone.utc).isoformat()})
            _save_s3_checkpoint(checkpoint_path, checkpoint)
            return _s3_replication_report(checkpoint, report)

        if checkpoint["state"] == "new":
            existing = _head_existing_object(client, bucket=bucket, key=key)
            if existing is not None:
                _assert_remote_object_identity(existing, source_sha256=source_sha256, source_size=source_size)
                version_id = existing.get("VersionId")
                if not version_id:
                    raise MemoryBackupError("versioned S3 object is missing its VersionId")
                checkpoint.update({"state": "remote_validation", "version_id": version_id, "upload_id": None})
                _save_s3_checkpoint(checkpoint_path, checkpoint)
                report = _verify_remote_s3_version(
                    client,
                    bucket=bucket,
                    key=key,
                    version_id=version_id,
                    source_sha256=source_sha256,
                    source_size=source_size,
                    passphrase=passphrase,
                    checkpoint_directory=checkpoint_dir,
                    expected_lock_mode=object_lock_mode,
                    expected_retain_until=checkpoint.get("retention_until"),
                )
                checkpoint.update({"state": "verified", "verified_at": datetime.now(timezone.utc).isoformat()})
                _save_s3_checkpoint(checkpoint_path, checkpoint)
                return _s3_replication_report(checkpoint, report)

            orphaned_upload_ids = _list_s3_multipart_uploads(client, bucket=bucket, key=key)
            if len(orphaned_upload_ids) > 1:
                raise MemoryBackupError("multiple incomplete multipart uploads exist for this backup key; resolve them before resuming")
            if orphaned_upload_ids:
                # Recover the narrow crash window after S3 accepted CreateMultipartUpload
                # but before its upload ID reached the local durable checkpoint.
                upload_id = orphaned_upload_ids[0]
                checkpoint.update({"state": "uploading", "upload_id": upload_id, "parts": {}})
                _save_s3_checkpoint(checkpoint_path, checkpoint)
            else:
                create_request: dict[str, Any] = {
                    "Bucket": bucket,
                    "Key": key,
                    "Metadata": {
                        "civilization-backup-format": "encrypted-v2",
                        "civilization-backup-sha256": source_sha256,
                        "civilization-backup-size": str(source_size),
                    },
                    "ChecksumAlgorithm": "SHA256",
                    "ChecksumType": "COMPOSITE",
                }
                if object_lock_mode is not None:
                    create_request["ObjectLockMode"] = object_lock_mode
                    create_request["ObjectLockRetainUntilDate"] = _retention_datetime(checkpoint["retention_until"])
                created = client.create_multipart_upload(**create_request)
                upload_id = created.get("UploadId")
                if not isinstance(upload_id, str) or not upload_id:
                    raise MemoryBackupError("S3 did not return a multipart upload ID")
                checkpoint.update({"state": "uploading", "upload_id": upload_id, "parts": {}})
                _save_s3_checkpoint(checkpoint_path, checkpoint)

        if checkpoint["state"] != "uploading" or not checkpoint.get("upload_id"):
            raise MemoryBackupError("S3 checkpoint is missing an active multipart upload ID")

        try:
            remote_parts = _list_s3_upload_parts(
                client,
                bucket=bucket,
                key=key,
                upload_id=checkpoint["upload_id"],
            )
        except Exception as error:
            existing = _head_existing_object(client, bucket=bucket, key=key)
            if existing is None:
                raise S3BackupError(f"could not resume S3 multipart upload: {_safe_s3_error(error)}") from error
            _assert_remote_object_identity(existing, source_sha256=source_sha256, source_size=source_size)
            version_id = existing.get("VersionId")
            if not version_id:
                raise MemoryBackupError("versioned S3 object is missing its VersionId") from error
            checkpoint.update({"state": "remote_validation", "version_id": version_id, "upload_id": None})
            _save_s3_checkpoint(checkpoint_path, checkpoint)
            report = _verify_remote_s3_version(
                client,
                bucket=bucket,
                key=key,
                version_id=version_id,
                source_sha256=source_sha256,
                source_size=source_size,
                passphrase=passphrase,
                checkpoint_directory=checkpoint_dir,
                expected_lock_mode=object_lock_mode,
                expected_retain_until=checkpoint.get("retention_until"),
            )
            checkpoint.update({"state": "verified", "verified_at": datetime.now(timezone.utc).isoformat()})
            _save_s3_checkpoint(checkpoint_path, checkpoint)
            return _s3_replication_report(checkpoint, report)

        part_records = dict(checkpoint.get("parts", {}))
        completed_parts: list[dict[str, Any]] = []
        source_after = _file_signature(source)
        if source_after != source_signature:
            raise MemoryBackupError("backup source changed before multipart replication")
        with source.open("rb") as source_stream:
            for part_number in range(1, part_count + 1):
                offset = (part_number - 1) * part_size_bytes
                source_stream.seek(offset)
                data = source_stream.read(part_size_bytes)
                if not data:
                    raise MemoryBackupError("backup source ended before all multipart parts were read")
                checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
                remote = remote_parts.get(part_number)
                remote_checksum = remote.get("ChecksumSHA256") if remote else None
                remote_size = remote.get("Size") if remote else None
                etag = remote.get("ETag") if remote else None

                if remote and remote_checksum == checksum and remote_size == len(data) and etag:
                    part_record = {
                        "PartNumber": part_number,
                        "ETag": etag,
                        "ChecksumSHA256": checksum,
                        "Size": len(data),
                    }
                    part_records[str(part_number)] = part_record
                    completed_parts.append(part_record)
                    checkpoint["parts"] = part_records
                    _save_s3_checkpoint(checkpoint_path, checkpoint)
                    continue

                response = client.upload_part(
                    Bucket=bucket,
                    Key=key,
                    UploadId=checkpoint["upload_id"],
                    PartNumber=part_number,
                    Body=data,
                    ContentLength=len(data),
                    ChecksumSHA256=checksum,
                )
                response_checksum = response.get("ChecksumSHA256")
                if response_checksum is not None and response_checksum != checksum:
                    raise MemoryBackupError(f"S3 part {part_number} checksum response did not match")
                etag = response.get("ETag")
                if not isinstance(etag, str) or not etag:
                    raise MemoryBackupError(f"S3 did not return an ETag for part {part_number}")
                part_record = {
                    "PartNumber": part_number,
                    "ETag": etag,
                    "ChecksumSHA256": checksum,
                    "Size": len(data),
                }
                part_records[str(part_number)] = part_record
                completed_parts.append(part_record)
                checkpoint["parts"] = part_records
                _save_s3_checkpoint(checkpoint_path, checkpoint)

        if _file_signature(source) != source_signature:
            raise MemoryBackupError("backup source changed during multipart replication")
        if len(completed_parts) != part_count:
            raise MemoryBackupError("not all S3 multipart parts were verified")

        completion_parts = [
            {key_name: item[key_name] for key_name in ("PartNumber", "ETag", "ChecksumSHA256")}
            for item in sorted(completed_parts, key=lambda item: item["PartNumber"])
        ]
        try:
            completed = client.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=checkpoint["upload_id"],
                MultipartUpload={"Parts": completion_parts},
                ChecksumType="COMPOSITE",
            )
            version_id = completed.get("VersionId")
        except Exception as error:
            existing = _head_existing_object(client, bucket=bucket, key=key)
            if existing is None:
                raise S3BackupError(f"S3 multipart completion failed: {_safe_s3_error(error)}") from error
            _assert_remote_object_identity(existing, source_sha256=source_sha256, source_size=source_size)
            version_id = existing.get("VersionId")
        if not isinstance(version_id, str) or not version_id:
            raise MemoryBackupError("versioned S3 object is missing its VersionId")

        checkpoint.update({"state": "remote_validation", "version_id": version_id, "upload_id": None})
        _save_s3_checkpoint(checkpoint_path, checkpoint)
        report = _verify_remote_s3_version(
            client,
            bucket=bucket,
            key=key,
            version_id=version_id,
            source_sha256=source_sha256,
            source_size=source_size,
            passphrase=passphrase,
            checkpoint_directory=checkpoint_dir,
            expected_lock_mode=object_lock_mode,
            expected_retain_until=checkpoint.get("retention_until"),
        )
        checkpoint.update({"state": "verified", "verified_at": datetime.now(timezone.utc).isoformat()})
        _save_s3_checkpoint(checkpoint_path, checkpoint)
        return _s3_replication_report(checkpoint, report)


def plan_s3_backup_retention(
    client: Any,
    *,
    bucket: str,
    prefix: str,
    keep_latest: int,
    older_than_days: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Plan exact-version deletions; versions protected by retention/holds are preserved."""

    if isinstance(keep_latest, bool) or not isinstance(keep_latest, int) or keep_latest < 1:
        raise ValueError("keep_latest must be a positive integer")
    if isinstance(older_than_days, bool) or not isinstance(older_than_days, int) or older_than_days < 1:
        raise ValueError("older_than_days must be a positive integer")
    prefix = _validate_s3_key_prefix(prefix)
    if client.get_bucket_versioning(Bucket=bucket).get("Status") != "Enabled":
        raise MemoryBackupError("S3 backup retention requires bucket versioning")
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    current_time = _as_utc(current_time)
    cutoff = current_time - timedelta(days=older_than_days)
    versions = _list_s3_object_versions(client, bucket=bucket, prefix=prefix)
    backups: list[tuple[datetime, dict[str, Any], dict[str, Any]]] = []
    for version in versions:
        key = version.get("Key")
        version_id = version.get("VersionId")
        last_modified = version.get("LastModified")
        if not isinstance(key, str) or not isinstance(version_id, str) or not isinstance(last_modified, datetime):
            continue
        head = client.head_object(Bucket=bucket, Key=key, VersionId=version_id)
        metadata = {str(k).casefold(): v for k, v in head.get("Metadata", {}).items()}
        if metadata.get("civilization-backup-format") != "encrypted-v2":
            continue
        backups.append((last_modified, version, head))

    backups.sort(key=lambda item: item[0], reverse=True)
    candidates: list[dict[str, Any]] = []
    preserved: list[dict[str, Any]] = []
    for index, (last_modified, version, head) in enumerate(backups):
        key = version["Key"]
        version_id = version["VersionId"]
        if index < keep_latest:
            preserved.append({"key": key, "version_id": version_id, "reason": "keep_latest"})
            continue
        if last_modified >= cutoff:
            preserved.append({"key": key, "version_id": version_id, "reason": "within_retention_window"})
            continue
        retain_until = head.get("ObjectLockRetainUntilDate")
        legal_hold = head.get("ObjectLockLegalHoldStatus") == "ON"
        if legal_hold or (isinstance(retain_until, datetime) and _as_utc(retain_until) > _as_utc(current_time)):
            preserved.append({"key": key, "version_id": version_id, "reason": "object_lock_or_legal_hold"})
            continue
        candidates.append({"key": key, "version_id": version_id, "last_modified": _as_utc(last_modified).isoformat()})
    return {
        "bucket": bucket,
        "prefix": prefix,
        "keep_latest": keep_latest,
        "older_than_days": older_than_days,
        "planned_at": current_time.isoformat(),
        "cutoff": cutoff.isoformat(),
        "candidate_versions": candidates,
        "preserved_versions": preserved,
        "delete_markers_preserved": True,
    }


def apply_s3_backup_retention(client: Any, plan: dict[str, Any], *, apply: bool = False) -> dict[str, Any]:
    """Delete only planned, rechecked encrypted backup versions when explicitly requested."""

    if not isinstance(apply, bool):
        raise ValueError("apply must be a boolean")
    bucket = plan.get("bucket")
    prefix = plan.get("prefix")
    candidates = plan.get("candidate_versions")
    if not isinstance(bucket, str) or not isinstance(prefix, str) or not isinstance(candidates, list):
        raise ValueError("retention plan is malformed")
    prefix = _validate_s3_key_prefix(prefix)
    if not apply:
        return {"dry_run": True, "deleted_versions": [], "candidate_versions": candidates}

    if client.get_bucket_versioning(Bucket=bucket).get("Status") != "Enabled":
        raise MemoryBackupError("S3 backup retention requires bucket versioning")
    keep_latest = plan.get("keep_latest")
    older_than_days = plan.get("older_than_days")
    planned_at = plan.get("planned_at")
    if (
        isinstance(keep_latest, bool)
        or not isinstance(keep_latest, int)
        or isinstance(older_than_days, bool)
        or not isinstance(older_than_days, int)
        or not isinstance(planned_at, str)
    ):
        raise ValueError("retention plan policy fields are invalid")
    planned_time = _retention_datetime(planned_at)
    current_plan = plan_s3_backup_retention(
        client,
        bucket=bucket,
        prefix=prefix,
        keep_latest=keep_latest,
        older_than_days=older_than_days,
        now=planned_time,
    )
    expected_candidates = {
        (item["key"], item["version_id"]) for item in current_plan["candidate_versions"]
    }
    supplied_candidates = {
        (item.get("key"), item.get("version_id"))
        for item in candidates
        if isinstance(item, dict)
    }
    if expected_candidates != supplied_candidates or len(supplied_candidates) != len(candidates):
        raise MemoryBackupError("retention plan changed; generate a fresh plan before applying")
    candidates = current_plan["candidate_versions"]
    deleted: list[dict[str, str]] = []
    preserved: list[dict[str, str]] = []
    now = datetime.now(timezone.utc)
    for candidate in candidates:
        key = candidate.get("key")
        version_id = candidate.get("version_id")
        if not isinstance(key, str) or not isinstance(version_id, str):
            raise ValueError("retention candidate is malformed")
        if not key.startswith(f"{prefix}/"):
            raise ValueError("retention candidate key is outside the planned prefix")
        head = client.head_object(Bucket=bucket, Key=key, VersionId=version_id)
        metadata = {str(k).casefold(): v for k, v in head.get("Metadata", {}).items()}
        if metadata.get("civilization-backup-format") != "encrypted-v2":
            preserved.append({"key": key, "version_id": version_id, "reason": "not_a_civilization_backup"})
            continue
        retain_until = head.get("ObjectLockRetainUntilDate")
        if head.get("ObjectLockLegalHoldStatus") == "ON" or (
            isinstance(retain_until, datetime) and _as_utc(retain_until) > now
        ):
            preserved.append({"key": key, "version_id": version_id, "reason": "object_lock_or_legal_hold"})
            continue
        client.delete_object(Bucket=bucket, Key=key, VersionId=version_id)
        deleted.append({"key": key, "version_id": version_id})
    return {"dry_run": False, "deleted_versions": deleted, "preserved_versions": preserved}


def _validate_s3_key_prefix(prefix: str) -> str:
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("key_prefix must be a non-empty S3 key prefix")
    normalized = prefix.strip().strip("/")
    components = normalized.split("/")
    if any(not _KEY_PREFIX_COMPONENT_RE.fullmatch(component) for component in components):
        raise ValueError("key_prefix components may contain only letters, digits, dot, underscore, equals, or hyphen")
    if len(normalized.encode("utf-8")) > 850:
        raise ValueError("key_prefix is too long for a content-addressed S3 key")
    return normalized


def s3_backup_object_key(key_prefix: str, source_sha256: str) -> str:
    """Build a stable content-addressed key without allowing path traversal."""

    prefix = _validate_s3_key_prefix(key_prefix)
    if not isinstance(source_sha256, str) or not _SHA256_HEX_RE.fullmatch(source_sha256):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    key = f"{prefix}/{source_sha256[:2]}/{source_sha256}.enc"
    if len(key.encode("utf-8")) > 1024:
        raise ValueError("generated S3 object key exceeds 1024 bytes")
    return key


def _prepare_checkpoint_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser()
    if directory.is_symlink():
        raise MemoryBackupError("S3 checkpoint directory must not be a symlink")
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    if directory.stat().st_mode & 0o077:
        raise MemoryBackupError("S3 checkpoint directory must not be accessible by group or others")
    return directory


@contextmanager
def _checkpoint_lock(path: Path) -> Iterator[None]:
    if path.is_symlink():
        raise MemoryBackupError("S3 checkpoint lock must not be a symlink")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MemoryBackupError("another S3 backup replication is using this checkpoint") from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def _load_s3_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink():
        raise MemoryBackupError("S3 checkpoint file must not be a symlink")
    if path.stat().st_mode & 0o077:
        raise MemoryBackupError("S3 checkpoint file must have mode 0600")
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MemoryBackupError("S3 checkpoint file is unreadable or invalid") from error
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != S3_CHECKPOINT_SCHEMA:
        raise MemoryBackupError("S3 checkpoint schema is unsupported")
    if not isinstance(checkpoint.get("parts"), dict):
        raise MemoryBackupError("S3 checkpoint parts are invalid")
    return checkpoint


def _save_s3_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".s3-checkpoint-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            payload = (json.dumps(checkpoint, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _list_s3_upload_parts(client: Any, *, bucket: str, key: str, upload_id: str) -> dict[int, dict[str, Any]]:
    parts: dict[int, dict[str, Any]] = {}
    marker: int | None = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket, "Key": key, "UploadId": upload_id, "MaxParts": 1000}
        if marker is not None:
            request["PartNumberMarker"] = marker
        response = client.list_parts(**request)
        for part in response.get("Parts", []):
            number = part.get("PartNumber")
            if isinstance(number, int):
                parts[number] = part
        if not response.get("IsTruncated"):
            return parts
        next_marker = response.get("NextPartNumberMarker")
        if not isinstance(next_marker, int) or next_marker <= (marker or 0):
            raise MemoryBackupError("S3 ListParts pagination is invalid")
        marker = next_marker


def _list_s3_multipart_uploads(client: Any, *, bucket: str, key: str) -> list[str]:
    upload_ids: list[str] = []
    key_marker: str | None = None
    upload_id_marker: str | None = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket, "Prefix": key, "MaxUploads": 1000}
        if key_marker is not None:
            request["KeyMarker"] = key_marker
        if upload_id_marker is not None:
            request["UploadIdMarker"] = upload_id_marker
        response = client.list_multipart_uploads(**request)
        for upload in response.get("Uploads", []):
            if upload.get("Key") == key and isinstance(upload.get("UploadId"), str):
                upload_ids.append(upload["UploadId"])
        if not response.get("IsTruncated"):
            return upload_ids
        next_key = response.get("NextKeyMarker")
        next_upload_id = response.get("NextUploadIdMarker")
        if not isinstance(next_key, str) or not isinstance(next_upload_id, str):
            raise MemoryBackupError("S3 ListMultipartUploads pagination is invalid")
        if (next_key, next_upload_id) == (key_marker, upload_id_marker):
            raise MemoryBackupError("S3 ListMultipartUploads pagination did not advance")
        key_marker, upload_id_marker = next_key, next_upload_id


def _is_s3_not_found(error: Exception) -> bool:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return False
    error_code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status == 404 or error_code in {"404", "NoSuchKey", "NotFound", "NoSuchUpload"}


def _head_existing_object(client: Any, *, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except Exception as error:
        if _is_s3_not_found(error):
            return None
        raise


def _assert_remote_object_identity(head: dict[str, Any], *, source_sha256: str, source_size: int) -> None:
    metadata = {str(key).casefold(): value for key, value in head.get("Metadata", {}).items()}
    if head.get("ContentLength") != source_size or metadata.get("civilization-backup-sha256") != source_sha256:
        raise MemoryBackupError("S3 destination key already exists with different backup content")
    if metadata.get("civilization-backup-format") != "encrypted-v2":
        raise MemoryBackupError("S3 destination key is not a Civilization encrypted v2 backup")


def _verify_remote_s3_version(
    client: Any,
    *,
    bucket: str,
    key: str,
    version_id: str | None,
    source_sha256: str,
    source_size: int,
    passphrase: str | None,
    checkpoint_directory: Path,
    expected_lock_mode: str | None,
    expected_retain_until: str | None,
) -> dict[str, Any]:
    if not version_id:
        raise MemoryBackupError("versioned S3 backup object is missing its VersionId")
    head = client.head_object(Bucket=bucket, Key=key, VersionId=version_id)
    _assert_remote_object_identity(head, source_sha256=source_sha256, source_size=source_size)
    if head.get("VersionId") not in {None, version_id}:
        raise MemoryBackupError("S3 returned a different VersionId during read-back verification")
    if expected_lock_mode is not None:
        if head.get("ObjectLockMode") != expected_lock_mode:
            raise MemoryBackupError("S3 object lock mode did not match the requested mode")
        expected_date = _retention_datetime(expected_retain_until or "")
        actual_date = head.get("ObjectLockRetainUntilDate")
        if not isinstance(actual_date, datetime) or _as_utc(actual_date) < expected_date:
            raise MemoryBackupError("S3 object lock retention did not match the requested retention")

    descriptor, temporary_name = tempfile.mkstemp(prefix=".s3-readback-", suffix=".enc", dir=checkpoint_directory)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    total = 0
    try:
        os.fchmod(descriptor, 0o600)
        response = client.get_object(Bucket=bucket, Key=key, VersionId=version_id)
        body = response["Body"]
        try:
            with os.fdopen(descriptor, "wb") as output:
                descriptor = -1
                while chunk := body.read(1024 * 1024):
                    total += len(chunk)
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if total != source_size or digest.hexdigest() != source_sha256:
            raise MemoryBackupError("remote S3 backup read-back checksum does not match its source")
        report = verify_memory_backup(temporary, passphrase=passphrase)
        if report.get("encrypted") is not True:
            raise MemoryBackupError("remote S3 backup read-back is not encrypted")
        return {**report, "version_id": version_id, "sha256": digest.hexdigest(), "total_bytes": total}
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _validate_object_lock_config(
    client: Any,
    bucket: str,
    *,
    object_lock_mode: str | None,
    retention_days: int | None,
) -> dict[str, Any] | None:
    if object_lock_mode is None:
        return None
    try:
        response = client.get_object_lock_configuration(Bucket=bucket)
    except Exception as error:
        raise MemoryBackupError(f"could not verify S3 Object Lock configuration: {_safe_s3_error(error)}") from error
    config = response.get("ObjectLockConfiguration", {})
    if config.get("ObjectLockEnabled") != "Enabled":
        raise MemoryBackupError("S3 bucket must have Object Lock enabled for retention requests")
    if retention_days is None:
        raise ValueError("retention_days is required when object_lock_mode is set")
    return {
        "mode": object_lock_mode,
        "retention_days": retention_days,
        "retain_until": (datetime.now(timezone.utc) + timedelta(days=retention_days)).replace(microsecond=0).isoformat(),
    }


def _retention_datetime(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise MemoryBackupError("S3 Object Lock retain-until date is invalid") from error
    if result.tzinfo is None:
        raise MemoryBackupError("S3 Object Lock retain-until date must include a timezone")
    return _as_utc(result)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _file_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _list_s3_object_versions(client: Any, *, bucket: str, prefix: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    key_marker: str | None = None
    version_marker: str | None = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if key_marker is not None:
            request["KeyMarker"] = key_marker
        if version_marker is not None:
            request["VersionIdMarker"] = version_marker
        response = client.list_object_versions(**request)
        results.extend(response.get("Versions", []))
        if not response.get("IsTruncated"):
            break
        next_key = response.get("NextKeyMarker")
        next_version = response.get("NextVersionIdMarker")
        if not isinstance(next_key, str) or not isinstance(next_version, str):
            raise MemoryBackupError("S3 ListObjectVersions pagination is invalid")
        if (next_key, next_version) == (key_marker, version_marker):
            raise MemoryBackupError("S3 ListObjectVersions pagination did not advance")
        key_marker, version_marker = next_key, next_version
    return results


def _s3_replication_report(checkpoint: dict[str, Any], remote_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "replicated": True,
        "verified": True,
        "encrypted": True,
        "bucket": checkpoint["bucket"],
        "key": checkpoint["key"],
        "version_id": checkpoint["version_id"],
        "source_sha256": checkpoint["source_sha256"],
        "total_bytes": checkpoint["source_size"],
        "entry_count": remote_report["entry_count"],
        "session_count": remote_report["session_count"],
        "checkpoint": checkpoint["checkpoint_path"],
        "object_lock_mode": checkpoint.get("object_lock_mode"),
        "retention_until": checkpoint.get("retention_until"),
    }


def _safe_s3_error(error: Exception) -> str:
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if isinstance(code, str):
            return code
    return type(error).__name__
