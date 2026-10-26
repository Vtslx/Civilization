from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import pytest

from civilization.engine.persistence import (
    MemoryBackupError,
    apply_s3_backup_retention,
    create_s3_client,
    create_memory_backup,
    plan_s3_backup_retention,
    replicate_encrypted_backup_to_s3,
    s3_backup_object_key,
    validate_s3_endpoint_url,
)
from civilization.cli import build_parser, main
from civilization.cli import build_parser, main


_TEST_PASSPHRASE = "test-only-s3-backup-passphrase"


class FakeS3Error(Exception):
    def __init__(self, code: str, status: int = 404):
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class FakeS3Client:
    def __init__(self):
        self.uploads = {}
        self.objects = {}
        self.versioning_enabled = True
        self.object_lock_enabled = True
        self.fail_after_part_once = False
        self.failed_part_once = False
        self.fail_after_create_once = False
        self.failed_create_once = False
        self.corrupt_readback = False
        self.part_upload_calls = []
        self.complete_calls = 0
        self.create_calls = 0
        self.delete_calls = []

    def get_bucket_versioning(self, *, Bucket):
        return {"Status": "Enabled" if self.versioning_enabled else "Suspended"}

    def get_object_lock_configuration(self, *, Bucket):
        if not self.object_lock_enabled:
            raise FakeS3Error("ObjectLockConfigurationNotFoundError")
        return {"ObjectLockConfiguration": {"ObjectLockEnabled": "Enabled"}}

    def head_object(self, *, Bucket, Key, VersionId=None):
        versions = self.objects.get((Bucket, Key), [])
        if VersionId is None:
            version = versions[-1] if versions else None
        else:
            version = next((item for item in versions if item["VersionId"] == VersionId), None)
        if version is None:
            raise FakeS3Error("NoSuchKey")
        result = {
            "ContentLength": len(version["Body"]),
            "Metadata": version["Metadata"],
            "VersionId": version["VersionId"],
        }
        if version.get("ObjectLockMode"):
            result["ObjectLockMode"] = version["ObjectLockMode"]
            result["ObjectLockRetainUntilDate"] = version["ObjectLockRetainUntilDate"]
        if version.get("ObjectLockLegalHoldStatus"):
            result["ObjectLockLegalHoldStatus"] = version["ObjectLockLegalHoldStatus"]
        return result

    def create_multipart_upload(self, **request):
        upload_id = f"upload-{len(self.uploads) + 1}"
        self.uploads[upload_id] = {**request, "UploadId": upload_id, "Parts": {}}
        self.create_calls += 1
        if self.fail_after_create_once and not self.failed_create_once:
            self.failed_create_once = True
            raise TimeoutError("simulated response loss after multipart creation")
        return {"UploadId": upload_id}

    def list_multipart_uploads(self, *, Bucket, Prefix, **_kwargs):
        uploads = [
            {"Key": upload["Key"], "UploadId": upload_id}
            for upload_id, upload in self.uploads.items()
            if upload["Bucket"] == Bucket and upload["Key"].startswith(Prefix)
        ]
        return {"Uploads": uploads, "IsTruncated": False}

    def list_parts(self, *, Bucket, Key, UploadId, **_kwargs):
        upload = self.uploads.get(UploadId)
        if upload is None:
            raise FakeS3Error("NoSuchUpload")
        return {
            "Parts": [dict(part) for _, part in sorted(upload["Parts"].items())],
            "IsTruncated": False,
        }

    def upload_part(self, *, Bucket, Key, UploadId, PartNumber, Body, ChecksumSHA256, **_kwargs):
        upload = self.uploads[UploadId]
        actual_checksum = __import__("base64").b64encode(hashlib.sha256(Body).digest()).decode("ascii")
        if actual_checksum != ChecksumSHA256:
            raise AssertionError("client supplied an incorrect part checksum")
        etag = f'"{hashlib.sha256(Body).hexdigest()}"'
        upload["Parts"][PartNumber] = {
            "PartNumber": PartNumber,
            "ETag": etag,
            "ChecksumSHA256": ChecksumSHA256,
            "Size": len(Body),
            "Body": Body,
        }
        self.part_upload_calls.append((UploadId, PartNumber))
        if self.fail_after_part_once and not self.failed_part_once:
            self.failed_part_once = True
            raise TimeoutError("simulated response loss after S3 accepted the part")
        return {"ETag": etag, "ChecksumSHA256": ChecksumSHA256}

    def complete_multipart_upload(self, *, Bucket, Key, UploadId, MultipartUpload, **_kwargs):
        upload = self.uploads[UploadId]
        ordered = MultipartUpload["Parts"]
        body = b"".join(upload["Parts"][part["PartNumber"]]["Body"] for part in ordered)
        version_id = f"version-{self.complete_calls + 1}"
        version = {
            "VersionId": version_id,
            "Body": body,
            "Metadata": upload["Metadata"],
            "LastModified": datetime.now(timezone.utc),
            "ObjectLockMode": upload.get("ObjectLockMode"),
            "ObjectLockRetainUntilDate": upload.get("ObjectLockRetainUntilDate"),
            "ObjectLockLegalHoldStatus": "OFF",
        }
        self.objects.setdefault((Bucket, Key), []).append(version)
        del self.uploads[UploadId]
        self.complete_calls += 1
        return {"VersionId": version_id}

    def get_object(self, *, Bucket, Key, VersionId):
        version = next(item for item in self.objects[(Bucket, Key)] if item["VersionId"] == VersionId)
        body = bytearray(version["Body"])
        if self.corrupt_readback and body:
            body[-1] ^= 1
        return {"Body": io.BytesIO(body), "VersionId": VersionId}

    def list_object_versions(self, *, Bucket, Prefix, **_kwargs):
        versions = []
        for (bucket, key), items in self.objects.items():
            if bucket != Bucket or not key.startswith(Prefix):
                continue
            for item in items:
                versions.append({
                    "Key": key,
                    "VersionId": item["VersionId"],
                    "LastModified": item["LastModified"],
                    "Size": len(item["Body"]),
                })
        return {"Versions": versions, "DeleteMarkers": [], "IsTruncated": False}

    def delete_object(self, *, Bucket, Key, VersionId):
        self.delete_calls.append((Bucket, Key, VersionId))
        items = self.objects[(Bucket, Key)]
        self.objects[(Bucket, Key)] = [item for item in items if item["VersionId"] != VersionId]

    def add_test_version(self, bucket, key, version_id, *, last_modified, retain_until=None, legal_hold=False):
        self.objects.setdefault((bucket, key), []).append({
            "VersionId": version_id,
            "Body": b"synthetic encrypted bundle placeholder",
            "Metadata": {
                "civilization-backup-format": "encrypted-v2",
                "civilization-backup-sha256": hashlib.sha256(version_id.encode()).hexdigest(),
            },
            "LastModified": last_modified,
            "ObjectLockMode": "COMPLIANCE" if retain_until else None,
            "ObjectLockRetainUntilDate": retain_until,
            "ObjectLockLegalHoldStatus": "ON" if legal_hold else "OFF",
        })


def _state_fixture(root: Path, *, journal_bytes: bytes = b'{"t":"cell","v":{"cell_id":"s3-test"}}\n') -> Path:
    state = root / "state"
    session = state / "sessions" / "alice"
    session.mkdir(parents=True)
    (session / "journal.ndjson").write_bytes(journal_bytes)
    (session / "snapshot.json").write_text(
        json.dumps({"version": 1, "session_id": "alice", "cells": [], "links": [], "trace_events": []}),
        encoding="utf-8",
    )
    (state / "global-memory.json").write_text(json.dumps({"cells": [], "links": []}), encoding="utf-8")
    return state


def _replicate(client, backup: Path, checkpoint_dir: Path, **kwargs):
    return replicate_encrypted_backup_to_s3(
        backup,
        client=client,
        bucket="memory-backups",
        key_prefix="civilization/orion",
        checkpoint_directory=checkpoint_dir,
        endpoint_identity="https://fake-s3.invalid",
        passphrase=_TEST_PASSPHRASE,
        **kwargs,
    )


def test_s3_endpoint_validation_and_content_addressed_key() -> None:
    assert validate_s3_endpoint_url("https://s3.example.test") == "https://s3.example.test"
    assert validate_s3_endpoint_url("http://127.0.0.1:9000") == "http://127.0.0.1:9000"
    assert validate_s3_endpoint_url("http://s3.example.test", allow_insecure_http=True) == "http://s3.example.test"
    with pytest.raises(ValueError, match="require HTTPS"):
        validate_s3_endpoint_url("http://s3.example.test")
    with pytest.raises(ValueError, match="credentials"):
        validate_s3_endpoint_url("https://user:secret@s3.example.test")
    with pytest.raises(ValueError, match="key_prefix components"):
        s3_backup_object_key("civilization/../escape", "0" * 64)
    assert s3_backup_object_key("civilization/orion", "a" * 64) == f"civilization/orion/aa/{'a' * 64}.enc"


def test_s3_multipart_checkpoint_resumes_and_readback_authenticates(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path, journal_bytes=b"x" * (5 * 1024 * 1024 + 257))
    backup = tmp_path / "memory.enc"
    checkpoint_dir = tmp_path / "private-checkpoints"
    create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)
    client = FakeS3Client()
    client.fail_after_part_once = True

    with pytest.raises(TimeoutError, match="response loss"):
        _replicate(client, backup, checkpoint_dir, part_size_bytes=5 * 1024 * 1024, object_lock_mode="COMPLIANCE", retention_days=30)

    checkpoint_files = list(checkpoint_dir.glob("s3-backup-*.json"))
    assert len(checkpoint_files) == 1
    checkpoint_text = checkpoint_files[0].read_text()
    assert _TEST_PASSPHRASE not in checkpoint_text
    assert checkpoint_files[0].stat().st_mode & 0o777 == 0o600

    report = _replicate(client, backup, checkpoint_dir, part_size_bytes=5 * 1024 * 1024, object_lock_mode="COMPLIANCE", retention_days=30)
    assert report["verified"] is True
    assert report["encrypted"] is True
    assert report["object_lock_mode"] == "COMPLIANCE"
    assert report["version_id"] == "version-1"
    assert client.part_upload_calls == [("upload-1", 1), ("upload-1", 2)]
    assert client.complete_calls == 1

    replay = _replicate(client, backup, checkpoint_dir, part_size_bytes=5 * 1024 * 1024, object_lock_mode="COMPLIANCE", retention_days=30)
    assert replay["version_id"] == report["version_id"]
    assert client.complete_calls == 1


def test_s3_recovers_if_create_multipart_response_is_lost_before_checkpoint(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.enc"
    checkpoint_dir = tmp_path / "checkpoints"
    create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)
    client = FakeS3Client()
    client.fail_after_create_once = True

    with pytest.raises(TimeoutError, match="response loss"):
        _replicate(client, backup, checkpoint_dir)
    assert list(checkpoint_dir.glob("s3-backup-*.json")) == []
    assert len(client.uploads) == 1

    report = _replicate(client, backup, checkpoint_dir)
    assert report["verified"] is True
    assert client.create_calls == 1
    assert client.complete_calls == 1
    assert client.uploads == {}


def test_s3_readback_failure_preserves_remote_version_and_checkpoint(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.enc"
    checkpoint_dir = tmp_path / "checkpoints"
    create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)
    client = FakeS3Client()
    client.corrupt_readback = True

    with pytest.raises(MemoryBackupError, match="read-back checksum"):
        _replicate(client, backup, checkpoint_dir)

    assert client.complete_calls == 1
    assert len(client.objects[("memory-backups", s3_backup_object_key("civilization/orion", hashlib.sha256(backup.read_bytes()).hexdigest()))]) == 1
    checkpoint = json.loads(next(checkpoint_dir.glob("s3-backup-*.json")).read_text())
    assert checkpoint["state"] == "remote_validation"
    assert checkpoint["version_id"] == "version-1"


def test_s3_retention_dry_run_preserves_latest_locked_and_held_versions() -> None:
    client = FakeS3Client()
    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    client.add_test_version("memory-backups", "civilization/orion/old.enc", "v-old", last_modified=now - timedelta(days=90))
    client.add_test_version("memory-backups", "civilization/orion/locked.enc", "v-locked", last_modified=now - timedelta(days=100), retain_until=now + timedelta(days=10))
    client.add_test_version("memory-backups", "civilization/orion/held.enc", "v-held", last_modified=now - timedelta(days=120), legal_hold=True)
    client.add_test_version("memory-backups", "civilization/orion/latest.enc", "v-latest", last_modified=now - timedelta(days=1))

    plan = plan_s3_backup_retention(
        client,
        bucket="memory-backups",
        prefix="civilization/orion",
        keep_latest=1,
        older_than_days=30,
        now=now,
    )
    dry_run = apply_s3_backup_retention(client, plan)
    assert [(v["key"], v["version_id"]) for v in plan["candidate_versions"]] == [
        ("civilization/orion/old.enc", "v-old")
    ]
    assert len(dry_run["candidate_versions"]) == 1
    assert client.delete_calls == []

    applied = apply_s3_backup_retention(client, plan, apply=True)
    assert applied["deleted_versions"] == [{"key": "civilization/orion/old.enc", "version_id": "v-old"}]
    assert client.delete_calls == [("memory-backups", "civilization/orion/old.enc", "v-old")]
    assert client.head_object(Bucket="memory-backups", Key="civilization/orion/locked.enc", VersionId="v-locked")
    assert client.head_object(Bucket="memory-backups", Key="civilization/orion/held.enc", VersionId="v-held")


def test_s3_cli_uses_environment_passphrase_and_emits_no_credentials(tmp_path: Path, monkeypatch, capsys) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.enc"
    checkpoint_dir = tmp_path / "checkpoints"
    create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)
    client = FakeS3Client()
    monkeypatch.setenv("CIVILIZATION_BACKUP_PASSPHRASE", _TEST_PASSPHRASE)
    monkeypatch.setattr(
        "civilization.engine.persistence.create_s3_client",
        lambda **_kwargs: client,
    )

    status = main([
        "replicate-s3-backup",
        "--backup",
        str(backup),
        "--bucket",
        "memory-backups",
        "--key-prefix",
        "civilization/orion",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--endpoint-url",
        "https://s3.example.test",
        "--region",
        "us-east-1",
        "--json",
    ])

    output = capsys.readouterr().out
    report = json.loads(output)
    assert status == 0
    assert report["verified"] is True
    assert _TEST_PASSPHRASE not in output
    subparsers = next(
        action
        for action in build_parser()._actions
        if "replicate-s3-backup" in (getattr(action, "choices", None) or {})
    )
    s3_help = subparsers.choices["replicate-s3-backup"].format_help()
    assert "--passphrase" not in s3_help
    assert "--access-key" not in s3_help
    assert "--secret-key" not in s3_help


def test_s3_client_fails_with_clear_optional_dependency_error_when_boto3_is_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(MemoryBackupError, match="backup-s3"):
        create_s3_client(endpoint_url="https://s3.example.test")


def test_s3_replication_cli_uses_environment_passphrase_and_has_no_credential_arguments(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.enc"
    checkpoint_dir = tmp_path / "checkpoints"
    create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)
    client = FakeS3Client()
    monkeypatch.setenv("CIVILIZATION_BACKUP_PASSPHRASE", _TEST_PASSPHRASE)
    monkeypatch.setattr("civilization.engine.persistence.create_s3_client", lambda **_kwargs: client)

    status = main([
        "replicate-s3-backup",
        "--backup",
        str(backup),
        "--bucket",
        "memory-backups",
        "--key-prefix",
        "civilization/orion",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--endpoint-url",
        "https://s3.example.test",
        "--json",
    ])

    output = capsys.readouterr().out
    report = json.loads(output)
    assert status == 0
    assert report["verified"] is True
    assert _TEST_PASSPHRASE not in output
    subparsers = next(
        action
        for action in build_parser()._actions
        if "replicate-s3-backup" in (getattr(action, "choices", None) or {})
    )
    help_text = subparsers.choices["replicate-s3-backup"].format_help()
    assert "--access-key" not in help_text
    assert "--secret-key" not in help_text
    assert "--passphrase" not in help_text


def test_s3_retention_cli_dry_runs_until_explicit_apply(tmp_path: Path, monkeypatch, capsys) -> None:
    client = FakeS3Client()
    now = datetime.now(timezone.utc)
    client.add_test_version(
        "memory-backups",
        "civilization/orion/old.enc",
        "v-old",
        last_modified=now - timedelta(days=60),
    )
    client.add_test_version(
        "memory-backups",
        "civilization/orion/latest.enc",
        "v-latest",
        last_modified=now - timedelta(days=1),
    )
    monkeypatch.setattr("civilization.engine.persistence.create_s3_client", lambda **_kwargs: client)
    args = [
        "s3-backup-retention",
        "--bucket",
        "memory-backups",
        "--prefix",
        "civilization/orion",
        "--keep-latest",
        "1",
        "--older-than-days",
        "30",
        "--json",
    ]

    assert main(args) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["dry_run"] is True
    assert client.delete_calls == []

    assert main([*args, "--apply"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["dry_run"] is False
    assert applied["deleted_versions"] == [{"key": "civilization/orion/old.enc", "version_id": "v-old"}]
    assert client.delete_calls == [("memory-backups", "civilization/orion/old.enc", "v-old")]
