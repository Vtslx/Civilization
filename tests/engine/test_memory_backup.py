from __future__ import annotations

from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

import pytest

from civilization.cli import build_parser, main
from civilization.engine.persistence import (
    MemoryBackupError,
    PersistenceConfig,
    SessionPersistence,
    create_memory_backup,
    replicate_memory_backup,
    restore_memory_backup,
    verify_memory_backup,
)


_TEST_PASSPHRASE = "test-only-backup-passphrase"


def _write_tar(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, mode="w") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o600
            archive.addfile(info, BytesIO(content))


def _read_tar(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, mode="r:*") as archive:
        return {member.name: archive.extractfile(member).read() for member in archive.getmembers()}


def _state_fixture(root: Path) -> Path:
    state = root / "state"
    session = state / "sessions" / "alice"
    session.mkdir(parents=True)
    (session / "journal.ndjson").write_text('{"t":"cell","v":{"cell_id":"episodic-000001"}}\n', encoding="utf-8")
    (session / "snapshot.json").write_text(
        json.dumps({"version": 1, "session_id": "alice", "cells": [], "links": [], "trace_events": []}),
        encoding="utf-8",
    )
    (state / "global-memory.json").write_text(json.dumps({"cells": [], "links": []}), encoding="utf-8")
    return state


def _write_legacy_backup(state: Path, path: Path) -> None:
    entries = []
    files: dict[str, bytes] = {}
    for session_dir in sorted((state / "sessions").iterdir()):
        for source in sorted(session_dir.iterdir()):
            relative = f"sessions/{session_dir.name}/{source.name}"
            payload = source.read_bytes()
            files[relative] = payload
            entries.append({"path": relative, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    global_store = state / "global-memory.json"
    if global_store.exists():
        payload = global_store.read_bytes()
        files["global-memory.json"] = payload
        entries.append({"path": "global-memory.json", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    files["manifest.json"] = json.dumps(
        {
            "format": "civilization-orion-memory-backup",
            "format_version": 1,
            "created_at": time.time(),
            "scope": ["sessions", "global-memory.json"],
            "entries": entries,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    _write_tar(path, files)


def test_memory_backup_verifies_and_restores_sessions_and_global_snapshot(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "backups" / "memory.tar"

    report = create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)
    restored = tmp_path / "restored-state"
    restore_report = restore_memory_backup(backup, restored, passphrase=_TEST_PASSPHRASE)

    assert report["verified"] is True
    assert report["session_count"] == 1
    assert report["includes_global_store"] is True
    assert verify_memory_backup(backup, passphrase=_TEST_PASSPHRASE)["total_bytes"] == report["total_bytes"]
    assert report["encrypted"] is True
    assert restore_report["restored_to"] == str(restored.resolve())
    assert restore_report["global_store_path"] == str(restored / "global-memory.json")
    assert (restored / "sessions/alice/journal.ndjson").read_bytes() == (state / "sessions/alice/journal.ndjson").read_bytes()
    assert (restored / "sessions/alice/snapshot.json").read_bytes() == (state / "sessions/alice/snapshot.json").read_bytes()
    assert (restored / "global-memory.json").read_bytes() == (state / "global-memory.json").read_bytes()
    if os.name == "posix":
        assert backup.stat().st_mode & 0o777 == 0o600


def test_memory_backup_restores_journal_into_live_persistence_reader(tmp_path: Path) -> None:
    state = tmp_path / "source-state"
    state.mkdir()
    persistence = SessionPersistence(PersistenceConfig(mode="in_process", state_dir=str(state), fsync="every_write"))
    store = persistence.hydrate("customer-42")
    cell = store.write_cell(memory_system="episodic", content="eight days camping", summary="camping duration", source="user")
    persistence.close()

    backup = tmp_path / "backup.tar"
    create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)
    restored = tmp_path / "restored-state"
    restore_memory_backup(backup, restored, passphrase=_TEST_PASSPHRASE)
    recovered_persistence = SessionPersistence(
        PersistenceConfig(mode="in_process", state_dir=str(restored), fsync="every_write")
    )
    recovered_store = recovered_persistence.hydrate("customer-42")
    results = recovered_store.read("camping duration")
    recovered_persistence.close()

    assert results and results[0].cell.cell_id == cell.cell_id
    assert results[0].cell.content == "eight days camping"


def test_memory_backup_rejects_output_inside_state_and_existing_backup(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    nested_output = state / "not-created" / "backup.tar"
    with pytest.raises(ValueError, match="outside"):
        create_memory_backup(state, nested_output, passphrase=_TEST_PASSPHRASE)
    assert not nested_output.parent.exists()

    backup = tmp_path / "memory.tar"
    create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)
    with pytest.raises(FileExistsError):
        create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)


def test_memory_backup_detects_payload_corruption(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.tar"
    _write_legacy_backup(state, backup)
    files = _read_tar(backup)
    payload_path = "sessions/alice/journal.ndjson"
    original = files[payload_path]
    files[payload_path] = original[:-1] + bytes([original[-1] ^ 1])
    corrupted = tmp_path / "corrupt.tar"
    _write_tar(corrupted, files)

    with pytest.raises(MemoryBackupError, match="checksum mismatch"):
        verify_memory_backup(corrupted)


@pytest.mark.parametrize(
    "manifest_change, expected_message",
    [
        ({"format_version": 99}, "unsupported backup format version"),
        ({"format_version": True}, "unsupported backup format version"),
        ({"entries": [{"path": "../outside", "size": 0, "sha256": "0" * 64}]}, "unsafe backup path"),
    ],
)
def test_memory_backup_rejects_unknown_version_and_traversal(
    tmp_path: Path, manifest_change: dict, expected_message: str
) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.tar"
    _write_legacy_backup(state, backup)
    files = _read_tar(backup)
    manifest = json.loads(files["manifest.json"])
    manifest.update(manifest_change)
    files["manifest.json"] = json.dumps(manifest).encode("utf-8")
    malformed = tmp_path / "malformed.tar"
    _write_tar(malformed, files)

    with pytest.raises(MemoryBackupError, match=expected_message):
        restore_memory_backup(malformed, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_memory_backup_does_not_promote_partial_restore_on_interruption(tmp_path: Path, monkeypatch) -> None:
    import civilization.engine.persistence.backup as backup_module

    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.tar"
    create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)
    destination = tmp_path / "restored"

    def fail_promotion(staging: Path, target: Path) -> None:
        raise OSError("simulated rename interruption")

    monkeypatch.setattr(backup_module, "_promote_restore", fail_promotion)
    with pytest.raises(OSError, match="simulated rename interruption"):
        restore_memory_backup(backup, destination, passphrase=_TEST_PASSPHRASE)

    assert not destination.exists()
    assert not list(tmp_path.glob(".restored.restore-*"))


def test_memory_backup_never_replaces_existing_restore_destination(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.tar"
    create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        restore_memory_backup(backup, destination, passphrase=_TEST_PASSPHRASE)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_memory_backup_cli_exposes_create_verify_and_restore() -> None:
    parser = build_parser()

    backup = parser.parse_args(["backup", "--output", "memory.tar"])
    verify = parser.parse_args(["verify-backup", "--backup", "memory.tar"])
    restore = parser.parse_args(["restore", "--backup", "memory.tar", "--destination", "restored-state"])

    assert backup.command == "backup"
    assert verify.command == "verify-backup"
    assert restore.command == "restore"


def test_memory_backup_cli_round_trip(tmp_path: Path, capsys, monkeypatch) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "cli-backup.tar"
    restored = tmp_path / "cli-restored"

    monkeypatch.setenv("CIVILIZATION_BACKUP_PASSPHRASE", _TEST_PASSPHRASE)
    assert main(["backup", "--state-dir", str(state), "--output", str(backup), "--json"]) == 0
    backup_output = capsys.readouterr().out
    assert json.loads(backup_output)["verified"] is True
    assert _TEST_PASSPHRASE not in backup_output
    assert main(["verify-backup", "--backup", str(backup), "--json"]) == 0
    verify_output = capsys.readouterr().out
    assert json.loads(verify_output)["verified"] is True
    assert _TEST_PASSPHRASE not in verify_output
    assert main(["restore", "--backup", str(backup), "--destination", str(restored), "--json"]) == 0
    restore_output = capsys.readouterr().out
    assert json.loads(restore_output)["restored_to"] == str(restored.resolve())
    assert _TEST_PASSPHRASE not in restore_output


def test_memory_backup_rejects_truncated_archive(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.tar"
    _write_legacy_backup(state, backup)
    truncated = tmp_path / "truncated.tar"
    with tarfile.open(backup, mode="r:*") as archive:
        manifest_member = archive.getmember("manifest.json")
    truncated.write_bytes(backup.read_bytes()[: manifest_member.offset_data + manifest_member.size - 1])

    with pytest.raises(MemoryBackupError):
        verify_memory_backup(truncated)


def test_memory_backup_rejects_symlinked_session_files(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    session = state / "sessions" / "alice"
    source = tmp_path / "outside.txt"
    source.write_text("must not be included", encoding="utf-8")
    linked = session / "linked.txt"
    try:
        linked.symlink_to(source)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="unexpected session state entry"):
        create_memory_backup(state, tmp_path / "backup.tar", passphrase=_TEST_PASSPHRASE)


def test_encrypted_memory_backup_hides_payload_and_rejects_wrong_passphrase(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    marker = b"private memory marker"
    (state / "sessions/alice/journal.ndjson").write_bytes(marker + b"\n")
    backup = tmp_path / "memory.enc"

    report = create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)

    encrypted = backup.read_bytes()
    assert marker not in encrypted
    assert _TEST_PASSPHRASE.encode("utf-8") not in encrypted
    assert _TEST_PASSPHRASE not in json.dumps(report)
    assert report["encrypted"] is True
    assert verify_memory_backup(backup, passphrase=_TEST_PASSPHRASE)["verified"] is True
    with pytest.raises(MemoryBackupError, match="authentication failed"):
        verify_memory_backup(backup, passphrase="wrong-passphrase")


def test_replicate_memory_backup_copies_and_authenticates_without_exposing_passphrase(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import civilization.engine.persistence.backup as backup_module

    state = _state_fixture(tmp_path)
    source = tmp_path / "memory.enc"
    target_directory = tmp_path / "replicas"
    target_directory.mkdir()
    create_memory_backup(state, source, passphrase=_TEST_PASSPHRASE)
    real_device_id = backup_module._filesystem_device_id
    target = target_directory.resolve()

    def simulated_distinct_device(path: Path) -> int:
        resolved = Path(path).resolve()
        device_id = real_device_id(resolved)
        return device_id + 1 if resolved == target else device_id

    monkeypatch.setattr(backup_module, "_filesystem_device_id", simulated_distinct_device)

    report = replicate_memory_backup(source, target_directory, passphrase=_TEST_PASSPHRASE)
    replica = target_directory / source.name

    assert report["replicated"] is True
    assert report["verified"] is True
    assert report["encrypted"] is True
    assert report["source_device_id"] != report["destination_device_id"]
    assert replica.read_bytes() == source.read_bytes()
    assert replica.stat().st_mode & 0o777 == 0o600
    assert verify_memory_backup(replica, passphrase=_TEST_PASSPHRASE)["verified"] is True
    assert _TEST_PASSPHRASE not in json.dumps(report)
    assert not list(target_directory.glob(".civ-replica-*.tmp"))


def test_replicate_memory_backup_rejects_same_filesystem_and_preserves_source(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    source = tmp_path / "memory.enc"
    target_directory = tmp_path / "same-volume-replicas"
    target_directory.mkdir()
    create_memory_backup(state, source, passphrase=_TEST_PASSPHRASE)
    original_bytes = source.read_bytes()

    with pytest.raises(MemoryBackupError, match="distinct filesystem"):
        replicate_memory_backup(source, target_directory, passphrase=_TEST_PASSPHRASE)

    assert source.read_bytes() == original_bytes
    assert list(target_directory.iterdir()) == []


def test_replicate_memory_backup_rejects_plaintext_and_existing_generations(tmp_path: Path, monkeypatch) -> None:
    import civilization.engine.persistence.backup as backup_module

    state = _state_fixture(tmp_path)
    plaintext = tmp_path / "legacy.tar"
    _write_legacy_backup(state, plaintext)
    target_directory = tmp_path / "replicas"
    target_directory.mkdir()
    with pytest.raises(MemoryBackupError, match="only encrypted v2"):
        replicate_memory_backup(plaintext, target_directory, passphrase=_TEST_PASSPHRASE)

    source = tmp_path / "memory.enc"
    create_memory_backup(state, source, passphrase=_TEST_PASSPHRASE)
    existing = target_directory / source.name
    existing.write_bytes(b"preserve this generation")
    real_device_id = backup_module._filesystem_device_id
    target = target_directory.resolve()

    def simulated_distinct_device(path: Path) -> int:
        resolved = Path(path).resolve()
        device_id = real_device_id(resolved)
        return device_id + 1 if resolved == target else device_id

    monkeypatch.setattr(backup_module, "_filesystem_device_id", simulated_distinct_device)
    with pytest.raises(FileExistsError):
        replicate_memory_backup(source, target_directory, passphrase=_TEST_PASSPHRASE)
    assert existing.read_bytes() == b"preserve this generation"


def test_replicate_memory_backup_rejects_corrupt_source_before_writing_target(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    source = tmp_path / "memory.enc"
    target_directory = tmp_path / "replicas"
    target_directory.mkdir()
    create_memory_backup(state, source, passphrase=_TEST_PASSPHRASE)
    changed = bytearray(source.read_bytes())
    changed[-1] ^= 1
    source.write_bytes(changed)

    with pytest.raises(MemoryBackupError, match="authentication failed"):
        replicate_memory_backup(source, target_directory, passphrase=_TEST_PASSPHRASE)

    assert list(target_directory.iterdir()) == []


def test_replicate_memory_backup_cleans_temporary_file_when_replica_verification_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import civilization.engine.persistence.backup as backup_module

    state = _state_fixture(tmp_path)
    source = tmp_path / "memory.enc"
    target_directory = tmp_path / "replicas"
    target_directory.mkdir()
    create_memory_backup(state, source, passphrase=_TEST_PASSPHRASE)
    real_device_id = backup_module._filesystem_device_id
    target = target_directory.resolve()

    def simulated_distinct_device(path: Path) -> int:
        resolved = Path(path).resolve()
        device_id = real_device_id(resolved)
        return device_id + 1 if resolved == target else device_id

    real_verify = backup_module.verify_memory_backup
    calls = 0

    def fail_replica_verification(path, *, passphrase=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryBackupError("simulated replica verification failure")
        return real_verify(path, passphrase=passphrase)

    monkeypatch.setattr(backup_module, "_filesystem_device_id", simulated_distinct_device)
    monkeypatch.setattr(backup_module, "verify_memory_backup", fail_replica_verification)

    with pytest.raises(MemoryBackupError, match="simulated replica verification failure"):
        replicate_memory_backup(source, target_directory, passphrase=_TEST_PASSPHRASE)

    assert list(target_directory.iterdir()) == []


def test_encrypted_memory_backup_rejects_tampering_and_truncation(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.enc"
    create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)
    original = backup.read_bytes()

    tampered = tmp_path / "tampered.enc"
    changed = bytearray(original)
    changed[-1] ^= 1
    tampered.write_bytes(changed)
    with pytest.raises(MemoryBackupError, match="authentication failed"):
        verify_memory_backup(tampered, passphrase=_TEST_PASSPHRASE)

    truncated = tmp_path / "truncated.enc"
    truncated.write_bytes(original[:-1])
    with pytest.raises(MemoryBackupError, match="truncated encrypted backup"):
        verify_memory_backup(truncated, passphrase=_TEST_PASSPHRASE)


def test_encrypted_memory_backup_rejects_unsupported_kdf_parameters_before_deriving(tmp_path: Path) -> None:
    import civilization.engine.persistence.backup as backup_module

    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.enc"
    create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)
    data = bytearray(backup.read_bytes())
    offset = len(backup_module._ENCRYPTED_BACKUP_MAGIC)
    header_size = int.from_bytes(data[offset : offset + 4], "big")
    header_start = offset + 4
    header = json.loads(data[header_start : header_start + header_size])
    header["kdf"]["iterations"] = 2
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert len(encoded) == header_size
    data[header_start : header_start + header_size] = encoded
    malformed = tmp_path / "malformed-kdf.enc"
    malformed.write_bytes(data)

    with pytest.raises(MemoryBackupError, match="unsupported encryption parameters"):
        verify_memory_backup(malformed, passphrase=_TEST_PASSPHRASE)


def test_encrypted_memory_backup_rejects_reordered_frame_counter(tmp_path: Path) -> None:
    import civilization.engine.persistence.backup as backup_module

    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.enc"
    create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)
    data = bytearray(backup.read_bytes())
    header_offset = len(backup_module._ENCRYPTED_BACKUP_MAGIC)
    header_size = int.from_bytes(data[header_offset : header_offset + 4], "big")
    first_frame = header_offset + 4 + header_size
    data[first_frame : first_frame + 4] = (1).to_bytes(4, "big")
    reordered = tmp_path / "reordered.enc"
    reordered.write_bytes(data)

    with pytest.raises(MemoryBackupError, match="frame sequence is invalid"):
        verify_memory_backup(reordered, passphrase=_TEST_PASSPHRASE)


def test_encrypted_backup_streams_multiple_frames_and_long_pax_paths(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    old_session = state / "sessions" / "alice"
    session_id = "s" * 120
    session = state / "sessions" / session_id
    old_session.rename(session)
    snapshot_path = session / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["session_id"] = session_id
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    journal = "".join(
        json.dumps({"t": "cell", "v": {"cell_id": f"episodic-{index:06d}"}}) + "\n"
        for index in range(30000)
    ).encode("utf-8")
    (session / "journal.ndjson").write_bytes(journal)
    backup = tmp_path / "multi-frame.enc"
    restored = tmp_path / "multi-frame-restored"

    report = create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)
    verified = verify_memory_backup(backup, passphrase=_TEST_PASSPHRASE)
    restore_memory_backup(backup, restored, passphrase=_TEST_PASSPHRASE)

    assert report["encrypted"] is True
    assert verified["total_bytes"] == report["total_bytes"]
    assert (restored / "sessions" / session_id / "journal.ndjson").read_bytes() == journal


def test_encrypted_verification_does_not_spool_plaintext_archive(tmp_path: Path, monkeypatch) -> None:
    import civilization.engine.persistence.backup as backup_module

    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.enc"
    create_memory_backup(state, backup, passphrase=_TEST_PASSPHRASE)

    def reject_plaintext_tempfile(*args, **kwargs):
        raise AssertionError("encrypted verification must not write a plaintext tar temporary file")

    monkeypatch.setattr(backup_module.tempfile, "TemporaryFile", reject_plaintext_tempfile)
    assert verify_memory_backup(backup, passphrase=_TEST_PASSPHRASE)["verified"] is True


def test_legacy_plaintext_bundle_remains_verifiable_and_restorable(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "legacy.tar"
    _write_legacy_backup(state, backup)

    report = verify_memory_backup(backup)
    restored = tmp_path / "legacy-restored"
    restore_report = restore_memory_backup(backup, restored)

    assert report["verified"] is True
    assert report["encrypted"] is False
    assert restore_report["restored_to"] == str(restored.resolve())
    assert (restored / "global-memory.json").read_bytes() == (state / "global-memory.json").read_bytes()


def test_backup_cli_requires_environment_passphrase_and_has_no_passphrase_argument(tmp_path: Path, monkeypatch) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.enc"
    monkeypatch.delenv("CIVILIZATION_BACKUP_PASSPHRASE", raising=False)

    with pytest.raises(SystemExit, match="CIVILIZATION_BACKUP_PASSPHRASE"):
        main(["backup", "--state-dir", str(state), "--output", str(backup)])
    with pytest.raises(SystemExit):
        main(["backup", "--state-dir", str(state), "--output", str(backup), "--passphrase", _TEST_PASSPHRASE])
    assert not backup.exists()


def test_replicate_backup_cli_requires_environment_passphrase_and_has_no_passphrase_argument(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backup = tmp_path / "memory.enc"
    destination = tmp_path / "replicas"
    destination.mkdir()
    monkeypatch.delenv("CIVILIZATION_BACKUP_PASSPHRASE", raising=False)

    with pytest.raises(SystemExit, match="CIVILIZATION_BACKUP_PASSPHRASE"):
        main(["replicate-backup", "--backup", str(backup), "--destination-dir", str(destination)])
    with pytest.raises(SystemExit):
        main([
            "replicate-backup",
            "--backup",
            str(backup),
            "--destination-dir",
            str(destination),
            "--passphrase",
            _TEST_PASSPHRASE,
        ])
    assert list(destination.iterdir()) == []


def test_replicate_backup_cli_reports_verified_replica_without_secret(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import civilization.engine.persistence.backup as backup_module

    state = _state_fixture(tmp_path)
    source = tmp_path / "memory.enc"
    target_directory = tmp_path / "replicas"
    target_directory.mkdir()
    create_memory_backup(state, source, passphrase=_TEST_PASSPHRASE)
    real_device_id = backup_module._filesystem_device_id
    target = target_directory.resolve()

    def simulated_distinct_device(path: Path) -> int:
        resolved = Path(path).resolve()
        device_id = real_device_id(resolved)
        return device_id + 1 if resolved == target else device_id

    monkeypatch.setattr(backup_module, "_filesystem_device_id", simulated_distinct_device)
    monkeypatch.setenv("CIVILIZATION_BACKUP_PASSPHRASE", _TEST_PASSPHRASE)

    assert main([
        "replicate-backup",
        "--backup",
        str(source),
        "--destination-dir",
        str(target_directory),
        "--json",
    ]) == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["verified"] is True
    assert report["encrypted"] is True
    assert _TEST_PASSPHRASE not in output


def test_memory_backup_rejects_short_passphrases(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    with pytest.raises(MemoryBackupError, match="non-empty backup passphrase is required"):
        create_memory_backup(state, tmp_path / "missing.enc")
    with pytest.raises(MemoryBackupError, match="at least 16 UTF-8 bytes"):
        create_memory_backup(state, tmp_path / "weak.enc", passphrase="too-short")
    assert not (tmp_path / "missing.enc").exists()
    assert not (tmp_path / "weak.enc").exists()


def test_new_encrypted_backup_fails_closed_without_optional_crypto_extra(tmp_path: Path, monkeypatch) -> None:
    import builtins

    state = _state_fixture(tmp_path)
    output = tmp_path / "no-crypto.enc"
    original_import = builtins.__import__

    def without_cryptography(name, *args, **kwargs):
        if name.startswith("cryptography"):
            raise ModuleNotFoundError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cryptography)
    with pytest.raises(MemoryBackupError, match="secure-backup"):
        create_memory_backup(state, output, passphrase=_TEST_PASSPHRASE)
    assert not output.exists()
