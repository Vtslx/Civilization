from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import tarfile

import pytest

from civilization.cli import build_parser, main
from civilization.engine.persistence import (
    MemoryBackupError,
    PersistenceConfig,
    SessionPersistence,
    create_memory_backup,
    restore_memory_backup,
    verify_memory_backup,
)


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


def test_memory_backup_verifies_and_restores_sessions_and_global_snapshot(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "backups" / "memory.tar"

    report = create_memory_backup(state, backup)
    restored = tmp_path / "restored-state"
    restore_report = restore_memory_backup(backup, restored)

    assert report["verified"] is True
    assert report["session_count"] == 1
    assert report["includes_global_store"] is True
    assert verify_memory_backup(backup)["total_bytes"] == report["total_bytes"]
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
    create_memory_backup(state, backup)
    restored = tmp_path / "restored-state"
    restore_memory_backup(backup, restored)
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
        create_memory_backup(state, nested_output)
    assert not nested_output.parent.exists()

    backup = tmp_path / "memory.tar"
    create_memory_backup(state, backup)
    with pytest.raises(FileExistsError):
        create_memory_backup(state, backup)


def test_memory_backup_detects_payload_corruption(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.tar"
    create_memory_backup(state, backup)
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
    create_memory_backup(state, backup)
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
    create_memory_backup(state, backup)
    destination = tmp_path / "restored"

    def fail_promotion(staging: Path, target: Path) -> None:
        raise OSError("simulated rename interruption")

    monkeypatch.setattr(backup_module, "_promote_restore", fail_promotion)
    with pytest.raises(OSError, match="simulated rename interruption"):
        restore_memory_backup(backup, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".restored.restore-*"))


def test_memory_backup_never_replaces_existing_restore_destination(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.tar"
    create_memory_backup(state, backup)
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        restore_memory_backup(backup, destination)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_memory_backup_cli_exposes_create_verify_and_restore() -> None:
    parser = build_parser()

    backup = parser.parse_args(["backup", "--output", "memory.tar"])
    verify = parser.parse_args(["verify-backup", "--backup", "memory.tar"])
    restore = parser.parse_args(["restore", "--backup", "memory.tar", "--destination", "restored-state"])

    assert backup.command == "backup"
    assert verify.command == "verify-backup"
    assert restore.command == "restore"


def test_memory_backup_cli_round_trip(tmp_path: Path, capsys) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "cli-backup.tar"
    restored = tmp_path / "cli-restored"

    assert main(["backup", "--state-dir", str(state), "--output", str(backup), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True
    assert main(["verify-backup", "--backup", str(backup), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True
    assert main(["restore", "--backup", str(backup), "--destination", str(restored), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["restored_to"] == str(restored.resolve())


def test_memory_backup_rejects_truncated_archive(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path)
    backup = tmp_path / "memory.tar"
    create_memory_backup(state, backup)
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
        create_memory_backup(state, tmp_path / "backup.tar")
