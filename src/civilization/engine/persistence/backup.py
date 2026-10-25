"""Checksummed offline backup bundles for Orion memory state.

Backups are deliberately offline: stop every writer for the state directory
before creating one. Restore validates the entire archive before atomically
promoting a new state directory and never overwrites an existing destination.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import time
from typing import Any


BACKUP_FORMAT = "civilization-orion-memory-backup"
BACKUP_FORMAT_VERSION = 1
BACKUP_MANIFEST = "manifest.json"
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}\Z")
_SESSION_FILES = frozenset({"journal.ndjson", "snapshot.json"})
_CHUNK_SIZE = 1024 * 1024
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024


class MemoryBackupError(ValueError):
    """The backup is incomplete, corrupt, unsupported, or unsafe to restore."""


def create_memory_backup(state_dir: str | Path, output_file: str | Path) -> dict[str, Any]:
    """Write a mode-0600 tar bundle of session journals and the global snapshot.

    The output must be outside ``state_dir``. The final filename is created
    atomically and is never replaced if it already exists.
    """

    root = Path(state_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    output = Path(output_file).expanduser().absolute()
    output_resolved = output.resolve(strict=False)
    if _is_relative_to(output_resolved, root):
        raise ValueError("backup output must be outside the memory state directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    files = _collect_memory_files(root)
    created_at = time.time()
    entries = [
        {"path": archive_path, "size": path.stat().st_size, "sha256": _sha256_file(path)}
        for archive_path, path in files
    ]
    manifest = {
        "format": BACKUP_FORMAT,
        "format_version": BACKUP_FORMAT_VERSION,
        "created_at": created_at,
        "scope": ["sessions", "global-memory.json"],
        "entries": entries,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")

    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as archive_file:
            with tarfile.open(fileobj=archive_file, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for archive_path, source_path in files:
                    info = tarfile.TarInfo(archive_path)
                    info.size = source_path.stat().st_size
                    info.mode = 0o600
                    info.mtime = int(created_at)
                    with source_path.open("rb") as source:
                        archive.addfile(info, source)
                manifest_info = tarfile.TarInfo(BACKUP_MANIFEST)
                manifest_info.size = len(manifest_bytes)
                manifest_info.mode = 0o600
                manifest_info.mtime = int(created_at)
                archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
            archive_file.flush()
            os.fsync(archive_file.fileno())
        report = verify_memory_backup(temporary)
        os.chmod(temporary, 0o600)
        # A hard-link publish is atomic and fails instead of replacing an
        # existing backup, including one created concurrently.
        os.link(temporary, output)
        _fsync_directory(output.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink(missing_ok=True)
    _fsync_directory(output.parent)
    return report


def verify_memory_backup(backup_file: str | Path) -> dict[str, Any]:
    """Validate archive paths, manifest schema, payload sizes, and SHA-256 sums."""

    backup = Path(backup_file).expanduser().resolve(strict=True)
    try:
        with tarfile.open(backup, mode="r:*") as archive:
            manifest, members = _load_archive_manifest(archive)
            for entry in manifest["entries"]:
                member = members[entry["path"]]
                payload = archive.extractfile(member)
                if payload is None:
                    raise MemoryBackupError(f"backup entry cannot be read: {entry['path']}")
                _verify_stream(payload, entry)
    except (tarfile.TarError, OSError) as error:
        raise MemoryBackupError(f"could not read backup: {error}") from error
    return _verification_report(manifest)


def restore_memory_backup(backup_file: str | Path, destination: str | Path) -> dict[str, Any]:
    """Restore a verified bundle into a new directory using atomic promotion.

    The destination must not exist. Existing data is never replaced or merged.
    """

    backup = Path(backup_file).expanduser().resolve(strict=True)
    destination_path = Path(destination).expanduser().absolute()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(destination_path)

    staging = Path(tempfile.mkdtemp(prefix=f".{destination_path.name}.restore-", dir=destination_path.parent))
    os.chmod(staging, 0o700)
    try:
        (staging / "sessions").mkdir(mode=0o700)
        try:
            with tarfile.open(backup, mode="r:*") as archive:
                manifest, members = _load_archive_manifest(archive)
                for entry in manifest["entries"]:
                    member = members[entry["path"]]
                    source = archive.extractfile(member)
                    if source is None:
                        raise MemoryBackupError(f"backup entry cannot be read: {entry['path']}")
                    target = staging.joinpath(*PurePosixPath(entry["path"]).parts)
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    _copy_and_verify(source, target, entry)
        except (tarfile.TarError, OSError) as error:
            raise MemoryBackupError(f"could not restore backup: {error}") from error
        _fsync_tree(staging)
        _promote_restore(staging, destination_path)
        _fsync_directory(destination_path.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    report = _verification_report(manifest)
    report["restored_to"] = str(destination_path.resolve())
    report["global_store_path"] = (
        str(destination_path / "global-memory.json")
        if any(entry["path"] == "global-memory.json" for entry in manifest["entries"])
        else None
    )
    return report


def _collect_memory_files(root: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    sessions = root / "sessions"
    if sessions.is_symlink():
        raise ValueError("sessions directory must not be a symlink")
    if sessions.exists():
        if not sessions.is_dir():
            raise NotADirectoryError(sessions)
        for session_dir in sorted(sessions.iterdir()):
            if session_dir.is_symlink() or not session_dir.is_dir():
                raise ValueError(f"unexpected entry in sessions directory: {session_dir.name}")
            if not _SESSION_ID_RE.fullmatch(session_dir.name):
                raise ValueError(f"invalid session directory name: {session_dir.name}")
            for item in sorted(session_dir.iterdir()):
                if item.name == ".snapshot.json.tmp":
                    # An interrupted atomic snapshot write leaves a disposable
                    # temp file; replay uses the prior snapshot and journal.
                    continue
                if item.is_symlink() or not item.is_file() or item.name not in _SESSION_FILES:
                    raise ValueError(f"unexpected session state entry: {session_dir.name}/{item.name}")
                files.append((f"sessions/{session_dir.name}/{item.name}", item))

    global_store = root / "global-memory.json"
    if global_store.is_symlink():
        raise ValueError("global-memory.json must not be a symlink")
    if global_store.exists():
        if not global_store.is_file():
            raise ValueError("global-memory.json must be a regular file")
        files.append(("global-memory.json", global_store))
    return files


def _load_archive_manifest(archive: tarfile.TarFile) -> tuple[dict[str, Any], dict[str, tarfile.TarInfo]]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        if not member.isfile():
            raise MemoryBackupError(f"backup contains a non-regular entry: {member.name}")
        if member.name in members:
            raise MemoryBackupError(f"backup contains a duplicate path: {member.name}")
        members[member.name] = member
    manifest_member = members.get(BACKUP_MANIFEST)
    if manifest_member is None or manifest_member.size > _MAX_MANIFEST_BYTES:
        raise MemoryBackupError("backup manifest is missing or too large")
    stream = archive.extractfile(manifest_member)
    if stream is None:
        raise MemoryBackupError("backup manifest cannot be read")
    try:
        manifest = json.loads(stream.read(_MAX_MANIFEST_BYTES + 1).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MemoryBackupError("backup manifest is not valid UTF-8 JSON") from error
    if not isinstance(manifest, dict) or manifest.get("format") != BACKUP_FORMAT:
        raise MemoryBackupError("backup format is not recognized")
    format_version = manifest.get("format_version")
    if type(format_version) is not int or format_version != BACKUP_FORMAT_VERSION:
        raise MemoryBackupError(f"unsupported backup format version: {manifest.get('format_version')!r}")
    created_at = manifest.get("created_at")
    if isinstance(created_at, bool) or not isinstance(created_at, (int, float)):
        raise MemoryBackupError("backup manifest has an invalid creation time")
    try:
        finite_created_at = math.isfinite(float(created_at))
    except (OverflowError, ValueError):
        finite_created_at = False
    if not finite_created_at:
        raise MemoryBackupError("backup manifest has an invalid creation time")
    if manifest.get("scope") != ["sessions", "global-memory.json"]:
        raise MemoryBackupError("backup manifest has an unsupported scope")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise MemoryBackupError("backup manifest entries must be a list")

    expected_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise MemoryBackupError("backup manifest entry must be an object")
        name, size, digest = entry.get("path"), entry.get("size"), entry.get("sha256")
        if not isinstance(name, str) or not _valid_archive_path(name):
            raise MemoryBackupError(f"unsafe backup path: {name!r}")
        if name in expected_paths:
            raise MemoryBackupError(f"duplicate manifest path: {name}")
        if type(size) is not int or size < 0:
            raise MemoryBackupError(f"invalid size in backup manifest: {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise MemoryBackupError(f"invalid SHA-256 in backup manifest: {name}")
        member = members.get(name)
        if member is None or member.size != size:
            raise MemoryBackupError(f"backup entry is missing or has a size mismatch: {name}")
        expected_paths.add(name)
    if set(members) != expected_paths | {BACKUP_MANIFEST}:
        raise MemoryBackupError("backup contains entries not declared in its manifest")
    return manifest, members


def _valid_archive_path(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or str(path) != name or any(part in {"", ".", ".."} for part in path.parts):
        return False
    parts = path.parts
    if len(parts) == 1:
        return parts[0] == "global-memory.json"
    return (
        len(parts) == 3
        and parts[0] == "sessions"
        and bool(_SESSION_ID_RE.fullmatch(parts[1]))
        and parts[2] in _SESSION_FILES
    )


def _verify_stream(stream: Any, entry: dict[str, Any]) -> None:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(_CHUNK_SIZE):
        size += len(chunk)
        digest.update(chunk)
    if size != entry["size"] or digest.hexdigest() != entry["sha256"]:
        raise MemoryBackupError(f"backup checksum mismatch: {entry['path']}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_and_verify(stream: Any, destination: Path, entry: dict[str, Any]) -> None:
    digest = hashlib.sha256()
    size = 0
    with destination.open("xb") as target:
        os.chmod(destination, 0o600)
        while chunk := stream.read(_CHUNK_SIZE):
            size += len(chunk)
            if size > entry["size"]:
                raise MemoryBackupError(f"backup entry exceeds its declared size: {entry['path']}")
            digest.update(chunk)
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
    if size != entry["size"] or digest.hexdigest() != entry["sha256"]:
        raise MemoryBackupError(f"backup checksum mismatch: {entry['path']}")


def _verification_report(manifest: dict[str, Any]) -> dict[str, Any]:
    entries = manifest["entries"]
    return {
        "format": manifest["format"],
        "format_version": manifest["format_version"],
        "created_at": manifest["created_at"],
        "verified": True,
        "entry_count": len(entries),
        "session_count": len({entry["path"].split("/")[1] for entry in entries if entry["path"].startswith("sessions/")}),
        "total_bytes": sum(entry["size"] for entry in entries),
        "includes_global_store": any(entry["path"] == "global-memory.json" for entry in entries),
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_file():
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        elif path.is_dir():
            directories.append(path)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _promote_restore(staging: Path, destination: Path) -> None:
    os.rename(staging, destination)
