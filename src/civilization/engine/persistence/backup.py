"""Authenticated encrypted backup bundles for Orion memory state.

Backups are deliberately offline: stop every writer for the state directory
before creating one. Restore validates the entire archive before atomically
promoting a new state directory and never overwrites an existing destination.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import tarfile
import tempfile
import time
from typing import Any, BinaryIO


BACKUP_FORMAT = "civilization-orion-memory-backup"
BACKUP_FORMAT_VERSION = 1
BACKUP_MANIFEST = "manifest.json"
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}\Z")
_SESSION_FILES = frozenset({"journal.ndjson", "snapshot.json"})
_CHUNK_SIZE = 1024 * 1024
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_ENCRYPTED_BACKUP_MAGIC = b"CIVORION\x00ENC\x02"
_ENCRYPTED_BACKUP_VERSION = 2
_ENCRYPTED_CHUNK_SIZE = 1024 * 1024
_ENCRYPTED_SALT_BYTES = 16
_ENCRYPTED_NONCE_PREFIX_BYTES = 8
_ENCRYPTED_TAG_BYTES = 16
_ENCRYPTED_MAX_HEADER_BYTES = 4096
_ENCRYPTED_FINAL_FRAME = 1
_ENCRYPTED_HEADER_LENGTH = struct.Struct(">I")
_ENCRYPTED_FRAME_HEADER = struct.Struct(">IIB")
_MIN_PASSPHRASE_BYTES = 16
_ARGON2ID_PARAMETERS = {"algorithm": "argon2id", "iterations": 1, "lanes": 4, "memory_cost_kib": 65536}


class MemoryBackupError(ValueError):
    """The backup is incomplete, corrupt, unsupported, or unsafe to restore."""


def create_memory_backup(
    state_dir: str | Path,
    output_file: str | Path,
    *,
    passphrase: str | None = None,
) -> dict[str, Any]:
    """Write a mode-0600 authenticated encrypted backup of Orion memory.

    The output must be outside ``state_dir``. The final filename is created
    atomically and is never replaced if it already exists. The passphrase is
    never stored in the archive or returned in the report.
    """

    passphrase = _validate_passphrase(passphrase)
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
        with os.fdopen(fd, "wb") as encrypted_file:
            writer = _EncryptedArchiveWriter(encrypted_file, passphrase)
            with tarfile.open(fileobj=writer, mode="w|", format=tarfile.PAX_FORMAT) as archive:
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
            writer.finish()
            encrypted_file.flush()
            os.fsync(encrypted_file.fileno())
        report = verify_memory_backup(temporary, passphrase=passphrase)
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


def verify_memory_backup(backup_file: str | Path, *, passphrase: str | None = None) -> dict[str, Any]:
    """Validate encryption authentication, archive paths, and payload checksums.

    Legacy v1 plaintext bundles remain readable without a passphrase. Encrypted
    v2 bundles require the passphrase used at creation.
    """

    backup = Path(backup_file).expanduser().resolve(strict=True)
    encrypted = _is_encrypted_backup(backup)
    try:
        if encrypted:
            with backup.open("rb") as source:
                reader = _AuthenticatedDecryptingReader(source, passphrase)
                manifest = _read_streaming_archive(reader)
        else:
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
    return _verification_report(manifest, encrypted=encrypted)


def replicate_memory_backup(
    backup_file: str | Path,
    destination_directory: str | Path,
    *,
    passphrase: str | None = None,
) -> dict[str, Any]:
    """Copy a verified encrypted backup to an existing distinct filesystem.

    The source remains untouched. The destination filename is the source
    filename; an existing destination is never replaced. A private temporary
    copy is authenticated before an atomic hard-link publishes it.
    """

    passphrase = _validate_passphrase(passphrase)
    source_input = Path(backup_file).expanduser()
    if source_input.is_symlink():
        raise MemoryBackupError("backup source must not be a symlink")
    source = source_input.resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(source)

    source_signature = _backup_file_signature(source)
    source_report = verify_memory_backup(source, passphrase=passphrase)
    if not source_report.get("encrypted"):
        raise MemoryBackupError("only encrypted v2 backups can be replicated")
    if _backup_file_signature(source) != source_signature:
        raise MemoryBackupError("backup source changed while it was being verified")

    target_input = Path(destination_directory).expanduser()
    if target_input.is_symlink():
        raise MemoryBackupError("backup destination directory must not be a symlink")
    target_directory_path = target_input.resolve(strict=True)
    if not target_directory_path.is_dir():
        raise NotADirectoryError(target_directory_path)
    source_device = _filesystem_device_id(source)
    target_device = _filesystem_device_id(target_directory_path)
    if source_device == target_device:
        raise MemoryBackupError("backup replica destination must be on a distinct filesystem")

    destination = target_directory_path / source.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)

    descriptor: int | None = None
    temporary: Path | None = None
    copied_digest = hashlib.sha256()
    copied_size = 0
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".civ-replica-",
            suffix=".tmp",
            dir=target_directory_path,
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as target_stream:
            descriptor = None
            with source.open("rb") as source_stream:
                while chunk := source_stream.read(_CHUNK_SIZE):
                    target_stream.write(chunk)
                    copied_digest.update(chunk)
                    copied_size += len(chunk)
            target_stream.flush()
            os.fsync(target_stream.fileno())

        if _backup_file_signature(source) != source_signature:
            raise MemoryBackupError("backup source changed while it was being replicated")
        if copied_size != source_signature[2]:
            raise MemoryBackupError("backup replica size does not match its source")

        copied_sha256 = copied_digest.hexdigest()
        replica_sha256 = _sha256_file(temporary)
        if replica_sha256 != copied_sha256:
            raise MemoryBackupError("backup replica checksum does not match its source stream")

        replica_report = verify_memory_backup(temporary, passphrase=passphrase)
        if not replica_report.get("encrypted"):
            raise MemoryBackupError("backup replica is not encrypted")

        # Same-directory hard-link publication is atomic and never replaces an
        # existing generation, including one created concurrently.
        os.link(temporary, destination)
        _fsync_directory(target_directory_path)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise

    assert temporary is not None
    temporary.unlink(missing_ok=True)
    _fsync_directory(target_directory_path)
    return {
        "replicated": True,
        "verified": True,
        "encrypted": True,
        "source": str(source),
        "replica": str(destination),
        "source_device_id": source_device,
        "destination_device_id": target_device,
        "total_bytes": copied_size,
        "sha256": copied_sha256,
        "entry_count": replica_report["entry_count"],
        "session_count": replica_report["session_count"],
    }


def restore_memory_backup(
    backup_file: str | Path,
    destination: str | Path,
    *,
    passphrase: str | None = None,
) -> dict[str, Any]:
    """Restore a verified bundle into a new directory using atomic promotion.

    The destination must not exist. Existing data is never replaced or merged.
    """

    backup = Path(backup_file).expanduser().resolve(strict=True)
    destination_path = Path(destination).expanduser().absolute()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(destination_path)
    encrypted = _is_encrypted_backup(backup)
    if encrypted and passphrase is None:
        raise MemoryBackupError("encrypted backup requires CIVILIZATION_BACKUP_PASSPHRASE")

    staging = Path(tempfile.mkdtemp(prefix=f".{destination_path.name}.restore-", dir=destination_path.parent))
    os.chmod(staging, 0o700)
    try:
        (staging / "sessions").mkdir(mode=0o700)
        try:
            if encrypted:
                with backup.open("rb") as source:
                    reader = _AuthenticatedDecryptingReader(source, passphrase)
                    manifest = _read_streaming_archive(reader, destination=staging)
            else:
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

    report = _verification_report(manifest, encrypted=encrypted)
    report["restored_to"] = str(destination_path.resolve())
    report["global_store_path"] = (
        str(destination_path / "global-memory.json")
        if any(entry["path"] == "global-memory.json" for entry in manifest["entries"])
        else None
    )
    return report


class _EncryptedArchiveWriter:
    """Write a tar stream as independently authenticated AES-GCM frames."""

    def __init__(self, target: BinaryIO, passphrase: str):
        aesgcm_type, _, _ = _crypto_primitives()
        salt = os.urandom(_ENCRYPTED_SALT_BYTES)
        self._nonce_prefix = os.urandom(_ENCRYPTED_NONCE_PREFIX_BYTES)
        key = _derive_backup_key(passphrase, salt)
        self._cipher = aesgcm_type(key)
        header = {
            "format": "civilization-orion-memory-backup-envelope",
            "format_version": _ENCRYPTED_BACKUP_VERSION,
            "cipher": "AES-256-GCM",
            "kdf": _ARGON2ID_PARAMETERS,
            "salt": base64.b64encode(salt).decode("ascii"),
            "nonce_prefix": base64.b64encode(self._nonce_prefix).decode("ascii"),
            "chunk_size": _ENCRYPTED_CHUNK_SIZE,
        }
        self._header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(self._header_bytes) > _ENCRYPTED_MAX_HEADER_BYTES:
            raise MemoryBackupError("encrypted backup header is too large")
        size_bytes = _ENCRYPTED_HEADER_LENGTH.pack(len(self._header_bytes))
        self._aad_prefix = _ENCRYPTED_BACKUP_MAGIC + size_bytes + self._header_bytes
        self._target = target
        self._counter = 0
        self._plain_offset = 0
        self._finished = False
        _write_all(target, self._aad_prefix)

    def write(self, data: bytes | bytearray | memoryview) -> int:
        if self._finished:
            raise ValueError("encrypted archive stream is already finalized")
        view = memoryview(data)
        total = len(view)
        for offset in range(0, total, _ENCRYPTED_CHUNK_SIZE):
            chunk = bytes(view[offset : offset + _ENCRYPTED_CHUNK_SIZE])
            self._write_frame(chunk, flags=0)
        self._plain_offset += total
        return total

    def tell(self) -> int:
        return self._plain_offset

    def flush(self) -> None:
        self._target.flush()

    def finish(self) -> None:
        if self._finished:
            raise ValueError("encrypted archive stream was already finalized")
        self._write_frame(b"", flags=_ENCRYPTED_FINAL_FRAME)
        self._finished = True

    def _write_frame(self, plaintext: bytes, *, flags: int) -> None:
        if self._counter > 0xFFFFFFFF:
            raise MemoryBackupError("encrypted backup exceeds the maximum frame count")
        if flags == 0 and not 0 < len(plaintext) <= _ENCRYPTED_CHUNK_SIZE:
            raise ValueError("invalid encrypted backup chunk size")
        if flags == _ENCRYPTED_FINAL_FRAME and plaintext:
            raise ValueError("final encrypted backup frame must be empty")
        frame_header = _ENCRYPTED_FRAME_HEADER.pack(self._counter, len(plaintext), flags)
        nonce = self._nonce_prefix + self._counter.to_bytes(4, "big")
        ciphertext = self._cipher.encrypt(nonce, plaintext, self._aad_prefix + frame_header)
        _write_all(self._target, frame_header)
        _write_all(self._target, ciphertext)
        self._counter += 1


def _is_encrypted_backup(backup: Path) -> bool:
    with backup.open("rb") as source:
        return source.read(len(_ENCRYPTED_BACKUP_MAGIC)) == _ENCRYPTED_BACKUP_MAGIC


class _AuthenticatedDecryptingReader:
    """Expose authenticated plaintext frames to tarfile without a plaintext temp file."""

    def __init__(self, source: BinaryIO, passphrase: str | None):
        if passphrase is None:
            raise MemoryBackupError("encrypted backup requires CIVILIZATION_BACKUP_PASSPHRASE")
        passphrase = _validate_passphrase(passphrase)
        magic = _read_exact(source, len(_ENCRYPTED_BACKUP_MAGIC), "envelope magic")
        if magic != _ENCRYPTED_BACKUP_MAGIC:
            raise MemoryBackupError("encrypted backup format is not recognized")
        header_size_bytes = _read_exact(source, _ENCRYPTED_HEADER_LENGTH.size, "header length")
        header_size = _ENCRYPTED_HEADER_LENGTH.unpack(header_size_bytes)[0]
        if not 0 < header_size <= _ENCRYPTED_MAX_HEADER_BYTES:
            raise MemoryBackupError("encrypted backup header size is invalid")
        header_bytes = _read_exact(source, header_size, "header")
        header = _decode_encryption_header(header_bytes)
        salt = _decode_header_bytes(header, "salt", _ENCRYPTED_SALT_BYTES)
        self._nonce_prefix = _decode_header_bytes(header, "nonce_prefix", _ENCRYPTED_NONCE_PREFIX_BYTES)
        aesgcm_type, _, self._invalid_tag = _crypto_primitives()
        self._cipher = aesgcm_type(_derive_backup_key(passphrase, salt))
        self._aad_prefix = magic + header_size_bytes + header_bytes
        self._source = source
        self._counter = 0
        self._buffer = bytearray()
        self._eof = False

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size < 0:
            raise ValueError("authenticated backup reader requires bounded read sizes")
        while len(self._buffer) < size and not self._eof:
            self._load_frame()
        result = bytes(self._buffer[:size])
        del self._buffer[: len(result)]
        return result

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def drain_and_validate_padding(self) -> None:
        while not self._eof:
            if any(self._buffer):
                raise MemoryBackupError("backup contains data after the tar terminator")
            self._buffer.clear()
            self._load_frame()
        if any(self._buffer):
            raise MemoryBackupError("backup contains data after the tar terminator")
        self._buffer.clear()

    def _load_frame(self) -> None:
        if self._eof:
            return
        frame_header = _read_exact(self._source, _ENCRYPTED_FRAME_HEADER.size, "frame header")
        counter, plaintext_size, flags = _ENCRYPTED_FRAME_HEADER.unpack(frame_header)
        if counter != self._counter:
            raise MemoryBackupError("encrypted backup frame sequence is invalid")
        if flags == _ENCRYPTED_FINAL_FRAME:
            if plaintext_size != 0:
                raise MemoryBackupError("encrypted backup final frame is invalid")
            ciphertext = _read_exact(self._source, _ENCRYPTED_TAG_BYTES, "final frame")
            plaintext = _decrypt_frame(
                self._cipher,
                self._nonce_prefix,
                counter,
                frame_header,
                self._aad_prefix,
                ciphertext,
                self._invalid_tag,
            )
            if plaintext:
                raise MemoryBackupError("encrypted backup final frame is invalid")
            if self._source.read(1):
                raise MemoryBackupError("encrypted backup has trailing data")
            self._eof = True
            return
        if flags != 0 or not 0 < plaintext_size <= _ENCRYPTED_CHUNK_SIZE:
            raise MemoryBackupError("encrypted backup frame metadata is invalid")
        ciphertext = _read_exact(self._source, plaintext_size + _ENCRYPTED_TAG_BYTES, "data frame")
        plaintext = _decrypt_frame(
            self._cipher,
            self._nonce_prefix,
            counter,
            frame_header,
            self._aad_prefix,
            ciphertext,
            self._invalid_tag,
        )
        if len(plaintext) != plaintext_size:
            raise MemoryBackupError("encrypted backup frame size is invalid")
        self._buffer.extend(plaintext)
        self._counter += 1


def _read_streaming_archive(
    source: _AuthenticatedDecryptingReader,
    *,
    destination: Path | None = None,
) -> dict[str, Any]:
    observed_entries: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] | None = None

    with tarfile.open(fileobj=source, mode="r|") as archive:
        for member in archive:
            if member.name == BACKUP_MANIFEST:
                if manifest is not None or not member.isfile() or member.size > _MAX_MANIFEST_BYTES:
                    raise MemoryBackupError("backup manifest is missing, duplicate, or too large")
                manifest_stream = archive.extractfile(member)
                if manifest_stream is None:
                    raise MemoryBackupError("backup manifest cannot be read")
                manifest_bytes = manifest_stream.read(_MAX_MANIFEST_BYTES + 1)
                if len(manifest_bytes) != member.size or len(manifest_bytes) > _MAX_MANIFEST_BYTES:
                    raise MemoryBackupError("backup manifest size is invalid")
                try:
                    manifest = json.loads(manifest_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise MemoryBackupError("backup manifest is not valid UTF-8 JSON") from error
                continue

            if manifest is not None:
                raise MemoryBackupError("backup contains entries after its manifest")
            if not member.isfile():
                raise MemoryBackupError(f"backup contains a non-regular entry: {member.name}")
            if not _valid_archive_path(member.name):
                raise MemoryBackupError(f"unsafe backup path: {member.name!r}")
            if member.name in observed_entries:
                raise MemoryBackupError(f"backup contains a duplicate path: {member.name}")
            if member.size < 0:
                raise MemoryBackupError(f"backup entry has an invalid size: {member.name}")
            payload = archive.extractfile(member)
            if payload is None:
                raise MemoryBackupError(f"backup entry cannot be read: {member.name}")
            if destination is None:
                observed_entries[member.name] = _hash_tar_member(payload, member.name, member.size)
            else:
                target = destination.joinpath(*PurePosixPath(member.name).parts)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                observed_entries[member.name] = _copy_tar_member(payload, target, member.name, member.size)

    source.drain_and_validate_padding()
    if manifest is None:
        raise MemoryBackupError("backup manifest is missing")
    _validate_manifest_entries(manifest, observed_entries)
    return manifest


def _hash_tar_member(stream: Any, name: str, expected_size: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(_CHUNK_SIZE):
        size += len(chunk)
        if size > expected_size:
            raise MemoryBackupError(f"backup entry exceeds its declared size: {name}")
        digest.update(chunk)
    if size != expected_size:
        raise MemoryBackupError(f"backup entry has a truncated payload: {name}")
    return {"size": size, "sha256": digest.hexdigest()}


def _copy_tar_member(stream: Any, destination: Path, name: str, expected_size: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with destination.open("xb") as target:
        os.chmod(destination, 0o600)
        while chunk := stream.read(_CHUNK_SIZE):
            size += len(chunk)
            if size > expected_size:
                raise MemoryBackupError(f"backup entry exceeds its declared size: {name}")
            digest.update(chunk)
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
    if size != expected_size:
        raise MemoryBackupError(f"backup entry has a truncated payload: {name}")
    return {"size": size, "sha256": digest.hexdigest()}


def _decode_encryption_header(header_bytes: bytes) -> dict[str, Any]:
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MemoryBackupError("encrypted backup header is not valid UTF-8 JSON") from error
    if not isinstance(header, dict):
        raise MemoryBackupError("encrypted backup header must be an object")
    expected_fields = {
        "format",
        "format_version",
        "cipher",
        "kdf",
        "salt",
        "nonce_prefix",
        "chunk_size",
    }
    if set(header) != expected_fields:
        raise MemoryBackupError("encrypted backup header has unsupported fields")
    if (
        header.get("format") != "civilization-orion-memory-backup-envelope"
        or type(header.get("format_version")) is not int
        or header["format_version"] != _ENCRYPTED_BACKUP_VERSION
        or header.get("cipher") != "AES-256-GCM"
        or type(header.get("chunk_size")) is not int
        or header["chunk_size"] != _ENCRYPTED_CHUNK_SIZE
    ):
        raise MemoryBackupError("encrypted backup header is unsupported")
    kdf = header.get("kdf")
    if not isinstance(kdf, dict) or set(kdf) != set(_ARGON2ID_PARAMETERS):
        raise MemoryBackupError("unsupported encryption parameters")
    if any(type(kdf.get(name)) is not type(value) or kdf.get(name) != value for name, value in _ARGON2ID_PARAMETERS.items()):
        raise MemoryBackupError("unsupported encryption parameters")
    return header


def _decode_header_bytes(header: dict[str, Any], field: str, expected_length: int) -> bytes:
    value = header.get(field)
    if not isinstance(value, str):
        raise MemoryBackupError(f"encrypted backup {field} is invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        raise MemoryBackupError(f"encrypted backup {field} is invalid") from error
    if len(decoded) != expected_length:
        raise MemoryBackupError(f"encrypted backup {field} is invalid")
    return decoded


def _crypto_primitives() -> tuple[type, type, type]:
    try:
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
    except ImportError as error:
        raise MemoryBackupError(
            "encrypted backups require the optional dependency; install with `pip install 'astreusn-civilization-v1[secure-backup]'`"
        ) from error
    return AESGCM, Argon2id, InvalidTag


def _derive_backup_key(passphrase: str, salt: bytes) -> bytes:
    _, argon2id_type, _ = _crypto_primitives()
    try:
        kdf = argon2id_type(
            salt=salt,
            length=32,
            iterations=_ARGON2ID_PARAMETERS["iterations"],
            lanes=_ARGON2ID_PARAMETERS["lanes"],
            memory_cost=_ARGON2ID_PARAMETERS["memory_cost_kib"],
        )
        return kdf.derive(passphrase.encode("utf-8"))
    except (UnicodeEncodeError, ValueError) as error:
        raise MemoryBackupError("could not derive encrypted backup key") from error


def _validate_passphrase(passphrase: str | None) -> str:
    if not isinstance(passphrase, str) or not passphrase:
        raise MemoryBackupError("a non-empty backup passphrase is required")
    try:
        encoded = passphrase.encode("utf-8")
    except UnicodeEncodeError as error:
        raise MemoryBackupError("backup passphrase must be valid UTF-8") from error
    if len(encoded) < _MIN_PASSPHRASE_BYTES:
        raise MemoryBackupError("backup passphrase must be at least 16 UTF-8 bytes")
    if len(encoded) > 4096:
        raise MemoryBackupError("backup passphrase is too long")
    return passphrase


def _decrypt_frame(
    cipher: Any,
    nonce_prefix: bytes,
    counter: int,
    frame_header: bytes,
    aad_prefix: bytes,
    ciphertext: bytes,
    invalid_tag: type[Exception],
) -> bytes:
    nonce = nonce_prefix + counter.to_bytes(4, "big")
    try:
        return cipher.decrypt(nonce, ciphertext, aad_prefix + frame_header)
    except invalid_tag as error:
        raise MemoryBackupError("backup authentication failed (wrong passphrase or damaged data)") from error


def _read_exact(stream: BinaryIO, length: int, description: str) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = stream.read(length - len(chunks))
        if not chunk:
            raise MemoryBackupError(f"truncated encrypted backup {description}")
        chunks.extend(chunk)
    return bytes(chunks)


def _write_all(stream: BinaryIO, data: bytes | bytearray | memoryview) -> None:
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if written is None or written <= 0:
            raise OSError("backup stream made no write progress")
        view = view[written:]


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
    actual_entries = {
        name: {"size": member.size}
        for name, member in members.items()
        if name != BACKUP_MANIFEST
    }
    _validate_manifest_entries(manifest, actual_entries)
    return manifest, members


def _validate_manifest_entries(manifest: Any, actual_entries: dict[str, dict[str, Any]]) -> None:
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
        actual = actual_entries.get(name)
        if actual is None or actual.get("size") != size:
            raise MemoryBackupError(f"backup entry is missing or has a size mismatch: {name}")
        if actual.get("sha256") is not None and actual["sha256"] != digest:
            raise MemoryBackupError(f"backup checksum mismatch: {name}")
        expected_paths.add(name)
    if set(actual_entries) != expected_paths:
        raise MemoryBackupError("backup contains entries not declared in its manifest")


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


def _backup_file_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _filesystem_device_id(path: Path) -> int:
    return path.stat().st_dev


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


def _verification_report(manifest: dict[str, Any], *, encrypted: bool = False) -> dict[str, Any]:
    entries = manifest["entries"]
    report = {
        "format": manifest["format"],
        "format_version": manifest["format_version"],
        "created_at": manifest["created_at"],
        "verified": True,
        "encrypted": encrypted,
        "entry_count": len(entries),
        "session_count": len({entry["path"].split("/")[1] for entry in entries if entry["path"].startswith("sessions/")}),
        "total_bytes": sum(entry["size"] for entry in entries),
        "includes_global_store": any(entry["path"] == "global-memory.json" for entry in entries),
    }
    if encrypted:
        report["encryption_format_version"] = _ENCRYPTED_BACKUP_VERSION
    return report


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
