"""Durable journal and snapshot storage for session memory.

The storage format is deliberately simple so that recovery is obviously correct:

- ``journal.ndjson``  append-only records, one JSON object per line, written with
  ``os.write`` on an ``O_APPEND`` file descriptor. Because the write goes to the
  operating system immediately, an abrupt **process** death loses nothing that
  was already acknowledged; a **power** loss loses at most the tail that the
  configured fsync policy had not yet committed.
- ``snapshot.json``   a compacted full state. Written to a temporary file, fsynced
  and atomically renamed, then the journal is truncated.

Every record is an idempotent upsert of one entity, so replaying a journal on top
of a snapshot — including the records a crash may have left behind after the
snapshot was written — can never corrupt state. A torn final line (a crash in the
middle of a write) is detected and ignored; everything before it is kept.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Iterator, Mapping

JOURNAL_NAME = "journal.ndjson"
SNAPSHOT_NAME = "snapshot.json"
FORMAT_VERSION = 1

# Record kinds. Each one is an upsert keyed by the entity's own id.
RECORD_CELL = "cell"
RECORD_LINK = "link"
RECORD_TRACE = "trace"
# Clears the store before the records that follow it. Used when a session's
# contents are replaced wholesale (for example a global-store load).
RECORD_RESET = "reset"


def session_directory(root: str | Path, session_id: str) -> Path:
    """Directory that holds one session's journal and snapshot.

    Session ids are validated by the API, but the path is sanitised defensively:
    anything outside ``[A-Za-z0-9_.:-]`` becomes ``_`` so a hostile id can never
    escape the state directory.
    """

    safe = "".join(character if (character.isalnum() or character in "_.:-") else "_" for character in session_id)
    if not safe or safe in {".", ".."}:
        safe = "_" + safe
    return Path(root) / "sessions" / safe


def _fsync_directory(path: Path) -> None:
    """Persist a rename or unlink in the directory entry itself."""

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


def encode_record(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


class SessionJournal:
    """Append-only journal for one session, plus its snapshot."""

    def __init__(self, directory: str | Path, *, fsync: bool = False) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.directory / JOURNAL_NAME
        self.snapshot_path = self.directory / SNAPSHOT_NAME
        self._fd: int | None = None
        self.records_written = 0
        self.records_since_snapshot = 0
        self.records_fsynced = 0
        self.last_fsync_at: float | None = None
        self._dirty = False

    # -- lifecycle ---------------------------------------------------------
    @property
    def fd(self) -> int:
        if self._fd is None:
            self._fd = os.open(self.journal_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        return self._fd

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.fsync(self._fd)
            except OSError:
                pass
            os.close(self._fd)
            self._fd = None

    # -- writing -----------------------------------------------------------
    def append(self, record: Mapping[str, Any], *, fsync: bool | None = None) -> None:
        """Append one record; the bytes leave the process immediately."""

        os.write(self.fd, encode_record(record))
        self.records_written += 1
        self.records_since_snapshot += 1
        self._dirty = True
        if fsync if fsync is not None else False:
            self.fsync()

    def append_many(self, records: Iterable[Mapping[str, Any]], *, fsync: bool | None = None) -> None:
        payload = b"".join(encode_record(record) for record in records)
        if not payload:
            return
        os.write(self.fd, payload)
        count = payload.count(b"\n")
        self.records_written += count
        self.records_since_snapshot += count
        self._dirty = True
        if fsync if fsync is not None else False:
            self.fsync()

    def fsync(self) -> bool:
        """Commit buffered writes to stable storage. Returns True when work was done."""

        if self._fd is None:
            return False
        try:
            os.fsync(self._fd)
        except OSError:
            return False
        self.records_fsynced = self.records_written
        self.last_fsync_at = time.time()
        self._dirty = False
        return True

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def pending_records(self) -> int:
        return max(0, self.records_written - self.records_fsynced)

    # -- snapshot and compaction ------------------------------------------
    def write_snapshot(self, payload: Mapping[str, Any]) -> None:
        """Atomically replace the snapshot, then truncate the journal.

        Order matters: the snapshot is durable before the journal is reset, and
        both steps are idempotent on replay, so a crash in between is harmless.
        """

        document = {"version": FORMAT_VERSION, "written_at": time.time(), **payload}
        temporary = self.snapshot_path.with_name(f".{SNAPSHOT_NAME}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.snapshot_path)
        _fsync_directory(self.directory)
        self.truncate_journal()

    def truncate_journal(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        with open(self.journal_path, "wb"):
            pass
        _fsync_directory(self.directory)
        self.records_since_snapshot = 0
        self.records_written = 0
        self.records_fsynced = 0
        self._dirty = False

    # -- reading -----------------------------------------------------------
    def load_snapshot(self) -> dict[str, Any] | None:
        if not self.snapshot_path.exists():
            return None
        try:
            with self.snapshot_path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        return document if isinstance(document, dict) else None

    def iter_journal(self) -> Iterator[dict[str, Any]]:
        """Yield journal records, stopping cleanly at a torn tail."""

        if not self.journal_path.exists():
            return
        with self.journal_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    # A crash can leave a partial final line; everything before it
                    # is still valid and is kept.
                    return
                if isinstance(record, dict):
                    yield record

    def usage(self) -> dict[str, Any]:
        journal_bytes = self.journal_path.stat().st_size if self.journal_path.exists() else 0
        snapshot_bytes = self.snapshot_path.stat().st_size if self.snapshot_path.exists() else 0
        return {
            "journal_bytes": journal_bytes,
            "snapshot_bytes": snapshot_bytes,
            "records_written": self.records_written,
            "records_since_snapshot": self.records_since_snapshot,
            "pending_records": self.pending_records,
            "last_fsync_at": self.last_fsync_at,
        }
