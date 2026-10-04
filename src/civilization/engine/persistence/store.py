"""Persistent session memory: hydrate on first use, journal every mutation.

The in-memory store stays the read path — retrieval, ranking, and the decision
context are served from memory exactly as before. Persistence is a write-behind
layer:

1. every mutation appends one idempotent record to the session journal,
2. a background thread commits the journal with ``fsync`` on the configured
   interval (or on every write, if the deployment asks for it),
3. once enough records accumulate, the journal is compacted into a snapshot,
4. on startup (or on the first request for a session) the store is rebuilt from
   the snapshot plus the journal tail.

Because appends use ``os.write`` on an append-only descriptor, a process crash
loses nothing that was acknowledged. A power loss loses at most the records
written since the last commit — bounded by ``fsync`` policy — and never
corrupts what was already committed: replay stops at a torn tail and every
record is an idempotent upsert.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import shutil
import threading
import time
from typing import Any, Iterator, Mapping, Sequence

from ..stages.stage73_orion_memory_kernel import (
    MemoryCell,
    MemoryLink,
    MemoryLinkType,
    MemoryTraceEvent,
    MemoryTraceAction,
    MemorySystem,
    OrionMemoryStore,
)
from .journal import (
    RECORD_CELL,
    RECORD_LINK,
    RECORD_RESET,
    RECORD_TRACE,
    SNAPSHOT_NAME,
    session_directory,
)
from .sink import (
    MODE_NONE,
    PersistenceConfig,
    PersistenceSink,
    build_sink,
)

GLOBAL_SESSION = "__global__"


def cell_from_dict(payload: Mapping[str, Any]) -> MemoryCell:
    return MemoryCell(
        cell_id=str(payload["cell_id"]),
        memory_system=MemorySystem(str(payload["memory_system"])),
        content=str(payload.get("content", "")),
        summary=str(payload.get("summary", "")),
        source=str(payload.get("source", "")),
        confidence=float(payload.get("confidence", 1.0)),
        importance=float(payload.get("importance", 1.0)),
        created_at=float(payload.get("created_at", 0.0)),
        updated_at=float(payload.get("updated_at", 0.0)),
        time_index=float(payload.get("time_index", 0.0)),
        decay_state=str(payload.get("decay_state", "active")),
        consolidation_state=str(payload.get("consolidation_state", "raw")),
        metadata=dict(payload.get("metadata") or {}),
    )


def link_from_dict(payload: Mapping[str, Any]) -> MemoryLink:
    return MemoryLink(
        source_cell_id=str(payload["source_cell_id"]),
        target_cell_id=str(payload["target_cell_id"]),
        link_type=MemoryLinkType(str(payload["link_type"])),
        weight=float(payload.get("weight", 1.0)),
        created_at=float(payload.get("created_at", 0.0)),
        metadata=dict(payload.get("metadata") or {}),
    )


def trace_from_dict(payload: Mapping[str, Any]) -> MemoryTraceEvent:
    return MemoryTraceEvent(
        event_id=str(payload["event_id"]),
        action=MemoryTraceAction(str(payload["action"])),
        cell_ids=tuple(str(value) for value in payload.get("cell_ids") or ()),
        created_at=float(payload.get("created_at", 0.0)),
        details=dict(payload.get("details") or {}),
    )


def _next_counter(cells: Sequence[MemoryCell], traces: Sequence[MemoryTraceEvent]) -> tuple[int, int]:
    next_cell = 1
    for cell in cells:
        _, _, suffix = cell.cell_id.rpartition("-")
        if suffix.isdigit():
            next_cell = max(next_cell, int(suffix) + 1)
    next_trace = 1
    for event in traces:
        _, _, suffix = event.event_id.rpartition("-")
        if suffix.isdigit():
            next_trace = max(next_trace, int(suffix) + 1)
    return next_cell, next_trace


def apply_records(store: OrionMemoryStore, records: Sequence[Mapping[str, Any]]) -> int:
    """Apply journal records to a store. Idempotent: re-applying changes nothing."""

    applied = 0
    link_keys = {
        (link.source_cell_id, link.target_cell_id, link.link_type.value, link.created_at) for link in store.links
    }
    for record in records:
        kind = record.get("t") or record.get("type")
        payload = record.get("v") or record.get("value")
        if kind == RECORD_RESET:
            store.cells.clear()
            store.links.clear()
            store.trace_events.clear()
            link_keys.clear()
            applied += 1
            continue
        if not isinstance(payload, Mapping):
            continue
        if kind == RECORD_CELL:
            cell = cell_from_dict(payload)
            store.cells[cell.cell_id] = cell
            applied += 1
        elif kind == RECORD_LINK:
            link = link_from_dict(payload)
            key = (link.source_cell_id, link.target_cell_id, link.link_type.value, link.created_at)
            if key not in link_keys:
                store.links.append(link)
                link_keys.add(key)
                applied += 1
        elif kind == RECORD_TRACE:
            event = trace_from_dict(payload)
            if all(existing.event_id != event.event_id for existing in store.trace_events):
                store.trace_events.append(event)
                applied += 1
    next_cell, next_trace = _next_counter(list(store.cells.values()), store.trace_events)
    store._next_cell_id = next_cell  # noqa: SLF001 - restoring the kernel's own counter
    store._next_event_id = next_trace  # noqa: SLF001
    return applied


def _records_from_snapshot(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cell in document.get("cells") or []:
        if isinstance(cell, Mapping):
            records.append({"t": RECORD_CELL, "v": dict(cell)})
    for link in document.get("links") or []:
        if isinstance(link, Mapping):
            records.append({"t": RECORD_LINK, "v": dict(link)})
    for event in document.get("trace_events") or []:
        if isinstance(event, Mapping):
            records.append({"t": RECORD_TRACE, "v": dict(event)})
    return records


class PersistentOrionMemoryStore(OrionMemoryStore):
    """An :class:`OrionMemoryStore` whose mutations are journaled."""

    def __init__(self, *, session_id: str, persistence: "SessionPersistence", now_fn: Any = time.time) -> None:
        super().__init__(now_fn=now_fn)
        self.session_id = session_id
        self._persistence = persistence

    # -- recording ---------------------------------------------------------
    def _record_cell(self, cell: MemoryCell) -> None:
        self._persistence.record(self.session_id, {"t": RECORD_CELL, "v": cell.to_dict()})
        self._persistence.after_record(self.session_id, self)

    def _record_link(self, link: MemoryLink) -> None:
        self._persistence.record(self.session_id, {"t": RECORD_LINK, "v": link.to_dict()})

    def _record_trace(self, event: MemoryTraceEvent) -> None:
        if self._persistence.config.persist_traces:
            self._persistence.record(self.session_id, {"t": RECORD_TRACE, "v": event.to_dict()})

    # -- kernel overrides --------------------------------------------------
    def _trace(self, action: MemoryTraceAction, cell_ids: tuple[str, ...], details: dict[str, Any] | None = None) -> None:
        super()._trace(action, cell_ids, details)
        if self.trace_events:
            self._record_trace(self.trace_events[-1])

    def write_cell(self, **kwargs: Any) -> MemoryCell:
        cell = super().write_cell(**kwargs)
        self._record_cell(cell)
        return cell

    def update_cell(self, cell_id: str, **kwargs: Any) -> MemoryCell:
        cell = super().update_cell(cell_id, **kwargs)
        self._record_cell(cell)
        return cell

    def link_cells(self, *args: Any, **kwargs: Any) -> MemoryLink:
        link = super().link_cells(*args, **kwargs)
        self._record_link(link)
        return link

    def mark_conflict(self, *args: Any, **kwargs: Any) -> MemoryLink:
        link = super().mark_conflict(*args, **kwargs)
        self._record_link(link)
        for cell_id in (link.source_cell_id, link.target_cell_id):
            cell = self.cells.get(cell_id)
            if cell is not None:
                self._record_cell(cell)
        return link

    def expire_working_memory(self) -> list[MemoryCell]:
        expired = super().expire_working_memory()
        for cell in expired:
            self._record_cell(cell)
        return expired

    def replace_contents(self, cells: Sequence[MemoryCell], links: Sequence[MemoryLink]) -> None:
        """Replace the whole store with the given contents, journaling the swap.

        A reset record is written first so that replay cannot resurrect cells the
        replacement removed. Used by the global store's explicit load.
        """

        self.cells.clear()
        self.links.clear()
        self.trace_events.clear()
        self._persistence.record(self.session_id, {"t": RECORD_RESET})
        for cell in cells:
            self.cells[cell.cell_id] = cell
            self._record_cell(cell)
        for link in links:
            self.links.append(link)
            self._record_link(link)
        next_cell, next_trace = _next_counter(list(self.cells.values()), self.trace_events)
        self._next_cell_id = next_cell  # noqa: SLF001 - keep ids monotonic
        self._next_event_id = max(self._next_event_id, next_trace)  # noqa: SLF001


@dataclass
class PersistenceStats:
    hydrated_sessions: int = 0
    replayed_records: int = 0
    snapshots: int = 0
    last_hydrate_at: float | None = None


class SessionPersistence:
    """Owns the sink, hydrates stores, and compacts journals into snapshots."""

    def __init__(self, config: PersistenceConfig, *, sink: PersistenceSink | None = None) -> None:
        self.config = config
        self.enabled = config.mode != MODE_NONE
        self.sink = sink if sink is not None else build_sink(config)
        self.stats = PersistenceStats()
        self._lock = threading.RLock()
        # Per-thread append buffer used by request_scope(): a request that writes
        # several cells becomes a single journal append.
        self._buffer_state = threading.local()
        self._buffer_limit = 256

    # -- store lifecycle ---------------------------------------------------
    def hydrate(self, session_id: str, *, now_fn: Any = time.time) -> PersistentOrionMemoryStore:
        """Build a store for a session from its snapshot and journal."""

        store = PersistentOrionMemoryStore(session_id=session_id, persistence=self, now_fn=now_fn)
        if not self.enabled or self.sink is None:
            return store
        journal = self._journal(session_id)
        replayed = 0
        if journal is not None:
            document = journal.load_snapshot()
            if document:
                replayed += apply_records(store, _records_from_snapshot(document))
            replayed += apply_records(store, list(journal.iter_journal()))
        with self._lock:
            self.stats.hydrated_sessions += 1
            self.stats.replayed_records += replayed
            self.stats.last_hydrate_at = time.time()
        return store

    def _journal(self, session_id: str) -> Any:
        journal_accessor = getattr(self.sink, "journal", None)
        if callable(journal_accessor):
            return journal_accessor(session_id)
        return None

    # -- recording ---------------------------------------------------------
    def record(self, session_id: str, record: Mapping[str, Any]) -> None:
        if not (self.enabled and self.sink is not None):
            return
        buffer = getattr(self._buffer_state, "records", None)
        if buffer is None:
            self.sink.append(session_id, record)
            return
        bucket = buffer.setdefault(session_id, [])
        bucket.append(dict(record))
        if len(bucket) >= self._buffer_limit:
            self.sink.append_many(session_id, bucket)
            bucket.clear()

    def _buffered_count(self, session_id: str) -> int:
        buffer = getattr(self._buffer_state, "records", None)
        return len(buffer.get(session_id, ())) if buffer is not None else 0

    @contextmanager
    def request_scope(self) -> Iterator[None]:
        """Coalesce this request's journal records into one append per session.

        Durability is unchanged for anything the caller was told about: records
        are written before the scope exits, so a crash mid-request loses only
        work that had not been acknowledged yet.
        """

        existing = getattr(self._buffer_state, "records", None)
        if not self.enabled or self.sink is None or existing is not None:
            yield
            return
        self._buffer_state.records = {}
        try:
            yield
        finally:
            pending = self._buffer_state.records
            self._buffer_state.records = None
            for session_id, records in pending.items():
                if records:
                    self.sink.append_many(session_id, records)

    def after_record(self, session_id: str, store: OrionMemoryStore) -> None:
        """Compact when a session has accumulated enough journal records."""

        if not self.enabled or self.sink is None:
            return
        counter = getattr(self.sink, "records_since_snapshot", None)
        written = counter(session_id) if callable(counter) else 0
        if written + self._buffered_count(session_id) >= self.config.snapshot_every_records:
            # Buffered records are part of the store's in-memory state already, so
            # snapshotting now captures them; replay of the same records is
            # idempotent.
            self.snapshot(session_id, store)

    def snapshot(self, session_id: str, store: OrionMemoryStore) -> None:
        if not self.enabled or self.sink is None:
            return
        payload = {
            "session_id": session_id,
            "cells": [cell.to_dict() for cell in store.cells.values()],
            "links": [link.to_dict() for link in store.links],
            "trace_events": [event.to_dict() for event in store.trace_events] if self.config.persist_traces else [],
        }
        self.sink.snapshot(session_id, payload)
        with self._lock:
            self.stats.snapshots += 1

    # -- operations --------------------------------------------------------
    def flush(self) -> None:
        if self.enabled and self.sink is not None:
            self.sink.flush()

    # -- retention ---------------------------------------------------------
    def _merged_state(self, session_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Read a session's durable state from disk without a live store."""

        journal = self._journal(session_id)
        if journal is None:
            return [], [], []
        store = OrionMemoryStore()
        document = journal.load_snapshot()
        if document:
            apply_records(store, _records_from_snapshot(document))
        apply_records(store, list(journal.iter_journal()))
        return (
            [cell.to_dict() for cell in store.cells.values()],
            [link.to_dict() for link in store.links],
            [event.to_dict() for event in store.trace_events],
        )

    def session_usage(self) -> list[dict[str, Any]]:
        """Per-session disk usage and last activity, newest first."""

        root = session_directory(self.config.state_dir, "_").parent
        if not root.exists():
            return []
        entries: list[dict[str, Any]] = []
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            files = [item for item in path.iterdir() if item.is_file()]
            if not files:
                continue
            entries.append(
                {
                    "session": path.name,
                    "bytes": sum(item.stat().st_size for item in files),
                    "last_activity": max(item.stat().st_mtime for item in files),
                    "files": sorted(item.name for item in files),
                }
            )
        entries.sort(key=lambda entry: entry["last_activity"], reverse=True)
        return entries

    def compact_session(self, session_id: str) -> dict[str, Any]:
        """Merge a session's journal into its snapshot without loading a store."""

        journal = self._journal(session_id)
        if journal is None:
            raise RuntimeError("compaction needs the in-process journal sink")
        before = journal.usage()
        cells, links, traces = self._merged_state(session_id)
        journal.write_snapshot(
            {"session_id": session_id, "cells": cells, "links": links, "trace_events": traces}
        )
        with self._lock:
            self.stats.snapshots += 1
        return {
            "session": session_id,
            "cells": len(cells),
            "links": len(links),
            "journal_bytes_before": before["journal_bytes"],
            "journal_bytes_after": journal.usage()["journal_bytes"],
        }

    def retention_sweep(
        self,
        *,
        older_than_seconds: float | None = None,
        keep_sessions: int | None = None,
        compact_over_records: int | None = None,
        skip_sessions: Sequence[str] = (),
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Compact large journals and drop sessions that are past their retention.

        ``skip_sessions`` protects sessions that a running service currently
        holds in memory. ``dry_run`` reports what would happen and changes
        nothing.
        """

        skip = set(skip_sessions)
        report: dict[str, Any] = {
            "dry_run": dry_run,
            "compacted": [],
            "pruned": [],
            "skipped": sorted(skip),
            "bytes_freed": 0,
        }

        for entry in self.session_usage():
            session_id = entry["session"]
            if session_id in skip:
                continue
            if compact_over_records is not None:
                journal = self._journal(session_id)
                if journal is not None and journal.records_since_snapshot >= compact_over_records:
                    if dry_run:
                        report["compacted"].append({"session": session_id, "records": journal.records_since_snapshot})
                    else:
                        report["compacted"].append(self.compact_session(session_id))

        usage = self.session_usage()
        candidates = [entry for entry in usage if entry["session"] not in skip]
        expired: list[dict[str, Any]] = []
        if older_than_seconds is not None:
            cutoff = time.time() - older_than_seconds
            expired = [entry for entry in candidates if entry["last_activity"] < cutoff]
        if keep_sessions is not None:
            protected = {entry["session"] for entry in usage[: max(0, keep_sessions)]}
            expired = [entry for entry in expired if entry["session"] not in protected]

        for entry in expired:
            if dry_run:
                report["pruned"].append({"session": entry["session"], "bytes": entry["bytes"]})
                continue
            directory = session_directory(self.config.state_dir, entry["session"])
            shutil.rmtree(directory, ignore_errors=True)
            report["pruned"].append({"session": entry["session"], "bytes": entry["bytes"]})
            report["bytes_freed"] += entry["bytes"]

        if not dry_run:
            report["bytes_freed"] = sum(item["bytes"] for item in report["pruned"])
        report["sessions_before"] = len(usage)
        report["sessions_after"] = len(self.session_usage())
        return report

    def known_sessions(self) -> list[str]:
        """Session ids that have durable state on disk."""

        root = session_directory(self.config.state_dir, "_").parent
        if not root.exists():
            return []
        return sorted(path.name for path in root.iterdir() if path.is_dir())

    def status(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "snapshot_every_records": self.config.snapshot_every_records,
            "persist_traces": self.config.persist_traces,
            "hydrated_sessions": self.stats.hydrated_sessions,
            "replayed_records": self.stats.replayed_records,
            "snapshots": self.stats.snapshots,
            "last_hydrate_at": self.stats.last_hydrate_at,
            "sessions_on_disk": self.known_sessions(),
        }
        if self.sink is not None:
            payload["sink"] = self.sink.status()
        return payload

    def close(self) -> None:
        if self.sink is not None:
            self.sink.close()


__all__ = [
    "GLOBAL_SESSION",
    "PersistenceConfig",
    "PersistenceStats",
    "PersistentOrionMemoryStore",
    "SNAPSHOT_NAME",
    "SessionPersistence",
    "apply_records",
    "cell_from_dict",
    "link_from_dict",
]
