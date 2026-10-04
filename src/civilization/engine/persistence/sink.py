"""Persistence sinks: who owns the journal files, and who calls fsync.

Three modes are supported, all with the same on-disk layout:

``in_process`` (default)
    The service writes records itself with ``os.write`` on an append-only file
    descriptor, and a background thread commits them with ``fsync`` on the
    configured interval. A process crash loses nothing that was written; a
    power loss loses at most the un-fsynced tail.

``none``
    Memory stays in-process only. Used by tests and by deployments that treat
    memory as cache.

``sidecar``
    A separate persistence process owns the journal files and the fsync loop.
    The service sends records over a Unix socket, so an abrupt service death
    still leaves the already-sent records with the sidecar, which drains and
    commits them. This isolates fsync stalls from request handling and keeps the
    durable stream continuous across service restarts. It is **not** what makes
    memory durable — the journal plus the fsync policy is; the sidecar decides
    *where* that work happens.

If the sidecar becomes unreachable the client degrades to local in-process
writing instead of failing memory writes, and reports ``degraded`` in the status
payload so the condition is visible rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any, Mapping, Protocol, Sequence

from .journal import SessionJournal, session_directory

FSYNC_EVERY_WRITE = "every_write"
FSYNC_INTERVAL = "interval"
FSYNC_NEVER = "never"
FSYNC_POLICIES = (FSYNC_EVERY_WRITE, FSYNC_INTERVAL, FSYNC_NEVER)

MODE_IN_PROCESS = "in_process"
MODE_SIDECAR = "sidecar"
MODE_NONE = "none"
PERSISTENCE_MODES = (MODE_IN_PROCESS, MODE_SIDECAR, MODE_NONE)


@dataclass(frozen=True)
class PersistenceConfig:
    """How session memory is persisted."""

    mode: str = MODE_IN_PROCESS
    state_dir: str = "var/state"
    fsync: str = FSYNC_INTERVAL
    fsync_interval_seconds: float = 0.25
    snapshot_every_records: int = 512
    sidecar_socket: str | None = None
    sidecar_reconnect_interval_seconds: float = 5.0
    persist_traces: bool = False

    def __post_init__(self) -> None:
        if self.mode not in PERSISTENCE_MODES:
            raise ValueError(f"unknown persistence mode {self.mode!r}; expected one of {PERSISTENCE_MODES}")
        if self.fsync not in FSYNC_POLICIES:
            raise ValueError(f"unknown fsync policy {self.fsync!r}; expected one of {FSYNC_POLICIES}")
        if self.fsync_interval_seconds <= 0:
            raise ValueError("fsync_interval_seconds must be > 0")
        if self.snapshot_every_records < 1:
            raise ValueError("snapshot_every_records must be >= 1")
        if self.mode == MODE_SIDECAR and not (self.sidecar_socket or "").strip():
            raise ValueError("the sidecar mode needs sidecar_socket")
        if self.sidecar_reconnect_interval_seconds <= 0:
            raise ValueError("sidecar_reconnect_interval_seconds must be > 0")


class PersistenceSink(Protocol):
    def append(self, session_id: str, record: Mapping[str, Any]) -> None: ...
    def append_many(self, session_id: str, records: Sequence[Mapping[str, Any]]) -> None: ...
    def snapshot(self, session_id: str, payload: Mapping[str, Any]) -> None: ...
    def flush(self) -> None: ...
    def status(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


class InProcessSink:
    """Owns the journal files and commits them from a background thread."""

    def __init__(self, config: PersistenceConfig) -> None:
        self.config = config
        self.root = Path(config.state_dir).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self._journals: dict[str, SessionJournal] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._flusher: threading.Thread | None = None
        if config.fsync == FSYNC_INTERVAL:
            self._flusher = threading.Thread(target=self._flush_loop, name="civilization-fsync", daemon=True)
            self._flusher.start()

    # -- journal access ----------------------------------------------------
    def journal(self, session_id: str) -> SessionJournal:
        with self._lock:
            journal = self._journals.get(session_id)
            if journal is None:
                journal = SessionJournal(session_directory(self.root, session_id))
                self._journals[session_id] = journal
            return journal

    def session_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._journals)

    def _flush_loop(self) -> None:
        while not self._closed:
            time.sleep(self.config.fsync_interval_seconds)
            try:
                self.flush()
            except Exception:  # noqa: BLE001 - a flusher must never kill the service
                continue

    # -- sink protocol -----------------------------------------------------
    def append(self, session_id: str, record: Mapping[str, Any]) -> None:
        journal = self.journal(session_id)
        journal.append(record, fsync=self.config.fsync == FSYNC_EVERY_WRITE)

    def append_many(self, session_id: str, records: Sequence[Mapping[str, Any]]) -> None:
        if not records:
            return
        self.journal(session_id).append_many(records, fsync=self.config.fsync == FSYNC_EVERY_WRITE)

    def snapshot(self, session_id: str, payload: Mapping[str, Any]) -> None:
        self.journal(session_id).write_snapshot(payload)

    def records_since_snapshot(self, session_id: str) -> int:
        journal = self._journals.get(session_id)
        return journal.records_since_snapshot if journal else 0

    def flush(self) -> None:
        with self._lock:
            journals = list(self._journals.values())
        for journal in journals:
            if journal.dirty:
                journal.fsync()

    def status(self) -> dict[str, Any]:
        with self._lock:
            journals = {name: journal.usage() for name, journal in self._journals.items()}
        return {
            "mode": MODE_IN_PROCESS,
            "state_dir": str(self.root),
            "fsync": self.config.fsync,
            "fsync_interval_seconds": self.config.fsync_interval_seconds,
            "snapshot_every_records": self.config.snapshot_every_records,
            "sessions": len(journals),
            "pending_records": sum(entry["pending_records"] for entry in journals.values()),
            "journals": journals,
            "degraded": False,
        }

    def close(self) -> None:
        self._closed = True
        self.flush()
        with self._lock:
            for journal in self._journals.values():
                journal.close()
            self._journals.clear()


class SidecarClient:
    """Sends records to a persistence process; falls back locally if it is gone."""

    def __init__(self, config: PersistenceConfig) -> None:
        self.config = config
        self.socket_path = Path(str(config.sidecar_socket)).expanduser()
        self._socket: socket.socket | None = None
        self._lock = threading.RLock()
        self._degraded_reason: str | None = None
        self._degraded_since: float | None = None
        self._fallback: InProcessSink | None = None
        self._probe: threading.Thread | None = None
        self._closed = False
        self.sent_records = 0
        self.reconnects = 0

    # -- connection --------------------------------------------------------
    def _connect(self) -> socket.socket:
        _check_socket_path_length(self.socket_path)
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.connect(str(self.socket_path))
        return connection

    def _new_fallback(self) -> InProcessSink:
        return InProcessSink(
            PersistenceConfig(
                mode=MODE_IN_PROCESS,
                state_dir=self.config.state_dir,
                fsync=self.config.fsync,
                fsync_interval_seconds=self.config.fsync_interval_seconds,
                snapshot_every_records=self.config.snapshot_every_records,
                persist_traces=self.config.persist_traces,
            )
        )

    def _attach(self) -> bool:
        """Try to attach to the sidecar. Returns True when the socket is usable."""

        with self._lock:
            if self._socket is not None:
                return True
            try:
                connection = self._connect()
            except (OSError, ValueError) as error:
                # A missing, refused, or unusable socket must not stop memory
                # writes: fall back to local journaling and report the reason.
                self._degrade(f"sidecar unreachable ({error})")
                return False
            # Only one writer may hold the journal files: commit and release the
            # local sink before handing ownership to the sidecar. Records written
            # while degraded are already in the same files, so the switch loses
            # nothing.
            self._release_fallback()
            self._socket = connection
            if self._degraded_reason is not None:
                self.reconnects += 1
            self._degraded_reason = None
            self._degraded_since = None
            return True

    def _release_fallback(self) -> None:
        fallback = self._fallback
        self._fallback = None
        if fallback is not None:
            try:
                fallback.close()
            except Exception:  # noqa: BLE001 - best-effort handover
                pass

    def _connection(self) -> socket.socket | None:
        if self._socket is not None:
            return self._socket
        if not self._attach():
            return None
        return self._socket

    def _degrade(self, reason: str) -> None:
        if self._fallback is None:
            self._fallback = self._new_fallback()
            self._degraded_since = time.time()
        self._degraded_reason = reason
        self._ensure_probe()

    def _ensure_probe(self) -> None:
        """Start the background thread that re-attaches when the sidecar returns."""

        if self._probe is not None or self._closed:
            return
        with self._lock:
            if self._probe is not None or self._closed:
                return
            self._probe = threading.Thread(target=self._probe_loop, name="civilization-sidecar-probe", daemon=True)
            self._probe.start()

    def _probe_loop(self) -> None:
        while not self._closed:
            time.sleep(self.config.sidecar_reconnect_interval_seconds)
            if self._closed:
                return
            if self._socket is None and self._fallback is not None:
                try:
                    self._attach()
                except Exception:  # noqa: BLE001 - a probe must never kill the service
                    continue

    def _send(self, frame: Mapping[str, Any]) -> bool:
        payload = (json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        connection = self._connection()
        if connection is None:
            return False
        try:
            connection.sendall(payload)
            return True
        except OSError as error:
            with self._lock:
                if self._socket is not None:
                    try:
                        self._socket.close()
                    except OSError:
                        pass
                    self._socket = None
            self._degrade(f"sidecar connection lost ({error})")
            return False

    # -- sink protocol -----------------------------------------------------
    def append(self, session_id: str, record: Mapping[str, Any]) -> None:
        if self._send({"op": "append", "session": session_id, "record": dict(record)}):
            self.sent_records += 1
        elif self._fallback is not None:
            self._fallback.append(session_id, record)

    def append_many(self, session_id: str, records: Sequence[Mapping[str, Any]]) -> None:
        if not records:
            return
        if self._send({"op": "append_many", "session": session_id, "records": [dict(r) for r in records]}):
            self.sent_records += len(records)
        elif self._fallback is not None:
            self._fallback.append_many(session_id, records)

    def snapshot(self, session_id: str, payload: Mapping[str, Any]) -> None:
        if not self._send({"op": "snapshot", "session": session_id, "payload": dict(payload)}):
            if self._fallback is not None:
                self._fallback.snapshot(session_id, payload)

    def flush(self) -> None:
        self._send({"op": "flush"})
        if self._fallback is not None:
            self._fallback.flush()

    def status(self) -> dict[str, Any]:
        if self._fallback is not None:
            return {
                **self._fallback.status(),
                "mode": MODE_SIDECAR,
                "socket": str(self.socket_path),
                "attached": False,
                "sent_records": self.sent_records,
                "reconnects": self.reconnects,
                "reconnect_interval_seconds": self.config.sidecar_reconnect_interval_seconds,
                "degraded": True,
                "degraded_reason": self._degraded_reason,
                "degraded_since": self._degraded_since,
            }
        return {
            "mode": MODE_SIDECAR,
            "socket": str(self.socket_path),
            "state_dir": str(Path(self.config.state_dir).expanduser()),
            "fsync": self.config.fsync,
            "attached": self._socket is not None,
            "sent_records": self.sent_records,
            "reconnects": self.reconnects,
            "reconnect_interval_seconds": self.config.sidecar_reconnect_interval_seconds,
            "degraded": self._degraded_reason is not None,
            "degraded_reason": self._degraded_reason,
        }

    def close(self) -> None:
        self._closed = True
        self.flush()
        with self._lock:
            if self._socket is not None:
                try:
                    self._socket.close()
                except OSError:
                    pass
                self._socket = None
        self._release_fallback()
        probe = self._probe
        if probe is not None and probe.is_alive() and probe is not threading.current_thread():
            probe.join(timeout=1.0)
        self._probe = None


class SidecarServer:
    """The persistence process: owns the files, commits them, survives restarts.

    It accepts one service connection at a time. When the service disconnects —
    including when it is killed abruptly — the server drains whatever the kernel
    already buffered, commits it, and keeps listening so a restarted service can
    re-attach to the same durable stream.
    """

    def __init__(self, config: PersistenceConfig) -> None:
        if not (config.sidecar_socket or "").strip():
            raise ValueError("the sidecar server needs sidecar_socket")
        self.config = config
        self.socket_path = Path(str(config.sidecar_socket)).expanduser()
        self.sink = InProcessSink(
            PersistenceConfig(
                mode=MODE_IN_PROCESS,
                state_dir=config.state_dir,
                fsync=config.fsync,
                fsync_interval_seconds=config.fsync_interval_seconds,
                snapshot_every_records=config.snapshot_every_records,
                persist_traces=config.persist_traces,
            )
        )
        self.received_records = 0
        self.connections = 0
        self._server: socket.socket | None = None
        self._stopped = threading.Event()

    def _handle_frame(self, frame: Mapping[str, Any]) -> None:
        operation = frame.get("op")
        session_id = str(frame.get("session", "default"))
        if operation == "append":
            record = frame.get("record")
            if isinstance(record, dict):
                self.sink.append(session_id, record)
                self.received_records += 1
        elif operation == "append_many":
            records = frame.get("records")
            if isinstance(records, list):
                payload = [record for record in records if isinstance(record, dict)]
                self.sink.append_many(session_id, payload)
                self.received_records += len(payload)
        elif operation == "snapshot":
            payload = frame.get("payload")
            if isinstance(payload, dict):
                self.sink.snapshot(session_id, payload)
        elif operation == "flush":
            self.sink.flush()

    def _serve_connection(self, connection: socket.socket) -> None:
        buffer = b""
        with connection:
            while True:
                try:
                    chunk = connection.recv(65536)
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, _, buffer = buffer.partition(b"\n")
                    if not line.strip():
                        continue
                    try:
                        frame = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    if isinstance(frame, dict):
                        self._handle_frame(frame)
            if buffer.strip():
                try:
                    frame = json.loads(buffer.decode("utf-8"))
                    if isinstance(frame, dict):
                        self._handle_frame(frame)
                except json.JSONDecodeError:
                    pass
        # The service is gone (cleanly or not): commit everything received.
        self.sink.flush()

    def serve_forever(self, *, on_ready: Any | None = None) -> None:
        _check_socket_path_length(self.socket_path)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.socket_path))
        self._server.listen(4)
        os.chmod(self.socket_path, 0o600)
        if on_ready is not None:
            on_ready(str(self.socket_path))
        try:
            while not self._stopped.is_set():
                try:
                    connection, _ = self._server.accept()
                except OSError:
                    break
                self.connections += 1
                self._serve_connection(connection)
        finally:
            self.sink.flush()
            self.sink.close()
            if self._server is not None:
                self._server.close()
            if self.socket_path.exists():
                self.socket_path.unlink()

    def stop(self) -> None:
        self._stopped.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass


def _check_socket_path_length(path: Path) -> None:
    """Unix sockets have a short path limit (about 104 bytes on macOS).

    Failing here with an explanation beats a silent bind error inside a thread.
    """

    encoded = len(str(path).encode("utf-8"))
    if encoded > 100:
        raise ValueError(
            f"unix socket path is {encoded} bytes, which exceeds the platform limit (~104): {path}. "
            "Use a shorter path, for example /tmp/civilization-persist.sock."
        )


def build_sink(config: PersistenceConfig) -> PersistenceSink | None:
    """Create the sink for a configuration, or ``None`` when persistence is off."""

    if config.mode == MODE_NONE:
        return None
    if config.mode == MODE_SIDECAR:
        return SidecarClient(config)
    return InProcessSink(config)
