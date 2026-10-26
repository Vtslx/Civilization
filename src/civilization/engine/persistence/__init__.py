"""Durable session memory: append-only journal, snapshots, and crash-safe replay.

Read path stays in memory; writes are appended to a per-session journal and
committed by policy. On startup a session is rebuilt from its snapshot plus the
journal tail. An optional sidecar process can own the files and the commit loop
so that fsync work and the durable stream survive service restarts.
"""

from __future__ import annotations

from .backup import (
    MemoryBackupError,
    create_memory_backup,
    replicate_memory_backup,
    restore_memory_backup,
    verify_memory_backup,
)
from .s3_backup import (
    apply_s3_backup_retention,
    create_s3_client,
    plan_s3_backup_retention,
    replicate_encrypted_backup_to_s3,
    s3_backup_object_key,
    validate_s3_endpoint_url,
)
from .journal import (
    FORMAT_VERSION,
    JOURNAL_NAME,
    RECORD_CELL,
    RECORD_LINK,
    RECORD_RESET,
    RECORD_TRACE,
    SNAPSHOT_NAME,
    SessionJournal,
    session_directory,
)
from .sink import (
    FSYNC_EVERY_WRITE,
    FSYNC_INTERVAL,
    FSYNC_NEVER,
    FSYNC_POLICIES,
    MODE_IN_PROCESS,
    MODE_NONE,
    MODE_SIDECAR,
    PERSISTENCE_MODES,
    InProcessSink,
    PersistenceConfig,
    PersistenceSink,
    SidecarClient,
    SidecarServer,
    build_sink,
)
from .store import (
    GLOBAL_SESSION,
    PersistenceStats,
    PersistentOrionMemoryStore,
    SessionPersistence,
    apply_records,
    cell_from_dict,
    link_from_dict,
    trace_from_dict,
)

__all__ = [
    "MemoryBackupError",
    "FORMAT_VERSION",
    "FSYNC_EVERY_WRITE",
    "FSYNC_INTERVAL",
    "FSYNC_NEVER",
    "FSYNC_POLICIES",
    "GLOBAL_SESSION",
    "InProcessSink",
    "JOURNAL_NAME",
    "MODE_IN_PROCESS",
    "MODE_NONE",
    "MODE_SIDECAR",
    "PERSISTENCE_MODES",
    "PersistenceConfig",
    "PersistenceSink",
    "PersistenceStats",
    "PersistentOrionMemoryStore",
    "RECORD_CELL",
    "RECORD_LINK",
    "RECORD_RESET",
    "RECORD_TRACE",
    "SNAPSHOT_NAME",
    "SessionJournal",
    "SessionPersistence",
    "SidecarClient",
    "SidecarServer",
    "apply_records",
    "apply_s3_backup_retention",
    "build_sink",
    "cell_from_dict",
    "create_memory_backup",
    "create_s3_client",
    "link_from_dict",
    "replicate_memory_backup",
    "restore_memory_backup",
    "replicate_encrypted_backup_to_s3",
    "plan_s3_backup_retention",
    "s3_backup_object_key",
    "session_directory",
    "trace_from_dict",
    "verify_memory_backup",
    "validate_s3_endpoint_url",
]
