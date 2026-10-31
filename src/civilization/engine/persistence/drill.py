"""Executable disaster-recovery drill: backup, replicate, restore, rehydrate, compare.

The backup, replication, and restore commands each have their own tests. A drill
is different: it runs them in one sequence against real files and then proves
that the restored journals hydrate to the same cells and links as the source.
That is the check that catches a path that is individually correct but does not
compose into a recovery.

The local drill deliberately requires the replica to be on a different
filesystem. A copy on the same device does not exercise the cross-device path
and is not evidence for a disaster that loses that device. Tests may opt out
because a temporary directory is usually on the same filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from .backup import (
    MemoryBackupError,
    create_memory_backup,
    replicate_memory_backup,
    restore_memory_backup,
    verify_memory_backup,
)
from .journal import session_directory
from .sink import PersistenceConfig
from .store import SessionPersistence


class DrillError(RuntimeError):
    """The drill could not complete or one of its integrity checks failed."""


@dataclass(frozen=True)
class DrillCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(path: Path) -> int:
    return Path(path).resolve().stat().st_dev


def _session_ids(state_dir: Path) -> list[str]:
    root = session_directory(state_dir, "_").parent
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def _hydrate(state_dir: Path, session_id: str) -> tuple[dict[str, str], dict[str, str]]:
    persistence = SessionPersistence(
        PersistenceConfig(mode="in_process", state_dir=str(state_dir), fsync="never")
    )
    store = persistence.hydrate(session_id)
    cells = {cell.cell_id: _canonical(cell.to_dict()) for cell in store.cells.values()}
    links = {
        _canonical(
            {
                "source": link.source_cell_id,
                "target": link.target_cell_id,
                "type": link.link_type.value,
                "weight": link.weight,
                "created_at": link.created_at,
                "metadata": link.metadata,
            }
        ): ""
        for link in store.links
    }
    return cells, links


def run_local_drill(
    *,
    state_dir: str | Path,
    backup_dir: str | Path,
    replica_dir: str | Path,
    passphrase: str | None,
    require_distinct_filesystem: bool = True,
) -> dict[str, Any]:
    """Run the full local recovery path and return a machine-readable report."""

    started = time.monotonic()
    source = Path(state_dir).expanduser().resolve(strict=True)
    if not source.is_dir():
        raise DrillError(f"state directory is not a directory: {source}")
    backup_root = Path(backup_dir).expanduser().resolve(strict=True)
    replica_root = Path(replica_dir).expanduser().resolve(strict=True)
    if not backup_root.is_dir() or not replica_root.is_dir():
        raise DrillError("backup_dir and replica_dir must already exist")
    if backup_root == replica_root:
        raise DrillError("backup_dir and replica_dir must differ")
    if require_distinct_filesystem and _device(backup_root) == _device(replica_root):
        raise DrillError("replica_dir must be on a different filesystem from backup_dir")

    session_ids = _session_ids(source)
    if not session_ids:
        raise DrillError(f"no session journals under {source}")
    source_cells: dict[str, dict[str, str]] = {}
    source_links: dict[str, dict[str, str]] = {}
    for session_id in session_ids:
        source_cells[session_id], source_links[session_id] = _hydrate(source, session_id)

    backup_path = backup_root / "drill-backup.civbackup"
    if backup_path.exists():
        raise DrillError(f"drill backup already exists: {backup_path}")
    checks: list[DrillCheck] = []

    create_report = create_memory_backup(source, backup_path, passphrase=passphrase)
    checks.append(DrillCheck("backup_created", backup_path.is_file(), f"{backup_path.name} ({create_report.get('total_bytes', '?')} bytes)"))
    checks.append(
        DrillCheck(
            "backup_is_private",
            (backup_path.stat().st_mode & 0o777) == 0o600,
            f"mode {oct(backup_path.stat().st_mode & 0o777)}",
        )
    )

    verify_report = verify_memory_backup(backup_path, passphrase=passphrase)
    checks.append(
        DrillCheck(
            "backup_authenticated",
            verify_report.get("encrypted") is True,
            f"encrypted={verify_report.get('encrypted')} entries={verify_report.get('entry_count', '?')}",
        )
    )

    replica_path = replica_root / backup_path.name
    if replica_path.exists():
        raise DrillError(f"drill replica already exists: {replica_path}")
    distinct = _device(backup_root) != _device(replica_root)
    if distinct:
        replicate_report = replicate_memory_backup(backup_path, replica_root, passphrase=passphrase)
        replica_mode = "authenticated_cross_filesystem_replication"
    else:
        # The production replication command correctly refuses a same-device
        # target. Tests and single-filesystem CI still need the rest of the
        # drill, so copy the already-verified encrypted bundle and keep the
        # mode explicit in the report.
        shutil.copy2(backup_path, replica_path)
        os.chmod(replica_path, 0o600)
        replicate_report = {"total_bytes": replica_path.stat().st_size}
        replica_mode = "plain_copy_same_filesystem"
    checks.append(
        DrillCheck(
            "replica_created",
            replica_path.is_file(),
            f"{replica_path} ({replicate_report.get('total_bytes', '?')} bytes; {replica_mode})",
        )
    )
    checks.append(
        DrillCheck(
            "replica_hash_matches",
            _sha256(replica_path) == _sha256(backup_path),
            "source and replica SHA-256 match",
        )
    )

    restored = backup_root / "restored-state"
    if restored.exists():
        raise DrillError(f"restore destination already exists: {restored}")
    restore_report = restore_memory_backup(replica_path, restored, passphrase=passphrase)
    checks.append(
        DrillCheck(
            "restore_completed",
            restored.is_dir(),
            f"{restore_report.get('entry_count', '?')} entries restored",
        )
    )

    restored_sessions = _session_ids(restored)
    checks.append(
        DrillCheck(
            "session_set_matches",
            restored_sessions == session_ids,
            f"{len(session_ids)} source sessions, {len(restored_sessions)} restored",
        )
    )

    mismatches: list[str] = []
    total_cells = 0
    total_links = 0
    for session_id in session_ids:
        restored_cells, restored_links = _hydrate(restored, session_id)
        total_cells += len(source_cells[session_id])
        total_links += len(source_links[session_id])
        if restored_cells != source_cells[session_id]:
            missing = sorted(set(source_cells[session_id]) - set(restored_cells))
            extra = sorted(set(restored_cells) - set(source_cells[session_id]))
            changed = sorted(
                cell_id
                for cell_id in set(source_cells[session_id]) & set(restored_cells)
                if source_cells[session_id][cell_id] != restored_cells[cell_id]
            )
            mismatches.append(
                f"{session_id}: missing={missing[:3]} extra={extra[:3]} changed={changed[:3]}"
            )
        if restored_links != source_links[session_id]:
            mismatches.append(f"{session_id}: link set differs")
    checks.append(
        DrillCheck(
            "cell_and_link_contents_match",
            not mismatches,
            f"{total_cells} cells and {total_links} links compared"
            + (f"; {len(mismatches)} mismatch(es): {'; '.join(mismatches[:3])}" if mismatches else ""),
        )
    )

    passed = all(check.passed for check in checks)
    elapsed = time.monotonic() - started
    report = {
        "drill": "local_backup_replicate_restore_rehydrate",
        "passed": passed,
        "state_dir": str(source),
        "backup_dir": str(backup_root),
        "replica_dir": str(replica_root),
        "distinct_filesystem": _device(backup_root) != _device(replica_root),
        "replica_mode": replica_mode,
        "session_count": len(session_ids),
        "cell_count": total_cells,
        "link_count": total_links,
        "duration_seconds": round(elapsed, 3),
        "backup_sha256": _sha256(backup_path),
        "checks": [check.to_dict() for check in checks],
        "limits": (
            "Local cross-filesystem evidence on one host. It verifies the backup, "
            "replication, restore, and rehydration path; it is not an offsite or "
            "independent-failure-domain restore."
        ),
    }
    return report
