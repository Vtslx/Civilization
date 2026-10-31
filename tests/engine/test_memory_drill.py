from __future__ import annotations

from pathlib import Path

import pytest

from civilization.engine.persistence.drill import DrillError, run_local_drill
from civilization.engine.persistence.sink import PersistenceConfig
from civilization.engine.persistence.store import SessionPersistence
from civilization.engine.stages.stage73_orion_memory_kernel import MemorySystem


def _state_with_session(root: Path) -> Path:
    state = root / "state"
    persistence = SessionPersistence(PersistenceConfig(mode="in_process", state_dir=str(state), fsync="every_write"))
    store = persistence.hydrate("drill-session")
    first = store.write_cell(
        memory_system=MemorySystem.EPISODIC,
        content="two blue chairs",
        summary="two blue chairs",
        source="drill-session",
        metadata={"provenance": {"source_id": "turn-1", "quote": "two blue chairs", "role": "user"}},
    )
    second = store.write_cell(
        memory_system=MemorySystem.SEMANTIC,
        content="the user is furnishing a home",
        summary="furnishing",
        source="drill-session",
        metadata={"derived_from_cell_ids": [first.cell_id]},
    )
    store.link_cells(first.cell_id, second.cell_id, link_type="semantic_similarity")
    persistence.flush()
    return state


def test_local_drill_backs_up_replicates_restores_and_compares(tmp_path: Path) -> None:
    state = _state_with_session(tmp_path)
    backup_dir = tmp_path / "backup"
    replica_dir = tmp_path / "replica"
    backup_dir.mkdir()
    replica_dir.mkdir()

    report = run_local_drill(
        state_dir=state,
        backup_dir=backup_dir,
        replica_dir=replica_dir,
        passphrase="drill-passphrase",
        require_distinct_filesystem=False,
    )

    assert report["passed"] is True
    assert report["session_count"] == 1
    assert report["cell_count"] == 2
    assert report["link_count"] == 1
    assert all(check["passed"] for check in report["checks"])
    assert (backup_dir / "drill-backup.civbackup").stat().st_mode & 0o777 == 0o600
    assert (replica_dir / "drill-backup.civbackup").is_file()
    assert (backup_dir / "restored-state" / "sessions" / "drill-session").is_dir()


def test_local_drill_rejects_same_filesystem_when_required(tmp_path: Path) -> None:
    state = _state_with_session(tmp_path)
    backup_dir = tmp_path / "backup"
    replica_dir = tmp_path / "replica"
    backup_dir.mkdir()
    replica_dir.mkdir()

    with pytest.raises(DrillError, match="different filesystem"):
        run_local_drill(
            state_dir=state,
            backup_dir=backup_dir,
            replica_dir=replica_dir,
            passphrase="drill-passphrase",
            require_distinct_filesystem=True,
        )


def test_local_drill_rejects_empty_state_directory(tmp_path: Path) -> None:
    state = tmp_path / "state"
    (state / "sessions").mkdir(parents=True)
    backup_dir = tmp_path / "backup"
    replica_dir = tmp_path / "replica"
    backup_dir.mkdir()
    replica_dir.mkdir()

    with pytest.raises(DrillError, match="no session journals"):
        run_local_drill(
            state_dir=state,
            backup_dir=backup_dir,
            replica_dir=replica_dir,
            passphrase="drill-passphrase",
            require_distinct_filesystem=False,
        )


def test_cli_drill_runs_end_to_end_on_one_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from civilization.cli import main

    state = _state_with_session(tmp_path)
    backup_dir = tmp_path / "cli-backup"
    replica_dir = tmp_path / "cli-replica"
    output = tmp_path / "drill-report.json"
    monkeypatch.setenv("DRILL_TEST_PASSPHRASE", "drill-passphrase")

    exit_code = main(
        [
            "dr-drill",
            "--state-dir",
            str(state),
            "--backup-dir",
            str(backup_dir),
            "--replica-dir",
            str(replica_dir),
            "--passphrase-env",
            "DRILL_TEST_PASSPHRASE",
            "--allow-same-filesystem",
            "--output",
            str(output),
            "--json",
        ]
    )

    assert exit_code == 0
    report = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["replica_mode"] == "plain_copy_same_filesystem"
    assert report["session_count"] == 1
    assert report["cell_count"] == 2
