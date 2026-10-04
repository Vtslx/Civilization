"""Durability tests for session memory: journal, snapshot, replay, sidecar."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from civilization import CivilizationRequest, EmbeddedCivilization, EmbeddedConfig
from civilization.engine.persistence import (
    FSYNC_EVERY_WRITE,
    MODE_IN_PROCESS,
    MODE_SIDECAR,
    PersistenceConfig,
    SessionPersistence,
    SidecarServer,
    apply_records,
    session_directory,
)
from civilization.engine.stages.stage60_queue_rate_limit import Stage60SlowFakeRuntime
from civilization.engine.stages.stage73_orion_memory_kernel import MemorySystem, OrionMemoryStore


def _config(tmp_path: Path, **overrides) -> PersistenceConfig:
    settings = {"mode": MODE_IN_PROCESS, "state_dir": str(tmp_path), "fsync": FSYNC_EVERY_WRITE}
    settings.update(overrides)
    return PersistenceConfig(**settings)


def _seed(store) -> None:
    episodic = store.write_cell(
        memory_system="episodic",
        content="The canary passed health checks.",
        summary="canary passed",
        source="test",
    )
    semantic = store.write_cell(
        memory_system="semantic",
        content="Releases require two approvals.",
        summary="release policy",
        source="test",
    )
    store.link_cells(episodic.cell_id, semantic.cell_id, link_type="temporal", weight=0.5)
    store.update_cell(episodic.cell_id, importance=2.5)
    store.expire_working_memory()


def test_session_survives_a_restart(tmp_path: Path):
    first = SessionPersistence(_config(tmp_path))
    store = first.hydrate("session-a")
    _seed(store)
    first.close()

    second = SessionPersistence(_config(tmp_path))
    restored = second.hydrate("session-a")

    assert sorted(restored.cells) == ["episodic-000001", "semantic-000002"]
    assert len(restored.links) == 1
    assert restored.cells["episodic-000001"].importance == 2.5
    assert [result.cell.cell_id for result in restored.read("canary", limit=3)] == ["episodic-000001"]
    # id counters continue instead of colliding with restored ids
    assert restored.write_cell(
        memory_system="episodic", content="second episode", summary="second", source="test"
    ).cell_id == "episodic-000003"
    second.close()


def test_a_torn_final_line_is_ignored_and_the_rest_is_kept(tmp_path: Path):
    persistence = SessionPersistence(_config(tmp_path))
    store = persistence.hydrate("session-a")
    _seed(store)
    persistence.flush()
    persistence.close()

    journal_path = session_directory(tmp_path, "session-a") / "journal.ndjson"
    with journal_path.open("ab") as handle:  # a crash in the middle of a write
        handle.write(b'{"t":"cell","v":{"cell_id":"episodic-')

    restored = SessionPersistence(_config(tmp_path)).hydrate("session-a")
    assert len(restored.cells) == 2
    assert restored.cells["semantic-000002"].summary == "release policy"


def test_replaying_stale_records_on_top_of_a_snapshot_is_idempotent(tmp_path: Path):
    persistence = SessionPersistence(_config(tmp_path, snapshot_every_records=1000))
    store = persistence.hydrate("session-a")
    _seed(store)

    records = list(session_directory(tmp_path, "session-a").joinpath("journal.ndjson").open("r", encoding="utf-8"))
    parsed = [json.loads(line) for line in records if line.strip()]

    persistence.snapshot("session-a", store)
    # Simulate a crash between snapshot write and journal truncation: replay the
    # snapshot plus the stale journal records into a fresh store.
    replica = OrionMemoryStore()
    apply_records(replica, [{"t": "cell", "v": cell.to_dict()} for cell in store.cells.values()])
    apply_records(replica, parsed)

    assert len(replica.cells) == 2
    assert len(replica.links) == 1
    persistence.close()


def test_link_replay_does_not_duplicate(tmp_path: Path):
    persistence = SessionPersistence(_config(tmp_path))
    store = persistence.hydrate("session-a")
    _seed(store)
    persistence.flush()
    persistence.close()

    journal_path = session_directory(tmp_path, "session-a") / "journal.ndjson"
    payload = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    target = OrionMemoryStore()
    apply_records(target, payload)
    apply_records(target, payload)

    assert len(target.links) == 1
    assert len(target.cells) == 2


def test_fsync_policies_commit_and_report(tmp_path: Path):
    every_write = SessionPersistence(_config(tmp_path / "a", fsync=FSYNC_EVERY_WRITE))
    store = every_write.hydrate("s")
    _seed(store)
    status = every_write.status()["sink"]
    assert status["pending_records"] == 0
    assert status["journals"]["s"]["last_fsync_at"] is not None
    every_write.close()

    interval = SessionPersistence(
        _config(tmp_path / "b", fsync="interval", fsync_interval_seconds=0.05)
    )
    store = interval.hydrate("s")
    _seed(store)
    assert interval.status()["sink"]["pending_records"] > 0
    deadline = time.time() + 2.0
    while time.time() < deadline and interval.status()["sink"]["pending_records"] > 0:
        time.sleep(0.05)
    assert interval.status()["sink"]["pending_records"] == 0
    interval.close()


def test_sidecar_owns_the_journal_and_survives_the_client(tmp_path: Path):
    # Unix socket paths are length-limited, so keep this one short.
    socket_dir = Path(tempfile.mkdtemp(prefix="civ-sidecar-"))
    socket_path = socket_dir / "s.sock"
    server = SidecarServer(
        PersistenceConfig(
            mode=MODE_IN_PROCESS,
            state_dir=str(tmp_path / "state"),
            fsync=FSYNC_EVERY_WRITE,
            sidecar_socket=str(socket_path),
        )
    )
    ready = threading.Event()
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"on_ready": lambda _path: ready.set()},
        daemon=True,
    )
    thread.start()
    assert ready.wait(5.0), "sidecar did not start"

    client = SessionPersistence(
        PersistenceConfig(
            mode=MODE_SIDECAR,
            state_dir=str(tmp_path / "state"),
            sidecar_socket=str(socket_path),
            fsync=FSYNC_EVERY_WRITE,
        )
    )
    store = client.hydrate("session-sidecar")
    _seed(store)
    client.flush()
    assert client.status()["sink"]["attached"] is True
    assert client.status()["sink"]["degraded"] is False

    # The service "dies" abruptly: the socket closes without a clean shutdown.
    # _seed writes four records: two cells, one link, and one cell update.
    client.sink._socket.close()  # noqa: SLF001 - simulating an abrupt process death
    deadline = time.time() + 5.0
    while time.time() < deadline and server.received_records < 4:
        time.sleep(0.05)
    server.stop()
    thread.join(timeout=5.0)

    assert server.received_records >= 4
    restored = SessionPersistence(_config(tmp_path / "state")).hydrate("session-sidecar")
    assert len(restored.cells) == 2
    assert len(restored.links) == 1


def test_sidecar_failure_degrades_to_local_writing(tmp_path: Path):
    client = SessionPersistence(
        PersistenceConfig(
            mode=MODE_SIDECAR,
            state_dir=str(tmp_path),
            sidecar_socket=str(tmp_path / "missing.sock"),
            fsync=FSYNC_EVERY_WRITE,
        )
    )
    store = client.hydrate("session-degraded")
    _seed(store)
    status = client.status()["sink"]
    assert status["degraded"] is True
    assert "unreachable" in status["degraded_reason"]

    client.close()
    restored = SessionPersistence(_config(tmp_path)).hydrate("session-degraded")
    assert len(restored.cells) == 2


def test_service_restart_restores_session_memory_over_http(tmp_path: Path):
    state_dir = tmp_path / "state"
    config = EmbeddedConfig(
        runtime="provider",
        provider_base_url="https://provider.invalid/v1",
        provider_model="test-model",
        state_dir=str(state_dir),
        bearer_token_env=None,
        persist_sessions=True,
        fsync_policy=FSYNC_EVERY_WRITE,
    )

    with EmbeddedCivilization(config, runtime_factory=lambda: Stage60SlowFakeRuntime(0.0)) as runtime:
        client = runtime.start()
        cell = client.write_memory(
            session_id="restart-session",
            memory_system="episodic",
            content="The staging rollout finished with no errors.",
            summary="staging rollout finished",
        )
        assert cell["cell_id"]
        flush = client._request("POST", "/admin/orion/memory/flush")  # noqa: SLF001 - exercising the endpoint
        assert flush["status"] == "ok"
        assert flush["persistence"]["enabled"] is True

    # A brand new process would do exactly this: same state_dir, new objects.
    with EmbeddedCivilization(config, runtime_factory=lambda: Stage60SlowFakeRuntime(0.0)) as runtime:
        client = runtime.start()
        results = client.read_memory(session_id="restart-session", query="staging rollout")
        assert results, "memory written before the restart must be readable after it"
        assert results[0]["cell"]["cell_id"] == cell["cell_id"]
        status = client.memory_status()
        assert status["persistence"]["enabled"] is True
        assert status["persistence"]["replayed_records"] >= 1


def test_persistence_can_be_disabled(tmp_path: Path):
    config = EmbeddedConfig(
        runtime="provider",
        provider_base_url="https://provider.invalid/v1",
        provider_model="test-model",
        state_dir=str(tmp_path / "state"),
        bearer_token_env=None,
        persist_sessions=False,
    )
    with EmbeddedCivilization(config, runtime_factory=lambda: Stage60SlowFakeRuntime(0.0)) as runtime:
        client = runtime.start()
        status = client.memory_status()
        assert status["persistence"]["enabled"] is False
        assert not (tmp_path / "state" / "sessions").exists()


def test_session_directory_cannot_escape_the_state_root(tmp_path: Path):
    for hostile in ("../../etc/passwd", "..", "a/b", ""):
        directory = session_directory(tmp_path, hostile)
        assert directory.parent.parent == tmp_path
        assert "/" not in directory.name


@pytest.mark.parametrize("policy", ["bogus"])
def test_invalid_persistence_settings_are_rejected(policy: str, tmp_path: Path):
    with pytest.raises(ValueError):
        PersistenceConfig(mode=MODE_IN_PROCESS, state_dir=str(tmp_path), fsync=policy)
    with pytest.raises(ValueError):
        PersistenceConfig(mode="elsewhere", state_dir=str(tmp_path))
    with pytest.raises(ValueError):
        PersistenceConfig(mode=MODE_SIDECAR, state_dir=str(tmp_path), sidecar_socket="")


def test_memory_system_enum_round_trips(tmp_path: Path):
    persistence = SessionPersistence(_config(tmp_path))
    store = persistence.hydrate("s")
    store.write_cell(memory_system=MemorySystem.WORKING, content="scratch", summary="scratch", source="test")
    persistence.close()
    restored = SessionPersistence(_config(tmp_path)).hydrate("s")
    assert restored.cells["working-000001"].memory_system is MemorySystem.WORKING



# --------------------------------------------------------------------------- #
# global (cross-session) memory uses the same durability path
# --------------------------------------------------------------------------- #
def _embedded(tmp_path: Path, **overrides) -> tuple[EmbeddedConfig, Path]:
    state_dir = tmp_path / "state"
    settings = {
        "runtime": "provider",
        "provider_base_url": "https://provider.invalid/v1",
        "provider_model": "test-model",
        "state_dir": str(state_dir),
        "bearer_token_env": None,
        "fsync_policy": FSYNC_EVERY_WRITE,
    }
    settings.update(overrides)
    return EmbeddedConfig(**settings), state_dir


def test_global_memory_survives_a_restart_without_an_explicit_save(tmp_path: Path):
    config, _ = _embedded(tmp_path)

    with EmbeddedCivilization(config, runtime_factory=lambda: Stage60SlowFakeRuntime(0.0)) as runtime:
        runtime.start()
        cell = runtime._service.global_memory_store.write_cell(  # noqa: SLF001 - the store under test
            memory_system="semantic",
            content="Every release needs two approvals.",
            summary="release policy",
            source="global",
        )
        assert cell.cell_id
        runtime._service.flush_session_memory()  # noqa: SLF001

    with EmbeddedCivilization(config, runtime_factory=lambda: Stage60SlowFakeRuntime(0.0)) as runtime:
        runtime.start()
        restored = runtime._service.global_memory_store  # noqa: SLF001
        assert "semantic-000001" in restored.cells
        assert restored.cells["semantic-000001"].summary == "release policy"


def test_global_store_load_replaces_contents_and_is_journaled(tmp_path: Path):
    config, state_dir = _embedded(tmp_path)
    global_file = state_dir / "global-memory.json"
    global_file.parent.mkdir(parents=True, exist_ok=True)
    global_file.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_id": "semantic-000001",
                        "memory_system": "semantic",
                        "content": "loaded from the export file",
                        "summary": "loaded",
                        "source": "global",
                        "confidence": 1.0,
                        "importance": 1.0,
                        "created_at": 1.0,
                        "updated_at": 1.0,
                        "time_index": 1.0,
                        "decay_state": "active",
                        "consolidation_state": "raw",
                        "metadata": {},
                    }
                ],
                "links": [],
            }
        ),
        encoding="utf-8",
    )

    with EmbeddedCivilization(config, runtime_factory=lambda: Stage60SlowFakeRuntime(0.0)) as runtime:
        runtime.start()
        store = runtime._service.global_memory_store  # noqa: SLF001
        store.write_cell(memory_system="semantic", content="stale", summary="stale", source="global")
        runtime._service.global_store_lifecycle("load")  # noqa: SLF001
        assert [cell.summary for cell in store.cells.values()] == ["loaded"]
        runtime._service.flush_session_memory()  # noqa: SLF001

    with EmbeddedCivilization(config, runtime_factory=lambda: Stage60SlowFakeRuntime(0.0)) as runtime:
        runtime.start()
        restored = runtime._service.global_memory_store  # noqa: SLF001
        # The reset record prevents the pre-load cell from being replayed back in.
        assert [cell.summary for cell in restored.cells.values()] == ["loaded"]


def test_global_store_load_still_assigns_a_plain_store_without_persistence(tmp_path: Path):
    config, state_dir = _embedded(tmp_path, persist_sessions=False)
    global_file = state_dir / "global-memory.json"
    global_file.parent.mkdir(parents=True, exist_ok=True)
    global_file.write_text(json.dumps({"cells": [], "links": []}), encoding="utf-8")

    with EmbeddedCivilization(config, runtime_factory=lambda: Stage60SlowFakeRuntime(0.0)) as runtime:
        runtime.start()
        runtime._service.global_store_lifecycle("load")  # noqa: SLF001
        assert isinstance(runtime._service.global_memory_store, OrionMemoryStore)  # noqa: SLF001


# --------------------------------------------------------------------------- #
# retention
# --------------------------------------------------------------------------- #
def _age_session(state_dir: Path, session_id: str, days: float) -> None:
    directory = session_directory(state_dir, session_id)
    stamp = time.time() - days * 86400
    for item in directory.iterdir():
        os.utime(item, (stamp, stamp))


def test_retention_prunes_expired_sessions_and_keeps_recent_ones(tmp_path: Path):
    persistence = SessionPersistence(_config(tmp_path))
    for name in ("old-session", "recent-session"):
        store = persistence.hydrate(name)
        store.write_cell(memory_system="episodic", content="fact", summary="fact", source="test")
    persistence.flush()
    _age_session(tmp_path, "old-session", days=30)

    report = persistence.retention_sweep(older_than_seconds=7 * 86400)

    assert [item["session"] for item in report["pruned"]] == ["old-session"]
    assert report["bytes_freed"] > 0
    assert not session_directory(tmp_path, "old-session").exists()
    assert session_directory(tmp_path, "recent-session").exists()
    persistence.close()


def test_retention_keeps_the_most_recent_sessions(tmp_path: Path):
    persistence = SessionPersistence(_config(tmp_path))
    for name in ("a", "b", "c"):
        persistence.hydrate(name).write_cell(memory_system="episodic", content="x", summary="x", source="t")
    persistence.flush()
    for index, name in enumerate(("a", "b", "c")):
        _age_session(tmp_path, name, days=10 + index)

    report = persistence.retention_sweep(older_than_seconds=86400, keep_sessions=1)

    assert [item["session"] for item in report["pruned"]] == ["b", "c"]
    assert session_directory(tmp_path, "a").exists()
    persistence.close()


def test_retention_compacts_large_journals_and_preserves_state(tmp_path: Path):
    persistence = SessionPersistence(_config(tmp_path, snapshot_every_records=100000))
    store = persistence.hydrate("busy")
    for _ in range(20):
        store.write_cell(memory_system="episodic", content="fact", summary="fact", source="test")
    persistence.flush()
    journal = session_directory(tmp_path, "busy") / "journal.ndjson"
    assert journal.stat().st_size > 0

    report = persistence.retention_sweep(compact_over_records=10)

    assert report["compacted"] and report["compacted"][0]["session"] == "busy"
    assert report["compacted"][0]["journal_bytes_after"] == 0
    persistence.close()

    restored = SessionPersistence(_config(tmp_path)).hydrate("busy")
    assert len(restored.cells) == 20


def test_retention_dry_run_changes_nothing(tmp_path: Path):
    persistence = SessionPersistence(_config(tmp_path, snapshot_every_records=100000))
    persistence.hydrate("old").write_cell(memory_system="episodic", content="x", summary="x", source="t")
    for _ in range(12):
        persistence.hydrate("busy").write_cell(memory_system="episodic", content="y", summary="y", source="t")
    persistence.flush()
    _age_session(tmp_path, "old", days=30)

    report = persistence.retention_sweep(older_than_seconds=86400, compact_over_records=5, dry_run=True)

    assert report["dry_run"] is True
    assert report["pruned"] and report["compacted"]
    assert session_directory(tmp_path, "old").exists()
    assert (session_directory(tmp_path, "busy") / "journal.ndjson").stat().st_size > 0
    persistence.close()


def test_retention_never_touches_protected_sessions(tmp_path: Path):
    persistence = SessionPersistence(_config(tmp_path))
    persistence.hydrate("live").write_cell(memory_system="episodic", content="x", summary="x", source="t")
    persistence.flush()
    _age_session(tmp_path, "live", days=90)

    report = persistence.retention_sweep(older_than_seconds=86400, skip_sessions=["live"])

    assert report["pruned"] == []
    assert report["skipped"] == ["live"]
    assert session_directory(tmp_path, "live").exists()
    persistence.close()


def test_service_retention_protects_the_sessions_it_is_serving(tmp_path: Path):
    config, state_dir = _embedded(tmp_path)

    with EmbeddedCivilization(config, runtime_factory=lambda: Stage60SlowFakeRuntime(0.0)) as runtime:
        client = runtime.start()
        for name in ("live-a", "live-b"):
            client.write_memory(session_id=name, memory_system="episodic", content="fact", summary="fact")
        report = client._request(  # noqa: SLF001 - admin endpoint under test
            "POST", "/admin/orion/memory/retention", {"older_than_days": 0, "dry_run": False}
        )["retention"]
        # Both sessions (and the global store) are loaded in memory, so none may be pruned.
        assert report["pruned"] == []
        assert set(report["skipped"]) >= {"live-a", "live-b", "__global__"}
        assert client.read_memory(session_id="live-a", query="fact")


def test_prune_cli_offline_prunes_and_reports(tmp_path: Path, capsys):
    from civilization.cli import main

    persistence = SessionPersistence(_config(tmp_path / "state"))
    persistence.hydrate("stale").write_cell(memory_system="episodic", content="x", summary="x", source="t")
    persistence.flush()
    persistence.close()
    _age_session(tmp_path / "state", "stale", days=45)

    exit_code = main(["prune", "--state-dir", str(tmp_path / "state"), "--older-than-days", "30"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "pruned:    1" in output
    assert not session_directory(tmp_path / "state", "stale").exists()


# --------------------------------------------------------------------------- #
# write coalescing
# --------------------------------------------------------------------------- #
def test_request_scope_coalesces_appends_and_writes_before_returning(tmp_path: Path):
    persistence = SessionPersistence(_config(tmp_path, snapshot_every_records=100000))
    store = persistence.hydrate("s")

    with persistence.request_scope():
        for index in range(5):
            store.write_cell(memory_system="episodic", content=f"fact {index}", summary=f"fact {index}", source="t")
        # Inside the scope the journal has not been written yet...
        journal_path = session_directory(tmp_path, "s") / "journal.ndjson"
        assert not journal_path.exists() or journal_path.stat().st_size == 0

    usage = persistence.status()["sink"]["journals"]["s"]
    assert usage["records_written"] == 5
    assert usage["appends"] == 1
    # ...and after the scope exits everything is on disk, so anything the caller
    # was told about is durable.
    assert journal_path.stat().st_size > 0
    assert len(list(persistence._journal("s").iter_journal())) == 5  # noqa: SLF001
    persistence.close()


def test_nested_request_scopes_share_one_buffer(tmp_path: Path):
    persistence = SessionPersistence(_config(tmp_path, snapshot_every_records=100000))
    store = persistence.hydrate("s")

    with persistence.request_scope():
        store.write_cell(memory_system="episodic", content="outer", summary="outer", source="t")
        with persistence.request_scope():  # inner scopes must not flush early
            store.write_cell(memory_system="episodic", content="inner", summary="inner", source="t")

    usage = persistence.status()["sink"]["journals"]["s"]
    assert usage["records_written"] == 2
    assert usage["appends"] == 1
    persistence.close()


def test_buffered_records_still_trigger_compaction(tmp_path: Path):
    persistence = SessionPersistence(_config(tmp_path, snapshot_every_records=3))
    store = persistence.hydrate("s")

    with persistence.request_scope():
        for index in range(3):
            store.write_cell(memory_system="episodic", content=f"f{index}", summary=f"f{index}", source="t")

    snapshot_path = session_directory(tmp_path, "s") / "snapshot.json"
    assert snapshot_path.exists(), "buffered records must count towards the snapshot threshold"
    restored = SessionPersistence(_config(tmp_path)).hydrate("s")
    assert len(restored.cells) == 3
    persistence.close()


def test_a_service_request_coalesces_its_memory_writes(tmp_path: Path):
    config, state_dir = _embedded(tmp_path)

    with EmbeddedCivilization(config, runtime_factory=lambda: Stage60SlowFakeRuntime(0.0)) as runtime:
        client = runtime.start()
        runtime._service.memory_store("coalesce")  # noqa: SLF001 - hydrate before the request
        client.predict(
            CivilizationRequest(
                text="The canary passed. Choose.",
                answer_options=("approve", "reject"),
                session_id="coalesce",
                task_name="coalesce_test",
            )
        )
        usage = runtime._service.session_persistence.status()["sink"]["journals"]["coalesce"]  # noqa: SLF001

    # A decision writes several cells and links; they leave in far fewer appends.
    assert usage["records_written"] >= 3
    assert usage["appends"] <= 2
    assert usage["records_per_append"] >= 2.0


# --------------------------------------------------------------------------- #
# sidecar reconnection
# --------------------------------------------------------------------------- #
def test_sidecar_reconnects_after_it_returns(tmp_path: Path):
    socket_dir = Path(tempfile.mkdtemp(prefix="civ-reconnect-"))
    socket_path = socket_dir / "s.sock"

    client = SessionPersistence(
        PersistenceConfig(
            mode=MODE_SIDECAR,
            state_dir=str(tmp_path / "state"),
            sidecar_socket=str(socket_path),
            fsync=FSYNC_EVERY_WRITE,
            sidecar_reconnect_interval_seconds=0.2,
        )
    )

    # No sidecar yet: writes degrade to local journaling rather than failing.
    store = client.hydrate("s")
    store.write_cell(memory_system="episodic", content="written while degraded", summary="degraded", source="t")
    assert client.status()["sink"]["degraded"] is True
    assert client.status()["sink"]["attached"] is False

    # The sidecar appears; the client re-attaches on its own.
    server = SidecarServer(
        PersistenceConfig(
            mode=MODE_IN_PROCESS,
            state_dir=str(tmp_path / "state"),
            fsync=FSYNC_EVERY_WRITE,
            sidecar_socket=str(socket_path),
        )
    )
    ready = threading.Event()
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"on_ready": lambda _path: ready.set()}, daemon=True
    )
    thread.start()
    assert ready.wait(5.0)

    deadline = time.time() + 5.0
    while time.time() < deadline and not client.status()["sink"]["attached"]:
        time.sleep(0.1)

    status = client.status()["sink"]
    assert status["attached"] is True, "the client must re-attach without a service restart"
    assert status["degraded"] is False
    assert status["reconnects"] >= 1

    # New writes go through the sidecar, and the records written while degraded
    # are still part of the same durable state.
    before = status["sent_records"]
    store.write_cell(memory_system="semantic", content="written after reattach", summary="reattached", source="t")
    client.flush()
    assert client.status()["sink"]["sent_records"] > before

    server.stop()
    thread.join(timeout=5.0)
    client.close()

    restored = SessionPersistence(_config(tmp_path / "state")).hydrate("s")
    assert len(restored.cells) == 2, "state written in both phases must survive"
