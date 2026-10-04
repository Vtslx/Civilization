"""Durability tests for session memory: journal, snapshot, replay, sidecar."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path

import pytest

from civilization import EmbeddedCivilization, EmbeddedConfig
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

