from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from civilization.engine.stages.stage48_production_jsonl_inference import Stage48Limits
from civilization.engine.stages.stage49_persistent_inference_service import Stage49ServiceConfig
from civilization.engine.stages.stage59_access_control_audit import Stage59SecurityConfig
from civilization.engine.stages.stage61_async_jobs import Stage61JobConfig, wait_for_job
from civilization.engine.stages.stage62_persistent_async_jobs import Stage62PersistenceConfig
from civilization.engine.stages.stage63_job_retention_listing import Stage63RetentionConfig
from civilization.engine.stages.stage64_external_result_store import Stage64ResultStoreConfig
from civilization.engine.stages.stage65_batch_jobs import Stage65BatchConfig
from civilization.engine.stages.stage66_batch_export import Stage66ExportConfig
from civilization.engine.stages.stage67_export_lifecycle import Stage67ExportLifecycleConfig
from civilization.engine.stages.stage68_export_package_delivery import Stage68PackageConfig
from civilization.engine.stages.stage69_streaming_package_delivery import Stage69StreamingConfig
from civilization.engine.stages.stage73_orion_memory_kernel import MemorySystem
from civilization.engine.stages.stage74_orion_memory_service import (
    Stage74MemoryConfig,
    build_stage74_fake_service,
    run_stage74_orion_memory_service_smoke,
)


def _payload(request_id: str = "stage74-request", session_id: str = "session-a") -> dict:
    return {
        "id": request_id,
        "text": "Use persistent Orion semantic memory for this operation.",
        "memory_items": ["caller provided memory"],
        "rule_items": ["keep Qwen frozen"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "orion_operation",
        "seed": 202,
        "controls": ["full"],
        "orion_memory": {"session_id": session_id},
    }


def _service(tmp_path, **overrides):
    defaults = {
        "job_config": Stage61JobConfig(job_log_path=str(tmp_path / "jobs.jsonl"), max_queued_jobs=8, result_ttl_seconds=3600),
        "persistence_config": Stage62PersistenceConfig(job_state_path=str(tmp_path / "jobs_state.json")),
        "retention_config": Stage63RetentionConfig(max_persisted_jobs=8, max_list_limit=20),
        "result_store_config": Stage64ResultStoreConfig(result_dir=str(tmp_path / "results"), inline_result_row_limit=0, max_result_page_limit=1),
        "batch_config": Stage65BatchConfig(max_batch_submit=4, max_batch_result_jobs=4),
        "export_config": Stage66ExportConfig(export_dir=str(tmp_path / "exports"), max_export_jobs=4),
        "lifecycle_config": Stage67ExportLifecycleConfig(max_persisted_exports=4, max_export_list_limit=10),
        "package_config": Stage68PackageConfig(max_package_bytes=1024 * 1024),
        "streaming_config": Stage69StreamingConfig(download_chunk_bytes=32),
    }
    defaults.update(overrides)
    return build_stage74_fake_service(port=0, **defaults)


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urlopen(request, timeout=10) as response:  # noqa: S310 - local test client
        return json.loads(response.read().decode())


def _get(url: str) -> dict:
    with urlopen(url, timeout=10) as response:  # noqa: S310 - local test client
        return json.loads(response.read().decode())


def test_stage74_retrieves_only_same_session_and_writes_trace(tmp_path) -> None:
    service = _service(tmp_path)
    seeded = service.write_memory_cell(
        "session-a",
        {"memory_system": "semantic", "content": "persistent Orion semantic memory", "summary": "persistent semantic rule"},
    )
    service.write_memory_cell(
        "session-b",
        {"memory_system": "semantic", "content": "private session B fact", "summary": "private B fact"},
    )

    rows = service.predict_record(_payload())
    trace = rows[0]["orion_memory"]

    assert rows[0]["status"] == "ok"
    assert seeded.cell_id in trace["retrieved_cell_ids"]
    assert trace["working_cell_id"] is not None
    assert trace["episodic_cell_id"] is not None
    assert trace["procedural_cell_id"] is not None
    assert all("session-b" not in item["cell"]["source"] for item in service.read_memory("session-a", {"query": "private", "limit": 10}))
    assert service.memory_store("session-a").summary()["trace_count"] > 0


def test_stage74_respects_stage48_context_budget(tmp_path) -> None:
    service = _service(tmp_path)
    service.config = Stage49ServiceConfig(limits=Stage48Limits(max_context_items=2))
    service.write_memory_cell(
        "session-a",
        {"memory_system": "semantic", "content": "persistent Orion semantic memory", "summary": "persistent semantic rule"},
    )

    rows = service.predict_record(_payload())

    assert rows[0]["status"] == "ok"
    assert rows[0]["orion_memory"]["injected_item_count"] == 0


def test_stage74_working_ttl_expires_without_cross_session_leak(tmp_path) -> None:
    service = _service(tmp_path, memory_config=Stage74MemoryConfig(working_ttl_seconds=0.0))
    rows = service.predict_record(_payload())
    working_cell_id = rows[0]["orion_memory"]["working_cell_id"]

    results = service.read_memory("session-a", {"query": "active request", "memory_system": MemorySystem.WORKING.value})

    assert working_cell_id not in [item["cell"]["cell_id"] for item in results]
    assert service.memory_store("session-b").summary()["cell_counts"]["working"] == 0


def test_stage77_replay_is_opt_in_and_consolidates_second_session_episode(tmp_path) -> None:
    service = _service(tmp_path, memory_config=Stage74MemoryConfig(replay_after_request=True, replay_min_episodes=2))
    first = service.predict_record(_payload("replay-1"))[0]["orion_memory"]
    second = service.predict_record(_payload("replay-2"))[0]["orion_memory"]

    assert first["replay"]["accepted"] is False
    assert first["replay"]["reason"] == "insufficient_episodes"
    assert second["replay"]["accepted"] is True
    assert second["replay"]["semantic_cell_id"] in service.memory_store("session-a").cells


def test_stage80_global_retrieval_is_request_opt_in(tmp_path) -> None:
    service = _service(tmp_path)
    service.global_memory_store.write_cell(memory_system="semantic", content="global Orion evidence", summary="global evidence", source="stage78")
    local = service.predict_record(_payload("local"))[0]["orion_memory"]
    global_payload = _payload("global")
    global_payload["orion_memory"]["include_global"] = True
    global_trace = service.predict_record(global_payload)[0]["orion_memory"]
    assert "global" not in local["retrieved_tiers"]
    assert "global" not in global_trace["retrieved_tiers"]
    assert global_trace["excluded_retrieval"]


def test_stage87_only_injects_resolved_global_winner_when_task_policy_allows(tmp_path) -> None:
    service = _service(tmp_path)
    winner = service.global_memory_store.write_cell(memory_system="semantic", content="global Orion evidence", summary="global evidence", source="stage78", confidence=0.9)
    loser = service.global_memory_store.write_cell(memory_system="semantic", content="global rejected evidence", summary="rejected", source="stage78", confidence=0.4)
    service.global_memory_store.mark_conflict(winner.cell_id, loser.cell_id, reason="test")
    from civilization.engine.stages.stage84_orion_conflict_resolution import OrionConflictResolutionPolicy
    OrionConflictResolutionPolicy().resolve(service.global_memory_store, winner.cell_id, loser.cell_id)
    payload = _payload("stage87")
    payload["orion_memory"].update({"include_global": True, "allow_resolved_global": False})
    denied = service.predict_record(payload)[0]["orion_memory"]
    payload["orion_memory"]["allow_resolved_global"] = True
    allowed = service.predict_record(payload)[0]["orion_memory"]
    assert "global" not in denied["retrieved_tiers"]
    assert any(entry["reason"] == "global_not_resolved_winner" for entry in denied["excluded_retrieval"])
    assert "global" in allowed["retrieved_tiers"]


def test_stage94_global_store_lifecycle_admin_api_requires_token(tmp_path) -> None:
    token = "stage94-token"
    service = _service(tmp_path, security=Stage59SecurityConfig(bearer_token=token, require_token_for_admin=True, audit_log_path=str(tmp_path / "audit.jsonl")), memory_config=Stage74MemoryConfig(global_store_path=str(tmp_path / "global.json")))
    service.global_memory_store.write_cell(memory_system="semantic", content="persist", summary="persist", source="global")
    server = service.start_background(); host, port = server.server_address; url = f"http://{host}:{port}/admin/orion/global-store/save"
    try:
        try:
            _post(url, {})
            raise AssertionError("admin lifecycle must require token")
        except HTTPError as error:
            assert error.code == 401
        saved = _post(url, {}, token=token)
        service.global_memory_store = type(service.global_memory_store)()
        loaded = _post(f"http://{host}:{port}/admin/orion/global-store/load", {}, token=token)
    finally:
        service.shutdown()
    assert saved["status"] == "ok" and loaded["status"] == "ok"
    assert len(service.global_memory_store.cells) == 1
    assert service.memory_metrics.global_store_saves == 1
    assert service.memory_metrics.global_store_loads == 1


def test_stage95_restart_requires_explicit_global_store_load(tmp_path) -> None:
    path = tmp_path / "global.json"
    first = _service(tmp_path, memory_config=Stage74MemoryConfig(global_store_path=str(path)))
    first.global_memory_store.write_cell(memory_system="semantic", content="persist", summary="persist", source="global")
    first.global_store_lifecycle("save")
    second = _service(tmp_path, memory_config=Stage74MemoryConfig(global_store_path=str(path)))
    assert not second.global_memory_store.cells
    second.global_store_lifecycle("load")
    assert len(second.global_memory_store.cells) == 1
    assert second.memory_metrics.global_store_loads == 1


def test_stage97_global_store_status_is_health_visible_and_admin_protected(tmp_path) -> None:
    token = "stage97-token"
    service = _service(tmp_path, security=Stage59SecurityConfig(bearer_token=token, require_token_for_admin=True, audit_log_path=str(tmp_path / "audit.jsonl")), memory_config=Stage74MemoryConfig(global_store_path=str(tmp_path / "global.json")))
    health = service.health()
    server = service.start_background(); host, port = server.server_address
    try:
        try:
            _get(f"http://{host}:{port}/admin/orion/global-store/status")
            raise AssertionError("status must require token")
        except HTTPError as error:
            assert error.code == 401
        request = Request(f"http://{host}:{port}/admin/orion/global-store/status", headers={"Authorization": f"Bearer {token}"})
        with urlopen(request, timeout=10) as response:
            status = json.loads(response.read().decode())
    finally:
        service.shutdown()
    assert health["global_store"]["configured"]
    assert status["global_store"]["configured"]


def test_stage74_admin_memory_api_follows_stage59_token_policy(tmp_path) -> None:
    token = "stage74-token"
    service = _service(
        tmp_path,
        security=Stage59SecurityConfig(bearer_token=token, require_token_for_admin=True, require_token_for_predict=True, audit_log_path=str(tmp_path / "audit.jsonl")),
    )
    server = service.start_background()
    host, port = server.server_address
    url = f"http://{host}:{port}/admin/orion/memory/write"
    payload = {"session_id": "session-a", "memory_system": "semantic", "content": "admin seeded memory"}
    try:
        try:
            _post(url, payload)
            raise AssertionError("admin write should require token")
        except HTTPError as error:
            assert error.code == 401
        written = _post(url, payload, token=token)
    finally:
        service.shutdown()

    assert written["status"] == "ok"
    assert written["cell"]["memory_system"] == "semantic"


def test_stage74_async_jobs_use_same_predict_hook(tmp_path) -> None:
    service = _service(tmp_path)
    service.write_memory_cell(
        "session-a",
        {"memory_system": "semantic", "content": "persistent Orion semantic memory", "summary": "persistent semantic rule"},
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        accepted = _post(f"{base}/v1/jobs", {"request": _payload("job-a")})
        finished = wait_for_job(base, accepted["job"]["job_id"], timeout_seconds=5)
        result_page = _get(f"{base}/v1/jobs/{accepted['job']['job_id']}/result?limit=1")
    finally:
        service.shutdown()

    assert finished["job"]["status"] == "completed"
    assert finished["job"]["result"]["result_ref"]["storage"] == "jsonl"
    row = result_page["result"]["rows"][0]
    assert row["orion_memory"]["injected_item_count"] >= 1


def test_stage74_smoke_writes_memory_artifacts(tmp_path) -> None:
    summary = run_stage74_orion_memory_service_smoke(output_dir=tmp_path)

    assert summary["passes_stage_gate"]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "memory_cells.json").exists()
    assert (tmp_path / "trace_events.jsonl").exists()
