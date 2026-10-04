from __future__ import annotations

from dataclasses import asdict, dataclass
from http import HTTPStatus
import json
from pathlib import Path
import re
import threading
from typing import Any

from .stage49_persistent_inference_service import Stage49ServiceConfig
from .stage51_runtime_management import Stage51RuntimeDescriptor
from .stage59_access_control_audit import Stage59SecurityConfig
from .stage60_queue_rate_limit import Stage60QueueConfig, Stage60SlowFakeRuntime
from .stage61_async_jobs import Stage61JobConfig, _get_json, _post_json
from .stage62_persistent_async_jobs import Stage62PersistenceConfig
from .stage63_job_retention_listing import Stage63RetentionConfig
from .stage64_external_result_store import Stage64ResultStoreConfig
from .stage65_batch_jobs import Stage65BatchConfig
from .stage66_batch_export import Stage66ExportConfig
from .stage67_export_lifecycle import Stage67ExportLifecycleConfig
from .stage68_export_package_delivery import Stage68PackageConfig
from .stage69_streaming_package_delivery import Stage69StreamingConfig
from .stage71_if_range_package_delivery import Stage71IfRangePackageService, build_stage71_real_service
from ..persistence import (
    FSYNC_INTERVAL,
    MODE_IN_PROCESS,
    PersistenceConfig,
    SessionPersistence,
)
from .stage73_orion_memory_kernel import MemoryCell, MemoryLinkType, MemorySystem, OrionMemoryStore
from .stage75_orion_adapter_context_bridge import OrionAdapterContextBridge
from .stage76_orion_replay_consolidation import OrionReplayConsolidationPolicy, Stage76ReplayConfig
from .stage79_orion_global_retrieval import OrionGlobalRetrievalRouter
from .stage86_orion_task_memory_policy import OrionTaskMemoryPolicy
from .stage93_orion_global_store_lifecycle import OrionGlobalStoreLifecycle
from civilization.engine.model_paths import DEFAULT_MODEL_PATH


DEFAULT_OUTPUT_DIR = Path("artifacts/civilization/stage74_orion_memory_service")
DEFAULT_JOB_STATE = Path("artifacts/civilization/logs/stage74_jobs_state.json")
DEFAULT_JOB_LOG = Path("artifacts/civilization/logs/stage74_jobs.jsonl")
DEFAULT_RESULT_DIR = Path("artifacts/civilization/logs/stage74_job_results")
DEFAULT_EXPORT_DIR = Path("artifacts/civilization/logs/stage74_exports")
DEFAULT_SESSION_ID = "default"


@dataclass(frozen=True)
class Stage74MemoryConfig:
    working_ttl_seconds: float = 300.0
    retrieval_limit: int = 4
    max_injected_items: int = 4
    max_injected_item_chars: int = 512
    write_working_memory: bool = True
    write_episodic_memory: bool = True
    write_procedural_memory: bool = True
    replay_after_request: bool = False
    replay_min_episodes: int = 2
    global_store_path: str | None = None
    # Session memory durability: when a directory is configured, every session
    # mutation is journaled, committed by the fsync policy, and replayed on the
    # next startup.
    session_persistence_dir: str | None = None
    session_persistence_mode: str = MODE_IN_PROCESS
    session_fsync: str = FSYNC_INTERVAL
    session_fsync_interval_seconds: float = 0.25
    session_snapshot_every_records: int = 512
    session_sidecar_socket: str | None = None
    persist_traces: bool = False


@dataclass
class Stage74MemoryMetrics:
    requests_with_memory: int = 0
    retrieved_cells: int = 0
    injected_items: int = 0
    working_writes: int = 0
    episodic_writes: int = 0
    procedural_writes: int = 0
    explicit_writes: int = 0
    explicit_reads: int = 0
    consolidations: int = 0
    global_store_saves: int = 0
    global_store_loads: int = 0
    global_store_lifecycle_failures: int = 0

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


def _session_source(session_id: str) -> str:
    return f"stage74_session:{session_id}"


def _valid_session_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value))


class Stage74OrionMemoryService(Stage71IfRangePackageService):
    def __init__(
        self,
        *,
        runtime_factory,
        descriptor: Stage51RuntimeDescriptor,
        config: Stage49ServiceConfig = Stage49ServiceConfig(),
        security: Stage59SecurityConfig = Stage59SecurityConfig(),
        queue_config: Stage60QueueConfig = Stage60QueueConfig(),
        job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
        persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(job_state_path=str(DEFAULT_JOB_STATE)),
        retention_config: Stage63RetentionConfig = Stage63RetentionConfig(),
        result_store_config: Stage64ResultStoreConfig = Stage64ResultStoreConfig(result_dir=str(DEFAULT_RESULT_DIR)),
        batch_config: Stage65BatchConfig = Stage65BatchConfig(),
        export_config: Stage66ExportConfig = Stage66ExportConfig(export_dir=str(DEFAULT_EXPORT_DIR)),
        lifecycle_config: Stage67ExportLifecycleConfig = Stage67ExportLifecycleConfig(),
        package_config: Stage68PackageConfig = Stage68PackageConfig(),
        streaming_config: Stage69StreamingConfig = Stage69StreamingConfig(),
        memory_config: Stage74MemoryConfig = Stage74MemoryConfig(),
    ) -> None:
        if memory_config.working_ttl_seconds < 0:
            raise ValueError("working_ttl_seconds must be >= 0")
        if memory_config.retrieval_limit < 1:
            raise ValueError("retrieval_limit must be >= 1")
        if memory_config.max_injected_items < 0:
            raise ValueError("max_injected_items must be >= 0")
        if memory_config.max_injected_item_chars < 1:
            raise ValueError("max_injected_item_chars must be >= 1")
        if memory_config.replay_min_episodes < 1:
            raise ValueError("replay_min_episodes must be >= 1")
        self.memory_config = memory_config
        self._adapter_bridge = OrionAdapterContextBridge(
            max_items=memory_config.max_injected_items,
            max_item_chars=memory_config.max_injected_item_chars,
        )
        self._replay_policy = OrionReplayConsolidationPolicy(Stage76ReplayConfig(min_episodes=memory_config.replay_min_episodes))
        self.memory_metrics = Stage74MemoryMetrics()
        self._memory_lock = threading.RLock()
        self._memory_stores: dict[str, OrionMemoryStore] = {}
        self.session_persistence = SessionPersistence(
            PersistenceConfig(
                mode=memory_config.session_persistence_mode if memory_config.session_persistence_dir else "none",
                state_dir=memory_config.session_persistence_dir or "var/state",
                fsync=memory_config.session_fsync,
                fsync_interval_seconds=memory_config.session_fsync_interval_seconds,
                snapshot_every_records=memory_config.session_snapshot_every_records,
                sidecar_socket=memory_config.session_sidecar_socket,
                persist_traces=memory_config.persist_traces,
            )
        )
        self.global_memory_store = OrionMemoryStore()
        self._global_router = OrionGlobalRetrievalRouter()
        super().__init__(
            runtime_factory=runtime_factory,
            descriptor=descriptor,
            config=config,
            security=security,
            queue_config=queue_config,
            job_config=job_config,
            persistence_config=persistence_config,
            retention_config=retention_config,
            result_store_config=result_store_config,
            batch_config=batch_config,
            export_config=export_config,
            lifecycle_config=lifecycle_config,
            package_config=package_config,
            streaming_config=streaming_config,
        )

    def memory_store(self, session_id: str = DEFAULT_SESSION_ID) -> OrionMemoryStore:
        if not _valid_session_id(session_id):
            raise ValueError("session_id must use 1-128 letters, digits, '.', '_', ':' or '-'")
        with self._memory_lock:
            store = self._memory_stores.get(session_id)
            if store is None:
                # First touch of this session in this process: rebuild it from the
                # durable snapshot and journal tail, then keep serving from memory.
                store = self.session_persistence.hydrate(session_id)
                self._memory_stores[session_id] = store
            return store

    def flush_session_memory(self) -> dict[str, Any]:
        """Commit pending journal records to stable storage."""

        self.session_persistence.flush()
        return self.session_persistence.status()

    def memory_status(self) -> dict[str, Any]:
        with self._memory_lock:
            sessions = {session_id: store.summary() for session_id, store in sorted(self._memory_stores.items())}
        return {
            "config": asdict(self.memory_config),
            "metrics": self.memory_metrics.snapshot(),
            "session_count": len(sessions),
            "sessions": sessions,
            "persistence": self.session_persistence.status(),
        }

    def shutdown(self) -> None:  # type: ignore[override]
        try:
            self.session_persistence.close()
        finally:
            super().shutdown()

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["stage"] = "stage74_orion_memory_service"
        payload["orion_memory"] = self.memory_status()
        payload["global_store"] = self.global_store_status()
        return payload

    def global_store_status(self) -> dict[str, Any]:
        path = Path(self.memory_config.global_store_path) if self.memory_config.global_store_path else None
        states: dict[str, int] = {}
        for cell in self.global_memory_store.cells.values():
            states[cell.decay_state] = states.get(cell.decay_state, 0) + 1
        return {"configured": path is not None, "path": str(path) if path else None, "exists": path.exists() if path else False, "in_memory_cells": len(self.global_memory_store.cells), "in_memory_links": len(self.global_memory_store.links), "decay_state_counts": states, "metrics": {"saves": self.memory_metrics.global_store_saves, "loads": self.memory_metrics.global_store_loads, "failures": self.memory_metrics.global_store_lifecycle_failures}}

    def global_store_lifecycle(self, action: str) -> dict[str, Any]:
        try:
            if not self.memory_config.global_store_path:
                raise ValueError("global_store_path is not configured")
            lifecycle = OrionGlobalStoreLifecycle(self.memory_config.global_store_path)
            if action == "save":
                self.memory_metrics.global_store_saves += 1
                return lifecycle.save(self.global_memory_store)
            if action == "load":
                store, status = lifecycle.load()
                self.global_memory_store = store
                self.memory_metrics.global_store_loads += 1
                return status
            raise ValueError("unsupported global store lifecycle action")
        except Exception:
            self.memory_metrics.global_store_lifecycle_failures += 1
            raise

    def _memory_options(self, payload: dict[str, Any]) -> tuple[str, bool, bool, bool, bool]:
        raw = payload.get("orion_memory", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("orion_memory must be an object when provided")
        session_id = str(raw.get("session_id", DEFAULT_SESSION_ID))
        if not _valid_session_id(session_id):
            raise ValueError("invalid orion_memory.session_id")
        read_enabled = raw.get("read", True)
        write_enabled = raw.get("write", True)
        include_global = raw.get("include_global", False)
        allow_resolved_global = raw.get("allow_resolved_global", False)
        if not isinstance(read_enabled, bool) or not isinstance(write_enabled, bool) or not isinstance(include_global, bool) or not isinstance(allow_resolved_global, bool):
            raise ValueError("orion_memory.read, write, include_global and allow_resolved_global must be booleans")
        return session_id, read_enabled, write_enabled, include_global, allow_resolved_global

    def _retrieve_for_request(self, store: OrionMemoryStore, *, session_id: str, query: str, payload: dict[str, Any], include_global: bool = False, allow_resolved_global: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        memory_items = payload.get("memory_items", [])
        rule_items = payload.get("rule_items", [])
        if not isinstance(memory_items, list) or not isinstance(rule_items, list):
            return [], []
        if any(not isinstance(item, str) for item in [*memory_items, *rule_items]):
            return [], []
        available = self.config.limits.max_context_items - len(memory_items) - len(rule_items)
        if available <= 0:
            return [], []
        with self._memory_lock:
            tiered = self._global_router.read(store, self.global_memory_store, query=query, include_global=include_global, limit=min(available, self.memory_config.max_injected_items))
        policy_rows = []
        for item in tiered:
            global_cell = self.global_memory_store.cells.get(item.result.cell.cell_id) if item.tier == "global" else None
            policy_rows.append({"tier": item.tier, "cell_id": f"{item.tier}:{item.result.cell.cell_id}", "resolution_status": global_cell.metadata.get("stage84_resolution", "unresolved") if global_cell else "not_applicable"})
        decision = OrionTaskMemoryPolicy(allow_resolved_global=allow_resolved_global).select(policy_rows)
        selected_refs = set(decision["selected_cell_ids"])
        selected = [{"cell_id": item.result.cell.cell_id, "item": self._adapter_bridge.items_for_results([item.result], limit=1)[0].text, "tier": item.tier} for item in tiered if f"{item.tier}:{item.result.cell.cell_id}" in selected_refs]
        excluded = [{**entry, "reference": entry["cell_id"]} for entry in decision["excluded"]]
        self.memory_metrics.retrieved_cells += len(selected)
        self.memory_metrics.injected_items += len(selected)
        return selected, excluded

    def _write_request_trace(
        self,
        *,
        store: OrionMemoryStore,
        session_id: str,
        payload: dict[str, Any],
        rows: list[dict[str, Any]],
        injected: list[dict[str, Any]],
        excluded: list[dict[str, Any]],
        enabled: bool,
    ) -> dict[str, Any]:
        trace: dict[str, Any] = {
            "session_id": session_id,
            "retrieved_cell_ids": [item["cell_id"] for item in injected],
            "retrieved_tiers": [item.get("tier", "session") for item in injected],
            "excluded_retrieval": excluded,
            "injected_item_count": len(injected),
        }
        if not enabled:
            trace["write_skipped"] = True
            return trace
        source = _session_source(session_id)
        request_id = str(payload.get("id", ""))
        task_name = str(payload.get("task_name", "custom"))
        text = str(payload.get("text", ""))
        outcome = "success" if rows and all(row.get("status") == "ok" for row in rows) else "failure"
        working: MemoryCell | None = None
        episodic: MemoryCell | None = None
        procedural: MemoryCell | None = None
        with self._memory_lock:
            if self.memory_config.write_working_memory:
                working = store.write_cell(
                    memory_system=MemorySystem.WORKING,
                    content=text,
                    summary=f"{task_name} active request",
                    source=source,
                    ttl_seconds=max(self.memory_config.working_ttl_seconds, 1e-9),
                    metadata={"request_id": request_id, "stage": 74},
                )
                self.memory_metrics.working_writes += 1
            if self.memory_config.write_episodic_memory:
                episodic = store.write_cell(
                    memory_system=MemorySystem.EPISODIC,
                    content=f"task={task_name}; request_id={request_id}; outcome={outcome}; injected={len(injected)}",
                    summary=f"{task_name} inference {outcome}",
                    source=source,
                    confidence=1.0 if outcome == "success" else 0.4,
                    metadata={"request_id": request_id, "task_name": task_name, "outcome": outcome, "row_count": len(rows), "injected_cell_ids": trace["retrieved_cell_ids"]},
                )
                self.memory_metrics.episodic_writes += 1
            if self.memory_config.write_procedural_memory:
                procedural = store.write_procedural_from_task_trace(
                    task_name=task_name,
                    steps=["retrieve session memory", "run frozen inference", f"record {outcome} outcome"],
                    outcome=outcome,
                    source=source,
                    confidence=1.0 if outcome == "success" else 0.4,
                    metadata={"request_id": request_id, "row_count": len(rows)},
                )
                self.memory_metrics.procedural_writes += 1
            if working is not None and episodic is not None:
                store.link_cells(working.cell_id, episodic.cell_id, link_type=MemoryLinkType.TEMPORAL, weight=1.0)
            if episodic is not None and procedural is not None:
                store.link_cells(episodic.cell_id, procedural.cell_id, link_type=MemoryLinkType.TASK, weight=1.0)
        trace.update(
            {
                "working_cell_id": working.cell_id if working else None,
                "episodic_cell_id": episodic.cell_id if episodic else None,
                "procedural_cell_id": procedural.cell_id if procedural else None,
                "outcome": outcome,
            }
        )
        if self.memory_config.replay_after_request and episodic is not None:
            with self._memory_lock:
                trace["replay"] = self._replay_policy.apply(store, source=source, task_name=task_name).to_dict()
        return trace

    def predict_record(self, payload: dict[str, Any], *, line_number: int = 1) -> list[dict[str, Any]]:
        try:
            session_id, read_enabled, write_enabled, include_global, allow_resolved_global = self._memory_options(payload)
        except ValueError as error:
            return [{"id": str(payload.get("id", "")), "line_number": line_number, "status": "error", "error_type": type(error).__name__, "error": str(error)}]
        store = self.memory_store(session_id)
        query = f"{payload.get('task_name', 'custom')} {payload.get('text', '')}"
        injected, excluded = self._retrieve_for_request(store, session_id=session_id, query=query, payload=payload, include_global=include_global, allow_resolved_global=allow_resolved_global) if read_enabled else ([], [])
        prepared_payload = dict(payload)
        original_items = payload.get("memory_items", [])
        if injected and isinstance(original_items, list):
            prepared_payload["memory_items"] = [*original_items, *(item["item"] for item in injected)]
        rows = super().predict_record(prepared_payload, line_number=line_number)
        trace = self._write_request_trace(
            store=store,
            session_id=session_id,
            payload=payload,
            rows=rows,
            injected=injected,
            excluded=excluded,
            enabled=write_enabled,
        )
        self.memory_metrics.requests_with_memory += 1
        for row in rows:
            row["orion_memory"] = trace
        return rows

    def write_memory_cell(self, session_id: str, payload: dict[str, Any]) -> MemoryCell:
        store = self.memory_store(session_id)
        with self._memory_lock:
            cell = store.write_cell(
                memory_system=payload["memory_system"],
                content=str(payload["content"]),
                summary=str(payload.get("summary", payload["content"])),
                source=_session_source(session_id),
                confidence=float(payload.get("confidence", 1.0)),
                importance=float(payload.get("importance", 1.0)),
                ttl_seconds=float(payload["ttl_seconds"]) if payload.get("ttl_seconds") is not None else None,
                metadata=dict(payload.get("metadata", {})),
            )
        self.memory_metrics.explicit_writes += 1
        return cell

    def read_memory(self, session_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        store = self.memory_store(session_id)
        with self._memory_lock:
            results = store.read(
                str(payload.get("query", "")),
                memory_system=payload.get("memory_system"),
                include_expired=bool(payload.get("include_expired", False)),
                limit=int(payload.get("limit", self.memory_config.retrieval_limit)),
            )
        self.memory_metrics.explicit_reads += 1
        return [result.to_dict() for result in results]

    def consolidate_memory(self, session_id: str, payload: dict[str, Any]) -> MemoryCell:
        store = self.memory_store(session_id)
        with self._memory_lock:
            cell = store.consolidate_episodic_to_semantic(
                list(payload["episodic_cell_ids"]),
                summary=str(payload["summary"]),
                content=str(payload["content"]),
                source=_session_source(session_id),
                metadata=dict(payload.get("metadata", {})),
            )
        self.memory_metrics.consolidations += 1
        return cell

    def make_handler(self):  # type: ignore[override]
        service = self
        base_handler = super().make_handler()

        class Stage74Handler(base_handler):
            server_version = "Stage74OrionMemoryQwenService/1.0"

            def do_GET(self) -> None:  # noqa: N802 - stdlib API
                if self.path == "/admin/orion/global-store/status":
                    if not self._check_access():
                        return
                    self._send_json(HTTPStatus.OK, {"status": "ok", "global_store": service.global_store_status()})
                    return
                if self.path == "/admin/orion/memory":
                    if not self._check_access():
                        return
                    self._send_json(HTTPStatus.OK, {"status": "ok", "orion_memory": service.memory_status()})
                    return
                super().do_GET()

            def do_POST(self) -> None:  # noqa: N802 - stdlib API
                if self.path in {"/admin/orion/global-store/save", "/admin/orion/global-store/load"}:
                    try:
                        payload = self._read_json()
                        if not self._check_access(payload):
                            return
                        action = "save" if self.path.endswith("/save") else "load"
                        self._send_json(HTTPStatus.OK, {"status": "ok", "global_store": service.global_store_lifecycle(action)})
                        return
                    except Exception as error:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "error_type": type(error).__name__, "error": str(error)})
                        return
                if self.path == "/admin/orion/memory/flush":
                    try:
                        if not self._check_access():
                            return
                        self._send_json(HTTPStatus.OK, {"status": "ok", "persistence": service.flush_session_memory()})
                    except Exception as error:
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"status": "error", "error_type": type(error).__name__, "error": str(error)},
                        )
                    return
                if self.path in {"/admin/orion/memory/write", "/admin/orion/memory/read", "/admin/orion/memory/consolidate"}:
                    try:
                        payload = self._read_json()
                        if not isinstance(payload, dict):
                            raise ValueError("memory payload must be an object")
                        if not self._check_access(payload):
                            return
                        session_id = str(payload.get("session_id", DEFAULT_SESSION_ID))
                        if self.path.endswith("/write"):
                            cell = service.write_memory_cell(session_id, payload)
                            self._send_json(HTTPStatus.OK, {"status": "ok", "cell": cell.to_dict()})
                            return
                        if self.path.endswith("/read"):
                            results = service.read_memory(session_id, payload)
                            self._send_json(HTTPStatus.OK, {"status": "ok", "results": results})
                            return
                        cell = service.consolidate_memory(session_id, payload)
                        self._send_json(HTTPStatus.OK, {"status": "ok", "cell": cell.to_dict()})
                        return
                    except Exception as error:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "error_type": type(error).__name__, "error": str(error)})
                        return
                super().do_POST()

        return Stage74Handler


def build_stage74_fake_service(
    *,
    port: int = 0,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
    persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(job_state_path=str(DEFAULT_JOB_STATE)),
    retention_config: Stage63RetentionConfig = Stage63RetentionConfig(),
    result_store_config: Stage64ResultStoreConfig = Stage64ResultStoreConfig(result_dir=str(DEFAULT_RESULT_DIR)),
    batch_config: Stage65BatchConfig = Stage65BatchConfig(),
    export_config: Stage66ExportConfig = Stage66ExportConfig(export_dir=str(DEFAULT_EXPORT_DIR)),
    lifecycle_config: Stage67ExportLifecycleConfig = Stage67ExportLifecycleConfig(),
    package_config: Stage68PackageConfig = Stage68PackageConfig(),
    streaming_config: Stage69StreamingConfig = Stage69StreamingConfig(),
    memory_config: Stage74MemoryConfig = Stage74MemoryConfig(),
) -> Stage74OrionMemoryService:
    return Stage74OrionMemoryService(
        runtime_factory=Stage60SlowFakeRuntime,
        descriptor=Stage51RuntimeDescriptor(package_manifest="fake", centroid_bundle="fake", label="stage74-fake"),
        config=Stage49ServiceConfig(port=port),
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        persistence_config=persistence_config,
        retention_config=retention_config,
        result_store_config=result_store_config,
        batch_config=batch_config,
        export_config=export_config,
        lifecycle_config=lifecycle_config,
        package_config=package_config,
        streaming_config=streaming_config,
        memory_config=memory_config,
    )


def build_stage74_real_service(
    *,
    port: int = 8765,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
    persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(job_state_path=str(DEFAULT_JOB_STATE)),
    retention_config: Stage63RetentionConfig = Stage63RetentionConfig(),
    result_store_config: Stage64ResultStoreConfig = Stage64ResultStoreConfig(result_dir=str(DEFAULT_RESULT_DIR)),
    batch_config: Stage65BatchConfig = Stage65BatchConfig(),
    export_config: Stage66ExportConfig = Stage66ExportConfig(export_dir=str(DEFAULT_EXPORT_DIR)),
    lifecycle_config: Stage67ExportLifecycleConfig = Stage67ExportLifecycleConfig(),
    package_config: Stage68PackageConfig = Stage68PackageConfig(),
    streaming_config: Stage69StreamingConfig = Stage69StreamingConfig(),
    memory_config: Stage74MemoryConfig = Stage74MemoryConfig(),
    package_manifest: str = "artifacts/civilization/stage45_adapter_package/package_manifest.json",
    centroid_bundle: str = "artifacts/civilization/stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt",
    model_path: str = str(DEFAULT_MODEL_PATH),
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> Stage74OrionMemoryService:
    base = build_stage71_real_service(
        port=port,
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        persistence_config=persistence_config,
        retention_config=retention_config,
        result_store_config=result_store_config,
        batch_config=batch_config,
        export_config=export_config,
        lifecycle_config=lifecycle_config,
        package_config=package_config,
        streaming_config=streaming_config,
        package_manifest=package_manifest,
        centroid_bundle=centroid_bundle,
        model_path=model_path,
        preferred_device=preferred_device,
        max_length=max_length,
    )
    return Stage74OrionMemoryService(
        runtime_factory=base._runtime_factory,  # noqa: SLF001 - preserve the validated Stage71 runtime factory
        descriptor=base.descriptor,
        config=base.config,
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        persistence_config=persistence_config,
        retention_config=retention_config,
        result_store_config=result_store_config,
        batch_config=batch_config,
        export_config=export_config,
        lifecycle_config=lifecycle_config,
        package_config=package_config,
        streaming_config=streaming_config,
        memory_config=memory_config,
    )


def _smoke_payload(request_id: str, *, session_id: str = "orion-smoke") -> dict[str, Any]:
    return {
        "id": request_id,
        "text": "Route the request through the Orion memory service.",
        "memory_items": ["baseline memory evidence"],
        "rule_items": ["keep the frozen adapter path unchanged"],
        "state_values": [1.0, 0.0, 0.5],
        "answer_options": ["approve", "reject"],
        "task_name": "orion_service_smoke",
        "seed": 202,
        "controls": ["full"],
        "orion_memory": {"session_id": session_id},
    }


def run_stage74_orion_memory_service_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    service = build_stage74_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(output / "jobs.jsonl"), max_queued_jobs=8, result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(output / "jobs_state.json")),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=8, max_list_limit=20),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(output / "results"), inline_result_row_limit=0, max_result_page_limit=1),
        batch_config=Stage65BatchConfig(max_batch_submit=4, max_batch_result_jobs=4),
        export_config=Stage66ExportConfig(export_dir=str(output / "exports"), max_export_jobs=4),
        lifecycle_config=Stage67ExportLifecycleConfig(max_persisted_exports=4, max_export_list_limit=10),
        package_config=Stage68PackageConfig(max_package_bytes=1024 * 1024),
        streaming_config=Stage69StreamingConfig(download_chunk_bytes=32),
    )
    seeded = service.write_memory_cell(
        "orion-smoke",
        {"memory_system": "semantic", "content": "Orion memory should be retrieved before frozen inference.", "summary": "Orion retrieval rule"},
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        health = _get_json(f"{base}/health")
        first = _post_json(f"{base}/v1/predict", _smoke_payload("stage74-first"))
        second = _post_json(f"{base}/v1/predict", _smoke_payload("stage74-second"))
        submitted = _post_json(f"{base}/v1/jobs", {"request": _smoke_payload("stage74-job")})
        job_id = submitted["job"]["job_id"]
        import time

        job = _get_json(f"{base}/v1/jobs/{job_id}")
        deadline = time.time() + 5.0
        while job["job"]["status"] in {"queued", "running"} and time.time() < deadline:
            time.sleep(0.02)
            job = _get_json(f"{base}/v1/jobs/{job_id}")
        job_result = _get_json(f"{base}/v1/jobs/{job_id}/result?limit=1")
        memory_admin = _get_json(f"{base}/admin/orion/memory")
    finally:
        service.shutdown()
    store = service.memory_store("orion-smoke")
    summary = {
        "stage": "stage74_orion_memory_service_smoke",
        "health": health,
        "first": first,
        "second": second,
        "job": job,
        "job_result": job_result,
        "memory_admin": memory_admin,
        "memory": store.summary(),
        "stage_gates": {
            "stage71_chain_ready": health.get("ready") is True and health.get("jobs") is not None,
            "semantic_seed_retrieved": seeded.cell_id in first["rows"][0].get("orion_memory", {}).get("retrieved_cell_ids", []),
            "request_trace_written": first["rows"][0].get("orion_memory", {}).get("episodic_cell_id") is not None,
            "second_request_reads_session_memory": second["rows"][0].get("orion_memory", {}).get("injected_item_count", 0) >= 1,
            "async_job_uses_memory_hook": job_result.get("result", {}).get("rows", [{}])[0].get("orion_memory") is not None,
            "memory_admin_visible": memory_admin.get("orion_memory", {}).get("session_count") == 1,
            "trace_written": store.summary()["trace_count"] > 0,
        },
    }
    summary["passes_stage_gate"] = all(summary["stage_gates"].values())
    store.write_artifacts(output, summary=summary)
    return summary
