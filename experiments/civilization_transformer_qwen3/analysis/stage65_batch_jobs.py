from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .stage49_persistent_inference_service import Stage49ServiceConfig
from .stage51_runtime_management import Stage51RuntimeDescriptor
from .stage59_access_control_audit import Stage59SecurityConfig
from .stage60_queue_rate_limit import Stage60QueueConfig, Stage60SlowFakeRuntime, build_stage60_real_service
from .stage61_async_jobs import Stage61JobConfig, _get_json, _payload, _post_json, wait_for_job
from .stage62_persistent_async_jobs import Stage62PersistenceConfig
from .stage63_job_retention_listing import Stage63RetentionConfig
from .stage64_external_result_store import (
    DEFAULT_JOB_LOG as DEFAULT_STAGE64_JOB_LOG,
    DEFAULT_JOB_STATE as DEFAULT_STAGE64_JOB_STATE,
    DEFAULT_OUTPUT_DIR as DEFAULT_STAGE64_OUTPUT_DIR,
    DEFAULT_RESULT_DIR as DEFAULT_STAGE64_RESULT_DIR,
    Stage64ExternalResultStoreService,
    Stage64ResultStoreConfig,
)


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage65_batch_jobs")
DEFAULT_JOB_STATE = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage65_jobs_state.json")
DEFAULT_JOB_LOG = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage65_jobs.jsonl")
DEFAULT_RESULT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/logs/stage65_job_results")


@dataclass(frozen=True)
class Stage65BatchConfig:
    max_batch_submit: int = 64
    max_batch_result_jobs: int = 64


@dataclass
class Stage65BatchMetrics:
    batch_submissions: int = 0
    batch_jobs_accepted: int = 0
    batch_jobs_rejected: int = 0
    batch_result_requests: int = 0
    batch_result_jobs_returned: int = 0

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


class Stage65BatchJobService(Stage64ExternalResultStoreService):
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
    ) -> None:
        if batch_config.max_batch_submit < 1:
            raise ValueError("max_batch_submit must be >= 1")
        if batch_config.max_batch_result_jobs < 1:
            raise ValueError("max_batch_result_jobs must be >= 1")
        self.batch_config = batch_config
        self.batch_metrics = Stage65BatchMetrics()
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
        )

    def submit_jobs_batch(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        if len(requests) > self.batch_config.max_batch_submit:
            raise OverflowError(f"batch contains {len(requests)} jobs; max_batch_submit={self.batch_config.max_batch_submit}")
        self.batch_metrics.batch_submissions += 1
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for index, payload in enumerate(requests):
            if not isinstance(payload, dict):
                rejected.append({"index": index, "error_type": "InvalidJobRequest", "error": "job request must be an object"})
                continue
            try:
                record = self.submit_job(payload)
                accepted.append({"index": index, "job": record.public(include_payload=True, security=self.security)})
            except Exception as error:
                rejected.append({"index": index, "error_type": type(error).__name__, "error": str(error)})
        self.batch_metrics.batch_jobs_accepted += len(accepted)
        self.batch_metrics.batch_jobs_rejected += len(rejected)
        self._append_job_event(
            {
                "event": "jobs_batch_submitted",
                "accepted": len(accepted),
                "rejected": len(rejected),
            }
        )
        return {
            "accepted": accepted,
            "rejected": rejected,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
        }

    def read_results_batch(self, job_ids: list[str], *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        if len(job_ids) > self.batch_config.max_batch_result_jobs:
            raise OverflowError(
                f"batch contains {len(job_ids)} job ids; max_batch_result_jobs={self.batch_config.max_batch_result_jobs}"
            )
        self.batch_metrics.batch_result_requests += 1
        results: list[dict[str, Any]] = []
        for job_id in job_ids:
            page = self.read_result_page(str(job_id), offset=offset, limit=limit)
            if page is None:
                results.append({"job_id": str(job_id), "status": "not_found", "result_available": False, "rows": []})
            else:
                results.append(page)
        self.batch_metrics.batch_result_jobs_returned += len(results)
        return {
            "job_ids": [str(job_id) for job_id in job_ids],
            "offset": max(0, offset),
            "limit": max(1, min(limit, self.result_store_config.max_result_page_limit)),
            "results": results,
        }

    def batch_status(self) -> dict[str, Any]:
        return {
            "config": asdict(self.batch_config),
            "metrics": self.batch_metrics.snapshot(),
        }

    def job_status(self) -> dict[str, Any]:
        payload = super().job_status()
        payload["batch"] = self.batch_status()
        return payload

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["stage"] = "stage65_batch_jobs"
        return payload

    def make_handler(self):  # type: ignore[override]
        service = self
        base_handler = super().make_handler()

        class Stage65Handler(base_handler):
            server_version = "Stage65BatchJobQwenService/1.0"

            def do_POST(self) -> None:  # noqa: N802 - stdlib API
                parsed = urlparse(self.path)
                if parsed.path == "/v1/jobs/batch":
                    try:
                        payload = self._read_json()
                    except Exception as error:
                        self._send_json(400, {"status": "error", "error_type": type(error).__name__, "error": str(error)})
                        return
                    if not self._check_access(payload):
                        return
                    requests = payload.get("requests") if isinstance(payload, dict) else None
                    if not isinstance(requests, list):
                        self._send_json(400, {"status": "error", "error": "requests must be a list"})
                        return
                    try:
                        batch = service.submit_jobs_batch(requests)
                    except OverflowError as error:
                        self._send_json(429, {"status": "error", "error_type": "BatchTooLarge", "error": str(error)})
                        return
                    status_code = 202 if batch["rejected_count"] == 0 else 207
                    self._send_json(status_code, {"status": "accepted" if status_code == 202 else "partial", "batch": batch})
                    return
                if parsed.path == "/v1/jobs/results":
                    try:
                        payload = self._read_json()
                    except Exception as error:
                        self._send_json(400, {"status": "error", "error_type": type(error).__name__, "error": str(error)})
                        return
                    if not self._check_access(payload):
                        return
                    job_ids = payload.get("job_ids") if isinstance(payload, dict) else None
                    if not isinstance(job_ids, list):
                        self._send_json(400, {"status": "error", "error": "job_ids must be a list"})
                        return
                    try:
                        offset = int(payload.get("offset", 0))
                        limit = int(payload.get("limit", 50))
                    except (TypeError, ValueError):
                        self._send_json(400, {"status": "error", "error": "offset and limit must be integers"})
                        return
                    try:
                        results = service.read_results_batch(job_ids, offset=offset, limit=limit)
                    except OverflowError as error:
                        self._send_json(429, {"status": "error", "error_type": "BatchTooLarge", "error": str(error)})
                        return
                    self._send_json(200, {"status": "ok", "batch_results": results})
                    return
                super().do_POST()

        return Stage65Handler


def build_stage65_fake_service(
    *,
    port: int = 0,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
    persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(job_state_path=str(DEFAULT_JOB_STATE)),
    retention_config: Stage63RetentionConfig = Stage63RetentionConfig(),
    result_store_config: Stage64ResultStoreConfig = Stage64ResultStoreConfig(result_dir=str(DEFAULT_RESULT_DIR)),
    batch_config: Stage65BatchConfig = Stage65BatchConfig(),
    runtime_delay_seconds: float = 0.0,
) -> Stage65BatchJobService:
    return Stage65BatchJobService(
        runtime_factory=lambda: Stage60SlowFakeRuntime(runtime_delay_seconds),
        descriptor=Stage51RuntimeDescriptor(package_manifest="fake", centroid_bundle="fake", label="stage65-fake"),
        config=Stage49ServiceConfig(port=port),
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        persistence_config=persistence_config,
        retention_config=retention_config,
        result_store_config=result_store_config,
        batch_config=batch_config,
    )


def build_stage65_real_service(
    *,
    port: int = 8765,
    security: Stage59SecurityConfig = Stage59SecurityConfig(),
    queue_config: Stage60QueueConfig = Stage60QueueConfig(),
    job_config: Stage61JobConfig = Stage61JobConfig(job_log_path=str(DEFAULT_JOB_LOG)),
    persistence_config: Stage62PersistenceConfig = Stage62PersistenceConfig(job_state_path=str(DEFAULT_JOB_STATE)),
    retention_config: Stage63RetentionConfig = Stage63RetentionConfig(),
    result_store_config: Stage64ResultStoreConfig = Stage64ResultStoreConfig(result_dir=str(DEFAULT_RESULT_DIR)),
    batch_config: Stage65BatchConfig = Stage65BatchConfig(),
    package_manifest: str = "experiments/civilization_transformer_qwen3/artifacts/stage45_adapter_package/package_manifest.json",
    centroid_bundle: str = (
        "experiments/civilization_transformer_qwen3/artifacts/"
        "stage47_centroid_batch_inference_official_full/centroid_bundle_seed_202.pt"
    ),
    model_path: str = "/home/yike/AoNeb-01/Models/Qwen3-0.6B",
    preferred_device: str = "cuda",
    max_length: int = 384,
) -> Stage65BatchJobService:
    base = build_stage60_real_service(
        port=port,
        security=security,
        queue_config=queue_config,
        package_manifest=package_manifest,
        centroid_bundle=centroid_bundle,
        model_path=model_path,
        preferred_device=preferred_device,
        max_length=max_length,
    )
    return Stage65BatchJobService(
        runtime_factory=base._runtime_factory,  # noqa: SLF001 - reuse shared Qwen backend factory
        descriptor=base.descriptor,
        config=base.config,
        security=security,
        queue_config=queue_config,
        job_config=job_config,
        persistence_config=persistence_config,
        retention_config=retention_config,
        result_store_config=result_store_config,
        batch_config=batch_config,
    )


def run_stage65_batch_jobs_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "jobs_state.json"
    job_log = output / "jobs.jsonl"
    result_dir = output / "results"
    service = build_stage65_fake_service(
        port=0,
        job_config=Stage61JobConfig(job_log_path=str(job_log), max_queued_jobs=8, result_ttl_seconds=3600),
        persistence_config=Stage62PersistenceConfig(job_state_path=str(state_path)),
        retention_config=Stage63RetentionConfig(max_persisted_jobs=8, max_list_limit=20),
        result_store_config=Stage64ResultStoreConfig(result_dir=str(result_dir), inline_result_row_limit=0, max_result_page_limit=1),
        batch_config=Stage65BatchConfig(max_batch_submit=4, max_batch_result_jobs=4),
    )
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        requests = [_payload(f"stage65-job-{index}") for index in range(3)]
        submitted = _post_json(f"{base}/v1/jobs/batch", {"requests": requests})
        job_ids = [item["job"]["job_id"] for item in submitted.get("batch", {}).get("accepted", [])]
        completed = [wait_for_job(base, job_id) for job_id in job_ids]
        batch_results = _post_json(f"{base}/v1/jobs/results", {"job_ids": job_ids, "limit": 1})
        admin_jobs = _get_json(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    stage_gates = {
        "batch_submit_accepted": submitted.get("batch", {}).get("accepted_count") == 3,
        "all_jobs_completed": all(item.get("job", {}).get("status") == "completed" for item in completed),
        "batch_results_available": len(batch_results.get("batch_results", {}).get("results", [])) == 3,
        "batch_result_limit_applied": all(
            len(item.get("rows", [])) <= 1 for item in batch_results.get("batch_results", {}).get("results", [])
        ),
        "result_files_written": len(list(result_dir.glob("*.jsonl"))) == 3,
        "batch_metrics_visible": "batch" in admin_jobs.get("jobs", {}),
    }
    summary = {
        "stage": "stage65_batch_jobs_smoke",
        "submitted": submitted,
        "completed": completed,
        "batch_results": batch_results,
        "admin_jobs": admin_jobs,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
