from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .stage61_async_jobs import Stage61JobConfig, _get_json, _payload, _post_json, wait_for_job
from .stage62_persistent_async_jobs import Stage62PersistenceConfig
from .stage63_job_retention_listing import Stage63RetentionConfig
from .stage64_external_result_store import Stage64ResultStoreConfig
from .stage65_batch_jobs import Stage65BatchConfig
from .stage66_batch_export import Stage66ExportConfig
from .stage67_export_lifecycle import Stage67ExportLifecycleConfig
from .stage68_export_package_delivery import Stage68PackageConfig
from .stage69_streaming_package_delivery import Stage69StreamingConfig
from .stage71_if_range_package_delivery import build_stage71_fake_service


DEFAULT_OUTPUT_DIR = Path("experiments/civilization_transformer_qwen3/artifacts/stage72_resumable_download_client")


@dataclass(frozen=True)
class Stage72DownloadConfig:
    chunk_bytes: int = 64 * 1024
    timeout_seconds: float = 30.0
    max_attempts: int = 3


@dataclass
class Stage72DownloadMetrics:
    attempts: int = 0
    full_downloads: int = 0
    resume_attempts: int = 0
    resumed_downloads: int = 0
    stale_etag_restarts: int = 0
    range_rejected_restarts: int = 0
    sha256_verified: int = 0
    partial_saves: int = 0
    bytes_written: int = 0
    last_status: int | None = None
    last_etag: str | None = None
    last_download_mode: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage72DownloadResult:
    status: str
    output_path: str
    partial_path: str
    metadata_path: str
    expected_sha256: str | None
    actual_sha256: str | None
    size_bytes: int
    etag: str | None
    metrics: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Stage72DownloadError(RuntimeError):
    pass


class Stage72ResumableDownloadClient:
    def __init__(self, config: Stage72DownloadConfig = Stage72DownloadConfig()) -> None:
        if config.chunk_bytes < 1:
            raise ValueError("chunk_bytes must be >= 1")
        if config.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.config = config
        self.metrics = Stage72DownloadMetrics()

    def metadata_path_for(self, output_path: str | Path) -> Path:
        return Path(output_path).with_suffix(Path(output_path).suffix + ".stage72.json")

    def partial_path_for(self, output_path: str | Path) -> Path:
        return Path(output_path).with_suffix(Path(output_path).suffix + ".part")

    def _read_sidecar(self, metadata_path: Path) -> dict[str, Any]:
        if not metadata_path.exists():
            return {}
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_sidecar(self, metadata_path: Path, payload: dict[str, Any]) -> None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _expected_from_package(self, package_url: str, headers: dict[str, str] | None) -> tuple[str | None, int | None]:
        request = Request(package_url, headers=headers or {}, method="GET")
        with urlopen(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310 - local/service client
            payload = json.loads(response.read().decode("utf-8"))
        package = payload.get("package", {})
        expected_sha = package.get("sha256")
        expected_size = package.get("size_bytes")
        return expected_sha, int(expected_size) if expected_size is not None else None

    def _open_download(self, download_url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], Any]:
        request = Request(download_url, headers=headers, method="GET")
        response = urlopen(request, timeout=self.config.timeout_seconds)  # noqa: S310 - local/service client
        return response.status, dict(response.headers), response

    def download(
        self,
        *,
        download_url: str,
        output_path: str | Path,
        package_url: str | None = None,
        expected_sha256: str | None = None,
        expected_size_bytes: int | None = None,
        headers: dict[str, str] | None = None,
        stop_after_bytes: int | None = None,
    ) -> Stage72DownloadResult:
        """Download a Stage71 package with local .part resume and If-Range validation.

        `stop_after_bytes` is intentionally exposed for deterministic smoke/tests; it
        simulates an interrupted transfer while preserving the sidecar metadata needed
        for the next call to resume.
        """
        request_headers = dict(headers or {})
        output = Path(output_path)
        partial = self.partial_path_for(output)
        sidecar = self.metadata_path_for(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        if package_url and (expected_sha256 is None or expected_size_bytes is None):
            package_sha, package_size = self._expected_from_package(package_url, request_headers)
            expected_sha256 = expected_sha256 or package_sha
            expected_size_bytes = expected_size_bytes or package_size

        if output.exists() and expected_sha256 and self._sha256(output) == expected_sha256:
            self.metrics.sha256_verified += 1
            return Stage72DownloadResult(
                status="already_verified",
                output_path=str(output),
                partial_path=str(partial),
                metadata_path=str(sidecar),
                expected_sha256=expected_sha256,
                actual_sha256=expected_sha256,
                size_bytes=output.stat().st_size,
                etag=self._read_sidecar(sidecar).get("etag"),
                metrics=self.metrics.snapshot(),
            )

        for _ in range(self.config.max_attempts):
            self.metrics.attempts += 1
            sidecar_payload = self._read_sidecar(sidecar)
            resume_from = partial.stat().st_size if partial.exists() else 0
            attempt_headers = dict(request_headers)
            if resume_from > 0 and sidecar_payload.get("etag"):
                attempt_headers["Range"] = f"bytes={resume_from}-"
                attempt_headers["If-Range"] = str(sidecar_payload["etag"])
                self.metrics.resume_attempts += 1

            try:
                status, response_headers, response = self._open_download(download_url, attempt_headers)
            except HTTPError as error:
                if error.code == 416 and partial.exists():
                    self.metrics.range_rejected_restarts += 1
                    partial.unlink(missing_ok=True)
                    self._write_sidecar(sidecar, {**sidecar_payload, "restart_reason": "range_rejected_416"})
                    continue
                raise

            with response:
                self.metrics.last_status = status
                self.metrics.last_etag = response_headers.get("ETag")
                self.metrics.last_download_mode = response_headers.get("X-Download-Mode")
                if expected_sha256 is None:
                    expected_sha256 = response_headers.get("X-Package-Sha256")
                if expected_size_bytes is None and response_headers.get("Content-Length"):
                    expected_size_bytes = int(response_headers["Content-Length"])

                append = status == 206 and resume_from > 0
                if append:
                    self.metrics.resumed_downloads += 1
                    mode = "ab"
                else:
                    if resume_from > 0:
                        self.metrics.stale_etag_restarts += 1
                    else:
                        self.metrics.full_downloads += 1
                    mode = "wb"

                bytes_this_call = 0
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(self.config.chunk_bytes)
                        if not chunk:
                            break
                        handle.write(chunk)
                        bytes_this_call += len(chunk)
                        self.metrics.bytes_written += len(chunk)
                        if stop_after_bytes is not None and bytes_this_call >= stop_after_bytes:
                            handle.flush()
                            self.metrics.partial_saves += 1
                            self._write_sidecar(
                                sidecar,
                                {
                                    "download_url": download_url,
                                    "package_url": package_url,
                                    "etag": response_headers.get("ETag"),
                                    "expected_sha256": expected_sha256,
                                    "expected_size_bytes": expected_size_bytes,
                                    "partial_size_bytes": partial.stat().st_size,
                                    "updated_at": time.time(),
                                    "status": "partial",
                                    "last_headers": response_headers,
                                },
                            )
                            return Stage72DownloadResult(
                                status="partial",
                                output_path=str(output),
                                partial_path=str(partial),
                                metadata_path=str(sidecar),
                                expected_sha256=expected_sha256,
                                actual_sha256=None,
                                size_bytes=partial.stat().st_size,
                                etag=response_headers.get("ETag"),
                                metrics=self.metrics.snapshot(),
                                headers=response_headers,
                            )

            actual_sha = self._sha256(partial)
            actual_size = partial.stat().st_size
            self._write_sidecar(
                sidecar,
                {
                    "download_url": download_url,
                    "package_url": package_url,
                    "etag": response_headers.get("ETag"),
                    "expected_sha256": expected_sha256,
                    "expected_size_bytes": expected_size_bytes,
                    "partial_size_bytes": actual_size,
                    "actual_sha256": actual_sha,
                    "updated_at": time.time(),
                    "status": "downloaded_unverified",
                    "last_headers": response_headers,
                },
            )
            if expected_size_bytes is not None and actual_size != expected_size_bytes:
                continue
            if expected_sha256 is not None and actual_sha != expected_sha256:
                partial.unlink(missing_ok=True)
                self._write_sidecar(
                    sidecar,
                    {
                        "download_url": download_url,
                        "package_url": package_url,
                        "etag": response_headers.get("ETag"),
                        "expected_sha256": expected_sha256,
                        "expected_size_bytes": expected_size_bytes,
                        "actual_sha256": actual_sha,
                        "actual_size_bytes": actual_size,
                        "updated_at": time.time(),
                        "status": "sha256_mismatch_restart",
                    },
                )
                continue

            partial.replace(output)
            self.metrics.sha256_verified += 1
            self._write_sidecar(
                sidecar,
                {
                    "download_url": download_url,
                    "package_url": package_url,
                    "etag": response_headers.get("ETag"),
                    "expected_sha256": expected_sha256,
                    "expected_size_bytes": expected_size_bytes,
                    "actual_sha256": actual_sha,
                    "actual_size_bytes": output.stat().st_size,
                    "updated_at": time.time(),
                    "status": "verified",
                    "last_headers": response_headers,
                },
            )
            return Stage72DownloadResult(
                status="verified",
                output_path=str(output),
                partial_path=str(partial),
                metadata_path=str(sidecar),
                expected_sha256=expected_sha256,
                actual_sha256=actual_sha,
                size_bytes=output.stat().st_size,
                etag=response_headers.get("ETag"),
                metrics=self.metrics.snapshot(),
                headers=response_headers,
            )

        raise Stage72DownloadError(f"download failed after {self.config.max_attempts} attempts")


def _build_stage72_fake_service(output: Path):
    return build_stage71_fake_service(
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


def run_stage72_resumable_download_smoke(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    service = _build_stage72_fake_service(output)
    server = service.start_background()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    export_id = "stage72-smoke-export"
    try:
        requests = [_payload(f"stage72-job-{index}") for index in range(4)]
        submitted = _post_json(f"{base}/v1/jobs/batch", {"requests": requests})
        job_ids = [item["job"]["job_id"] for item in submitted.get("batch", {}).get("accepted", [])]
        completed = [wait_for_job(base, job_id) for job_id in job_ids]
        exported = _post_json(f"{base}/v1/jobs/export", {"job_ids": job_ids, "export_id": export_id})
        packaged = _post_json(f"{base}/v1/jobs/export/{export_id}/package", {})

        client = Stage72ResumableDownloadClient(Stage72DownloadConfig(chunk_bytes=32, max_attempts=3))
        target = output / "downloads" / f"{export_id}.tar.gz"
        package_url = f"{base}/v1/jobs/export/{export_id}/package"
        download_url = f"{base}/v1/jobs/export/{export_id}/download"
        first = client.download(download_url=download_url, package_url=package_url, output_path=target, stop_after_bytes=64)
        second = client.download(download_url=download_url, package_url=package_url, output_path=target)

        stale_target = output / "downloads" / f"{export_id}-stale.tar.gz"
        stale_client = Stage72ResumableDownloadClient(Stage72DownloadConfig(chunk_bytes=32, max_attempts=3))
        stale_first = stale_client.download(download_url=download_url, package_url=package_url, output_path=stale_target, stop_after_bytes=64)
        stale_sidecar = stale_client.metadata_path_for(stale_target)
        stale_payload = json.loads(stale_sidecar.read_text(encoding="utf-8"))
        stale_payload["etag"] = "\"stale\""
        stale_sidecar.write_text(json.dumps(stale_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        stale_second = stale_client.download(download_url=download_url, package_url=package_url, output_path=stale_target)
        admin_jobs = _get_json(f"{base}/admin/jobs")
    finally:
        service.shutdown()

    stage_gates = {
        "batch_submit_accepted": submitted.get("batch", {}).get("accepted_count") == 4,
        "all_jobs_completed": all(item.get("job", {}).get("status") == "completed" for item in completed),
        "export_created": exported.get("status") == "created",
        "package_created": packaged.get("status") == "created",
        "partial_saved": first.status == "partial" and first.size_bytes > 0,
        "resume_verified": second.status == "verified" and second.actual_sha256 == second.expected_sha256,
        "resume_used_if_range": second.metrics.get("resumed_downloads", 0) >= 1,
        "stale_restart_verified": stale_second.status == "verified" and stale_second.actual_sha256 == stale_second.expected_sha256,
        "stale_restart_recorded": stale_second.metrics.get("stale_etag_restarts", 0) >= 1,
        "final_file_exists": target.exists() and stale_target.exists(),
        "service_if_range_visible": admin_jobs.get("jobs", {})
        .get("exports", {})
        .get("if_range_delivery", {})
        .get("metrics", {})
        .get("if_range_checks", 0)
        >= 2,
    }
    summary = {
        "stage": "stage72_resumable_download_client_smoke",
        "submitted": submitted,
        "completed": completed,
        "exported": exported,
        "packaged": packaged,
        "partial_result": first.to_dict(),
        "resumed_result": second.to_dict(),
        "stale_partial_result": stale_first.to_dict(),
        "stale_resumed_result": stale_second.to_dict(),
        "admin_jobs": admin_jobs,
        "stage_gates": stage_gates,
        "passes_stage_gate": all(stage_gates.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
