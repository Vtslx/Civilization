"""Dependency-free HTTP client for a deployed Civilization v1 service."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .models import CivilizationRequest, Job, MemorySystem, Prediction


class CivilizationSDKError(RuntimeError):
    """Base SDK exception."""


class CivilizationTransportError(CivilizationSDKError):
    """The service could not be reached or returned invalid transport data."""


class CivilizationAPIError(CivilizationSDKError):
    def __init__(self, status_code: int, method: str, path: str, body: str) -> None:
        self.status_code = status_code
        self.method = method
        self.path = path
        self.body = body
        super().__init__(f"Civilization API {method} {path} returned HTTP {status_code}: {body[:500]}")


class CivilizationInferenceError(CivilizationSDKError):
    """The HTTP request succeeded but the inference row failed validation."""


class CivilizationClient:
    """Client for prediction, memory, job, and export APIs.

    Authentication can be supplied directly or through ``token_env``. The
    token is never included in ``repr`` or exception messages.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        token_env: str | None = None,
        timeout: float = 120.0,
        allow_insecure_http: bool = False,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an absolute HTTP(S) URL without credentials, query, or fragment")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"} and not allow_insecure_http:
            raise ValueError("plain HTTP is allowed only for loopback addresses unless allow_insecure_http=True")
        if token is not None and token_env is not None:
            raise ValueError("provide token or token_env, not both")
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._token_env = token_env
        self.timeout = timeout

    def __repr__(self) -> str:
        auth = "configured" if self._token is not None or self._token_env is not None else "none"
        return f"CivilizationClient(base_url={self.base_url!r}, auth={auth!r}, timeout={self.timeout!r})"

    def _authorization(self) -> str | None:
        token = self._token
        if token is None and self._token_env is not None:
            token = os.environ.get(self._token_env)
        if token is not None and not token.strip():
            raise CivilizationSDKError("configured bearer token is empty")
        return f"Bearer {token}" if token is not None else None

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Accept": "application/json", "User-Agent": "aoneb-civilization-v1-sdk/0.1"}
        authorization = self._authorization()
        if authorization:
            headers["Authorization"] = authorization
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        request = Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - validated URL
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            raise CivilizationAPIError(error.code, method, path, body) from error
        except (URLError, TimeoutError, OSError) as error:
            raise CivilizationTransportError(f"Civilization API {method} {path} failed: {error}") from error
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CivilizationTransportError(f"Civilization API {method} {path} returned invalid JSON") from error
        if not isinstance(decoded, dict):
            raise CivilizationTransportError(f"Civilization API {method} {path} returned a non-object response")
        return decoded

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def ready(self) -> dict[str, Any]:
        return self._request("GET", "/ready")

    def metrics(self) -> dict[str, Any]:
        return self._request("GET", "/metrics")

    def predict_raw(self, request: CivilizationRequest) -> dict[str, Any]:
        return self._request("POST", "/v1/predict", request.to_payload())

    def predict(self, request: CivilizationRequest) -> Prediction:
        payload = self.predict_raw(request)
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise CivilizationInferenceError("prediction response must contain exactly one row")
        try:
            return Prediction.from_row(request, rows[0])
        except (TypeError, ValueError) as error:
            raise CivilizationInferenceError(str(error)) from error

    def batch_raw(self, requests: Sequence[CivilizationRequest]) -> dict[str, Any]:
        values = tuple(requests)
        if not values:
            raise ValueError("requests must not be empty")
        return self._request("POST", "/v1/batch", {"requests": [item.to_payload() for item in values]})

    def batch(self, requests: Sequence[CivilizationRequest]) -> list[Prediction]:
        values = tuple(requests)
        identifiers = [item.request_id for item in values]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("batch request_id values must be unique")
        payload = self.batch_raw(values)
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != len(values):
            raise CivilizationInferenceError("batch response row count does not match request count")
        by_id = {item.request_id: item for item in values}
        predictions: list[Prediction] = []
        try:
            for row in rows:
                if not isinstance(row, Mapping) or str(row.get("id", "")) not in by_id:
                    raise ValueError("batch response contains an unknown request id")
                predictions.append(Prediction.from_row(by_id[str(row["id"])], row))
        except (TypeError, ValueError) as error:
            raise CivilizationInferenceError(str(error)) from error
        return predictions

    def write_memory(
        self,
        *,
        session_id: str,
        memory_system: MemorySystem | str,
        content: str,
        summary: str | None = None,
        confidence: float = 1.0,
        importance: float = 1.0,
        ttl_seconds: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        system = memory_system.value if isinstance(memory_system, MemorySystem) else str(memory_system)
        payload: dict[str, Any] = {
            "session_id": session_id,
            "memory_system": system,
            "content": content,
            "summary": summary if summary is not None else content,
            "confidence": confidence,
            "importance": importance,
            "metadata": dict(metadata or {}),
        }
        if ttl_seconds is not None:
            payload["ttl_seconds"] = ttl_seconds
        response = self._request("POST", "/admin/orion/memory/write", payload)
        return dict(response.get("cell", {}))

    def read_memory(
        self,
        *,
        session_id: str,
        query: str,
        memory_system: MemorySystem | str | None = None,
        include_expired: bool = False,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "query": query,
            "include_expired": include_expired,
            "limit": limit,
        }
        if memory_system is not None:
            payload["memory_system"] = memory_system.value if isinstance(memory_system, MemorySystem) else str(memory_system)
        response = self._request("POST", "/admin/orion/memory/read", payload)
        results = response.get("results", [])
        if not isinstance(results, list):
            raise CivilizationTransportError("memory response results must be a list")
        return [dict(item) for item in results if isinstance(item, Mapping)]

    def consolidate_memory(
        self,
        *,
        session_id: str,
        episodic_cell_ids: Sequence[str],
        summary: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._request(
            "POST",
            "/admin/orion/memory/consolidate",
            {
                "session_id": session_id,
                "episodic_cell_ids": list(episodic_cell_ids),
                "summary": summary,
                "content": content,
                "metadata": dict(metadata or {}),
            },
        )
        return dict(response.get("cell", {}))

    def memory_status(self) -> dict[str, Any]:
        response = self._request("GET", "/admin/orion/memory")
        return dict(response.get("orion_memory", {}))

    def global_store_status(self) -> dict[str, Any]:
        response = self._request("GET", "/admin/orion/global-store/status")
        return dict(response.get("global_store", {}))

    def save_global_store(self) -> dict[str, Any]:
        return self._request("POST", "/admin/orion/global-store/save", {})

    def load_global_store(self) -> dict[str, Any]:
        return self._request("POST", "/admin/orion/global-store/load", {})

    def submit_job(self, request: CivilizationRequest) -> Job:
        return Job.from_payload(self._request("POST", "/v1/jobs", {"request": request.to_payload()}))

    def get_job(self, job_id: str) -> Job:
        return Job.from_payload(self._request("GET", f"/v1/jobs/{job_id}"))

    def wait_job(self, job_id: str, *, timeout: float = 120.0, poll_interval: float = 0.1) -> Job:
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("timeout and poll_interval must be > 0")
        deadline = time.monotonic() + timeout
        while True:
            job = self.get_job(job_id)
            if job.status in {"completed", "failed", "canceled"}:
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(f"job {job_id!r} did not finish within {timeout} seconds")
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))

    def job_result(self, job_id: str, *, offset: int = 0, limit: int = 200) -> dict[str, Any]:
        query = urlencode({"offset": offset, "limit": limit})
        return self._request("GET", f"/v1/jobs/{job_id}/result?{query}")

    def submit_batch_jobs(self, requests: Sequence[CivilizationRequest]) -> dict[str, Any]:
        values = tuple(requests)
        if not values:
            raise ValueError("requests must not be empty")
        return self._request("POST", "/v1/jobs/batch", {"requests": [item.to_payload() for item in values]})

    def create_export(self, job_ids: Sequence[str], *, export_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"job_ids": list(job_ids)}
        if export_id is not None:
            payload["export_id"] = export_id
        return self._request("POST", "/v1/jobs/export", payload)

    def create_package(self, export_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/jobs/export/{export_id}/package", {})

    def get_package(self, export_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/jobs/export/{export_id}/package")

    def download_package(
        self,
        export_id: str,
        output_path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.part")
        headers = {"User-Agent": "aoneb-civilization-v1-sdk/0.1"}
        authorization = self._authorization()
        if authorization:
            headers["Authorization"] = authorization
        request = Request(f"{self.base_url}/v1/jobs/export/{export_id}/download", headers=headers, method="GET")
        digest = hashlib.sha256()
        size = 0
        try:
            with urlopen(request, timeout=self.timeout) as response, temporary.open("wb") as writer:  # noqa: S310
                while chunk := response.read(64 * 1024):
                    writer.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        except HTTPError as error:
            temporary.unlink(missing_ok=True)
            body = error.read().decode("utf-8", "replace")
            raise CivilizationAPIError(error.code, "GET", f"/v1/jobs/export/{export_id}/download", body) from error
        except (URLError, TimeoutError, OSError) as error:
            temporary.unlink(missing_ok=True)
            raise CivilizationTransportError(f"package download failed: {error}") from error
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            temporary.unlink(missing_ok=True)
            raise CivilizationTransportError("downloaded package SHA-256 does not match expected_sha256")
        temporary.replace(destination)
        return {"path": str(destination), "bytes": size, "sha256": actual_sha256, "verified": expected_sha256 is None or actual_sha256 == expected_sha256}
